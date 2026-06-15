import asyncio
import base64
import gc
import json
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger("app")

import cv2
import fal_client
import numpy as np
from openai import AsyncOpenAI
from scipy.spatial.transform import Rotation

from app.core.config import settings
from app.core.storage import upload_to_s3
from app.models.response import ComparisonDefectItem, DefectItem, DefectType, Point3D, Severity
from app.services.frame_service import extract_frames
from app.services.pose_service import polygon_to_3d

_openai = AsyncOpenAI(api_key=settings.openai_api_key)
os.environ.setdefault("FAL_KEY", settings.fal_api_key)

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "defect_detection.txt").read_text(
    encoding="utf-8"
)

_MAX_FRAMES = 20
_MATCH_THRESHOLD_M = 0.3


def _build_poses(odometry_path: Path) -> dict[int, np.ndarray]:
    data = np.loadtxt(odometry_path, delimiter=",", skiprows=1)
    poses: dict[int, np.ndarray] = {}
    for row in data:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[5:]).as_matrix()
        T[:3, 3] = row[2:5]
        poses[int(row[1])] = T
    return poses


def _load_depth(depth_dir: Path, conf_dir: Path, frame_idx: int) -> np.ndarray | None:
    depth = cv2.imread(str(depth_dir / f"{frame_idx:06d}.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth_m = depth.astype(np.float32) / 1000.0
    conf_path = conf_dir / f"{frame_idx:06d}.png"
    if conf_path.exists():
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        depth_m[conf < 1] = 0.0
    return depth_m



def _centroid(region_3d: list[Point3D]) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in region_3d])
    return pts.mean(axis=0)



async def _gpt_detect(frames: list[tuple[int, str]]) -> list[dict]:
    content: list = []
    for frame_idx, data_uri in frames:
        content.append({"type": "text", "text": f"[frame_index: {frame_idx}]"})
        content.append({"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}})

    response = await _openai.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content).get("defects", [])


def _sam3_segment(image_bytes: bytes, bbox: list[float]) -> list[tuple[float, float]]:
    import httpx

    image_url = fal_client.upload(image_bytes, "image/jpeg")
    result = fal_client.subscribe(
        "fal-ai/sam-3/image",
        arguments={
            "image_url": image_url,
            "box_prompts": [{"x_min": int(bbox[0]), "y_min": int(bbox[1]), "x_max": int(bbox[2]), "y_max": int(bbox[3])}],
            "apply_mask": False,
            "output_format": "png",
        },
    )

    masks = result.get("masks", [])
    if not masks:
        return []

    mask_bytes = httpx.get(masks[0]["url"]).content
    mask_arr = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask_arr is None:
        return []

    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 3] if mask_arr.shape[2] == 4 else cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask_arr, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


async def _run_detection(
    video_path: Path,
    depth_dir: Path,
    conf_dir: Path,
    camera_matrix_path: Path,
    odometry_path: Path,
    every_n_frames: int = 30,
) -> tuple[list[DefectItem], dict[int, np.ndarray], dict[int, str], np.ndarray]:
    poses = _build_poses(odometry_path)
    K = np.loadtxt(camera_matrix_path, delimiter=",")
    frames = extract_frames(video_path, every_n_frames)

    if len(frames) > _MAX_FRAMES:
        step = len(frames) // _MAX_FRAMES
        frames = frames[::step][:_MAX_FRAMES]

    frame_dict: dict[int, str] = dict(frames)
    raw_defects = await _gpt_detect(frames)
    logger.info("[D01] GPT 탐지 결과 %d개 (프레임 %d개 분석)", len(raw_defects), len(frames))
    del frames

    items: list[DefectItem] = []
    for d in raw_defects:
        frame_idx: int = d["frame_index"]
        bbox: list[float] = d["bbox"]

        data_uri = frame_dict.get(frame_idx)
        T_WC = poses.get(frame_idx)
        if data_uri is None:
            logger.warning("[D01] 프레임 없음 frame_idx=%s (GPT가 잘못된 frame_index 반환)", frame_idx)
            continue
        if T_WC is None:
            logger.warning("[D01] pose 없음 frame_idx=%s — odometry에 해당 프레임 없어 skip", frame_idx)
            continue

        _, encoded = data_uri.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # 이미지 실제 해상도 확인 (SAM3 polygon 좌표계 기준)
        img_arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_arr is None:
            logger.warning("[D01] 이미지 디코딩 실패 frame_idx=%s", frame_idx)
            continue
        img_h, img_w = img_arr.shape[:2]

        polygon_2d = await asyncio.to_thread(_sam3_segment, image_bytes, bbox)
        if not polygon_2d:
            logger.warning("[D01] SAM3 세그멘테이션 실패 frame_idx=%s bbox=%s", frame_idx, bbox)

        depth_map = _load_depth(depth_dir, conf_dir, frame_idx)
        if depth_map is None:
            logger.warning("[D01] depth 파일 없음 frame_idx=%s", frame_idx)
            continue
        depth_h, depth_w = depth_map.shape

        # SAM3 polygon 좌표(이미지 해상도)를 depth map 해상도로 스케일
        sx = depth_w / img_w
        sy = depth_h / img_h
        polygon_depth = [(u * sx, v * sy) for u, v in polygon_2d]

        # camera matrix도 depth map 해상도 기준으로 스케일
        K_depth = K.copy()
        K_depth[0, 0] *= sx
        K_depth[1, 1] *= sy
        K_depth[0, 2] *= sx
        K_depth[1, 2] *= sy

        region_3d = polygon_to_3d(polygon_depth, depth_map, K_depth, T_WC)
        if not region_3d and polygon_2d:
            depth_at_polygon = [float(depth_map[int(v * sy), int(u * sx)]) for u, v in polygon_2d[:3] if 0 <= int(v * sy) < depth_h and 0 <= int(u * sx) < depth_w]
            logger.warning("[D01] region_3d 비어있음 frame_idx=%s polygon=%d개 depth_nonzero=%d depth_sample=%s",
                           frame_idx, len(polygon_2d), int(np.count_nonzero(depth_map)), depth_at_polygon)

        # 하자 이미지 S3 업로드 (bbox 시각화 / 마스크 시각화 / 크롭)
        image_url = None
        try:
            # bbox 시각화 (S3 저장만, 콜백 미포함)
            bbox_vis = img_arr.copy()
            cv2.rectangle(bbox_vis, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
            _, buf = cv2.imencode(".jpg", bbox_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            await asyncio.to_thread(upload_to_s3, buf.tobytes(), f"defects/{uuid.uuid4().hex}_bbox.jpg")

            # SAM3 마스크 시각화 (S3 저장만, 콜백 미포함)
            if polygon_2d:
                mask_vis = img_arr.copy()
                pts = np.array([[int(x), int(y)] for x, y in polygon_2d], dtype=np.int32)
                overlay = mask_vis.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.4, mask_vis, 0.6, 0, mask_vis)
                cv2.polylines(mask_vis, [pts], True, (0, 255, 0), 2)
                _, buf = cv2.imencode(".jpg", mask_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
                await asyncio.to_thread(upload_to_s3, buf.tobytes(), f"defects/{uuid.uuid4().hex}_mask.jpg")

            # 크롭 이미지 (1:1.6 비율, 콜백 포함)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            w = max(bw, bh / 1.6)
            h = w * 1.6
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(img_w, int(cx + w / 2))
            y2 = min(img_h, int(cy + h / 2))
            crop = img_arr[y1:y2, x1:x2]
            if crop.size > 0:
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_url = await asyncio.to_thread(upload_to_s3, buf.tobytes(), f"defects/{uuid.uuid4().hex}.jpg")
        except Exception as e:
            logger.warning("[D01] 하자 이미지 업로드 실패 frame_idx=%s error=%s", frame_idx, e)
        finally:
            del img_arr

        items.append(
            DefectItem(
                type=DefectType(d["type"]),
                severity=Severity(d["severity"]),
                location=d["location"],
                area=float(d["area"]),
                description=d["description"],
                region_3d=region_3d,
                image_url=image_url,
            )
        )

    gc.collect()
    return items, poses, frame_dict, K


async def detect_defects(
    video_path: Path,
    depth_dir: Path,
    conf_dir: Path,
    camera_matrix_path: Path,
    odometry_path: Path,
    every_n_frames: int = 30,
) -> list[DefectItem]:
    items, _, _, _ = await _run_detection(
        video_path, depth_dir, conf_dir, camera_matrix_path, odometry_path, every_n_frames
    )
    return items


async def compare_defects(
    in_defects: list[DefectItem],
    out_defects: list[DefectItem],
) -> list[ComparisonDefectItem]:
    in_centroids = [_centroid(d.region_3d) for d in in_defects if d.region_3d]

    results: list[ComparisonDefectItem] = []
    for d in out_defects:
        if not d.region_3d:
            continue
        out_centroid = _centroid(d.region_3d)

        is_existing = any(
            float(np.linalg.norm(out_centroid - c)) < _MATCH_THRESHOLD_M
            for c in in_centroids
        )
        if is_existing:
            continue

        results.append(
            ComparisonDefectItem(
                type=d.type,
                severity=d.severity,
                location=d.location,
                area=d.area,
                description=d.description,
                region_3d=d.region_3d,
                image_url=d.image_url,
            )
        )

    return results
