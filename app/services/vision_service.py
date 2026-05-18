import asyncio
import base64
import json
import os
from pathlib import Path

import cv2
import fal_client
import numpy as np
from openai import AsyncOpenAI
from scipy.spatial.transform import Rotation

from app.core.config import settings
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


def _load_depth(depth_dir: Path, conf_dir: Path, frame_idx: int) -> np.ndarray:
    depth = cv2.imread(str(depth_dir / f"{frame_idx:06d}.png"), cv2.IMREAD_UNCHANGED)
    depth_m = depth.astype(np.float32) / 1000.0
    conf_path = conf_dir / f"{frame_idx:06d}.png"
    if conf_path.exists():
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        depth_m[conf < 1] = 0.0
    return depth_m


def _crop_base64(data_uri: str, bbox: list[float]) -> str:
    _, encoded = data_uri.split(",", 1)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop = img[y1:y2, x1:x2]
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


def _centroid(region_3d: list[Point3D]) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in region_3d])
    return pts.mean(axis=0)


def _find_before_image(
    centroid: np.ndarray,
    poses: dict[int, np.ndarray],
    frame_dict: dict[int, str],
    K: np.ndarray,
) -> str:
    best_idx = min(
        (i for i in frame_dict if i in poses),
        key=lambda i: float(np.linalg.norm(poses[i][:3, 3] - centroid)),
        default=None,
    )
    if best_idx is None:
        return ""
    data_uri = frame_dict.get(best_idx)
    if data_uri is None:
        return ""

    T_CW = np.linalg.inv(poses[best_idx])
    p_c = T_CW @ np.append(centroid, 1.0)
    if p_c[2] <= 0:
        return ""

    u = K[0, 0] * p_c[0] / p_c[2] + K[0, 2]
    v = K[1, 1] * p_c[1] / p_c[2] + K[1, 2]

    _, encoded = data_uri.split(",", 1)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    half = 100
    x1, y1 = max(0, int(u) - half), max(0, int(v) - half)
    x2, y2 = min(W, int(u) + half), min(H, int(v) + half)
    crop = img[y1:y2, x1:x2]
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


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

    items: list[DefectItem] = []
    for d in raw_defects:
        frame_idx: int = d["frame_index"]
        bbox: list[float] = d["bbox"]

        data_uri = frame_dict.get(frame_idx)
        T_WC = poses.get(frame_idx)
        if data_uri is None or T_WC is None:
            continue

        _, encoded = data_uri.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        polygon_2d = await asyncio.to_thread(_sam3_segment, image_bytes, bbox)
        depth_map = _load_depth(depth_dir, conf_dir, frame_idx)
        region_3d = polygon_to_3d(polygon_2d, depth_map, K, T_WC)
        image_data = _crop_base64(data_uri, bbox)

        items.append(
            DefectItem(
                type=DefectType(d["type"]),
                severity=Severity(d["severity"]),
                location=d["location"],
                area=float(d["area"]),
                description=d["description"],
                image_data=image_data,
                region_3d=region_3d,
            )
        )

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
    in_video_path: Path,
    in_depth_dir: Path,
    in_conf_dir: Path,
    in_camera_matrix_path: Path,
    in_odometry_path: Path,
    out_video_path: Path,
    out_depth_dir: Path,
    out_conf_dir: Path,
    out_camera_matrix_path: Path,
    out_odometry_path: Path,
    every_n_frames: int = 30,
) -> list[ComparisonDefectItem]:
    (in_defects, in_poses, in_frame_dict, in_K), (out_defects, _, _, _) = await asyncio.gather(
        _run_detection(in_video_path, in_depth_dir, in_conf_dir, in_camera_matrix_path, in_odometry_path, every_n_frames),
        _run_detection(out_video_path, out_depth_dir, out_conf_dir, out_camera_matrix_path, out_odometry_path, every_n_frames),
    )

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

        before_data = _find_before_image(out_centroid, in_poses, in_frame_dict, in_K)
        results.append(
            ComparisonDefectItem(
                type=d.type,
                severity=d.severity,
                location=d.location,
                area=d.area,
                description=d.description,
                before_image_data=before_data,
                after_image_data=d.image_data,
                region_3d=d.region_3d,
            )
        )

    return results
