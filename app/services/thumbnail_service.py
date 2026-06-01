import asyncio
import base64
import io
import logging
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")  # headless 환경 — pyplot 임포트 전에 반드시 설정
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger("app")

_openai = AsyncOpenAI(api_key=settings.openai_api_key)
_PROMPT = (Path(__file__).parent.parent / "prompts" / "room_thumbnail.txt").read_text(encoding="utf-8")

_MAX_TRIANGLES = 8000


def _project_mesh_to_isometric(mesh_path: Path) -> bytes:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0 or len(vertices) == 0:
        raise ValueError("mesh가 비어있음")

    # 아이소메트릭 회전 행렬 (x축 35.26도, z축 45도)
    ax = np.radians(35.264)
    az = np.radians(45.0)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax), np.cos(ax)],
    ])
    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az), np.cos(az), 0],
        [0, 0, 1],
    ])
    projected = (Rx @ Rz @ vertices.T).T  # (N, 3)

    # 삼각형 샘플링
    if len(triangles) > _MAX_TRIANGLES:
        idx = np.random.choice(len(triangles), _MAX_TRIANGLES, replace=False)
        triangles = triangles[idx]

    # LineCollection으로 배치 렌더링 (루프 대신)
    pts = projected[triangles]  # (M, 3, 3) — M개 삼각형, 각 3개 꼭짓점, xyz
    xy = pts[:, :, :2]          # (M, 3, 2) — x, y만 사용
    # 삼각형 3개 엣지: 0→1, 1→2, 2→0
    edges = np.concatenate([
        xy[:, [0, 1]],
        xy[:, [1, 2]],
        xy[:, [2, 0]],
    ], axis=0)  # (3M, 2, 2)

    fig, ax_plot = plt.subplots(figsize=(6, 6), facecolor="white")
    ax_plot.set_aspect("equal")
    ax_plot.axis("off")

    lc = LineCollection(edges, colors="#4a90d9", linewidths=0.15, alpha=0.6)
    ax_plot.add_collection(lc)
    ax_plot.autoscale()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


async def generate_thumbnail(mesh_path: Path) -> bytes:
    logger.info("[Thumbnail] mesh 투영 시작")
    projection_bytes = await asyncio.to_thread(_project_mesh_to_isometric, mesh_path)
    logger.info("[Thumbnail] gpt-image-1 생성 요청")

    response = await _openai.images.edit(
        model="gpt-image-1",
        image=("projection.png", projection_bytes, "image/png"),
        prompt=_PROMPT,
        size="1024x1024",
        response_format="b64_json",
    )

    image_bytes = base64.b64decode(response.data[0].b64_json)
    logger.info("[Thumbnail] 생성 완료 size=%d bytes", len(image_bytes))
    return image_bytes
