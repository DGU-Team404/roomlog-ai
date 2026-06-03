import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from app.core.callback import post_reconstruction_result
from app.core.storage import download_scan_zip, upload_to_s3
from app.models.request import ReconstructionRequest
from app.models.response import APIResponse, ErrorResponse
from app.services.thumbnail_service import generate_thumbnail
from app.services.tsdf_service import run_reconstruction

logger = logging.getLogger(__name__)
router = APIRouter(tags=["AI-R01. 3D 재구성"])


async def _run(body: ReconstructionRequest) -> None:
    logger.info("[R01] 시작 scan_id=%s", body.scan_id)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            logger.info("[R01] ZIP 다운로드 중")
            scan = await asyncio.to_thread(download_scan_zip, body.scan_url, str(tmp_path / "scan"))

            logger.info("[R01] 재구성 처리 중")
            pcd_path, mesh_path = await asyncio.to_thread(
                run_reconstruction,
                scan.video_path,
                scan.depth_dir,
                scan.conf_dir,
                scan.camera_matrix_path,
                scan.odometry_path,
                tmp_path / "output",
            )

            logger.info("[R01] S3 업로드 중")
            prefix = f"scans/{body.scan_id}/{uuid.uuid4().hex}"
            _, mesh_url = await asyncio.gather(
                asyncio.to_thread(upload_to_s3, pcd_path.read_bytes(), f"{prefix}/point_cloud.ply"),
                asyncio.to_thread(upload_to_s3, mesh_path.read_bytes(), f"{prefix}/mesh.ply"),
            )

            logger.info("[R01] 썸네일 생성 중")
            thumbnail_bytes = await generate_thumbnail(mesh_path)
            thumbnail_url = await asyncio.to_thread(
                upload_to_s3, thumbnail_bytes, f"{prefix}/thumbnail.jpg"
            )
            logger.info("[R01] 썸네일 S3 업로드 완료")

        logger.info("[R01] 콜백 전송 중 mesh_url=%s", mesh_url)
        await post_reconstruction_result(body.callback_url, body.scan_id, mesh_url, thumbnail_url)
        logger.info("[R01] 완료 scan_id=%s", body.scan_id)
    except Exception as e:
        logger.error("[R01] 실패 scan_id=%s error=%s", body.scan_id, e, exc_info=True)


@router.post(
    "/reconstruction",
    summary="AI-R01. 3D 재구성",
    status_code=202,
    response_model=APIResponse[None],
    responses={
        500: {"model": ErrorResponse, "description": "재구성 처리 중 오류 (AI_REC_001)"},
    },
)
async def reconstruct(body: ReconstructionRequest, background_tasks: BackgroundTasks) -> APIResponse[None]:
    background_tasks.add_task(_run, body)
    return APIResponse(code=202, message="처리 중")
