import asyncio
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from app.core.callback import post_result
from app.core.storage import download_scan_zip, upload_to_s3
from app.models.request import ReconstructionRequest
from app.models.response import APIResponse, ErrorResponse, ReconstructionData
from app.services.tsdf_service import run_reconstruction

router = APIRouter(tags=["AI-R01. 3D 재구성"])


async def _run(body: ReconstructionRequest) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            scan = await asyncio.to_thread(download_scan_zip, body.scan_url, str(tmp_path / "scan"))

            pcd_path, mesh_path = await asyncio.to_thread(
                run_reconstruction,
                scan.video_path,
                scan.depth_dir,
                scan.conf_dir,
                scan.camera_matrix_path,
                scan.odometry_path,
                tmp_path / "output",
            )

            prefix = f"scans/{body.scan_id}/{uuid.uuid4().hex}"
            pcd_url, mesh_url = await asyncio.gather(
                asyncio.to_thread(upload_to_s3, pcd_path.read_bytes(), f"{prefix}/point_cloud.ply"),
                asyncio.to_thread(upload_to_s3, mesh_path.read_bytes(), f"{prefix}/mesh.ply"),
            )

        await post_result(
            body.analysis_id,
            {
                "success": True,
                "code": 200,
                "message": "요청 성공",
                "data": ReconstructionData(
                    scan_id=body.scan_id,
                    point_cloud_url=pcd_url,
                    mesh_url=mesh_url,
                ).model_dump(),
            },
        )
    except Exception as e:
        await post_result(
            body.analysis_id,
            {
                "success": False,
                "code": 500,
                "message": str(e),
                "error": {"code": "AI_REC_001"},
                "data": None,
            },
        )


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
