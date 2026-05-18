import asyncio
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.storage import download_and_extract_zip, download_to_tempdir, upload_to_s3
from app.models.request import ReconstructionRequest
from app.models.response import APIResponse, ErrorResponse, ReconstructionData
from app.services.tsdf_service import run_reconstruction

router = APIRouter(tags=["AI-R01. 3D 재구성"])


@router.post(
    "/reconstruction",
    summary="AI-R01. 3D 재구성",
    response_model=APIResponse[ReconstructionData],
    responses={
        500: {"model": ErrorResponse, "description": "재구성 처리 중 오류 (AI_REC_001)"},
    },
)
async def reconstruct(body: ReconstructionRequest) -> APIResponse[ReconstructionData]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            video_path, depth_dir, conf_dir, camera_matrix_path, odometry_path = await asyncio.gather(
                asyncio.to_thread(download_to_tempdir, body.rgb_video_url, "video.mp4", tmp),
                asyncio.to_thread(download_and_extract_zip, body.depth_dir_url, str(tmp_path / "depth")),
                asyncio.to_thread(download_and_extract_zip, body.confidence_dir_url, str(tmp_path / "conf")),
                asyncio.to_thread(download_to_tempdir, body.camera_matrix_url, "camera_matrix.csv", tmp),
                asyncio.to_thread(download_to_tempdir, body.odometry_url, "odometry.csv", tmp),
            )

            pcd_path, mesh_path = await asyncio.to_thread(
                run_reconstruction,
                video_path,
                depth_dir,
                conf_dir,
                camera_matrix_path,
                odometry_path,
                tmp_path / "output",
            )

            prefix = f"scans/{body.scan_id}/{uuid.uuid4().hex}"
            pcd_url, mesh_url = await asyncio.gather(
                asyncio.to_thread(upload_to_s3, pcd_path.read_bytes(), f"{prefix}/point_cloud.ply"),
                asyncio.to_thread(upload_to_s3, mesh_path.read_bytes(), f"{prefix}/mesh.ply"),
            )

        return APIResponse(
            data=ReconstructionData(
                scan_id=body.scan_id,
                point_cloud_url=pcd_url,
                mesh_url=mesh_url,
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "code": 500, "message": str(e), "error": {"code": "AI_REC_001"}, "data": None},
        ) from e
