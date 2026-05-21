import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from app.core.callback import post_result
from app.core.storage import download_scan_zip
from app.models.request import DefectDetectionRequest
from app.models.response import APIResponse, DefectDetectionData, ErrorResponse
from app.services.vision_service import detect_defects as _detect_defects

router = APIRouter(tags=["AI-D01. 하자 탐지"])


async def _run(body: DefectDetectionRequest) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            scan = await asyncio.to_thread(download_scan_zip, body.scan_url, str(tmp_path / "scan"))

            defects = await _detect_defects(
                video_path=scan.video_path,
                depth_dir=scan.depth_dir,
                conf_dir=scan.conf_dir,
                camera_matrix_path=scan.camera_matrix_path,
                odometry_path=scan.odometry_path,
            )

        await post_result(
            body.analysis_id,
            {
                "success": True,
                "code": 200,
                "message": "요청 성공",
                "data": DefectDetectionData(defects=defects).model_dump(),
            },
        )
    except Exception as e:
        await post_result(
            body.analysis_id,
            {
                "success": False,
                "code": 500,
                "message": str(e),
                "error": {"code": "AI_DET_001"},
                "data": None,
            },
        )


@router.post(
    "/defect-detection",
    summary="AI-D01. 하자 탐지",
    status_code=202,
    response_model=APIResponse[None],
    responses={
        500: {"model": ErrorResponse, "description": "하자 탐지 처리 중 오류 (AI_DET_001)"},
    },
)
async def detect_defects(body: DefectDetectionRequest, background_tasks: BackgroundTasks) -> APIResponse[None]:
    background_tasks.add_task(_run, body)
    return APIResponse(code=202, message="처리 중")
