import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from app.core.callback import post_result
from app.core.storage import download_scan_zip
from app.models.request import DefectComparisonRequest
from app.models.response import APIResponse, DefectComparisonData, DefectItem, ErrorResponse
from app.services.vision_service import compare_defects as _compare_defects
from app.services.vision_service import detect_defects as _detect_defects


async def _resolve_defects(
    defects_json: list[DefectItem] | None,
    scan_url: str | None,
    tmp_path: Path,
    scan_name: str,
) -> list[DefectItem]:
    if defects_json is not None:
        return defects_json
    scan = await asyncio.to_thread(download_scan_zip, scan_url, str(tmp_path / scan_name))
    return await _detect_defects(
        video_path=scan.video_path,
        depth_dir=scan.depth_dir,
        conf_dir=scan.conf_dir,
        camera_matrix_path=scan.camera_matrix_path,
        odometry_path=scan.odometry_path,
    )


router = APIRouter(tags=["AI-D02. 입주/퇴거 하자 비교"])


async def _run(body: DefectComparisonRequest) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            in_defects, out_defects = await asyncio.gather(
                _resolve_defects(body.in_defects_json, body.in_scan_url, tmp_path, "in_scan"),
                _resolve_defects(body.out_defects_json, body.out_scan_url, tmp_path, "out_scan"),
            )

            defects = await _compare_defects(in_defects=in_defects, out_defects=out_defects)

        await post_result(
            body.analysis_id,
            {
                "success": True,
                "code": 200,
                "message": "요청 성공",
                "data": DefectComparisonData(defects=defects).model_dump(),
            },
        )
    except Exception as e:
        await post_result(
            body.analysis_id,
            {
                "success": False,
                "code": 500,
                "message": str(e),
                "error": {"code": "AI_DET_002"},
                "data": None,
            },
        )


@router.post(
    "/defect-comparison",
    summary="AI-D02. 입주/퇴거 하자 비교",
    status_code=202,
    response_model=APIResponse[None],
    responses={
        500: {"model": ErrorResponse, "description": "하자 비교 처리 중 오류 (AI_DET_002)"},
    },
)
async def compare_defects(body: DefectComparisonRequest, background_tasks: BackgroundTasks) -> APIResponse[None]:
    background_tasks.add_task(_run, body)
    return APIResponse(code=202, message="처리 중")
