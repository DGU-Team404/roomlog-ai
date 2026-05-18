import tempfile
from pathlib import Path

from fastapi import APIRouter

from app.core.storage import download_and_extract_zip, download_to_tempdir
from app.models.request import DefectComparisonRequest
from app.models.response import APIResponse, DefectComparisonData, ErrorResponse
from app.services.vision_service import compare_defects as _compare_defects

router = APIRouter(tags=["AI-D02. 입주/퇴거 하자 비교"])


@router.post(
    "/defect-comparison",
    summary="AI-D02. 입주/퇴거 하자 비교",
    response_model=APIResponse[DefectComparisonData],
    responses={
        500: {"model": ErrorResponse, "description": "하자 비교 처리 중 오류 (AI_DET_002)"},
    },
)
async def compare_defects(body: DefectComparisonRequest) -> APIResponse[DefectComparisonData]:
    """입주/퇴거 원시 데이터 각각에 하자 탐지를 적용한 뒤 신규 발생 하자(퇴거 - 입주)만 반환"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        in_video = download_to_tempdir(body.in_rgb_video_url, "in_rgb.mp4", tmp)
        in_camera = download_to_tempdir(body.in_camera_matrix_url, "in_camera_matrix.csv", tmp)
        in_odom = download_to_tempdir(body.in_odometry_url, "in_odometry.csv", tmp)
        in_depth = download_and_extract_zip(body.in_depth_dir_url, str(base / "in_depth"))
        in_conf = download_and_extract_zip(body.in_confidence_dir_url, str(base / "in_confidence"))

        out_video = download_to_tempdir(body.out_rgb_video_url, "out_rgb.mp4", tmp)
        out_camera = download_to_tempdir(body.out_camera_matrix_url, "out_camera_matrix.csv", tmp)
        out_odom = download_to_tempdir(body.out_odometry_url, "out_odometry.csv", tmp)
        out_depth = download_and_extract_zip(body.out_depth_dir_url, str(base / "out_depth"))
        out_conf = download_and_extract_zip(body.out_confidence_dir_url, str(base / "out_confidence"))

        defects = await _compare_defects(
            in_video_path=in_video,
            in_depth_dir=in_depth,
            in_conf_dir=in_conf,
            in_camera_matrix_path=in_camera,
            in_odometry_path=in_odom,
            out_video_path=out_video,
            out_depth_dir=out_depth,
            out_conf_dir=out_conf,
            out_camera_matrix_path=out_camera,
            out_odometry_path=out_odom,
        )

    return APIResponse(data=DefectComparisonData(defects=defects))
