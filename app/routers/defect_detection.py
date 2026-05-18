import tempfile
from pathlib import Path

from fastapi import APIRouter

from app.core.storage import download_and_extract_zip, download_to_tempdir
from app.models.request import DefectDetectionRequest
from app.models.response import APIResponse, DefectDetectionData, ErrorResponse
from app.services.vision_service import detect_defects as _detect_defects

router = APIRouter(tags=["AI-D01. 하자 탐지"])


@router.post(
    "/defect-detection",
    summary="AI-D01. 하자 탐지",
    response_model=APIResponse[DefectDetectionData],
    responses={
        500: {"model": ErrorResponse, "description": "하자 탐지 처리 중 오류 (AI_DET_001)"},
    },
)
async def detect_defects(body: DefectDetectionRequest) -> APIResponse[DefectDetectionData]:
    """RGB 영상에서 프레임을 추출하고 GPT Vision으로 하자를 탐지한 뒤 3D 좌표와 함께 반환"""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = download_to_tempdir(body.rgb_video_url, "rgb.mp4", tmp)
        camera_matrix_path = download_to_tempdir(body.camera_matrix_url, "camera_matrix.csv", tmp)
        odometry_path = download_to_tempdir(body.odometry_url, "odometry.csv", tmp)
        depth_dir = download_and_extract_zip(body.depth_dir_url, str(Path(tmp) / "depth"))
        conf_dir = download_and_extract_zip(body.confidence_dir_url, str(Path(tmp) / "confidence"))

        defects = await _detect_defects(
            video_path=video_path,
            depth_dir=depth_dir,
            conf_dir=conf_dir,
            camera_matrix_path=camera_matrix_path,
            odometry_path=odometry_path,
        )

    return APIResponse(data=DefectDetectionData(defects=defects))
