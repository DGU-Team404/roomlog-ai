from pydantic import BaseModel, Field


class ReconstructionRequest(BaseModel):
    scan_id: int = Field(description="백엔드 스캔 ID")
    rgb_video_url: str = Field(description="RGB 영상 URL (.mp4)")
    depth_dir_url: str = Field(description="Depth 이미지 디렉토리 URL")
    confidence_dir_url: str = Field(description="Confidence 이미지 디렉토리 URL")
    camera_matrix_url: str = Field(description="카메라 파라미터 CSV URL")
    odometry_url: str = Field(description="오도메트리 CSV URL")
    imu_url: str = Field(description="IMU 데이터 CSV URL")


class DefectDetectionRequest(BaseModel):
    scan_id: int = Field(description="백엔드 스캔 ID")
    rgb_video_url: str = Field(description="RGB 영상 URL (.mp4)")
    depth_dir_url: str = Field(description="Depth 이미지 디렉토리 URL")
    confidence_dir_url: str = Field(description="Confidence 이미지 디렉토리 URL")
    camera_matrix_url: str = Field(description="카메라 파라미터 CSV URL")
    odometry_url: str = Field(description="오도메트리 CSV URL")


class DefectComparisonRequest(BaseModel):
    in_scan_id: int = Field(description="입주 시 스캔 ID")
    in_rgb_video_url: str = Field(description="입주 시 RGB 영상 URL (.mp4)")
    in_depth_dir_url: str = Field(description="입주 시 Depth 이미지 디렉토리 URL")
    in_confidence_dir_url: str = Field(description="입주 시 Confidence 이미지 디렉토리 URL")
    in_camera_matrix_url: str = Field(description="입주 시 카메라 파라미터 CSV URL")
    in_odometry_url: str = Field(description="입주 시 오도메트리 CSV URL")
    out_scan_id: int = Field(description="퇴거 시 스캔 ID")
    out_rgb_video_url: str = Field(description="퇴거 시 RGB 영상 URL (.mp4)")
    out_depth_dir_url: str = Field(description="퇴거 시 Depth 이미지 디렉토리 URL")
    out_confidence_dir_url: str = Field(description="퇴거 시 Confidence 이미지 디렉토리 URL")
    out_camera_matrix_url: str = Field(description="퇴거 시 카메라 파라미터 CSV URL")
    out_odometry_url: str = Field(description="퇴거 시 오도메트리 CSV URL")
