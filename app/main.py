from fastapi import FastAPI

from app.routers import defect_comparison, defect_detection, reconstruction

tags_metadata = [
    {
        "name": "AI-R01. 3D 재구성",
        "description": "스캔 원시 데이터를 받아 TSDF 기반 3D 재구성 후 결과 반환",
    },
    {
        "name": "AI-D01. 하자 탐지",
        "description": "RGB 영상 원시 데이터를 받아 GPT Vision으로 하자 탐지 후 결과 반환",
    },
    {
        "name": "AI-D02. 입주/퇴거 하자 비교",
        "description": "입주/퇴거 원시 데이터 두 개를 받아 새로 생긴 하자 반환",
    },
]

app = FastAPI(
    title="RoomLog AI API",
    version="0.1.0",
    openapi_tags=tags_metadata,
)

app.include_router(reconstruction.router)
app.include_router(defect_detection.router)
app.include_router(defect_comparison.router)
