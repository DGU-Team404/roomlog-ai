import httpx

from app.core.config import settings


async def post_result(analysis_id: int, payload: dict) -> None:
    if not settings.backend_url:
        return
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{settings.backend_url}/analyses/{analysis_id}/result",
            json=payload,
            headers={"X-Api-Key": settings.api_key},
            timeout=30,
        )
