import httpx

from app.core.config import settings


async def post_reconstruction_result(scan_id: int, scan_url: str) -> None:
    if not settings.backend_url:
        return
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{settings.backend_url}/scans/{scan_id}/result",
            json={"scan_id": scan_id, "scan_url": scan_url},
            headers={"X-Api-Key": settings.api_key},
            timeout=30,
        )


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
