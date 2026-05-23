import logging

import httpx

from app.core.config import settings

logger = logging.getLogger("app")


async def _post_callback(url: str, payload: dict, label: str) -> None:
    async with httpx.AsyncClient(follow_redirects=False) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={"X-Api-Key": settings.api_key},
            timeout=30,
        )
        if resp.is_redirect:
            logger.warning("[%s] 리다이렉트 감지 %s → %s", label, resp.status_code, resp.headers.get("location"))
        else:
            logger.info("[%s] 콜백 응답 %s", label, resp.status_code)


async def post_reconstruction_result(scan_id: int, scan_url: str) -> None:
    if not settings.backend_url:
        return
    await _post_callback(
        url=f"{settings.backend_url}/scans/{scan_id}/result",
        payload={"scan_id": scan_id, "scan_url": scan_url},
        label="R01 콜백",
    )


async def post_result(analysis_id: int, payload: dict) -> None:
    if not settings.backend_url:
        return
    await _post_callback(
        url=f"{settings.backend_url}/analyses/{analysis_id}/result",
        payload=payload,
        label="D01/D02 콜백",
    )
