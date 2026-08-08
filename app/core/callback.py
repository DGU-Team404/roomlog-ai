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


async def post_reconstruction_result(
    callback_url: str, scan_id: int, scan_url: str, thumbnail_url: str | None = None
) -> None:
    """R01 1차 콜백 — 백엔드가 요청 시 전달한 callback_url로 직접 POST"""
    logger.info("[R01 콜백] url=%s scan_id=%s thumbnail=%s", callback_url, scan_id, thumbnail_url is not None)
    payload: dict = {"scan_id": scan_id, "scan_url": scan_url}
    if thumbnail_url is not None:
        payload["thumbnail_url"] = thumbnail_url
    await _post_callback(url=callback_url, payload=payload, label="R01 콜백")


async def post_thumbnail_result(callback_url: str, scan_id: int, thumbnail_url: str) -> None:
    """R01 2차 콜백 — mesh 완료 후 늦게 생성된 썸네일만 전달"""
    logger.info("[R01 썸네일 콜백] url=%s scan_id=%s", callback_url, scan_id)
    await _post_callback(
        url=callback_url,
        payload={"scan_id": scan_id, "thumbnail_url": thumbnail_url},
        label="R01 썸네일 콜백",
    )


async def post_result(callback_url: str, payload: dict) -> None:
    """D01/D02 콜백 — 백엔드가 요청 시 전달한 callback_url로 직접 POST"""
    logger.info("[D01/D02 콜백] url=%s", callback_url)
    await _post_callback(
        url=callback_url,
        payload=payload,
        label="D01/D02 콜백",
    )
