import zipfile
from pathlib import Path

import boto3
import httpx


def download_to_tempdir(url: str, filename: str, tmp_dir: str) -> Path:
    dest = Path(tmp_dir) / filename
    with httpx.stream("GET", url) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest


def download_and_extract_zip(url: str, extract_dir: str) -> Path:
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    tmp_zip = Path(extract_dir) / "_tmp.zip"
    download_to_tempdir(url, "_tmp.zip", extract_dir)
    out_dir = Path(extract_dir)
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(out_dir)
    tmp_zip.unlink()
    return out_dir


_CONTENT_TYPES = {
    ".ply": "application/octet-stream",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


_PRESIGNED_EXPIRES = 86400  # 24h (S3 lifecycle 1일과 일치)


def upload_to_s3(data: bytes, key: str) -> str:
    from app.core.config import settings

    boto_kwargs = {}
    if settings.aws_access_key_id:
        boto_kwargs["aws_access_key_id"] = settings.aws_access_key_id
        boto_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

    s3 = boto3.client(
        "s3",
        region_name=settings.s3_region,
        endpoint_url=f"https://s3.{settings.s3_region}.amazonaws.com",
        **boto_kwargs,
    )
    s3.put_object(
        Bucket=settings.s3_bucket_name,
        Key=key,
        Body=data,
        ContentType=_CONTENT_TYPES.get(Path(key).suffix.lower(), "application/octet-stream"),
    )
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket_name, "Key": key},
        ExpiresIn=_PRESIGNED_EXPIRES,
    )
