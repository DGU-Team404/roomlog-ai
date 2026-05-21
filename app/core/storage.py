import zipfile
from pathlib import Path
from typing import NamedTuple

import boto3
import httpx



class ScanFiles(NamedTuple):
    video_path: Path
    camera_matrix_path: Path
    odometry_path: Path
    depth_dir: Path
    conf_dir: Path


def download_scan_zip(url: str, extract_dir: str) -> ScanFiles:
    base = Path(extract_dir)
    base.mkdir(parents=True, exist_ok=True)
    tmp_zip = base / "_scan.zip"
    with httpx.stream("GET", url) as r:
        r.raise_for_status()
        with open(tmp_zip, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(base)
    tmp_zip.unlink()
    return ScanFiles(
        video_path=base / "rgb.mp4",
        camera_matrix_path=base / "camera_matrix.csv",
        odometry_path=base / "odometry.csv",
        depth_dir=base / "depth",
        conf_dir=base / "confidence",
    )


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
