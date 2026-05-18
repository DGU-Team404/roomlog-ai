from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    fal_api_key: str
    s3_bucket_name: str
    s3_region: str = "ap-northeast-2"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    model_config = {"env_file": ".env"}


settings = Settings()
