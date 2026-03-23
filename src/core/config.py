import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "dev"
    anthropic_api_key: str = ""
    aws_region: str = "us-east-1"
    llm_model_dev: str = "claude-sonnet-4-5"
    llm_model_prod: str = "anthropic.claude-sonnet-4-5"

    model_config = {
        "env_file": os.getenv("ENV_FILE", ".env.dev"),
        "env_file_encoding": "utf-8",
    }


settings = Settings()
