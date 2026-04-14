import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "dev"
    anthropic_api_key: str = ""
    aws_region: str = "us-east-1"
    llm_model_dev: str = "claude-sonnet-4-5"
    llm_model_prod: str = "anthropic.claude-sonnet-4-5"

    # OpenSearch (prod RAG store)
    opensearch_host: str = ""
    opensearch_port: int = 443
    opensearch_index: str = "lending-intent-converters"
    opensearch_username: str = ""
    opensearch_password: str = ""

    # Bedrock embedding model (prod)
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    model_config = {
        "env_file": os.getenv("ENV_FILE", ".env.dev"),
        "env_file_encoding": "utf-8",
    }


settings = Settings()
