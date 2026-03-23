from abc import ABC, abstractmethod
from functools import lru_cache

import anthropic
import boto3
from botocore.exceptions import ClientError, EndpointResolutionError

from src.core.config import settings
from src.core.exceptions import LLMException


class BaseLLMClient(ABC):
    """Common interface for all LLM backends."""

    DEFAULT_MAX_TOKENS = 1024

    @abstractmethod
    def ask(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Send a prompt and return the text response."""
        pass


class AnthropicClient(BaseLLMClient):
    """Calls Claude directly via the Anthropic API. Use for local dev."""

    def __init__(self, model_id: str = settings.llm_model_dev):
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)

    def ask(self, prompt: str, max_tokens: int = BaseLLMClient.DEFAULT_MAX_TOKENS) -> str:
        try:
            message = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except anthropic.AuthenticationError as e:
            raise LLMException(f"Invalid Anthropic API key: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMException(f"Anthropic rate limit exceeded: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMException(f"Anthropic API error [{e.status_code}]: {e}") from e


class BedrockClient(BaseLLMClient):
    """Calls Claude via AWS Bedrock. Use for production."""

    def __init__(self, model_id: str = settings.llm_model_prod):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    def ask(self, prompt: str, max_tokens: int = BaseLLMClient.DEFAULT_MAX_TOKENS) -> str:
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens},
            )
            return response["output"]["message"]["content"][0]["text"]
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            raise LLMException(f"Bedrock API error [{error_code}]: {e}") from e
        except EndpointResolutionError as e:
            raise LLMException(f"Could not reach Bedrock endpoint: {e}") from e
        except (KeyError, IndexError) as e:
            raise LLMException(f"Unexpected response format from Bedrock: {e}") from e


@lru_cache(maxsize=1)
def get_llm_client() -> BaseLLMClient:
    """Returns the appropriate LLM client based on APP_ENV setting.

    lru_cache ensures only one instance is created and reused across all
    requests — equivalent to a singleton @Bean in Spring.
    """
    if settings.app_env == "prod":
        return BedrockClient()
    return AnthropicClient()
