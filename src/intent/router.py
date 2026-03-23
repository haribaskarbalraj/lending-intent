import logging

from fastapi import APIRouter, Depends, HTTPException

from src.core.exceptions import LLMException
from src.intent.schemas import IntentAnalysisResponse, SpendingRequest
from src.intent.service import IntentService
from src.llm.client import BaseLLMClient, get_llm_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intent", tags=["Lending Intent"])


def get_intent_service(llm: BaseLLMClient = Depends(get_llm_client)) -> IntentService:
    return IntentService(llm)


@router.post("/analyse", response_model=IntentAnalysisResponse)
async def analyse_spending(
    request: SpendingRequest,
    service: IntentService = Depends(get_intent_service),
) -> IntentAnalysisResponse:
    """Analyse a customer's recurring spending pattern and classify their lending intent."""
    try:
        return service.analyse(request)
    except LLMException as e:
        logger.error("LLM backend failure: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except (ValueError, TypeError) as e:
        logger.error("Failed to parse LLM response: %s", e)
        raise HTTPException(status_code=502, detail=f"Unexpected response from LLM: {e}")
