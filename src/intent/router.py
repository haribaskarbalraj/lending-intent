import logging

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.runnables import Runnable

from src.core.exceptions import LLMException
from src.intent.schemas import IntentAnalysisResponse, SpendingRequest
from src.intent.service import IntentService
from src.llm.chain import get_intent_chain
from src.rag.store import RAGStore, get_rag_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intent", tags=["Lending Intent"])


def get_intent_service(
    chain: Runnable = Depends(get_intent_chain),
    rag: RAGStore = Depends(get_rag_store),
) -> IntentService:
    """Construct IntentService with its dependencies.

    Spring analogy: this is the @Bean factory method that wires
    the @Service with its @Autowired dependencies.
    FastAPI's Depends() calls get_intent_chain() and get_rag_store()
    once (both are lru_cache singletons) and injects the results.
    """
    return IntentService(chain=chain, rag=rag)


@router.post("/analyse", response_model=IntentAnalysisResponse)
async def analyse_spending(
    request: SpendingRequest,
    service: IntentService = Depends(get_intent_service),
) -> IntentAnalysisResponse:
    """Analyse a customer's transaction history and classify their lending intent."""
    try:
        return service.analyse(request)
    except LLMException as e:
        logger.error("Service failure: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except (ValueError, TypeError) as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=502, detail=f"Unexpected error: {e}")
