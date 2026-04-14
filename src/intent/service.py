import logging

from langchain_core.runnables import Runnable

from src.core.exceptions import LLMException
from src.intent.guardrails import ComplianceError, check_compliance
from src.intent.schemas import IntentAnalysisResponse, SpendingRequest
from src.rag.store import RAGStore

logger = logging.getLogger(__name__)


class IntentService:
    """Business logic for lending intent analysis.

    Full pipeline:
        transactions → narrative → RAG enrich → LangChain chain → guardrails → response

    Spring analogy: this is your @Service class. Its dependencies (chain, rag)
    are injected via the constructor — equivalent to @Autowired constructor injection.
    The router's get_intent_service() factory acts as the @Bean provider.
    """

    def __init__(self, chain: Runnable, rag: RAGStore) -> None:
        self.chain = chain
        self.rag = rag

    def analyse(self, request: SpendingRequest) -> IntentAnalysisResponse:
        logger.info("Analysing intent for %d transactions", len(request.transactions))

        # Step 1: Build a human-readable narrative from raw transactions.
        # Used for both embedding (RAG) and LLM prompting.
        narrative = self.build_transaction_narrative(request)
        logger.debug("Narrative:\n%s", narrative)

        # Step 2: RAG — retrieve similar past converter narratives to enrich the prompt.
        # If the store is empty (e.g. first run) this returns [] gracefully.
        similar = self.rag.retrieve_similar(narrative)
        similar_context = self._format_similar_context(similar)
        logger.info("RAG: retrieved %d similar converter examples", len(similar))

        # Step 3: LangChain chain — structured output, no manual JSON parsing.
        # chain.invoke() returns a validated IntentAnalysisResponse directly.
        try:
            result: IntentAnalysisResponse = self.chain.invoke({
                "narrative": narrative,
                "similar_context": similar_context,
            })
        except Exception as e:
            raise LLMException(f"LangChain chain failed: {e}") from e

        # Step 4: Guardrails — compliance check before the result leaves the service.
        try:
            result = check_compliance(result)
        except ComplianceError as e:
            raise LLMException(f"Compliance violation: {e}") from e

        logger.info(
            "score=%.2f product='%s' pitch='%s'",
            result.propensity_score,
            result.recommended_product,
            result.pitch,
        )
        return result

    def build_transaction_narrative(self, request: SpendingRequest) -> str:
        """Convert raw transactions into a human-readable paragraph.

        This prose form is used for both LLM prompting and vector embedding.
        LLMs and embedding models understand narrative text better than raw JSON.

        Example output:
            Customer 3-month transaction history:
            2024-01-15: DEBIT $45.00 at Starbucks
            2024-01-16: CREDIT $2500.00 at Employer Payroll
        """
        lines = [
            f"{t.date}: {t.transaction_type.upper()} ${abs(t.amount):.2f} "
            f"at {t.merchant_description}"
            for t in request.transactions
        ]
        return "Customer 3-month transaction history:\n" + "\n".join(lines)

    def _format_similar_context(self, similar: list[str]) -> str:
        """Format RAG results into a prompt-ready context block.

        Returns an empty string when no examples exist —
        the system prompt handles the missing placeholder gracefully.
        """
        if not similar:
            return ""
        examples = "\n---\n".join(similar)
        return (
            f"\nFor context, here are {len(similar)} similar customers "
            f"who converted to a lending product — use these as soft reference:\n"
            f"{examples}\n"
        )
