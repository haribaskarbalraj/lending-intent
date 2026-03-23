import json
import logging

from src.core.exceptions import LLMException
from src.intent.schemas import IntentAnalysisResponse, SpendingRequest
from src.llm.client import BaseLLMClient

logger = logging.getLogger(__name__)


class IntentService:
    """Business logic for lending intent analysis."""

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def analyse(self, request: SpendingRequest) -> IntentAnalysisResponse:
        logger.info("Analysing intent for %d transactions", len(request.transactions))
        prompt = self._build_prompt(request)
        logger.debug("Prompt sent to LLM:\n%s", prompt)
        raw = self.llm.ask(prompt)
        logger.debug("Raw response from LLM:\n%s", raw)
        data = json.loads(self._strip_markdown(raw))
        result = IntentAnalysisResponse(**data)
        logger.info(
            "Propensity score=%.2f recommended_product='%s'",
            result.propensity_score,
            result.recommended_product,
        )
        return result

    def _strip_markdown(self, raw: str) -> str:
        """Remove ```json ... ``` wrapper that LLMs sometimes add despite instructions."""
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]
        return stripped.strip()

    def _build_prompt(self, req: SpendingRequest) -> str:
        transactions_json = json.dumps(
            [t.model_dump() for t in req.transactions], indent=2
        )
        return f"""You are a credit risk analyst. Analyze the customer's 60-day transaction history and predict lending intent.

<analysis_instructions>
1. Identify financial stress signals, life events, or debt consolidation patterns.
2. Choose up to 3 keyword flags that best summarize the customer's financial state (e.g., high_utilization, recent_relocation, min_payment_pattern).
3. Recommend the single best product from: ["Personal Loan", "Debt Consolidation Loan", "Balance Transfer Credit Card", "Auto Loan", "Overdraft Line of Credit", "None"].
4. Score lending propensity 0.00–1.00 (0.00–0.30 no need, 0.31–0.70 moderate, 0.71–1.00 high intent).
</analysis_instructions>

<output_format>
Respond ONLY with a valid JSON object — no markdown, no explanation:
{{
  "financial_flags": ["flag1", "flag2"],
  "recommended_product": "Product Name",
  "propensity_score": 0.00
}}
</output_format>

<transactions>
{transactions_json}
</transactions>"""
