"""
Compliance guardrails — post-LLM validation gate.

Sits between the LangChain chain output and the API response.
Even with structured output, the LLM can still produce values that
pass Pydantic type checks but violate business rules (e.g. an invented
product name, a score slightly outside range due to float precision).

Spring analogy:
    Like a @Validator or a @ControllerAdvice that fires *inside* the
    @Service layer before the result leaves the business tier —
    not at the HTTP boundary, but earlier, where you can still
    apply remediation (clamp, reject, flag) cleanly.

Rules applied (in order):
    1. recommended_product must be in the bank's approved product list
    2. propensity_score must be within [0.0, 1.0]
    3. financial_flags must not exceed 3 items
    4. pitch must not be empty
"""

import logging

from src.intent.schemas import IntentAnalysisResponse

logger = logging.getLogger(__name__)

# Single source of truth for allowed products.
# Any product the LLM returns that isn't here is a compliance violation.
ALLOWED_PRODUCTS: frozenset[str] = frozenset({
    "Personal Loan",
    "Debt Consolidation Loan",
    "Balance Transfer Credit Card",
    "Auto Loan",
    "Overdraft Line of Credit",
    "None",
})


class ComplianceError(ValueError):
    """Raised when LLM output violates a compliance rule.

    Caught in IntentService and re-raised as LLMException so the
    router returns a 503 — same error surface as an LLM backend failure.
    """


def check_compliance(result: IntentAnalysisResponse) -> IntentAnalysisResponse:
    """Validate LLM output against compliance rules.

    Returns the result unchanged if all rules pass.
    Raises ComplianceError on the first violation found.

    Spring analogy: like javax.validation.Validator.validate() —
    runs a set of constraint checks and surfaces violations.
    """
    _check_product(result)
    _check_score(result)
    _check_flags(result)
    _check_pitch(result)
    logger.debug("Compliance check passed (product=%s, score=%.2f)",
                 result.recommended_product, result.propensity_score)
    return result


# ---------------------------------------------------------------------------
# Individual rule functions — each has a single responsibility
# ---------------------------------------------------------------------------

def _check_product(result: IntentAnalysisResponse) -> None:
    if result.recommended_product not in ALLOWED_PRODUCTS:
        raise ComplianceError(
            f"Unauthorized product recommendation: '{result.recommended_product}'. "
            f"Allowed: {sorted(ALLOWED_PRODUCTS)}"
        )


def _check_score(result: IntentAnalysisResponse) -> None:
    if not (0.0 <= result.propensity_score <= 1.0):
        raise ComplianceError(
            f"Propensity score {result.propensity_score:.4f} is outside [0.0, 1.0]"
        )


def _check_flags(result: IntentAnalysisResponse) -> None:
    if len(result.financial_flags) > 3:
        raise ComplianceError(
            f"Too many financial flags: got {len(result.financial_flags)}, max is 3. "
            f"Flags: {result.financial_flags}"
        )


def _check_pitch(result: IntentAnalysisResponse) -> None:
    if not result.pitch or not result.pitch.strip():
        raise ComplianceError("Pitch must not be empty")
