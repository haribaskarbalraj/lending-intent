from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """A single transaction entry in the customer's history."""

    date: str = Field(..., description="Transaction date (YYYY-MM-DD)")
    amount: float = Field(..., description="Transaction amount in dollars")
    transaction_type: str = Field(..., description="debit or credit")
    merchant_description: str = Field(..., description="Merchant or transaction description")


class SpendingRequest(BaseModel):
    """Request body — 60-day transaction history for a customer."""

    transactions: list[Transaction] = Field(..., description="List of recent transactions")


class IntentAnalysisResponse(BaseModel):
    """LLM analysis result predicting lending intent from transaction history."""

    # Placed first so the model generates these tokens before the score,
    # acting as a compressed Chain of Thought.
    financial_flags: list[str] = Field(
        description="Max 3 keywords summarizing the financial state (e.g., ['high_utilization', 'recent_relocation'])"
    )
    recommended_product: str
    propensity_score: float
