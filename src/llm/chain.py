"""
LangChain LCEL chain for lending intent analysis.

Replaces the manual prompt → llm.ask() → json.loads() → _strip_markdown() flow
with a proper chain that guarantees structured output via Pydantic.

Pipeline:
    ChatPromptTemplate | ChatLLM.with_structured_output(IntentAnalysisResponse)

Spring analogy:
    Think of LCEL (|) like Java's Stream.of(...).map(...).collect(...)
    Each | is a transformation step. The chain is lazy — nothing runs
    until you call .invoke().

Dev:  ChatAnthropic  (Anthropic API, no AWS needed)
Prod: ChatBedrockConverse (AWS Bedrock, same Claude model)
Both return an IntentAnalysisResponse — caller never sees the difference.
"""

import logging
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.core.config import settings
from src.intent.schemas import IntentAnalysisResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# The {similar_context} placeholder is filled at runtime by IntentService.
# When the RAG store has no examples yet it will be an empty string —
# the LLM handles that gracefully.
_SYSTEM_PROMPT = """\
You are a credit risk analyst at a retail bank. \
Analyze the customer's transaction history and predict their lending intent.

Instructions:
1. Identify financial stress signals, life events, or debt-consolidation patterns.
2. Choose up to 3 keyword flags that best summarize the customer's financial state \
(e.g. high_utilization, recent_relocation, min_payment_pattern).
3. Recommend the single best product from: \
["Personal Loan", "Debt Consolidation Loan", "Balance Transfer Credit Card", \
"Auto Loan", "Overdraft Line of Credit", "None"].
4. Score lending propensity 0.00–1.00 \
(0.00–0.30 = no need, 0.31–0.70 = moderate intent, 0.71–1.00 = high intent).
5. Write one personalized sales pitch sentence for the recommended product, \
addressed directly to the customer.
{similar_context}"""

_HUMAN_PROMPT = "{narrative}"


# ---------------------------------------------------------------------------
# Chain factory — mirrors get_llm_client() and get_rag_store() pattern
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_intent_chain() -> Runnable:
    """Build and return the LCEL intent-analysis chain.

    lru_cache makes this a singleton — built once, reused across all requests.
    Spring analogy: a @Bean method that returns a singleton @Service.

    The chain is:
        prompt_template | llm.with_structured_output(IntentAnalysisResponse)

    .with_structured_output() instructs the LLM to return JSON that matches
    the IntentAnalysisResponse Pydantic schema — equivalent to Jackson's
    @JsonProperty deserialization in Spring, but enforced at the LLM level.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ])

    llm = _build_llm()

    # .with_structured_output() passes the Pydantic schema to the LLM as a
    # tool definition, then parses and validates the response automatically.
    # No more json.loads(), no more _strip_markdown().
    structured_llm = llm.with_structured_output(IntentAnalysisResponse)

    chain = prompt | structured_llm

    logger.info(
        "Intent chain built (backend=%s, model=%s)",
        "bedrock" if settings.app_env == "prod" else "anthropic",
        settings.llm_model_prod if settings.app_env == "prod" else settings.llm_model_dev,
    )
    return chain


def _build_llm():
    """Instantiate the correct LangChain chat model for the current environment.

    Separated from get_intent_chain() to keep the factory readable.
    """
    if settings.app_env == "prod":
        # ChatBedrockConverse — AWS Bedrock with the Converse API.
        # Uses boto3 under the hood, picks up IAM role or env credentials
        # automatically — no explicit key needed in prod.
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(
            model=settings.llm_model_prod,
            region_name=settings.aws_region,
            max_tokens=1024,
        )

    # ChatAnthropic — calls Claude directly via the Anthropic API.
    # Used for local dev where AWS credentials aren't required.
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=settings.llm_model_dev,
        api_key=settings.anthropic_api_key,
        max_tokens=1024,
    )
