# Lending Intent Engine

A GenAI application that analyses customer transaction history and classifies lending intent using an agentic LangGraph workflow, Amazon Bedrock (Claude), and RAG-enriched retrieval — deployed behind AWS Bedrock Guardrails for production safety.

## What it does

Given a customer's recent transactions, the engine returns:
- **Financial flags** — up to 3 keyword signals (e.g. `high_utilization`, `cash_advance_pattern`)
- **Recommended product** — the most suitable lending product
- **Propensity score** — 0.00–1.00 likelihood of conversion
- **Pitch** — one personalized sales sentence

## Tech Stack

| Layer | Dev | Prod |
|---|---|---|
| API | FastAPI (async) | FastAPI (async) |
| LLM | Anthropic API (Claude Sonnet) | AWS Bedrock (Claude Sonnet) |
| RAG store | ChromaDB in-memory | AWS OpenSearch k-NN |
| Embeddings | ONNX MiniLM (local) | Bedrock Titan Embeddings v2 |
| Validation | Pydantic v2 | Pydantic v2 |

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy `.env.dev` and fill in your Anthropic API key:
```
APP_ENV=dev
ANTHROPIC_API_KEY=sk-ant-...
AWS_REGION=us-east-1
LLM_MODEL_DEV=claude-sonnet-4-5
```

Start the server:
```bash
fastapi dev main.py
```

## Endpoints

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /intent/analyse`

**Request:**
```json
{
  "transactions": [
    {
      "date": "2026-06-01",
      "amount": 4500.00,
      "transaction_type": "debit",
      "merchant_description": "VISA CASH ADVANCE - BANK OF AMERICA"
    },
    {
      "date": "2026-06-03",
      "amount": 1200.00,
      "transaction_type": "debit",
      "merchant_description": "MINIMUM PAYMENT - CHASE CREDIT CARD"
    }
  ]
}
```

**Response:**
```json
{
  "financial_flags": ["high_utilization", "cash_advance_pattern", "min_payment_pattern"],
  "recommended_product": "Debt Consolidation Loan",
  "propensity_score": 0.92,
  "pitch": "Consolidate your high-interest balances into one low monthly payment and take control of your finances today."
}
```

**Valid products:** `Personal Loan`, `Debt Consolidation Loan`, `Balance Transfer Credit Card`, `Auto Loan`, `Overdraft Line of Credit`, `None`

**Propensity score bands:**
| Range | Meaning |
|---|---|
| 0.00 – 0.30 | No lending intent |
| 0.31 – 0.70 | Moderate intent |
| 0.71 – 1.00 | High intent |

## Local vs Production

```
APP_ENV=dev   →  Anthropic API + ChromaDB in-memory
APP_ENV=prod  →  AWS Bedrock  + AWS OpenSearch k-NN
```

See [docs/architecture.md](docs/architecture.md) for the full system design.

## Architecture Decisions

- [ADR-001: Dual-environment LLM and RAG backends](docs/adr/001-dual-env-llm-rag.md)
- [ADR-002: Hybrid RAG + LLM recommendation approach](docs/adr/002-hybrid-rag-approach.md)
- [ADR-003: Multi-agent architecture for post-scoring actions](docs/adr/003-multi-agent-architecture.md)
