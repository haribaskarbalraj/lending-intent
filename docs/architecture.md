# Architecture Overview

## System Purpose

The Lending Intent Engine is a GenAI classification service. It ingests raw customer transaction data and outputs a structured lending recommendation enriched by historical conversion patterns via RAG (Retrieval Augmented Generation).

## Request Pipeline

```
POST /intent/analyse
        │
        ▼
┌─────────────────────┐
│   SpendingRequest   │  Pydantic validation — list of Transaction objects
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  build_narrative()  │  Converts transactions to human-readable prose
│                     │  (LLMs understand narrative better than raw JSON)
└─────────────────────┘
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
┌──────────────────┐                ┌────────────────────┐
│  RAG retrieve    │                │  (future) RAG store │
│  similar past    │                │  narrative after    │
│  converters      │                │  conversion         │
└──────────────────┘                └────────────────────┘
        │
        ▼
┌─────────────────────┐
│  LangChain LCEL     │  ChatPromptTemplate | LLM.with_structured_output()
│  chain.invoke()     │  Returns validated IntentAnalysisResponse directly
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Compliance         │  Post-LLM guardrails — product allowlist,
│  Guardrails         │  score range, flag count, non-empty pitch
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ IntentAnalysisResponse │
│  financial_flags    │
│  recommended_product│
│  propensity_score   │
│  pitch              │
└─────────────────────┘
```

## Component Map

```
lending-intent/
├── main.py                    FastAPI app entry point
├── src/
│   ├── core/
│   │   ├── config.py          Pydantic Settings — loads .env.dev or .env.prod
│   │   └── exceptions.py      LLMException — wraps all backend failures
│   ├── llm/
│   │   ├── client.py          BaseLLMClient + AnthropicClient + BedrockClient
│   │   └── chain.py           LangChain LCEL chain + prompt templates
│   ├── rag/
│   │   └── store.py           RAGStore interface + ChromaRAGStore + OpenSearchRAGStore
│   ├── intent/
│   │   ├── schemas.py         Pydantic models: Transaction, SpendingRequest, IntentAnalysisResponse
│   │   ├── service.py         IntentService — orchestrates the full pipeline
│   │   ├── guardrails.py      Compliance validation (post-LLM)
│   │   └── router.py          FastAPI route: POST /intent/analyse
│   └── health/
│       └── router.py          GET /health
```

## Dual Environment Design

The `APP_ENV` setting toggles three infrastructure layers simultaneously:

```
                 APP_ENV=dev              APP_ENV=prod
                 ───────────              ────────────
LLM backend      Anthropic API            AWS Bedrock
                 ANTHROPIC_API_KEY        IAM Role (no key)
                 claude-sonnet-4-5        anthropic.claude-sonnet-4-5

RAG store        ChromaDB in-memory       AWS OpenSearch k-NN
                 ONNX MiniLM embeddings   Bedrock Titan Embeddings v2 (1024-dim)
                 wiped on restart         persistent across restarts

Auth             API key in .env.dev      IAM role / instance profile
```

The service layer (`IntentService`) and chain (`get_intent_chain`) are identical in both environments — only the infrastructure underneath swaps.

## RAG: How It Works

### Embedding

Text is converted into a vector (list of numbers) that captures semantic meaning.

```
"Cash advance, minimum payment only"   →  [ 0.12, -0.87, 0.34, ... ]  (1024 numbers)
"Paid full balance, savings deposit"   →  [ 0.71,  0.41, -0.21, ... ]
```

Vectors that are mathematically close = narratives with similar financial behavior.

### Storage and Retrieval

```
Past converter stored:
  { vector: [...], text: "DEBIT $4500 at VISA CASH ADVANCE...", metadata: { product: "Debt Consolidation Loan" } }

New customer arrives:
  embed their narrative → query vector store for top-3 nearest vectors
  → retrieve those past converter narratives
  → inject into LLM prompt as context examples
```

### Current State

The RAG store `retrieve_similar()` is wired up and called on every request. However, `store()` is not yet called after a successful analysis. The store is always empty — the LLM runs on raw transactions only. See [ADR-002](adr/002-hybrid-rag-approach.md) for the planned hybrid approach.

## Guardrails

Compliance validation runs **after** the LLM responds, before the result is returned:

| Check | Rule |
|---|---|
| Product | Must be in the approved product allowlist |
| Score | Must be between 0.0 and 1.0 |
| Flags | Maximum 3 financial flags |
| Pitch | Must be non-empty |

A `ComplianceError` is raised on violation, which is caught and returned as HTTP 503.
