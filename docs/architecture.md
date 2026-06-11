# Architecture Overview

## System Purpose

The Lending Intent Engine is a GenAI classification service built for **proactive customer solicitation**. It analyses the spending patterns of existing credit card customers and identifies who should be targeted with a lending product offer — before they come looking for a loan.

The engine does not wait for a customer to apply. It watches card activity, scores lending intent, and when the propensity crosses a threshold it triggers downstream agents to generate an invitation and send a personalised offer.

## Full System Flow

```
Card Activity Feed (batch / real-time)
        │
        ▼
┌───────────────────────────────────────┐
│        Lending Intent Engine          │  ← Orchestrator Agent
│                                       │
│  1. build_narrative()                 │  raw transactions → prose
│  2. RAG retrieve_similar()            │  find past solicitation patterns
│  3. LangChain chain.invoke()          │  LLM scores intent
│  4. Compliance guardrails             │  product + score + flag checks
│                                       │
│  propensity_score >= threshold?       │
└──────────────┬────────────────────────┘
               │ YES
               ▼
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌───────────────┐   ┌─────────────────────┐
│  Invitation   │   │  Communication      │
│    Agent      │   │      Agent          │
│               │   │                     │
│ - gen INV ID  │   │ - LLM: write email  │
│ - write CRM   │   │   body from pitch   │
│ - return ID   │   │ - send email to     │
└───────┬───────┘   │   customer          │
        │           └─────────────────────┘
        ▼
┌───────────────┐
│  RAG Store    │  store(narrative, customer_id, { product, score, invitation_id })
│  store()      │  ← stored at solicitation time, NOT after loan acceptance
└───────────────┘
        │
        ▼  (async — when customer responds)
┌───────────────┐
│ Feedback loop │  POST /intent/feedback { invitation_id, applied: true }
│               │  → update RAG entry → confirmed signal for future customers
└───────────────┘
```

## Scoring Pipeline (inside the Orchestrator)

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
        ▼
┌──────────────────────┐
│  RAG retrieve        │  embed narrative → find top-N similar past
│  retrieve_similar()  │  solicitation patterns from vector store
└──────────────────────┘
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
┌──────────────────────────┐
│  IntentAnalysisResponse  │
│  financial_flags         │
│  recommended_product     │
│  propensity_score        │
│  pitch                   │
│  invitation_id (future)  │
└──────────────────────────┘
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
│   ├── invitation/            (planned) Invitation Agent
│   │   ├── schemas.py         InvitationRequest, InvitationResponse
│   │   ├── service.py         generate ID, write to CRM/DB
│   │   └── router.py          POST /invitation/create
│   ├── communication/         (planned) Communication Agent
│   │   ├── schemas.py         EmailRequest, EmailResponse
│   │   ├── service.py         LLM email body generation + send
│   │   └── router.py          POST /communication/send
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

### What Gets Stored and When

The correct trigger for storing a pattern is **when a solicitation is sent**, not when a loan is accepted. By the time a loan is accepted the recommendation decision is already made — storing at that point teaches the wrong lesson.

```
Solicitation sent  → store immediately (baseline signal)
Customer applied   → update metadata: applied=true (strong signal)
Customer accepted  → outcome data only, decision already made
```

This means the RAG store learns: *"this spending pattern, when solicited with this product, resulted in a customer applying"* — which is exactly the signal needed for future recommendations.

### Current State

The RAG store `retrieve_similar()` is wired up and called on every request. However, `store()` is not yet called — the store is always empty and the LLM runs on raw transactions only. See [ADR-002](adr/002-hybrid-rag-approach.md) for the planned hybrid approach and [ADR-003](adr/003-multi-agent-architecture.md) for the downstream agent design.

## Guardrails

Compliance validation runs **after** the LLM responds, before the result is returned:

| Check | Rule |
|---|---|
| Product | Must be in the approved product allowlist |
| Score | Must be between 0.0 and 1.0 |
| Flags | Maximum 3 financial flags |
| Pitch | Must be non-empty |

A `ComplianceError` is raised on violation, which is caught and returned as HTTP 503.
