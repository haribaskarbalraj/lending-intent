# ADR-001: Dual-Environment LLM and RAG Backends

**Status:** Accepted  
**Date:** 2026-06-11

## Context

The engine needs to run in two distinct environments:

- **Local development** — no AWS account or VPC access needed; fast iteration cycle; API key-based auth is fine
- **Production (AWS)** — must use managed services (Bedrock, OpenSearch) for security, scalability, and compliance; no long-lived API keys in prod

Both environments must expose identical behaviour to the service layer. Developers should not need to change any business logic when deploying to prod.

## Decision

Use a factory pattern (`lru_cache`-backed singletons) to swap infrastructure implementations based on `APP_ENV`:

| Concern | Dev | Prod |
|---|---|---|
| LLM | `ChatAnthropic` (Anthropic API) | `ChatBedrockConverse` (AWS Bedrock) |
| Auth | `ANTHROPIC_API_KEY` in `.env.dev` | IAM role / instance profile |
| RAG store | `ChromaRAGStore` (in-memory) | `OpenSearchRAGStore` (AWS OpenSearch k-NN) |
| Embeddings | ONNX MiniLM L6 V2 (local) | Bedrock Titan Embeddings v2 (1024-dim) |

The `IntentService`, `get_intent_chain()`, and all business logic are environment-agnostic. Only the factories (`get_llm_client`, `get_rag_store`, `get_intent_chain`) branch on `APP_ENV`.

## Consequences

**Positive:**
- Developers can run the full pipeline locally with just an Anthropic API key
- No mocking of AWS services in development
- Prod uses IAM — no secret rotation risk for LLM credentials
- ChromaDB in-memory eliminates local infra setup (no Docker, no OpenSearch)

**Negative:**
- Local embeddings (ONNX MiniLM) and prod embeddings (Titan v2) have different vector dimensions and semantic spaces — RAG data populated in dev cannot be reused in prod
- ChromaDB is wiped on restart — no persistence testing locally
- Any drift between the two LLM models (Anthropic vs Bedrock Claude) must be validated before prod deployment
