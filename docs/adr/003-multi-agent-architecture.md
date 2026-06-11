# ADR-003: Multi-Agent Architecture for Post-Scoring Actions

**Status:** Proposed  
**Date:** 2026-06-11

## Context

When the Lending Intent Engine scores a customer above the propensity threshold, two things need to happen:

1. An invitation ID must be generated and written to the internal CRM / loan origination DB
2. A personalised email must be composed and sent to the customer

The naive approach is to do both inside `IntentService.analyse()`. This creates a bloated service that owns scoring, invitation management, and email delivery — unrelated concerns that scale, fail, and change independently.

## Decision

Adopt a **multi-agent architecture** where the Intent Engine acts as the orchestrator and delegates post-scoring actions to two purpose-built agents:

```
Lending Intent Engine (Orchestrator)
        │
        │  score >= threshold
        │
        ├──────────────────────────┐
        ▼                          ▼
Invitation Agent           Communication Agent
- generate invitation ID   - LLM: compose email body
- write record to CRM DB   - personalise from pitch
- return invitation_id     - send email to customer
        │
        ▼
  RAG store: store(narrative, invitation_id, metadata)
```

### Agent Responsibilities

**Invitation Agent** (`src/invitation/`)
- Deterministic — no LLM needed
- Generates a unique `invitation_id` (e.g. `INV-YYYYMMDD-UUID`)
- Writes a record to the internal DB: `{ customer_id, product, score, invitation_id, sent_at }`
- Returns `invitation_id` to the orchestrator so it can be stored in the RAG entry

**Communication Agent** (`src/communication/`)
- Generative — uses LLM to compose the email body
- Input: `{ customer_name, product, pitch, invitation_id }`
- LLM call: turns the one-line pitch into a full, compliant email
- Sends via email provider (SES / SendGrid)
- Does not know anything about scoring or invitations

### Why separate agents and not one service

| Concern | Invitation Agent | Communication Agent |
|---|---|---|
| LLM needed? | No | Yes |
| Failure impact | Cannot send — skip email | Email fails, invitation still recorded |
| Scales with | CRM write throughput | Email send volume |
| Changes when | CRM schema changes | Email template / provider changes |

Each agent can fail, retry, or be replaced independently without affecting the others.

## Agentic AI Principles Applied

**Single responsibility** — each agent has exactly one job.  
**Tool use** — the Invitation Agent's "tools" are `generate_id()` and `write_to_crm()`. The Communication Agent's tools are `compose_email()` (LLM) and `send_email()`.  
**Orchestrator pattern** — the Intent Engine decides *when* and *whether* to trigger agents. It does not know *how* they do their work.  
**Async by design** — in production, agent dispatch should be async (e.g. via SQS/SNS) so a slow email provider does not block the scoring response.

## Consequences

**Positive:**
- Intent Engine response time is not affected by CRM write latency or email delivery
- Each agent is independently testable, deployable, and replaceable
- Invitation ID is generated before the email is sent — audit trail is complete even if email fails
- RAG store entry is tied to `invitation_id`, enabling the feedback loop when customer responds

**Negative:**
- More services to deploy and monitor
- Distributed failure modes — need dead-letter queues or retry logic for agent calls
- `invitation_id` must be threaded through from Invitation Agent → Communication Agent → RAG store → feedback endpoint

## What Needs to Be Built

1. `src/invitation/` — Invitation Agent router + service + schema
2. `src/communication/` — Communication Agent router + service + LLM email chain
3. Add `invitation_id` to `IntentAnalysisResponse` (returned to caller when threshold crossed)
4. Orchestrator logic in `IntentService` — after guardrails pass, call Invitation Agent if score >= threshold
5. `POST /intent/feedback` — receives `{ invitation_id, applied: bool }` from CRM webhook, updates RAG entry
