# ADR-002: Hybrid RAG + LLM Recommendation Approach

**Status:** Proposed  
**Date:** 2026-06-11

## Context

The current implementation asks the LLM to do everything: classify financial signals, choose a product, score propensity, and write a pitch — on every request, from scratch.

Two alternative approaches exist:

### Option A — Pure LLM (current)
```
transactions → narrative → LLM reasons everything → recommendation + score + pitch
```
Works from day one. No historical data required. But expensive per request and the recommendation quality is bounded by the LLM's general financial knowledge, not your actual conversion data.

### Option B — Pure Similarity Search (no LLM for recommendation)
```
transactions → embed → find top-N similar past customers
                               ↓
                    majority vote on what they converted to
                               ↓
                    similarity score becomes propensity score
```
Fast and cheap. Gets better with more data. But:
- Broken on cold start (empty store = no recommendations)
- Cannot generate a personalized pitch
- Cannot handle genuinely new financial patterns it has never seen

### Option C — Hybrid (proposed)
```
Phase 1 (cold start):  LLM does everything — no store data yet
Phase 2 (at scale):    Similarity search → product + score
                        LLM → pitch only (given the known product)
```

## Decision

Adopt the **Hybrid approach (Option C)** as the target architecture.

**Phase 1** (current): LLM-led with RAG context injection. `store()` to be wired in after each successful analysis to begin populating the vector store.

**Phase 2** (after sufficient conversions): route the recommendation and propensity score to similarity search. Pass only the product name to the LLM and ask for a pitch sentence. This cuts LLM reasoning cost significantly while keeping the personalized pitch.

The threshold for switching to Phase 2 is a business decision based on: number of stored converters, product distribution coverage, and A/B validation of similarity-vs-LLM recommendation quality.

## Consequences

**Positive:**
- Works from day one with zero historical data (Phase 1)
- LLM cost per request drops in Phase 2 (pitch-only prompt is much shorter)
- Recommendations in Phase 2 are grounded in real conversion data, not just LLM priors
- The pitch remains personalized regardless of phase

**Negative:**
- Two code paths to maintain (LLM-led vs similarity-led)
- Requires a decision threshold and A/B testing infrastructure to trigger Phase 2
- Similarity-based propensity score (cosine distance) needs calibration against actual conversion rates to be meaningful

## Correct RAG Store Trigger

Storing after **loan acceptance** is the wrong trigger for this use case. The engine's job is proactive solicitation — by the time a loan is accepted, the recommendation decision is already done.

The correct event funnel:

```
Card activity observed
        ↓
Solicitation sent (invitation ID generated)   ← store here
        ↓
Customer engaged / clicked offer              ← update metadata
        ↓
Customer applied                              ← update metadata (strong signal)
        ↓
Customer accepted                             ← outcome only, too late for RAG
```

The RAG store learns: *"this card spending pattern, solicited with this product, led to an application."* That is the signal that improves future recommendations.

## What Needs to Be Built

1. Wire `rag.store(narrative, customer_id, metadata)` into the Invitation Agent — store at solicitation time, not in `IntentService.analyse()`
2. Add `customer_id` to `SpendingRequest` so stored narratives are identifiable
3. Add `retrieve_similar_with_metadata()` to `RAGStore` — current `retrieve_similar()` returns text only, not the associated product metadata
4. Add `POST /intent/feedback` endpoint to accept `{ invitation_id, applied: true }` from CRM and update the RAG entry
5. Implement the Phase 2 routing logic with a feature flag or config threshold
