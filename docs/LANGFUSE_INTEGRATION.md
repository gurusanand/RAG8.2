# Langfuse Observability Integration — Banking RAG Application

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup Guide](#setup-guide)
   - [Option A: Self-Hosted Langfuse (Docker)](#option-a-self-hosted-langfuse-docker)
   - [Option B: Langfuse Cloud](#option-b-langfuse-cloud)
4. [Configuration Reference](#configuration-reference)
5. [What Gets Traced](#what-gets-traced)
6. [Governance Metrics in Langfuse](#governance-metrics-in-langfuse)
7. [Monitoring Dashboards](#monitoring-dashboards)
8. [Troubleshooting](#troubleshooting)
9. [Files Changed](#files-changed)

---

## Overview

The Langfuse integration provides **full observability** for the 7-Layer Banking RAG pipeline. Every query is traced end-to-end, from the FAQ router through all 7 layers, governance checks, and innovations — with LLM token usage, latency, and governance scores emitted to Langfuse for monitoring.

**Key Capabilities:**

| Capability | Description |
|---|---|
| **Pipeline Tracing** | One trace per query, with spans for each of the 7 layers + governance |
| **LLM Generation Tracking** | Every OpenAI API call tracked with model, tokens, cost, latency |
| **Governance Scores** | Hallucination, bias, PII, and compliance scores emitted as Langfuse scores |
| **FAQ Routing Decisions** | 3-tier routing (exact/fuzzy/novel) tracked with similarity scores |
| **Innovation Tracking** | RAGAS, Self-RAG, Hybrid Search, GraphRAG, RAPTOR results captured |
| **User Feedback** | Thumbs up/down from the UI linked to traces |
| **Session Tracking** | Multi-turn conversations grouped by session |
| **Cost Monitoring** | Per-query and aggregate LLM cost tracking |

**Design Principles:**

- **Non-invasive**: All tracing is additive — the RAG pipeline works identically with Langfuse disabled
- **Graceful degradation**: If the Langfuse SDK is not installed or the server is unreachable, all tracing methods become silent no-ops
- **Feature toggle**: Master switch in `config/settings.py` (`langfuse.enabled = True/False`)
- **Production-safe**: Tracing errors are caught and logged — they never break the RAG pipeline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Banking RAG Application                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              seven_layer_rag.py (process_query)          │   │
│  │                                                          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ Layer 0 │→│ Layer 1 │→│ Layer 2 │→│ Layer 3 │→ ...  │   │
│  │  │  Orch   │ │  Cache  │ │  HyDE   │ │Retrieval│       │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │   │
│  │       │           │           │           │              │   │
│  │  ┌────▼───────────▼───────────▼───────────▼──────────┐  │   │
│  │  │         RAGPipelineTracer (observability/)         │  │   │
│  │  │  • trace_layer_start/end()                        │  │   │
│  │  │  • trace_llm_call()                               │  │   │
│  │  │  • trace_governance()                             │  │   │
│  │  │  • emit_governance_scores()                       │  │   │
│  │  │  • emit_pipeline_scores()                         │  │   │
│  │  └────────────────────┬──────────────────────────────┘  │   │
│  └───────────────────────┼──────────────────────────────────┘   │
│                          │                                       │
│  ┌───────────────────────▼──────────────────────────────────┐   │
│  │         LangfuseTracker (observability/)                  │   │
│  │  • Langfuse Python SDK wrapper                           │   │
│  │  • Singleton pattern, thread-safe                        │   │
│  │  • Graceful degradation (no-op if disabled)              │   │
│  └────────────────────────┬─────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │ HTTPS / HTTP
                            ▼
              ┌──────────────────────────┐
              │    Langfuse Server        │
              │  (Self-Hosted or Cloud)   │
              │                          │
              │  Port: 3100 (default)    │
              │  DB: PostgreSQL :5433    │
              └──────────────────────────┘
```

---

## Setup Guide

### Prerequisites

Install the Langfuse Python SDK:

```bash
pip install langfuse
```

### Option A: Self-Hosted Langfuse (Docker)

This is the recommended approach for development and on-premise deployments. Langfuse runs on **port 3100** by default (configurable).

**Step 1: Start Langfuse**

```bash
cd /path/to/banking-rag-app

# Start Langfuse + PostgreSQL
docker compose -f docker-compose-langfuse.yml up -d

# Verify it's running
docker compose -f docker-compose-langfuse.yml ps
```

**Step 2: Create a Project and API Keys**

1. Open **http://localhost:3100** in your browser
2. Create an account (first user becomes admin)
3. Create a new project (e.g., "Banking RAG")
4. Navigate to **Settings → API Keys → Create New**
5. Copy the **Public Key** and **Secret Key**

**Step 3: Configure the Application**

```bash
# Copy the template
cp .env.langfuse .env

# Edit with your API keys
nano .env
```

Set these values in `.env`:

```env
LANGFUSE_BASE_URL=http://localhost:3100
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
LANGFUSE_ENABLED=true
```

> **Note:** Langfuse Python SDK v4 uses `LANGFUSE_BASE_URL` (not `LANGFUSE_HOST`). The integration supports both for backward compatibility.

**Step 4: Verify the Integration**

Start the RAG application. You should see:

```
[LANGFUSE] ✅ Initialized v4.0.0 (base_url=http://localhost:3100, auth=OK)
[RAG] Langfuse observability initialized
```

Run a query, then check the Langfuse UI at **http://localhost:3100** → **Traces**.

**Changing the Port:**

To run Langfuse on a different port (e.g., 4000):

```bash
# Option 1: Set in .env
LANGFUSE_PORT=4000

# Option 2: Set inline
LANGFUSE_PORT=4000 docker compose -f docker-compose-langfuse.yml up -d
```

Then update your application config:

```env
LANGFUSE_BASE_URL=http://localhost:4000
```

**Stopping Langfuse:**

```bash
docker compose -f docker-compose-langfuse.yml down

# To also remove data:
docker compose -f docker-compose-langfuse.yml down -v
```

### Option B: Langfuse Cloud

For teams that prefer a managed solution.

**Step 1: Sign Up**

1. Go to [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Create an account and a project

**Step 2: Get API Keys**

1. Navigate to **Settings → API Keys**
2. Copy the **Public Key** and **Secret Key**

**Step 3: Configure**

```env
LANGFUSE_BASE_URL=                          # Leave empty for cloud
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_ENABLED=true
```

---

## Configuration Reference

All Langfuse settings are in `config/settings.py` under the `LangfuseConfig` dataclass:

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Master toggle — set `False` to disable all tracing |
| `host` | `str` | `""` (env) | Langfuse server URL. Empty = Cloud. Set URL for self-hosted |
| `public_key` | `str` | `""` (env) | Project public key from Langfuse UI |
| `secret_key` | `str` | `""` (env) | Project secret key from Langfuse UI |
| `flush_interval_seconds` | `int` | `5` | How often to flush events to Langfuse |
| `trace_sample_rate` | `float` | `1.0` | Sampling rate: `1.0` = every query, `0.5` = 50% |
| `trace_llm_calls` | `bool` | `True` | Track individual LLM API calls (tokens, cost) |
| `trace_embeddings` | `bool` | `True` | Track embedding operations |
| `trace_governance` | `bool` | `True` | Emit governance scores |
| `trace_innovations` | `bool` | `True` | Track RAG innovation results |
| `trace_faq_routing` | `bool` | `True` | Track FAQ 3-tier routing decisions |
| `trace_document_indexing` | `bool` | `True` | Track document upload/indexing |

**Environment Variable Overrides:**

| Variable | Overrides |
|---|---|
| `LANGFUSE_BASE_URL` | `langfuse.host` (SDK v4 uses `base_url`; `LANGFUSE_HOST` also supported) |
| `LANGFUSE_PUBLIC_KEY` | `langfuse.public_key` |
| `LANGFUSE_SECRET_KEY` | `langfuse.secret_key` |
| `LANGFUSE_ENABLED` | `langfuse.enabled` |

---

## What Gets Traced

### Per-Query Trace Structure

Each user query creates one Langfuse **trace** with the following hierarchy:

```
Trace: "rag-pipeline"
├── Span: "FAQ Smart Router"           (if FAQ pairs loaded)
├── Span: "Layer 0: Product Orchestration"
│   └── Generation: "Orchestration LLM"  (if LLM used for classification)
├── Span: "Layer 1: Semantic Cache"
├── Span: "Layer 2: HyDE Query Transformation"
│   └── Generation: "HyDE Generation"
├── Span: "Layer 3: Semantic Retrieval"
├── Span: "Layer 4: Corrective RAG (CRAG)"
│   └── Generation: "CRAG Grading"
├── Span: "Layer 5: LLM Re-Ranking"
│   └── Generation: "Re-Ranking"
├── Span: "Layer 6: Agentic RAG (ReAct)"
│   └── Generation: "ReAct Iteration 1"
│   └── Generation: "ReAct Iteration 2"
├── Span: "Layer 7: Response Validation"
│   └── Generation: "Answer Generation"
│   └── Generation: "Hallucination Check"
├── Span: "Layer 8: AI Governance (Four-Check)"
│   └── Generation: "Governance Check 1: Hallucination"
│   └── Generation: "Governance Check 2: Bias"
│   └── Generation: "Governance Check 3: PII"
│   └── Generation: "Governance Check 4: Compliance"
├── Span: "RAG Innovations"
│
├── Score: "response_confidence"        (0.0–1.0)
├── Score: "gov_hallucination_score"    (0.0–1.0)
├── Score: "gov_bias_score"             (0.0–1.0)
├── Score: "gov_pii_score"             (0.0–1.0)
├── Score: "gov_compliance_score"       (0.0–1.0)
├── Score: "gov_overall_status"         (0.0–1.0)
├── Score: "faq_routing_tier"           (0.0–1.0)
├── Score: "ragas_faithfulness"         (0.0–1.0)
├── Score: "ragas_answer_relevancy"     (0.0–1.0)
└── Score: "user_feedback"              (0 or 1)
```

### Turbo Mode Traces

When turbo mode is active, traces are lighter:

```
Trace: "rag-pipeline" [tags: turbo-mode]
├── Span: "TURBO: FAQ Exact Match" or "TURBO: Vector Search"
├── Generation: "TURBO: Direct Answer"  (if vector search used)
└── Score: "response_confidence"
```

### LLM Generation Details

Each LLM call captures:

| Field | Description |
|---|---|
| `model` | Model name (e.g., `gpt-4.1-mini`) |
| `input` | Prompt text (truncated to 2000 chars) |
| `output` | Response text (truncated to 2000 chars) |
| `usage.prompt_tokens` | Input token count |
| `usage.completion_tokens` | Output token count |
| `usage.total_tokens` | Total token count |
| `duration_ms` | API call latency |

---

## Governance Metrics in Langfuse

The governance engine's four checks are emitted as **Langfuse scores** on each trace:

| Score Name | Range | Description |
|---|---|---|
| `gov_hallucination_score` | 0.0–1.0 | Factual correctness (1.0 = fully correct) |
| `gov_bias_score` | 0.0–1.0 | Bias/toxicity check (1.0 = no bias detected) |
| `gov_pii_score` | 0.0–1.0 | PII detection (1.0 = no PII found) |
| `gov_compliance_score` | 0.0–1.0 | Regulatory compliance (1.0 = fully compliant) |
| `gov_overall_status` | 0.0–1.0 | Overall verdict: 1.0=approved, 0.5=warning, 0.25=escalated, 0.0=blocked |

### Viewing Governance Metrics

1. Open Langfuse UI → **Traces**
2. Click on any trace
3. Scroll to the **Scores** section
4. Filter by score name (e.g., `gov_hallucination_score`)

### Setting Up Governance Alerts

In Langfuse, you can create alerts based on score thresholds:

1. Go to **Settings → Alerts**
2. Create a new alert:
   - Score: `gov_overall_status`
   - Condition: `value < 0.5`
   - Action: Email/Slack notification
3. This will alert you whenever a response is escalated or blocked

---

## Monitoring Dashboards

### Built-in Langfuse Dashboards

Langfuse provides several built-in views:

| Dashboard | What It Shows |
|---|---|
| **Traces** | All pipeline executions with latency, tokens, scores |
| **Generations** | All LLM calls with model, tokens, cost |
| **Scores** | Governance and quality metrics over time |
| **Sessions** | Multi-turn conversation groupings |
| **Users** | Per-user query patterns and costs |
| **Models** | Model-level cost and latency analytics |

### Recommended Filters

| Filter | Purpose |
|---|---|
| `tags: turbo-mode` | View only turbo mode queries |
| `tags: banking-rag` | View all RAG queries |
| `scores.gov_overall_status < 0.5` | Find escalated/blocked responses |
| `scores.response_confidence < 0.3` | Find low-confidence responses |
| `scores.gov_hallucination_score < 0.7` | Find potential hallucinations |

### Custom Analytics Queries

Langfuse supports SQL-like queries for custom analytics:

```sql
-- Average governance scores over the last 7 days
SELECT
  DATE(timestamp) as date,
  AVG(CASE WHEN name = 'gov_hallucination_score' THEN value END) as avg_hallucination,
  AVG(CASE WHEN name = 'gov_bias_score' THEN value END) as avg_bias,
  AVG(CASE WHEN name = 'gov_pii_score' THEN value END) as avg_pii,
  AVG(CASE WHEN name = 'gov_compliance_score' THEN value END) as avg_compliance
FROM scores
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY DATE(timestamp)
ORDER BY date;
```

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---|---|---|
| `[LANGFUSE] SDK not installed` | `langfuse` package missing | Run `pip install langfuse` |
| `[LANGFUSE] Initialization failed` | Wrong host URL or keys | Check `.env` values, ensure Langfuse is running |
| `[LANGFUSE] Disabled in config` | `langfuse.enabled = False` | Set to `True` in `config/settings.py` or env |
| `[RAG] Langfuse configured but not connected` | Server unreachable | Check `docker compose ps`, verify port |
| Traces not appearing in UI | Flush delay | Wait 5–10 seconds, or check `flush_interval_seconds` |
| Missing governance scores | `trace_governance = False` | Set to `True` in config |

### Verifying the Connection

```python
# Quick test script
import os
os.environ['LANGFUSE_HOST'] = 'http://localhost:3100'
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-...'
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-...'

from langfuse import Langfuse
client = Langfuse()
trace = client.trace(name="test-trace", input={"test": True})
client.flush()
print(f"✅ Trace created: {trace.id}")
```

### Checking Docker Status

```bash
# Check if Langfuse is running
docker compose -f docker-compose-langfuse.yml ps

# View Langfuse logs
docker compose -f docker-compose-langfuse.yml logs langfuse-server

# Restart Langfuse
docker compose -f docker-compose-langfuse.yml restart
```

---

## Files Changed

### New Files

| File | Purpose |
|---|---|
| `observability/__init__.py` | Module exports |
| `observability/langfuse_integration.py` | Core Langfuse SDK wrapper (singleton tracker) |
| `observability/rag_pipeline_tracer.py` | High-level RAG pipeline instrumentation |
| `docker-compose-langfuse.yml` | Self-hosted Langfuse Docker Compose |
| `.env.langfuse` | Environment variable template |
| `docs/LANGFUSE_INTEGRATION.md` | This documentation |

### Modified Files

| File | Changes |
|---|---|
| `config/settings.py` | Added `LangfuseConfig` dataclass with all toggle settings |
| `rag_engine/seven_layer_rag.py` | Added import, `self.tracer` init, and tracing calls in `process_query()`, `_call_llm()`, FAQ router, and turbo pipeline |

### No Files Removed or Broken

The integration is fully additive. If Langfuse is disabled or the SDK is not installed, the application behaves exactly as before — zero performance impact.
