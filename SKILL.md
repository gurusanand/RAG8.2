---
name: banking-rag-pipeline
description: "Build production-grade RAG applications with multi-strategy document extraction, 7-layer retrieval pipeline, MongoDB/FAISS persistence, and AI governance. Use for building RAG systems, document Q&A apps, banking/financial AI assistants, enterprise search with governance, complex PDF extraction (tables, formulas, bilingual content)."
---

# Banking RAG Pipeline

Build a complete RAG application with 4-engine extraction, 7-layer retrieval, persistence, and governance. Adaptable to any domain (banking, insurance, legal, healthcare).

## Workflow

Building a RAG pipeline involves these steps:

1. Set up the project (run `setup_project.sh`)
2. Implement the extraction pipeline (4 engines)
3. Implement the 7-layer RAG pipeline
4. Add persistence (MongoDB + FAISS)
5. Add governance engine (4 checks)
6. Build the Streamlit UI
7. Test with domain documents

## Step 1: Project Setup

Run the setup script to create the directory structure and install dependencies:

```bash
bash /home/ubuntu/skills/banking-rag-pipeline/scripts/setup_project.sh <project-name>
```

Set the OpenAI API key:
```bash
export OPENAI_API_KEY="your-key"
```

Copy templates as starting points:
```bash
cp /home/ubuntu/skills/banking-rag-pipeline/templates/config_settings.py <project>/config/settings.py
cp /home/ubuntu/skills/banking-rag-pipeline/templates/prompt_manager.py <project>/prompts/prompt_manager.py
cp /home/ubuntu/skills/banking-rag-pipeline/templates/governance_engine.py <project>/governance/governance_engine.py
```

## Step 2: Extraction Pipeline

Implement 4 extraction engines in `rag_engine/extractors/`. Each engine handles a different document challenge:

| Engine | File | Handles | Key Algorithm |
|--------|------|---------|---------------|
| Vision | `vision_extractor.py` | Scanned PDFs, image tables | GPT-4 Vision API with page sampling |
| Table | `table_extractor.py` | Structured tables | 3-tier: Docling → pdfplumber → PyMuPDF coordinate |
| Graph | `knowledge_graph_builder.py` | Entity relationships | LLM extraction → NetworkX DiGraph |
| Enrichment | `contextual_enrichment.py` | Context-poor chunks | Document summary → chunk context prefix |

**Orchestrator pattern** (`extraction_orchestrator.py`):
```python
class ExtractionOrchestrator:
    def extract(self, pdf_path, fast_mode=True):
        results = {}
        if not fast_mode:
            results["vision"] = self.vision_extractor.extract(pdf_path)
            results["graph"] = self.graph_builder.build(chunks)
        results["tables"] = self.table_extractor.extract(pdf_path)  # Always run
        results["enriched"] = self.enrichment.enrich(chunks)         # Always run
        return self._fuse_results(results)
```

**Critical fix for product comparison tables**: When pdfplumber returns misaligned headers (card names missing from table cells), always run coordinate-based PyMuPDF extraction alongside it — not as a fallback. Group text blocks by Y-position (rows) and X-position (columns).

**Fast Mode** (default): Table + Enrichment only (~30s). **Full Mode**: All 4 engines (~10min).

For architecture details, see `references/architecture.md`.

## Step 3: 7-Layer RAG Pipeline

Implement in `rag_engine/seven_layer_rag.py`. Each layer is independently toggleable:

| Layer | Name | What It Does |
|-------|------|-------------|
| 0 | Product Orchestration | LLM classifies query → routes to correct domain |
| 1 | Semantic Cache | Cosine similarity ≥ 0.92 → return cached answer |
| 2 | HyDE | Generate hypothetical answer → embed for better retrieval |
| 3 | Hybrid Retrieval | BM25 keyword + Vector semantic + Reciprocal Rank Fusion |
| 4 | CRAG | LLM grades chunks as Correct/Ambiguous/Incorrect → filter |
| 5 | Re-Ranking | Cross-encoder scores chunks 0-10 → reorder |
| 6 | Agentic RAG | Check context sufficiency → trigger sub-queries if needed |
| 7 | Response Validation | Run governance 4-check system |

**Implementation pattern** — each layer follows:
```python
if self.config.layers.layer_N_enabled:
    result = self._layer_N(query, context)
    layer_log.append({"layer": N, "status": "done", "details": result})
else:
    layer_log.append({"layer": N, "status": "skipped"})
```

**Hybrid search (Layer 3)** combines BM25 and vector scores via Reciprocal Rank Fusion:
```python
rrf_score = sum(1.0 / (k + rank) for each ranking)  # k=60 typical
```

## Step 4: Persistence

**MongoDB** (`persistence/mongo_store.py`): Store documents, chunks, tables, graph entities in separate collections. Check `document_exists(filename)` before re-indexing.

**FAISS** (`persistence/faiss_store.py`): Use `IndexFlatIP` on normalized vectors for cosine similarity. Save index + metadata JSON sidecar to disk after every add.

```python
# Load on startup
if os.path.exists(index_path):
    self.index = faiss.read_index(index_path)
    self.metadata = json.load(open(metadata_path))
```

For configuration details, see `references/configuration.md`.

## Step 5: Governance Engine

Use the template at `templates/governance_engine.py`. 4 checks aligned to CBUAE framework:

1. **Hallucination**: LLM cross-references answer with source chunks
2. **Bias/Toxicity**: LLM checks for discriminatory language
3. **PII**: Regex patterns (credit cards, account numbers, Emirates ID)
4. **Compliance**: LLM checks for unauthorized financial advice

**Critical: scoring-based escalation** (not single-fail):
- Approved: 0-1 fails, avg score ≥ 0.2
- Warning: 2 fails, avg score ≥ 0.4 → deliver with disclaimer
- Escalated: 3+ fails or 2+ fails with avg score < 0.4 → deliver with human review flag
- Blocked: 2+ checks return "block" action

**Key rule**: Even when escalated, deliver the answer with a disclaimer. Never silently replace a valid answer.

Governance prompts must explicitly state: standard product information (fees, rates, features) MUST pass all checks.

## Step 6: Streamlit UI

Build `app.py` with these components:

```python
# Login with role-based access
if role == "admin":
    show_admin_dashboard()   # Analytics, layer config, document library
    show_layer_toggles()     # Enable/disable each RAG layer
else:
    show_chat_interface()    # Query input + response with layer trace
    show_document_upload()   # PDF upload with progress bar
```

**Admin features**: Upload documents, toggle Fast/Full mode, enable/disable individual layers, view analytics (query count, avg response time, governance stats).

**User features**: Chat interface with expandable layer-by-layer trace, document upload with real-time progress.

Cache the embedding model with `@st.cache_resource` to avoid reloading on every interaction.

## Step 7: Testing

Test with these query types to validate the pipeline:

| Query Type | Example | Tests |
|-----------|---------|-------|
| Specific data | "Annual fee for PlatinumPlus card" | Table extraction accuracy |
| Comparison | "Compare Gold vs Platinum benefits" | Multi-chunk retrieval |
| Calculation | "Interest on AED 50,000 at 3.5%" | Formula extraction |
| General | "What credit cards do you offer?" | Product orchestration |
| Edge case | "Tell me about XYZ product" (not in docs) | Graceful "not found" handling |

For common issues and fixes, see `references/troubleshooting.md`.

## Resources

### scripts/
- `setup_project.sh` — Create project directory structure and install all dependencies

### references/
- `architecture.md` — Full architecture details: project structure, 7-layer pipeline algorithms, 4-engine extraction, persistence flow, governance logic
- `configuration.md` — All settings, feature toggles, MongoDB setup, dependency list
- `troubleshooting.md` — Common issues with table extraction, retrieval quality, governance false positives, persistence, and performance

### templates/
- `config_settings.py` — Complete settings module with dataclasses and feature toggles
- `governance_engine.py` — 4-check governance engine with scoring-based escalation
- `prompt_manager.py` — Centralized prompt management with all extraction, RAG, and governance prompts
