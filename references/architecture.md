# Architecture Reference

## Table of Contents
1. [Project Structure](#project-structure)
2. [7-Layer RAG Pipeline](#7-layer-rag-pipeline)
3. [4-Engine Extraction Pipeline](#4-engine-extraction-pipeline)
4. [Persistence Layer](#persistence-layer)
5. [Governance Engine](#governance-engine)

## Project Structure

```
banking-rag-app/
├── app.py                          # Streamlit UI (login, chat, admin, upload)
├── config/
│   └── settings.py                 # Centralized config with feature toggles
├── prompts/
│   └── prompt_manager.py           # All LLM prompt templates (centralized)
├── rag_engine/
│   ├── seven_layer_rag.py          # Core pipeline: SevenLayerRAG, DocumentIndexer
│   ├── document_chunker.py         # UniversalDocumentChunker (semantic + overlap)
│   ├── product_orchestrator.py     # Query classification + product routing
│   ├── innovations/                # 9 RAG technique modules
│   │   ├── hybrid_search.py        # BM25 + Vector with Reciprocal Rank Fusion
│   │   ├── contextual_retrieval.py # Anthropic-style chunk enrichment
│   │   ├── graph_rag.py            # Knowledge graph entity reasoning
│   │   ├── adaptive_rag.py         # Query complexity routing
│   │   ├── query_decomposition.py  # Multi-hop query splitting
│   │   ├── self_rag.py             # Self-reflective retrieval
│   │   ├── speculative_rag.py      # Parallel draft generation
│   │   ├── raptor_indexing.py      # Hierarchical summarization tree
│   │   └── ragas_evaluation.py     # RAGAS metric evaluation
│   └── extractors/                 # Multi-strategy document extraction
│       ├── extraction_orchestrator.py  # Strategy selection + fusion
│       ├── vision_extractor.py     # GPT-4 Vision + OCR fallback
│       ├── table_extractor.py      # Docling + pdfplumber + PyMuPDF coordinate
│       ├── knowledge_graph_builder.py  # LLM entity extraction + NetworkX
│       └── contextual_enrichment.py    # Chunk context injection + formula linearization
├── governance/
│   └── governance_engine.py        # 4-check system (Hallucination, Bias, PII, Compliance)
├── persistence/
│   ├── mongo_store.py              # MongoDB document/chunk store
│   ├── faiss_store.py              # FAISS persistent vector index
│   └── persistence_manager.py      # Integration bridge
├── feedback/
│   └── feedback_service.py         # User feedback collection + analytics
└── ui/
    └── detailed_layer_display.py   # Pipeline execution visualization
```

## 7-Layer RAG Pipeline

Each query passes through 7 sequential layers. Each layer is toggleable via config.

| Layer | Name | Algorithm | Purpose |
|-------|------|-----------|---------|
| 0 | Product Orchestration | LLM classifier + confidence scoring | Route query to correct product domain |
| 1 | Semantic Cache | Cosine similarity (threshold 0.92) | Skip pipeline for repeated queries |
| 2 | HyDE | Hypothetical Document Embedding | Generate hypothetical answer, embed it for better retrieval |
| 3 | Hybrid Retrieval | BM25 + Vector + Reciprocal Rank Fusion | Retrieve chunks using keyword + semantic search |
| 4 | CRAG | Corrective RAG with LLM grading | Grade chunks as Correct/Ambiguous/Incorrect, filter bad ones |
| 5 | Re-Ranking | Cross-encoder scoring (0-10 scale) | Re-score chunks for final relevance ordering |
| 6 | Agentic RAG | Context sufficiency check + sub-queries | Self-healing: triggers additional retrieval if context insufficient |
| 7 | Response Validation | Governance 4-check system | Hallucination, bias, PII, compliance validation |

### Key Implementation Pattern

```python
# Each layer follows this pattern in process_query():
if self.config.layers.layer_N_enabled:
    result = self._layerN_method(query, context)
    layer_results.append({"layer": N, "status": "approved", "details": result})
else:
    layer_results.append({"layer": N, "status": "skipped"})
```

## 4-Engine Extraction Pipeline

The extraction orchestrator profiles each document and selects strategies.

### Engine 1: Vision Extractor
- **Algorithm**: Converts PDF pages to images, sends to GPT-4 Vision API
- **Prompt**: Structured extraction of text, tables, formulas, visual elements
- **Optimization**: Page sampling (max 3 representative pages in fast mode)
- **Fallback**: Tesseract OCR when API unavailable
- **Key use case**: Scanned PDFs, image-based tables, color-coded layouts

### Engine 2: Table Extractor
- **Algorithm**: 3-tier extraction (Docling → pdfplumber → PyMuPDF coordinate-based)
- **Coordinate-based**: Groups text blocks by Y-position (rows) and X-position (columns)
- **Key fix**: Always runs coordinate-based extraction alongside pdfplumber (not fallback)
- **Output**: Markdown tables + per-row chunks for precise retrieval

### Engine 3: Knowledge Graph Builder
- **Algorithm**: LLM entity extraction → NetworkX DiGraph → BFS traversal
- **Entities**: Products, fees, rates, policies, regulations
- **Relationships**: has_fee, requires, applies_to, governed_by
- **Query**: BFS from matched entities to find related information
- **Export**: JSON serializable for persistence

### Engine 4: Contextual Enrichment
- **Algorithm**: Anthropic-style contextual retrieval
- **Process**: Generate document summary → prepend context to each chunk
- **Formula handling**: Linearize mathematical expressions (e.g., "APR = (interest/principal) * 100")
- **Output**: Enriched chunks with document-level context

### Orchestrator Strategy Selection

```python
# Fast Mode (default): Table + Enrichment only (~30s)
# Full Mode: All 4 engines (~10min for 12-page PDF)
# Targeted Vision: Auto-triggers for pages with misaligned product tables
```

## Persistence Layer

### MongoDB Collections
- `documents`: Metadata (filename, hash, upload_time, page_count, status)
- `chunks`: Text, embeddings reference, source, page, extraction_method
- `tables`: Structured table data with page reference
- `graph_entities`: Entity nodes from knowledge graph
- `graph_relationships`: Entity relationships

### FAISS Vector Store
- Index type: `IndexFlatIP` (inner product for cosine similarity on normalized vectors)
- Metadata: JSON sidecar file mapping FAISS IDs to chunk metadata
- Auto-save: Writes to disk after every add operation
- Deduplication: Content hash check before adding

### Persistence Flow
```
Upload PDF → Extract → Chunk → Embed → Store in MongoDB + FAISS → Save to disk
Server restart → Load FAISS from disk + MongoDB retains data → No re-processing
```

## Governance Engine

4-check system aligned with CBUAE framework:

| Check | Method | What It Detects |
|-------|--------|----------------|
| 1. Hallucination | LLM cross-reference with sources | Fabricated data, unsupported claims |
| 2. Bias/Toxicity | LLM content analysis | Discriminatory language, unfair targeting |
| 3. PII Detection | Regex patterns | Credit card numbers, account numbers, Emirates ID |
| 4. Compliance | LLM regulatory check | Unauthorized financial advice, missing disclaimers |

### Escalation Logic (scoring-based, not single-fail)
- **Approved**: 0-1 fails with score >= 0.2
- **Warning**: 2 fails with avg score >= 0.4
- **Escalated**: 3+ fails, or 2+ fails with avg score < 0.4
- **Blocked**: 2+ checks return "block" action
- **Key rule**: Even when escalated, deliver the answer with a disclaimer (never silently replace)
