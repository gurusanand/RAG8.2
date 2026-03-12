# Troubleshooting Reference

## Table of Contents
1. [Document Extraction Issues](#document-extraction-issues)
2. [Retrieval Quality Issues](#retrieval-quality-issues)
3. [Governance False Positives](#governance-false-positives)
4. [Persistence Issues](#persistence-issues)
5. [Performance Optimization](#performance-optimization)

## Document Extraction Issues

### Problem: Table data not associated with correct entity
**Symptom**: Query "annual fee for PlatinumPlus" returns wrong fee or no fee.
**Root cause**: Visual tables have card names and fees in separate text blocks at different coordinates.
**Fix**: Enable targeted Vision extraction for pages with product comparison tables:
```python
# In extraction_orchestrator.py, _run_targeted_vision()
# Detects when multiple product names appear in one table cell (misaligned data)
# Auto-triggers GPT-4 Vision on just that page (~10s, not full extraction)
```

### Problem: pdfplumber returns garbled table headers
**Symptom**: Table headers are `['', 'Free for life', '', ...]` with no card names.
**Root cause**: Complex visual layouts with merged cells, card images, multi-line content.
**Fix**: Always run coordinate-based PyMuPDF extraction alongside pdfplumber (not as fallback):
```python
# In table_extractor.py extract():
# 1. Run pdfplumber (good for simple tables)
# 2. Always also run _extract_tables_pymupdf_coordinate() (catches visual tables)
# 3. Deduplicate results
```

### Problem: Bilingual content (EN/AR) creates mixed chunks
**Symptom**: Arabic and English text interleaved in same chunk.
**Fix**: Vision extractor prompt includes instruction to separate languages. Coordinate-based extraction groups by Y-position which naturally separates bilingual rows.

## Retrieval Quality Issues

### Problem: Correct chunk exists but not retrieved
**Symptom**: Document contains the answer but retriever returns irrelevant chunks.
**Fix checklist**:
1. Check if HyDE is enabled (Layer 2) — generates hypothetical answer for better embedding match
2. Check if Hybrid Search is working — BM25 catches keyword matches that vector search misses
3. Verify chunk size — too large chunks dilute relevance, too small lose context
4. Check if contextual enrichment ran — chunks without document context are harder to match

### Problem: Retrieved chunks are relevant but answer is wrong
**Symptom**: Good chunks in context but LLM generates incorrect answer.
**Fix checklist**:
1. Check CRAG layer (Layer 4) — should filter irrelevant chunks before generation
2. Check Re-Ranking (Layer 5) — should prioritize most relevant chunks
3. Check Agentic RAG (Layer 6) — should trigger sub-queries if context insufficient
4. Review prompt templates in `prompts/prompt_manager.py`

## Governance False Positives

### Problem: Valid answers flagged for "Human Agent Review"
**Symptom**: Standard banking product info (fees, cashback) triggers escalation.
**Root cause**: Governance checks too aggressive — single LLM check returning "fail" triggers escalation.
**Fix applied**:
1. Governance prompts explicitly state: standard banking product info MUST pass
2. Escalation requires 3+ fails or 2+ fails with avg score < 0.4 (not single fail)
3. Even when escalated, answer is delivered with disclaimer (never silently replaced)

### Problem: Governance check returns unparseable JSON
**Symptom**: LLM returns malformed JSON, check defaults to "fail".
**Fix**: Each check has try/except with default to "pass" (score 0.5) on parse failure:
```python
try:
    result = json.loads(llm_response)
except json.JSONDecodeError:
    result = {"status": "pass", "score": 0.5, "issues": [], "action": "approve"}
```

## Persistence Issues

### Problem: Documents lost after server restart
**Symptom**: Uploaded documents disappear, need to re-upload.
**Fix checklist**:
1. Verify MongoDB is running: `pgrep -x mongod`
2. Verify FAISS index directory exists: `ls data/faiss_index/`
3. Check PersistenceManager initialization in `seven_layer_rag.py __init__`
4. Verify `initialize()` calls `self.persistence.load_all_to_indexer()`

### Problem: Duplicate documents after re-upload
**Symptom**: Same document indexed twice, duplicate chunks in results.
**Fix**: `index_uploaded_document()` checks `persistence.document_exists(filename)` before processing. If exists, returns early with "already indexed" message.

## Performance Optimization

| Bottleneck | Cause | Fix |
|-----------|-------|-----|
| Upload takes 10+ min | Full extraction (Vision + Graph) on all pages | Enable Fast Mode (default) — ~30s |
| First query slow | SentenceTransformer model loading | Model cached after first load via `@st.cache_resource` |
| Vision API slow | Analyzing all 12 pages | Page sampling: max 3 representative pages |
| Graph construction slow | LLM entity extraction on 20 chunks | Limit to 5 chunks in fast mode |
| MongoDB connection slow | Cold start | Connection pooling via `MongoClient` singleton |
