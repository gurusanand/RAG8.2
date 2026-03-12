[README.md](https://github.com/user-attachments/files/25928115/README.md)
# Local Setup Guide — Banking RAG Application

## Prerequisites

- **Python 3.10+** (recommended: 3.11)
- **MongoDB 7.0+** (Community Edition)
- **OpenAI API Key** (for LLM calls and embeddings)

---

## Step 1: Install MongoDB

### Windows
1. Download MongoDB Community Server from: https://www.mongodb.com/try/download/community
2. Run the installer (choose "Complete" installation)
3. Check "Install MongoDB as a Service" during setup
4. MongoDB will auto-start on boot and listen on `localhost:27017`

### macOS
```bash
brew tap mongodb/brew
brew install mongodb-community@7.0
brew services start mongodb-community@7.0
```

### Linux (Ubuntu/Debian)
```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] http://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update && sudo apt-get install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
```

### Verify MongoDB is running
```bash
mongosh --eval "db.runCommand({ ping: 1 })"
```

---

## Step 2: Install Python Dependencies

```bash
cd banking-rag-app
pip install -r requirements.txt
```

### Key dependencies installed:

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `openai` | LLM API client |
| `sentence-transformers` | Local embedding model (all-MiniLM-L6-v2) |
| `torch` | PyTorch (required by sentence-transformers) |
| `pymongo` | MongoDB driver |
| `faiss-cpu` | Vector similarity search |
| `PyMuPDF` | PDF text extraction |
| `pdfplumber` | Table extraction from PDFs |
| `pdf2image` | PDF page to image conversion (for Vision extraction) |
| `Pillow` | Image processing |
| `pytesseract` | OCR fallback (optional) |
| `networkx` | Knowledge graph construction |
| `tabulate` | Table formatting |

---

## Step 3: Set Environment Variables

### Option A: Using .env.example (Recommended)
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

Then load before running:
- **Windows CMD:** `for /f "tokens=1,2 delims==" %a in (.env) do set %a=%b`
- **Windows PowerShell:** `Get-Content .env | ForEach-Object { $k,$v = $_ -split '=',2; [System.Environment]::SetEnvironmentVariable($k,$v) }`
- **macOS / Linux:** `export $(cat .env | xargs)`

### Option B: Set Directly

#### Windows (Command Prompt)
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

#### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
```

#### macOS / Linux
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Optional: Custom MongoDB URI (default: mongodb://localhost:27017)
```bash
export MONGODB_URI="mongodb://localhost:27017"
```

---

## Step 4: Run the Application

```bash
cd banking-rag-app
streamlit run app.py
```

The app will open at: **http://localhost:8501**

### Login credentials:
- **Admin:** Username `admin`, Password `admin88$`
- **User:** Any username and password

---

## Step 5: Upload Documents

1. Log in as admin
2. Go to the sidebar, find "Upload Documents (Admin Only)"
3. Upload your FAQ PDF files (supports TXT, PDF, MD)
4. Wait for processing (approximately 15-60 seconds in Fast Mode, longer for scanned PDFs)
5. Documents are persisted in MongoDB — no re-upload needed after restarting

### Scanned PDF Support
If your PDF is scanned (image-only, no text layer), the system automatically detects this and activates the Vision API fallback to extract text from images. This works even in Fast Mode.

---

## Step 6: Document Management (Admin Only)

1. Log in as admin
2. Click the **Admin Panel** tab
3. Scroll to the **Document Management** section
4. You will see all uploaded documents with details (chunks, pages, size, upload time, strategies)
5. Click **Delete** on any document to remove it from all stores (MongoDB, FAISS, in-memory index, FAQ pairs, hybrid search)
6. A confirmation dialog will appear before deletion

---

## How Persistence Works

| Component | What it stores | Where |
|-----------|---------------|-------|
| **MongoDB** | Document metadata, text chunks, tables, graph entities, FAQ Q&A pairs | `banking_rag` database |
| **FAISS** | Vector embeddings (for similarity search) | `data/faiss_index/` directory |

### On restart:
1. FAISS index loads from disk (no re-embedding needed)
2. Chunk metadata loads from MongoDB
3. FAQ Q&A pairs load from MongoDB (FAQ Smart Router works immediately)
4. No re-upload required!

### To verify persistence:
```bash
mongosh banking_rag --eval "
  print('Documents:', db.documents.countDocuments({}));
  print('Chunks:', db.chunks.countDocuments({}));
  print('FAQ pairs:', db.faq_pairs.countDocuments({}));
"
```

---

## Project Structure

```
banking-rag-app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── LOCAL_SETUP.md                  # This setup guide
├── config/
│   ├── __init__.py
│   └── settings.py                 # All configuration (toggles, thresholds, etc.)
├── rag_engine/
│   ├── __init__.py
│   ├── seven_layer_rag.py          # Core 7-Layer RAG engine
│   ├── faq_exact_match.py          # FAQ Smart Router (3-tier matching)
│   ├── document_chunker.py         # Document chunking logic
│   ├── product_orchestrator.py     # Product classification & routing
│   ├── extractors/
│   │   ├── extraction_orchestrator.py  # Multi-strategy extraction pipeline
│   │   ├── vision_extractor.py     # Vision API extraction (scanned PDFs)
│   │   ├── table_extractor.py      # Table extraction (pdfplumber/Docling)
│   │   ├── contextual_enrichment.py # Contextual prefix enrichment
│   │   └── knowledge_graph_builder.py # Knowledge graph construction
│   └── innovations/
│       ├── hybrid_search.py        # BM25 + Vector hybrid search
│       ├── contextual_retrieval.py # Anthropic-style contextual retrieval
│       ├── graph_rag.py            # Knowledge graph RAG
│       ├── adaptive_rag.py         # Dynamic strategy selection
│       ├── query_decomposition.py  # Multi-part query breakdown
│       ├── self_rag.py             # Self-reflective retrieval
│       ├── speculative_rag.py      # Parallel draft generation
│       ├── raptor_indexing.py      # Hierarchical tree indexing
│       └── ragas_evaluation.py     # RAGAS quality metrics
├── persistence/
│   ├── __init__.py
│   ├── persistence_manager.py      # Orchestrates MongoDB + FAISS persistence
│   ├── mongo_store.py              # MongoDB document/chunk/FAQ storage
│   └── faiss_store.py              # FAISS vector index persistence
├── prompts/
│   ├── __init__.py
│   └── prompt_manager.py           # All LLM prompts (response generation, etc.)
├── governance/
│   ├── __init__.py
│   └── governance_engine.py        # 4-check governance (hallucination, bias, PII, compliance)
├── feedback/
│   └── feedback_service.py         # User feedback collection
├── ui/
│   └── detailed_layer_display.py   # Detailed chunk display UI
└── data/
    ├── faiss_index/                # FAISS index files (auto-generated)
    ├── audit/                      # Audit trail logs
    └── page_images/                # Extracted page images (for vision)
```

---

## Configuration

All settings are in `config/settings.py`. Key toggles:

| Setting | Default | Description |
|---------|---------|-------------|
| `faq_exact_match_enabled` | `True` | FAQ Smart Router (3-tier) |
| `faq_exact_threshold` | `0.85` | Tier 1: Exact match threshold |
| `faq_fuzzy_threshold` | `0.60` | Tier 2: Fuzzy match threshold |
| `mongodb.enabled` | `True` | MongoDB persistence |
| `extraction_config.fast_mode` | `True` | Fast Mode (fewer LLM calls) |
| `governance_enabled` | `True` | CBUAE governance checks |
| `hybrid_search_enabled` | `True` | BM25 + Vector hybrid search |
| `contextual_retrieval_enabled` | `True` | Anthropic contextual retrieval |
| `graph_rag_enabled` | `True` | Knowledge graph RAG |
| `adaptive_rag_enabled` | `True` | Dynamic strategy selection |
| `self_rag_enabled` | `True` | Self-reflective retrieval |
| `speculative_rag_enabled` | `True` | Parallel draft generation |
| `raptor_enabled` | `True` | Hierarchical tree indexing |

---

## Turbo Mode

Turbo Mode provides instant FAQ answers with zero LLM calls for exact matches:

1. Enable Turbo Mode in the sidebar under "Layer Configuration"
2. Queries matching a stored FAQ at 85%+ similarity get instant answers (~50ms)
3. Queries matching at 60-85% get a lightweight LLM adaptation (~2s)
4. All other queries fall through to the full 7-layer pipeline

---

## Troubleshooting

### "pymongo not installed" error
```bash
pip install pymongo
```

### "faiss-cpu not installed" error
```bash
pip install faiss-cpu
```

### "PyMuPDF not installed" error
```bash
pip install PyMuPDF
```

### MongoDB connection refused
- Ensure MongoDB is running: `mongosh --eval "db.runCommand({ping:1})"`
- Windows: Check Services (Win+R → `services.msc`) → "MongoDB Server" is running
- Linux: `sudo systemctl status mongod`

### Slow document upload (>2 minutes)
- Ensure `fast_mode: True` in `config/settings.py` → `extraction_config`
- Fast Mode skips vision extraction, graph building, and per-chunk LLM enrichment
- Scanned PDFs (image-only) will take longer as they require Vision API extraction

### FAQ answers are truncated or incomplete
- Re-upload the document to re-extract FAQ pairs with the latest extraction logic
- Old FAQ pairs stored in MongoDB from a previous version may have truncated answers
- To clear old data: `mongosh banking_rag --eval "db.faq_pairs.drop()"`
- Then re-upload the document

### FAQ answers show question text instead of answer
- This was a bug in the question/answer splitting logic for multi-part questions
- The fix is included in this version — re-upload the document to re-extract

### Check stored FAQ pairs
```bash
mongosh banking_rag --eval "db.faq_pairs.find({question_number:'Q30'}).pretty()"
```

### Reset everything (start fresh)
```bash
mongosh banking_rag --eval "
  db.documents.drop();
  db.chunks.drop();
  db.faq_pairs.drop();
  db.graph_entities.drop();
  print('All collections dropped.');
"
# Also delete FAISS index files:
rm -rf data/faiss_index/*
```
