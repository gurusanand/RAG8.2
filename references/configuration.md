# Configuration Reference

## Table of Contents
1. [Settings Structure](#settings-structure)
2. [Feature Toggles](#feature-toggles)
3. [MongoDB Setup](#mongodb-setup)
4. [Dependencies](#dependencies)

## Settings Structure

All config lives in `config/settings.py` using Python dataclasses. Access via `get_settings()`.

```python
@dataclass
class RAGLayerConfig:
    layer_0_enabled: bool = True   # Product Orchestration
    layer_1_enabled: bool = True   # Semantic Cache
    layer_2_enabled: bool = True   # HyDE
    layer_3_enabled: bool = True   # Hybrid Retrieval
    layer_4_enabled: bool = True   # CRAG
    layer_5_enabled: bool = True   # Re-Ranking
    layer_6_enabled: bool = True   # Agentic RAG
    layer_7_enabled: bool = True   # Response Validation

@dataclass
class ExtractionConfig:
    vision_enabled: bool = True
    table_enabled: bool = True
    graph_enabled: bool = True
    enrichment_enabled: bool = True
    fast_mode: bool = True          # Default ON: skips Vision + Graph
    vision_max_pages: int = 3       # Max pages for Vision sampling
    graph_max_chunks: int = 5       # Max chunks for graph construction

@dataclass
class MongoDBConfig:
    enabled: bool = True
    uri: str = "mongodb://localhost:27017"
    database: str = "banking_rag"
```

## Feature Toggles

Every feature is toggleable via the admin sidebar UI or config file:

| Feature | Config Key | Default | Effect When Disabled |
|---------|-----------|---------|---------------------|
| Fast Mode | `extraction_config.fast_mode` | `True` | Enables Vision + Graph (slow, thorough) |
| Vision Extraction | `extraction_config.vision_enabled` | `True` | Falls back to text-only extraction |
| Table Extraction | `extraction_config.table_enabled` | `True` | No structured table parsing |
| Graph Construction | `extraction_config.graph_enabled` | `True` | No entity-relationship reasoning |
| Contextual Enrichment | `extraction_config.enrichment_enabled` | `True` | Raw chunks without context prefix |
| MongoDB Persistence | `mongodb.enabled` | `True` | Falls back to file-based persistence |
| Each RAG Layer | `layers.layer_N_enabled` | `True` | Layer skipped in pipeline |

## MongoDB Setup

```bash
# Install MongoDB
sudo apt-get install -y gnupg curl
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg
echo "deb [signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update && sudo apt-get install -y mongodb-org

# Start MongoDB
sudo mkdir -p /data/db
sudo mongod --dbpath /data/db --fork --logpath /tmp/mongod.log

# Verify
mongosh --eval "db.runCommand({ping:1})"
```

## Dependencies

```
# Core
streamlit>=1.28.0
openai>=1.0.0
sentence-transformers>=2.2.0
pymongo>=4.6.0
faiss-cpu>=1.7.4

# Extraction
pdfplumber>=0.10.0
PyMuPDF>=1.23.0
pytesseract>=0.3.10
pdf2image>=1.16.3
Pillow>=10.0.0
networkx>=3.0

# RAG Innovations
numpy>=1.24.0
scikit-learn>=1.3.0
rank-bm25>=0.2.2

# System
tesseract-ocr (apt package)
poppler-utils (apt package)
```

### Install Commands

```bash
# Python packages
pip3 install streamlit openai sentence-transformers pymongo faiss-cpu \
    pdfplumber PyMuPDF pytesseract pdf2image Pillow networkx \
    numpy scikit-learn rank-bm25

# System packages
sudo apt-get install -y tesseract-ocr poppler-utils
```
