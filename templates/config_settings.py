"""
Banking RAG Pipeline - Configuration Settings Template
Centralized configuration with feature toggles for all extraction strategies and RAG layers.
"""
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# RAG Layer Configuration
# ============================================================
@dataclass
class RAGLayerConfig:
    """Toggle individual RAG pipeline layers on/off."""
    layer_0_enabled: bool = True   # Product Orchestration (query routing)
    layer_1_enabled: bool = True   # Semantic Cache (cosine similarity ≥ 0.92)
    layer_2_enabled: bool = True   # HyDE (Hypothetical Document Embedding)
    layer_3_enabled: bool = True   # Hybrid Retrieval (BM25 + Vector + RRF)
    layer_4_enabled: bool = True   # CRAG (Corrective RAG with LLM grading)
    layer_5_enabled: bool = True   # Re-Ranking (cross-encoder scoring)
    layer_6_enabled: bool = True   # Agentic RAG (self-healing sub-queries)
    layer_7_enabled: bool = True   # Response Validation (governance checks)

# ============================================================
# Extraction Configuration
# ============================================================
@dataclass
class ExtractionConfig:
    """Toggle individual extraction engines and set parameters."""
    vision_enabled: bool = True         # GPT-4 Vision API extraction
    table_enabled: bool = True          # 3-tier table extraction
    graph_enabled: bool = True          # Knowledge graph construction
    enrichment_enabled: bool = True     # Contextual enrichment
    fast_mode: bool = True              # Default ON: skips Vision + Graph for speed
    vision_max_pages: int = 3           # Max pages for Vision sampling
    graph_max_chunks: int = 5           # Max chunks for graph entity extraction
    coordinate_extraction: bool = True  # Always run PyMuPDF coordinate-based (recommended)

# ============================================================
# MongoDB Configuration
# ============================================================
@dataclass
class MongoDBConfig:
    """MongoDB connection and storage settings."""
    enabled: bool = True
    uri: str = "mongodb://localhost:27017"
    database: str = "banking_rag"
    # Collection names
    documents_collection: str = "documents"
    chunks_collection: str = "chunks"
    tables_collection: str = "tables"
    graph_entities_collection: str = "graph_entities"
    graph_relationships_collection: str = "graph_relationships"

# ============================================================
# FAISS Configuration
# ============================================================
@dataclass
class FAISSConfig:
    """FAISS vector store settings."""
    index_dir: str = "data/faiss_index"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    similarity_metric: str = "cosine"  # cosine or l2

# ============================================================
# Governance Configuration
# ============================================================
@dataclass
class GovernanceConfig:
    """4-check governance system settings."""
    enabled: bool = True
    hallucination_check: bool = True
    bias_check: bool = True
    pii_check: bool = True
    compliance_check: bool = True
    # Scoring-based escalation thresholds
    escalation_fail_threshold: int = 3       # Fails needed for auto-escalate
    warning_fail_threshold: int = 2          # Fails needed for warning
    min_avg_score_for_warning: float = 0.4   # Min avg score to stay at warning (not escalate)
    block_threshold: int = 2                 # Block actions needed for block
    framework: str = "CBUAE"                 # Regulatory framework alignment

# ============================================================
# LLM Configuration
# ============================================================
@dataclass
class LLMConfig:
    """LLM provider settings."""
    model: str = "gpt-4.1-mini"
    vision_model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    max_tokens: int = 2000
    # API key loaded from environment: OPENAI_API_KEY

# ============================================================
# Chunking Configuration
# ============================================================
@dataclass
class ChunkingConfig:
    """Document chunking parameters."""
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    separator: str = "\n\n"

# ============================================================
# Application Settings
# ============================================================
@dataclass
class AppSettings:
    """Top-level application settings."""
    app_title: str = "Banking RAG Assistant"
    layers: RAGLayerConfig = field(default_factory=RAGLayerConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    mongodb: MongoDBConfig = field(default_factory=MongoDBConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    # Paths
    upload_dir: str = "data/uploaded_docs"
    index_dir: str = "data/index"

# Singleton
_settings: Optional[AppSettings] = None

def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings
