"""
Configuration settings for the 7-Layer Advanced RAG Banking Application.
All settings are centralized here for easy modification.
"""
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    """Application-level configuration."""
    app_title: str = "7-Layer Advanced RAG — Banking Assistant"
    app_icon: str = "🏦"
    version: str = "1.0.0"

@dataclass
class AuthConfig:
    """Authentication configuration."""
    admin_username: str = "admin"
    admin_password: str = "admin88$"
    session_timeout_minutes: int = 60

@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    temperature: float = 0.1
    max_tokens: int = 2048
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))

@dataclass
class RAGLayerConfig:
    """Configuration for each of the 7 RAG layers."""
    # Layer 1: Semantic Caching
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_max_entries: int = 500

    # Layer 2: HyDE (Query Transformation)
    hyde_enabled: bool = True
    hyde_min_query_length: int = 10  # Skip HyDE for very short queries

    # Layer 3: Semantic Chunking & Retrieval
    chunk_size_min: int = 200
    chunk_size_max: int = 800
    chunk_overlap: int = 50
    retrieval_top_k: int = 20  # Initial retrieval count

    # Layer 4: Corrective RAG (CRAG)
    crag_enabled: bool = True
    crag_quality_threshold_correct: float = 0.7
    crag_quality_threshold_ambiguous: float = 0.4
    crag_web_fallback_enabled: bool = True
    crag_trusted_urls: List[str] = field(default_factory=lambda: [
        "https://www.mashreqbank.com",
        "https://www.centralbank.ae",
    ])

    # Layer 5: LLM Re-Ranking
    rerank_enabled: bool = True
    rerank_top_k: int = 5  # Final top-K after re-ranking
    rerank_diversity_enabled: bool = True

    # Layer 6: Agentic RAG (ReAct)
    agentic_enabled: bool = True
    agentic_max_iterations: int = 5
    agentic_complexity_threshold: float = 0.6  # Activate for complex queries

    # Layer 7: Response Validation
    validation_enabled: bool = True
    validation_confidence_high: float = 0.70  # Deliver
    validation_confidence_medium: float = 0.30  # Warn
    # Below 0.30 = Block + Human Handoff
    hallucination_check_enabled: bool = True

    # Detailed Layer Display (chunk-level info for Layers 3-7)
    detailed_display_enabled: bool = True

    # Orchestration Layer (Product Classification + Intent + Risk + Routing)
    orchestrator_enabled: bool = True
    orchestrator_use_llm: bool = True  # Use LLM for ambiguous classifications
    orchestrator_keyword_confidence_threshold: float = 0.75  # Above this, skip LLM

    # ═══════════════════════════════════════════════════════════════
    # RAG INNOVATIONS — 9 Cutting-Edge Techniques (All Toggleable)
    # ═══════════════════════════════════════════════════════════════

    # Horizon 1: High Feasibility
    hybrid_search_enabled: bool = True  # BM25 + Vector with Reciprocal Rank Fusion
    hybrid_search_bm25_weight: float = 0.3  # Weight for BM25 in RRF (0.0-1.0)
    hybrid_search_vector_weight: float = 0.7  # Weight for vector search in RRF
    hybrid_search_rrf_k: int = 60  # RRF constant (higher = more equal weighting)

    contextual_retrieval_enabled: bool = True  # Anthropic's context-prefix technique

    ragas_evaluation_enabled: bool = True  # RAGAS quality metrics per query
    ragas_sample_rate: float = 1.0  # 1.0 = every query, 0.5 = 50% of queries

    # Horizon 2: Medium Feasibility
    graph_rag_enabled: bool = True  # Knowledge Graph for cross-document reasoning
    graph_rag_max_hops: int = 2  # Max traversal hops in knowledge graph

    adaptive_rag_enabled: bool = True  # Dynamic strategy selection by query complexity

    query_decomposition_enabled: bool = True  # Multi-part query breakdown
    query_decomposition_max_sub_queries: int = 5  # Max sub-queries

    # Horizon 3: Research-Grade
    self_rag_enabled: bool = True  # Self-reflective retrieval with correction loops
    self_rag_max_iterations: int = 3  # Max self-correction iterations

    speculative_rag_enabled: bool = True  # Parallel draft generation + verification
    speculative_rag_num_drafts: int = 3  # Number of parallel drafts

    raptor_enabled: bool = True  # Hierarchical tree of document summaries
    raptor_max_levels: int = 3  # Max tree depth

    # FAQ Smart Router — 3-Tier Query Routing via Embedding Similarity
    # Tier 1 (EXACT):  sim >= 0.85 → Return exact FAQ answer instantly (0 LLM calls, ~50ms)
    # Tier 2 (FUZZY):  sim 0.60-0.85 → Adapt FAQ answer with 1 lightweight LLM call (~2s)
    # Tier 3 (NOVEL):  sim < 0.60 → Fall through to full 7-layer pipeline (~10-20s)
    faq_exact_match_enabled: bool = True   # Master switch for FAQ smart routing
    faq_exact_threshold: float = 0.85      # Tier 1: Return exact answer, no LLM
    faq_fuzzy_threshold: float = 0.60      # Tier 2: Adapt answer with 1 LLM call
    faq_fuzzy_model: str = "gpt-4.1-nano"  # Lightweight model for fuzzy adaptation

    # ═══════════════════════════════════════════════════════════════
    # ADVANCED DOCUMENT EXTRACTION — Multi-Strategy Pipeline (All Toggleable)
    # ═══════════════════════════════════════════════════════════════
    extraction_orchestrator_enabled: bool = True  # Master switch for advanced extraction
    extraction_config: dict = field(default_factory=lambda: {
        'fast_mode': True,            # Fast mode: skip vision + graph, use table + enrichment only (~30s)
        'vision_enabled': True,       # ColPali-style LLM vision extraction (tables, images, formulas)
        'table_enabled': True,        # Docling/pdfplumber structured table extraction
        'graph_enabled': True,        # Knowledge graph construction (entity-relationship)
        'contextual_enabled': True,   # Contextual prefix enrichment + formula linearization
        'vision_max_pages': 3,        # Max pages to analyze with vision (first, middle, last)
        'graph_max_chunks': 5,        # Max chunks to process for graph construction
    })

    # Governance Engine (Four-Check System — CBUAE Aligned)
    governance_enabled: bool = True
    governance_check1_hallucination: bool = True  # Factual Correctness & Hallucination
    governance_check2_bias: bool = True  # Bias, Toxicity & Fairness
    governance_check3_pii: bool = True  # PII & Sensitive Data Redaction
    governance_check4_compliance: bool = True  # Regulatory & Compliance Validation
    governance_audit_trail_enabled: bool = True  # 14-field audit record per query
    governance_max_retries: int = 3  # Max retries on hallucination failure
    governance_ai_disclosure: bool = True  # Show AI disclosure banner to users

@dataclass
class LangfuseConfig:
    """Langfuse Observability configuration.
    
    Supports both self-hosted Langfuse (Docker) and Langfuse Cloud.
    
    Self-Hosted Setup:
      1. Run: docker compose -f docker-compose-langfuse.yml up -d
      2. Set host = "http://localhost:3100" (or your custom port)
      3. Create a project in the Langfuse UI and copy the API keys
    
    Cloud Setup:
      1. Sign up at https://cloud.langfuse.com
      2. Create a project and copy the API keys
      3. Leave host empty (defaults to cloud)
    
    Environment Variables (override config values):
      LANGFUSE_HOST         — Langfuse server URL (self-hosted)
      LANGFUSE_PUBLIC_KEY   — Project public key
      LANGFUSE_SECRET_KEY   — Project secret key
      LANGFUSE_ENABLED      — "true" or "false" to toggle
    """
    enabled: bool = True  # Master toggle — set False to disable all tracing
    host: str = field(default_factory=lambda: os.environ.get(
        "LANGFUSE_BASE_URL", os.environ.get("LANGFUSE_HOST", "")))  # Empty = Langfuse Cloud; set URL for self-hosted (SDK v4 uses LANGFUSE_BASE_URL)
    public_key: str = field(default_factory=lambda: os.environ.get(
        "LANGFUSE_PUBLIC_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.environ.get(
        "LANGFUSE_SECRET_KEY", ""))
    flush_interval_seconds: int = 5  # How often to flush events to Langfuse
    trace_sample_rate: float = 1.0  # 1.0 = trace every query, 0.5 = 50% sampling
    trace_llm_calls: bool = True  # Track individual LLM API calls (tokens, cost)
    trace_embeddings: bool = True  # Track embedding operations
    trace_governance: bool = True  # Emit governance scores (hallucination, bias, PII, compliance)
    trace_innovations: bool = True  # Track RAG innovation results (RAGAS, Self-RAG, etc.)
    trace_faq_routing: bool = True  # Track FAQ 3-tier routing decisions
    trace_document_indexing: bool = True  # Track document upload/indexing operations


@dataclass
class MongoDBConfig:
    """MongoDB configuration for product-specific vector stores."""
    enabled: bool = True
    connection_string: str = field(default_factory=lambda: os.environ.get(
        "MONGODB_URI", "mongodb://localhost:27017"))
    database_name: str = "banking_rag"
    # Collection names are defined in product_orchestrator.PRODUCT_CATALOG
    embedding_dimension: int = 1536
    batch_size: int = 100  # Batch size for bulk inserts
    # Ingestion pipeline settings
    ingestion_auto_detect_product: bool = True  # Auto-classify uploaded docs to product
    ingestion_require_approval: bool = True  # Require admin approval before indexing
    ingestion_pii_scan_enabled: bool = True  # Scan for PII before indexing
    # Refresh strategy
    refresh_strategy: str = "event_driven"  # event_driven, periodic, manual
    refresh_periodic_days: int = 180  # For periodic strategy


@dataclass
class FeedbackConfig:
    """Feedback configuration."""
    enabled: bool = True
    comments_enabled: bool = True
    feedback_file: str = "data/feedback/feedback.json"

@dataclass
class PathConfig:
    """File path configuration."""
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    documents_dir: str = "documents"
    sample_policies_dir: str = "documents/sample_policies"
    faiss_index_dir: str = "data/faiss_index"
    cache_dir: str = "data/cache"
    feedback_dir: str = "data/feedback"

    def get_abs_path(self, relative_path: str) -> str:
        return os.path.join(self.base_dir, relative_path)


class Settings:
    """Singleton settings manager."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.app = AppConfig()
            cls._instance.auth = AuthConfig()
            cls._instance.llm = LLMConfig()
            cls._instance.rag = RAGLayerConfig()
            cls._instance.feedback = FeedbackConfig()
            cls._instance.mongodb = MongoDBConfig()
            cls._instance.paths = PathConfig()
            cls._instance.langfuse = LangfuseConfig()
        return cls._instance


def get_settings() -> Settings:
    """Get the singleton settings instance."""
    return Settings()
