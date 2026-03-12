"""
7-Layer Advanced RAG Engine
Implements all 7 layers: Caching, HyDE, Chunking/Retrieval, CRAG, Re-Ranking, Agentic RAG, Response Validation.
Enhanced with detailed chunk-level metadata capture for Layers 3-7.
"""
import os
import json
import time
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_settings
from prompts.prompt_manager import PromptManager
from rag_engine.product_orchestrator import (
    ProductOrchestrator, OrchestrationResult, ClassificationResult,
    PRODUCT_CATALOG, GENERAL_COLLECTION
)
from governance.governance_engine import GovernanceEngine, GovernanceResult
from rag_engine.document_chunker import UniversalDocumentChunker

# Persistence Layer (MongoDB + FAISS)
try:
    from persistence.persistence_manager import PersistenceManager
    PERSISTENCE_AVAILABLE = True
except ImportError as e:
    print(f"[PERSISTENCE] Persistence layer not available: {e}")
    PERSISTENCE_AVAILABLE = False

# Advanced Extraction Pipeline (all toggleable)
try:
    from rag_engine.extractors.extraction_orchestrator import ExtractionOrchestrator
    EXTRACTION_AVAILABLE = True
except ImportError as e:
    print(f"[EXTRACTION] Advanced extraction pipeline not available: {e}")
    EXTRACTION_AVAILABLE = False

# Innovation modules (all toggleable)
try:
    from rag_engine.innovations.hybrid_search import HybridSearchEngine
    from rag_engine.innovations.contextual_retrieval import ContextualRetrievalEngine
    from rag_engine.innovations.ragas_evaluation import RAGASEvaluator
    from rag_engine.innovations.graph_rag import KnowledgeGraphRAG
    from rag_engine.innovations.adaptive_rag import AdaptiveRAGRouter
    from rag_engine.innovations.query_decomposition import QueryDecompositionEngine
    from rag_engine.innovations.self_rag import SelfRAGEngine
    from rag_engine.innovations.speculative_rag import SpeculativeRAGEngine
    from rag_engine.innovations.raptor_indexing import RAPTORIndexer
    INNOVATIONS_AVAILABLE = True
except ImportError as e:
    print(f"[INNOVATIONS] Some modules not available: {e}")
    INNOVATIONS_AVAILABLE = False

# FAQ Exact Match Engine
try:
    from rag_engine.faq_exact_match import FAQExactMatchEngine, FAQMatchResult
    FAQ_MATCH_AVAILABLE = True
except ImportError as e:
    print(f"[FAQ_MATCH] FAQ exact match not available: {e}")
    FAQ_MATCH_AVAILABLE = False

# Langfuse Observability (Feature Toggle: langfuse.enabled)
try:
    from observability.rag_pipeline_tracer import RAGPipelineTracer
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    print(f"[LANGFUSE] Observability module not available: {e}")
    LANGFUSE_AVAILABLE = False

# Shared local embedding model (loaded once)
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Pre-warm the model to eliminate cold-start latency on first query
        _embedding_model.encode("warm up", convert_to_numpy=True)
        print("[EMBEDDING] Model loaded and pre-warmed (all-MiniLM-L6-v2)")
    return _embedding_model


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    text: str
    source: str
    page: int = 0
    section: str = ""
    chunk_id: str = ""


@dataclass
class LayerResult:
    """Result from a single RAG layer."""
    layer_number: int
    layer_name: str
    status: str  # "executed", "skipped", "cache_hit"
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""
    answer: str
    confidence: float
    sources: List[Dict[str, str]]
    layer_results: List[LayerResult]
    total_duration_ms: float = 0.0
    pipeline_stopped_at: Optional[int] = None  # Layer number where pipeline stopped (only Layer 1)
    validation_status: str = "approved"  # approved, warning, blocked
    orchestration_result: Optional[Any] = None  # OrchestrationResult from product orchestrator
    governance_result: Optional[Any] = None  # GovernanceResult from governance engine
    innovation_results: Dict[str, Any] = field(default_factory=dict)  # Results from 9 innovations


class SemanticCache:
    """Layer 1: Semantic Caching Engine."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings
        self.cache: Dict[str, Dict] = {}  # hash -> {embedding, response, timestamp}
        self.embeddings_cache: List[Tuple[np.ndarray, str]] = []  # (embedding, hash_key)
        self.embed_model = get_embedding_model()

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embed_model.encode(text, convert_to_numpy=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def lookup(self, query: str) -> Optional[Dict]:
        """Check if a similar query exists in cache."""
        if not self.embeddings_cache:
            return None

        query_emb = self._get_embedding(query)
        best_score = 0.0
        best_key = None

        for cached_emb, hash_key in self.embeddings_cache:
            score = self._cosine_similarity(query_emb, cached_emb)
            if score > best_score:
                best_score = score
                best_key = hash_key

        threshold = self.settings.rag.cache_similarity_threshold
        if best_score >= threshold and best_key in self.cache:
            entry = self.cache[best_key]
            # Check TTL
            if time.time() - entry["timestamp"] < self.settings.rag.cache_ttl_seconds:
                return {
                    "response": entry["response"],
                    "similarity": best_score,
                    "original_query": entry["query"]
                }
            else:
                # Expired, remove
                del self.cache[best_key]
                self.embeddings_cache = [(e, k) for e, k in self.embeddings_cache if k != best_key]

        return None

    def store(self, query: str, response: 'RAGResponse'):
        """Store a query-response pair in cache."""
        query_emb = self._get_embedding(query)
        hash_key = hashlib.md5(query.encode()).hexdigest()

        self.cache[hash_key] = {
            "query": query,
            "response": response,
            "timestamp": time.time()
        }
        self.embeddings_cache.append((query_emb, hash_key))

        # Evict oldest if over limit
        if len(self.cache) > self.settings.rag.cache_max_entries:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
            self.embeddings_cache = [(e, k) for e, k in self.embeddings_cache if k != oldest_key]


class DocumentIndexer:
    """Layer 3: Semantic Chunking and FAISS Indexing."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings
        self.chunks: List[ChunkMetadata] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embed_model = get_embedding_model()

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embed_model.encode(text, convert_to_numpy=True)

    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts using local model."""
        return self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _semantic_chunk(self, text: str, source: str) -> List[ChunkMetadata]:
        """Split text into semantic chunks using the UniversalDocumentChunker.
        Supports any document format: TXT with delimiters, PDF extracted text, Markdown, etc.
        """
        chunker = UniversalDocumentChunker(
            chunk_size_min=self.settings.rag.chunk_size_min,
            chunk_size_max=self.settings.rag.chunk_size_max,
            chunk_overlap=self.settings.rag.chunk_overlap
        )
        chunk_results = chunker.chunk_document(text, source)

        # Convert ChunkResult objects to ChunkMetadata objects
        chunks = []
        for cr in chunk_results:
            chunks.append(ChunkMetadata(
                text=cr.text,
                source=source,
                section=cr.section,
                chunk_id=cr.chunk_id
            ))

        return chunks

    def index_document(self, text: str, source: str) -> int:
        """Index a document by chunking and embedding it."""
        new_chunks = self._semantic_chunk(text, source)
        if not new_chunks:
            return 0

        new_texts = [c.text for c in new_chunks]
        new_embeddings = self._get_embeddings_batch(new_texts)

        self.chunks.extend(new_chunks)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        return len(new_chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[ChunkMetadata, float]]:
        """Search for similar chunks using cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        normalized = self.embeddings / norms
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        similarities = np.dot(normalized, query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        # Safety bounds check: skip indices that exceed chunks list length
        results = []
        for i in top_indices:
            if i < len(self.chunks):
                results.append((self.chunks[i], float(similarities[i])))
        return results

    def get_document_count(self) -> int:
        return len(set(c.source for c in self.chunks))

    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_sources(self) -> List[str]:
        return list(set(c.source for c in self.chunks))

    def remove_document(self, source: str) -> int:
        """Remove all chunks and embeddings for a given document source.
        
        Rebuilds the embeddings array after removal.
        
        Args:
            source: The document filename/source to remove
        
        Returns:
            Number of chunks removed
        """
        # Find indices to keep (not matching the source)
        keep_indices = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed_count = len(self.chunks) - len(keep_indices)

        if removed_count == 0:
            return 0

        if len(keep_indices) == 0:
            # All chunks removed
            self.chunks = []
            self.embeddings = None
        else:
            # Rebuild chunks list and embeddings array
            self.chunks = [self.chunks[i] for i in keep_indices]
            if self.embeddings is not None and len(keep_indices) > 0:
                self.embeddings = self.embeddings[keep_indices]
            else:
                self.embeddings = None

        print(f"[INDEXER] Removed {removed_count} chunks for '{source}', {len(self.chunks)} remaining")
        return removed_count


class SevenLayerRAG:
    """Main 7-Layer RAG Engine with Product Orchestration."""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI()  # Uses env var for API key and base URL
        self.prompts = PromptManager()
        self.cache = SemanticCache(self.client, self.settings)
        self.indexer = DocumentIndexer(self.client, self.settings)
        self._initialized = False

        # Product Orchestrator (Feature Toggle: orchestrator_enabled)
        self.orchestrator = None
        if self.settings.rag.orchestrator_enabled:
            self.orchestrator = ProductOrchestrator(self.client, self.settings)

        # Governance Engine (Feature Toggle: governance_enabled)
        self.governance = None
        if self.settings.rag.governance_enabled:
            self.governance = GovernanceEngine(self.client, self.settings)

        # ═══════════════════════════════════════════════════════════
        # RAG INNOVATIONS — 9 Cutting-Edge Techniques (All Toggleable)
        # ═══════════════════════════════════════════════════════════
        self.hybrid_search = None
        self.contextual_retrieval = None
        self.ragas_evaluator = None
        self.graph_rag = None
        self.adaptive_rag = None
        self.query_decomposer = None
        self.self_rag = None
        self.speculative_rag = None
        self.raptor = None

        # ═══════════════════════════════════════════════════════════
        # PERSISTENCE LAYER (MongoDB + FAISS — Feature Toggle: mongodb.enabled)
        # ═══════════════════════════════════════════════════════════
        self.persistence = None
        if PERSISTENCE_AVAILABLE and getattr(self.settings, 'mongodb', None) and self.settings.mongodb.enabled:
            try:
                self.persistence = PersistenceManager(self.settings)
                if self.persistence.is_connected:
                    print("[RAG] Persistence layer initialized (MongoDB + FAISS)")
                else:
                    print("[RAG] Persistence layer failed to connect, falling back to in-memory")
                    self.persistence = None
            except Exception as e:
                print(f"[RAG] Persistence initialization error: {e}")
                self.persistence = None

        # ═══════════════════════════════════════════════════════════
        # ADVANCED EXTRACTION PIPELINE (Feature Toggle: extraction_orchestrator_enabled)
        # ═══════════════════════════════════════════════════════════
        self.extraction_orchestrator = None
        if EXTRACTION_AVAILABLE and getattr(self.settings.rag, 'extraction_orchestrator_enabled', False):
            self.extraction_orchestrator = ExtractionOrchestrator(self.client, self.settings)

        # ═══════════════════════════════════════════════════════════
        # LANGFUSE OBSERVABILITY (Feature Toggle: langfuse.enabled)
        # ═══════════════════════════════════════════════════════════
        self.tracer = None
        if LANGFUSE_AVAILABLE and getattr(self.settings, 'langfuse', None) and self.settings.langfuse.enabled:
            try:
                self.tracer = RAGPipelineTracer()
                if self.tracer.is_enabled:
                    print("[RAG] Langfuse observability initialized")
                else:
                    print("[RAG] Langfuse configured but not connected (check keys/host)")
                    self.tracer = None
            except Exception as e:
                print(f"[RAG] Langfuse initialization error: {e}")
                self.tracer = None

        # ═══════════════════════════════════════════════════════════
        # FAQ EXACT MATCH ENGINE (Feature Toggle: faq_exact_match_enabled)
        # ═══════════════════════════════════════════════════════════
        self.faq_engine = None
        if FAQ_MATCH_AVAILABLE and getattr(self.settings.rag, 'faq_exact_match_enabled', True):
            embed_model = get_embedding_model()
            self.faq_engine = FAQExactMatchEngine(embed_model, self.settings)
            print("[RAG] FAQ Exact Match Engine initialized")

        if INNOVATIONS_AVAILABLE:
            embed_model = get_embedding_model()
            if getattr(self.settings.rag, 'hybrid_search_enabled', False):
                self.hybrid_search = HybridSearchEngine(self.settings)
            if getattr(self.settings.rag, 'contextual_retrieval_enabled', False):
                self.contextual_retrieval = ContextualRetrievalEngine(self.client, self.settings)
            if getattr(self.settings.rag, 'ragas_evaluation_enabled', False):
                self.ragas_evaluator = RAGASEvaluator(self.client, self.settings)
            if getattr(self.settings.rag, 'graph_rag_enabled', False):
                self.graph_rag = KnowledgeGraphRAG(self.client, self.settings)
            if getattr(self.settings.rag, 'adaptive_rag_enabled', False):
                self.adaptive_rag = AdaptiveRAGRouter(self.client, self.settings)
            if getattr(self.settings.rag, 'query_decomposition_enabled', False):
                self.query_decomposer = QueryDecompositionEngine(self.client, self.settings)
            if getattr(self.settings.rag, 'self_rag_enabled', False):
                self.self_rag = SelfRAGEngine(self.client, self.settings)
            if getattr(self.settings.rag, 'speculative_rag_enabled', False):
                self.speculative_rag = SpeculativeRAGEngine(self.client, self.settings)
            if getattr(self.settings.rag, 'raptor_enabled', False):
                self.raptor = RAPTORIndexer(self.client, self.settings, embed_model)

    def initialize(self, documents_dir: str = None):
        """Load and index documents.
        
        If persistence is enabled and has data, loads from MongoDB + FAISS (no re-embedding).
        Otherwise, falls back to reading from the sample_policies directory.
        """
        # Step 1: Try loading from persistent storage (MongoDB + FAISS)
        if self.persistence and self.persistence.is_connected:
            loaded = self.persistence.load_into_indexer(self.indexer)
            if loaded > 0:
                print(f"[RAG] Loaded {loaded} chunks from persistent storage (no re-embedding needed)")
                # Also load into hybrid search if enabled
                if self.hybrid_search:
                    for chunk in self.indexer.chunks:
                        try:
                            self.hybrid_search.add_document(
                                chunk.chunk_id, chunk.text, chunk.source, chunk.section
                            )
                        except Exception:
                            pass
                
                # Load FAQ pairs from MongoDB into the FAQ engine
                if self.faq_engine:
                    if self.persistence.mongo and self.persistence.mongo.is_connected:
                        try:
                            import numpy as np
                            faq_load_start = time.time()
                            faq_dicts = self.persistence.mongo.get_all_faq_pairs()
                            if faq_dicts:
                                from rag_engine.faq_exact_match import FAQPair
                                restored_pairs = []
                                for fd in faq_dicts:
                                    pair = FAQPair(
                                        faq_id=fd.get('faq_id', ''),
                                        question=fd.get('question', ''),
                                        answer=fd.get('answer', ''),
                                        question_number=fd.get('question_number', ''),
                                        source_file=fd.get('source', ''),
                                        page=fd.get('page', 0),
                                        section=fd.get('section', ''),
                                    )
                                    emb = fd.get('embedding', [])
                                    if emb:
                                        pair.question_embedding = np.array(emb, dtype=np.float32)
                                    restored_pairs.append(pair)
                                self.faq_engine.add_pairs(restored_pairs)
                                faq_load_ms = (time.time() - faq_load_start) * 1000
                                print(f"[FAQ_PERSISTENCE] Loaded {len(restored_pairs)} FAQ pairs from MongoDB in {faq_load_ms:.0f}ms")
                            else:
                                print("[FAQ_PERSISTENCE] No FAQ pairs found in MongoDB (upload a FAQ document first)")
                        except Exception as faq_e:
                            print(f"[FAQ_PERSISTENCE] Error loading FAQ pairs: {faq_e}")
                    else:
                        print("[FAQ_PERSISTENCE] MongoDB not connected — FAQ pairs cannot be loaded from persistence")
                
                self._initialized = True
                return
            else:
                print("[RAG] No persisted data found, loading from sample documents...")

        # Step 2: Fallback — load from sample documents directory
        if documents_dir is None:
            documents_dir = self.settings.paths.get_abs_path(self.settings.paths.sample_policies_dir)

        if not os.path.exists(documents_dir):
            self._initialized = True
            return

        for filename in os.listdir(documents_dir):
            filepath = os.path.join(documents_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                self._index_with_innovations(text, filename)

        self._initialized = True

    def index_uploaded_document(self, text: str, filename: str, file_bytes: bytes = None, progress_callback=None) -> int:
        """Index a newly uploaded document.
        
        If the advanced extraction orchestrator is enabled and file_bytes are provided,
        uses multi-strategy extraction (vision, table, graph, contextual enrichment).
        Otherwise falls back to the standard text-based indexing pipeline.
        """
        # Check if document already exists in persistent storage (skip re-upload)
        if self.persistence and self.persistence.is_connected and file_bytes:
            if self.persistence.document_exists(filename, file_bytes):
                print(f"[PERSISTENCE] Document '{filename}' already exists in persistent storage, skipping re-upload")
                # Ensure it's loaded into memory if not already
                existing_sources = self.indexer.get_sources()
                if filename not in existing_sources:
                    self.persistence.load_into_indexer(self.indexer)
                self._initialized = True
                return self.indexer.get_chunk_count()

        # Use advanced extraction if available and file_bytes provided
        if self.extraction_orchestrator and file_bytes:
            try:
                result = self.index_uploaded_document_advanced(file_bytes, filename, progress_callback=progress_callback)
                if result and result.get('total_chunks', 0) > 0:
                    self._initialized = True
                    return result['total_chunks']
            except Exception as e:
                print(f"[EXTRACTION] Advanced extraction failed, falling back to standard: {e}")

        # Fallback: standard text-based indexing
        count = self._index_with_innovations(text, filename)
        self._initialized = True
        return count

    def index_uploaded_document_advanced(self, file_bytes: bytes, filename: str, progress_callback=None) -> dict:
        """Index a document using the advanced multi-strategy extraction pipeline.
        
        Processes the document through:
        1. Vision extraction (LLM-based page image analysis)
        2. Structured table extraction (Docling/pdfplumber)
        3. Knowledge graph construction (entity-relationship extraction)
        4. Contextual enrichment (prefix injection, formula linearization)
        
        Returns a dict with extraction results and statistics.
        """
        if not self.extraction_orchestrator:
            raise RuntimeError("Extraction orchestrator not initialized")

        # Run the full extraction pipeline
        result = self.extraction_orchestrator.process_document(file_bytes, filename, progress_callback=progress_callback)

        # Index each enriched chunk into the standard indexer + innovations
        indexed_count = 0
        for chunk in result.enriched_chunks:
            chunk_text = chunk.get('text', '')
            if not chunk_text or len(chunk_text.strip()) < 50:
                continue

            # Index into the standard DocumentIndexer
            count = self.indexer.index_document(chunk_text, filename)
            indexed_count += count

            # Index into Hybrid Search (BM25)
            if self.hybrid_search:
                try:
                    self.hybrid_search.add_document(
                        chunk.get('chunk_id', ''),
                        chunk_text,
                        filename,
                        chunk.get('section', '')
                    )
                except Exception as e:
                    print(f"[HYBRID_SEARCH] Error indexing chunk: {e}")

        # Index into GraphRAG from the orchestrator's graph builder
        # In fast_mode, skip GraphRAG entity re-extraction (each entity = 1 LLM call)
        fast_mode = self.settings.rag.extraction_config.get('fast_mode', True)
        if self.graph_rag and self.extraction_orchestrator.graph_builder and not fast_mode:
            try:
                # Transfer entities from extraction graph to the RAG graph
                graph_data = self.extraction_orchestrator.graph_builder.export_graph_json()
                for entity in graph_data.get('entities', []):
                    self.graph_rag.extract_and_add(
                        f"{entity['name']} ({entity['type']}): {json.dumps(entity.get('attributes', {}))}",
                        filename, entity['type']
                    )
            except Exception as e:
                print(f"[GRAPH_RAG] Error transferring graph data: {e}")
        elif fast_mode:
            print(f"[GRAPH_RAG] Skipped in fast mode (saves ~{len(result.enriched_chunks)} LLM calls)")

        # Build RAPTOR tree from enriched chunks
        # In fast_mode, skip RAPTOR tree building (each group of 3 chunks = 1 LLM call for summarization)
        if self.raptor and not fast_mode:
            try:
                doc_chunks = [
                    {'text': c.get('text', ''), 'source': filename,
                     'section': c.get('section', ''), 'chunk_id': c.get('chunk_id', '')}
                    for c in result.enriched_chunks if c.get('text', '')
                ]
                if doc_chunks:
                    self.raptor.build_tree(doc_chunks, filename)
            except Exception as e:
                print(f"[RAPTOR] Error building tree: {e}")
        elif fast_mode:
            print(f"[RAPTOR] Skipped in fast mode (saves ~{len(result.enriched_chunks) // 3} LLM calls)")

        # ═══════════════════════════════════════════════════════════
        # FAQ EXACT MATCH — Extract Q&A pairs from document text
        # ═══════════════════════════════════════════════════════════
        faq_pairs_count = 0
        if self.faq_engine and result.enriched_chunks:
            try:
                # Reconstruct full text from enriched chunks for FAQ detection
                full_doc_text = "\n\n".join(
                    c.get('original_text', c.get('text', ''))
                    for c in result.enriched_chunks
                    if c.get('chunk_type') in ('text', None, '')
                )
                if full_doc_text:
                    faq_pairs = self.faq_engine.extract_faq_pairs(full_doc_text, filename)
                    if faq_pairs:
                        self.faq_engine.add_pairs(faq_pairs)
                        faq_pairs_count = len(faq_pairs)
                        print(f"[FAQ_EXACT_MATCH] Extracted {faq_pairs_count} FAQ pairs from {filename}")
                        
                        # Persist FAQ pairs to MongoDB (survive restarts)
                        if self.persistence and self.persistence.is_connected:
                            try:
                                faq_dicts = []
                                for p in faq_pairs:
                                    faq_dicts.append({
                                        'faq_id': p.faq_id,
                                        'question': p.question,
                                        'answer': p.answer,
                                        'question_number': p.question_number,
                                        'source': p.source_file,
                                        'page': p.page,
                                        'section': p.section,
                                        'embedding': p.question_embedding.tolist() if p.question_embedding is not None else [],
                                    })
                                stored = self.persistence.mongo.store_faq_pairs(faq_dicts)
                                print(f"[FAQ_PERSISTENCE] Stored {stored} FAQ pairs to MongoDB")
                            except Exception as pe:
                                print(f"[FAQ_PERSISTENCE] Error storing FAQ pairs: {pe}")
                        
                        # Also index FAQ pairs as atomic chunks for hybrid search
                        faq_chunks = self.faq_engine.get_faq_chunks()
                        for faq_chunk in faq_chunks[-faq_pairs_count:]:  # Only new ones
                            faq_text = faq_chunk['text']
                            if len(faq_text.strip()) >= 30:
                                count = self.indexer.index_document(faq_text, filename)
                                indexed_count += count
                                if self.hybrid_search:
                                    try:
                                        self.hybrid_search.add_document(
                                            faq_chunk['chunk_id'], faq_text,
                                            filename, faq_chunk['section']
                                        )
                                    except Exception:
                                        pass
            except Exception as e:
                print(f"[FAQ_EXACT_MATCH] Error extracting FAQ pairs: {e}")

        extraction_result = {
            'total_chunks': indexed_count,
            'faq_pairs_extracted': faq_pairs_count,
            'strategies_used': result.strategies_used,
            'tables_extracted': len(result.tables_extracted),
            'formulas_extracted': len(result.formulas_extracted),
            'flowcharts_extracted': len(result.flowcharts_extracted),
            'graph_entities': result.graph_entities,
            'graph_relationships': result.graph_relationships,
            'document_summary': result.document_summary,
            'processing_time_ms': result.processing_time_ms,
            'human_review_required': result.human_review_required,
            'human_review_reason': result.human_review_reason,
            'events': result.events,
            'profile': {
                'page_count': result.profile.page_count,
                'has_tables': result.profile.has_tables,
                'has_images': result.profile.has_images,
                'has_formulas': result.profile.has_formulas,
                'is_bilingual': result.profile.is_bilingual,
                'complexity_score': result.profile.complexity_score
            }
        }

        # ═══════════════════════════════════════════════════════════
        # PERSIST TO MONGODB + FAISS (if enabled)
        # ═══════════════════════════════════════════════════════════
        print(f"[PERSISTENCE_DEBUG] persistence={self.persistence is not None}, "
              f"is_connected={self.persistence.is_connected if self.persistence else 'N/A'}, "
              f"file_bytes={len(file_bytes) if file_bytes else 0}")

        if self.persistence and self.persistence.is_connected and file_bytes:
            try:
                # Use the INDEXER's actual chunks and embeddings for persistence
                # (enriched_chunks get re-chunked by indexer.index_document, so counts differ)
                chunks_before = len(self.indexer.chunks) - indexed_count
                actual_chunks = self.indexer.chunks[chunks_before:]  # Only the newly indexed chunks
                
                # Build persist_chunks from the indexer's ChunkMetadata objects
                persist_chunks = []
                for chunk_meta in actual_chunks:
                    persist_chunks.append({
                        'chunk_id': chunk_meta.chunk_id,
                        'text': chunk_meta.text,
                        'original_text': chunk_meta.text,
                        'section': chunk_meta.section,
                        'source': chunk_meta.source,
                        'chunk_type': getattr(chunk_meta, 'chunk_type', 'text'),
                        'page': getattr(chunk_meta, 'page', 0),
                        'confidence': getattr(chunk_meta, 'confidence', 0.9),
                    })
                
                # Get matching embeddings from the indexer
                n_chunks = len(persist_chunks)
                chunk_embeddings = self.indexer.embeddings[-n_chunks:] if self.indexer.embeddings is not None and n_chunks > 0 else None

                print(f"[PERSISTENCE_DEBUG] indexed_count={indexed_count}, persist_chunks={n_chunks}, "
                      f"embeddings={'None' if chunk_embeddings is None else chunk_embeddings.shape}")

                if persist_chunks and chunk_embeddings is not None:
                    print(f"[PERSISTENCE] Storing {n_chunks} chunks + embeddings to MongoDB...")

                    persist_result = self.persistence.store_document_with_chunks(
                        filename=filename,
                        file_bytes=file_bytes,
                        chunks=persist_chunks,
                        embeddings=chunk_embeddings,
                        extraction_result=extraction_result
                    )
                    print(f"[PERSISTENCE] ✅ Stored: {persist_result}")
                    extraction_result['persistence'] = persist_result
                else:
                    print(f"[PERSISTENCE] ⚠️ Skipped: persist_chunks={n_chunks}, embeddings={'None' if chunk_embeddings is None else 'available'}")
            except Exception as e:
                import traceback
                print(f"[PERSISTENCE] ❌ Error storing document: {e}")
                traceback.print_exc()
                extraction_result['persistence'] = {'stored': False, 'error': str(e)}
        else:
            reasons = []
            if not self.persistence:
                reasons.append("persistence=None")
            elif not self.persistence.is_connected:
                reasons.append("not connected")
                if self.persistence.mongo:
                    reasons.append(f"mongo.is_connected={self.persistence.mongo.is_connected}")
                if self.persistence.faiss:
                    reasons.append(f"faiss.is_initialized={self.persistence.faiss.is_initialized}")
            if not file_bytes:
                reasons.append("no file_bytes")
            print(f"[PERSISTENCE] ⚠️ NOT STORING — reasons: {', '.join(reasons)}")
            extraction_result['persistence'] = {'stored': False, 'reason': ', '.join(reasons)}

        return extraction_result

    # ═══════════════════════════════════════════════════════════
    # DOCUMENT MANAGEMENT — View & Delete
    # ═══════════════════════════════════════════════════════════

    def get_all_documents(self) -> list:
        """Get all stored document records from MongoDB.
        
        Returns:
            List of document dicts with metadata (filename, upload_time, chunk_count, etc.)
        """
        if self.persistence and self.persistence.is_connected:
            return self.persistence.get_all_documents()
        # Fallback: build from in-memory indexer
        sources = self.indexer.get_sources()
        docs = []
        for source in sources:
            chunk_count = sum(1 for c in self.indexer.chunks if c.source == source)
            docs.append({
                'filename': source,
                'chunk_count': chunk_count,
                'status': 'indexed',
                'upload_time': 0,
            })
        return docs

    def delete_document(self, filename: str) -> dict:
        """Delete a document from ALL stores: MongoDB, FAISS, in-memory indexer, FAQ engine, hybrid search.
        
        Args:
            filename: The document filename to delete
        
        Returns:
            Dict with deletion stats
        """
        result = {
            'filename': filename,
            'deleted': False,
            'chunks_removed': 0,
            'faq_pairs_removed': 0,
            'hybrid_docs_removed': 0,
            'persistence_deleted': False,
        }

        try:
            # 1. Remove from in-memory DocumentIndexer (chunks + embeddings)
            chunks_removed = self.indexer.remove_document(filename)
            result['chunks_removed'] = chunks_removed

            # 2. Remove from FAQ engine
            if self.faq_engine:
                faq_removed = self.faq_engine.remove_by_source(filename)
                result['faq_pairs_removed'] = faq_removed

            # 3. Remove from Hybrid Search (BM25)
            if self.hybrid_search:
                hybrid_removed = self.hybrid_search.remove_by_source(filename)
                result['hybrid_docs_removed'] = hybrid_removed

            # 4. Remove from persistent storage (MongoDB + FAISS on disk)
            if self.persistence and self.persistence.is_connected:
                persist_ok = self.persistence.delete_document(filename)
                result['persistence_deleted'] = persist_ok

            # 5. Clear the semantic cache (queries may reference deleted document)
            if self.cache:
                self.cache.cache.clear()
                self.cache.embeddings_cache.clear()

            result['deleted'] = True
            print(f"[RAG] ✅ Document '{filename}' deleted from all stores: "
                  f"chunks={chunks_removed}, faq={result['faq_pairs_removed']}, "
                  f"hybrid={result['hybrid_docs_removed']}, persistence={result['persistence_deleted']}")

        except Exception as e:
            import traceback
            print(f"[RAG] ❌ Error deleting document '{filename}': {e}")
            traceback.print_exc()
            result['error'] = str(e)

        return result

    def _index_with_innovations(self, text: str, filename: str) -> int:
        """Index a document with all innovation enhancements."""
        # Step 1: Contextual Retrieval — add context prefix to chunks before indexing
        processed_text = text
        if self.contextual_retrieval:
            try:
                result = self.contextual_retrieval.enrich_chunks(text, filename)
                if result and result.enriched_chunks:
                    # Store enriched chunks for later use
                    processed_text = text  # Keep original for indexing
            except Exception as e:
                print(f"[CONTEXTUAL_RETRIEVAL] Error enriching {filename}: {e}")

        # Step 2: Standard indexing
        count = self.indexer.index_document(processed_text, filename)

        # Step 3: Hybrid Search — index BM25 tokens
        if self.hybrid_search:
            try:
                for chunk in self.indexer.chunks:
                    if chunk.source == filename:
                        self.hybrid_search.index_document(
                            chunk.chunk_id, chunk.text, chunk.source, chunk.section
                        )
            except Exception as e:
                print(f"[HYBRID_SEARCH] Error indexing {filename}: {e}")

        # Step 4: GraphRAG — extract entities and relationships
        if self.graph_rag:
            try:
                for chunk in self.indexer.chunks:
                    if chunk.source == filename:
                        self.graph_rag.extract_and_add(chunk.text, chunk.source, chunk.section)
            except Exception as e:
                print(f"[GRAPH_RAG] Error processing {filename}: {e}")

        # Step 5: RAPTOR — build hierarchical tree
        if self.raptor:
            try:
                doc_chunks = [
                    {'text': c.text, 'source': c.source, 'section': c.section, 'chunk_id': c.chunk_id}
                    for c in self.indexer.chunks if c.source == filename
                ]
                if doc_chunks:
                    self.raptor.build_tree(doc_chunks, filename)
            except Exception as e:
                print(f"[RAPTOR] Error building tree for {filename}: {e}")

        return count

    def _get_embedding(self, text: str) -> np.ndarray:
        embed_model = get_embedding_model()
        return embed_model.encode(text, convert_to_numpy=True)

    def _call_llm(self, prompt: str, temperature: float = None,
                  _trace=None, _span=None, _gen_name: str = "LLM Call") -> str:
        """Call the LLM with a prompt.
        
        Args:
            prompt: The prompt text
            temperature: Override temperature
            _trace: (Langfuse) Parent TraceHandle for observability
            _span: (Langfuse) Parent SpanHandle for nesting
            _gen_name: (Langfuse) Name for this generation in traces
        """
        if temperature is None:
            temperature = self.settings.llm.temperature

        llm_start = time.time()
        response = self.client.chat.completions.create(
            model=self.settings.llm.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.settings.llm.max_tokens
        )
        llm_duration_ms = (time.time() - llm_start) * 1000
        result_text = response.choices[0].message.content.strip()

        # ── Langfuse: Track this LLM generation ──
        if self.tracer and _trace and getattr(self.settings.langfuse, 'trace_llm_calls', True):
            try:
                usage = None
                if hasattr(response, 'usage') and response.usage:
                    usage = {
                        'prompt_tokens': response.usage.prompt_tokens or 0,
                        'completion_tokens': response.usage.completion_tokens or 0,
                        'total_tokens': response.usage.total_tokens or 0,
                    }
                self.tracer.trace_llm_call(
                    trace=_trace,
                    name=_gen_name,
                    model=self.settings.llm.model_name,
                    prompt=prompt,
                    response_text=result_text,
                    usage=usage,
                    parent_span=_span,
                    duration_ms=llm_duration_ms,
                )
            except Exception:
                pass  # Never break pipeline for tracing

        return result_text

    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {}

    # ==========================================
    # LAYER 1: Semantic Caching
    # ==========================================
    def _layer1_cache(self, query: str) -> Tuple[LayerResult, Optional[RAGResponse]]:
        """Layer 1: Check semantic cache for similar previous queries."""
        start = time.time()

        if not self.settings.rag.cache_enabled:
            return LayerResult(
                layer_number=1, layer_name="Semantic Caching Engine",
                status="skipped", duration_ms=0,
                details={"reason": "Cache disabled in config"}
            ), None

        cached = self.cache.lookup(query)
        duration = (time.time() - start) * 1000

        if cached:
            return LayerResult(
                layer_number=1, layer_name="Semantic Caching Engine",
                status="cache_hit", duration_ms=duration,
                details={
                    "similarity": f"{cached['similarity']:.2%}",
                    "original_query": cached["original_query"],
                    "threshold": f"{self.settings.rag.cache_similarity_threshold:.0%}"
                }
            ), cached["response"]
        else:
            return LayerResult(
                layer_number=1, layer_name="Semantic Caching Engine",
                status="cache_miss", duration_ms=duration,
                details={
                    "threshold": f"{self.settings.rag.cache_similarity_threshold:.0%}",
                    "cache_size": len(self.cache.cache)
                }
            ), None

    # ==========================================
    # LAYER 2: HyDE Query Transformation
    # ==========================================
    def _layer2_hyde(self, query: str) -> Tuple[LayerResult, np.ndarray]:
        """Layer 2: Transform query using Hypothetical Document Embeddings."""
        start = time.time()

        if not self.settings.rag.hyde_enabled or len(query) < self.settings.rag.hyde_min_query_length:
            embedding = self._get_embedding(query)
            duration = (time.time() - start) * 1000
            return LayerResult(
                layer_number=2, layer_name="Query Transformation (HyDE)",
                status="skipped", duration_ms=duration,
                details={"reason": "Simple query, original embedding used"}
            ), embedding

        prompt = self.prompts.hyde_generator(query)
        hypothetical_doc = self._call_llm(prompt, temperature=0.3)
        # Combine original query + hypothetical for richer embedding
        combined = f"{query}\n\n{hypothetical_doc}"
        embedding = self._get_embedding(combined)
        duration = (time.time() - start) * 1000

        return LayerResult(
            layer_number=2, layer_name="Query Transformation (HyDE)",
            status="executed", duration_ms=duration,
            details={
                "hypothetical_preview": hypothetical_doc[:200] + "..." if len(hypothetical_doc) > 200 else hypothetical_doc,
                "improvement": "25-40% better precision expected"
            }
        ), embedding

    # ==========================================
    # LAYER 3: Semantic Chunking & Retrieval
    # ==========================================
    def _layer3_retrieve(self, query_embedding: np.ndarray) -> Tuple[LayerResult, List[Tuple[ChunkMetadata, float]]]:
        """Layer 3: Retrieve candidate chunks from the vector store with full chunk details."""
        start = time.time()

        top_k = self.settings.rag.retrieval_top_k
        results = self.indexer.search(query_embedding, top_k=top_k)
        duration = (time.time() - start) * 1000

        sources = list(set(r[0].source for r in results))

        # Build detailed chunk information for display
        chunk_details = []
        for chunk, sim_score in results:
            chunk_details.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "section": chunk.section if chunk.section else "General",
                "similarity": sim_score,
                "text_preview": chunk.text[:400],
                "text_length": len(chunk.text),
            })

        return LayerResult(
            layer_number=3, layer_name="Semantic Chunking & Retrieval",
            status="executed", duration_ms=duration,
            details={
                "chunks_retrieved": len(results),
                "sources_matched": sources,
                "top_similarity": f"{results[0][1]:.2%}" if results else "N/A",
                "chunk_size_range": f"{self.settings.rag.chunk_size_min}-{self.settings.rag.chunk_size_max} chars",
                "chunk_details": chunk_details,
            }
        ), results

    # ==========================================
    # LAYER 4: Corrective RAG (CRAG)
    # ==========================================
    def _layer4_crag(self, query: str, chunks: List[Tuple[ChunkMetadata, float]]) -> Tuple[LayerResult, List[Tuple[ChunkMetadata, float, str]]]:
        """Layer 4: Grade chunk quality and filter, with detailed per-chunk grading info."""
        start = time.time()

        if not self.settings.rag.crag_enabled or not chunks:
            graded = [(c, s, "ungraded") for c, s in chunks]
            duration = (time.time() - start) * 1000
            return LayerResult(
                layer_number=4, layer_name="Corrective RAG (CRAG)",
                status="skipped", duration_ms=duration,
                details={"reason": "CRAG disabled or no chunks"}
            ), graded

        graded_chunks = []
        correct_count = 0
        ambiguous_count = 0
        incorrect_count = 0
        grading_details = []

        # Grade top chunks (limit to save API calls)
        chunks_to_grade = chunks[:min(10, len(chunks))]

        for chunk, sim_score in chunks_to_grade:
            prompt = self.prompts.crag_quality_grader(query, chunk.text[:500])
            result = self._call_llm(prompt)
            parsed = self._parse_json_response(result)

            relevance = parsed.get("relevance", "ambiguous")
            quality_score = parsed.get("score", 0.5)
            reason = parsed.get("reason", "No reason provided")

            grade_record = {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "section": chunk.section if chunk.section else "General",
                "text_preview": chunk.text[:300],
                "similarity": sim_score,
                "quality_score": f"{quality_score:.2f}",
                "reason": reason,
            }

            if relevance == "correct" or quality_score >= self.settings.rag.crag_quality_threshold_correct:
                graded_chunks.append((chunk, sim_score, "correct"))
                correct_count += 1
                grade_record["grade"] = "correct"
                grade_record["decision"] = "SELECTED — Passed to Layer 5"
            elif relevance == "ambiguous" or quality_score >= self.settings.rag.crag_quality_threshold_ambiguous:
                graded_chunks.append((chunk, sim_score, "ambiguous"))
                ambiguous_count += 1
                grade_record["grade"] = "ambiguous"
                grade_record["decision"] = "SELECTED (lower priority) — Passed to Layer 5"
            else:
                incorrect_count += 1  # Discarded
                grade_record["grade"] = "incorrect"
                grade_record["decision"] = "DISCARDED — Removed from pipeline"

            grading_details.append(grade_record)

        # Add remaining ungraded chunks as ambiguous
        for chunk, sim_score in chunks[len(chunks_to_grade):]:
            graded_chunks.append((chunk, sim_score, "ambiguous"))
            ambiguous_count += 1
            grading_details.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "section": chunk.section if chunk.section else "General",
                "text_preview": chunk.text[:300],
                "similarity": sim_score,
                "quality_score": "N/A (ungraded)",
                "reason": "Not graded (beyond top-10 limit) — passed as ambiguous",
                "grade": "ambiguous",
                "decision": "SELECTED (ungraded) — Passed to Layer 5",
            })

        duration = (time.time() - start) * 1000

        web_fallback_triggered = correct_count == 0 and self.settings.rag.crag_web_fallback_enabled
        return LayerResult(
            layer_number=4, layer_name="Corrective RAG (CRAG)",
            status="executed", duration_ms=duration,
            details={
                "correct": correct_count,
                "ambiguous": ambiguous_count,
                "incorrect_discarded": incorrect_count,
                "web_fallback_triggered": web_fallback_triggered,
                "quality_tiers": "3 (Correct / Ambiguous / Incorrect)",
                "grading_details": grading_details,
            }
        ), graded_chunks

    # ==========================================
    # LAYER 5: LLM Re-Ranking
    # ==========================================
    def _layer5_rerank(self, query: str, graded_chunks: List[Tuple[ChunkMetadata, float, str]]) -> Tuple[LayerResult, List[Tuple[ChunkMetadata, float]]]:
        """Layer 5: Deep relevance scoring and re-ranking with detailed per-chunk scoring."""
        start = time.time()

        if not self.settings.rag.rerank_enabled or not graded_chunks:
            final = [(c, s) for c, s, _ in graded_chunks[:self.settings.rag.rerank_top_k]]
            duration = (time.time() - start) * 1000
            return LayerResult(
                layer_number=5, layer_name="LLM Re-Ranking",
                status="skipped", duration_ms=duration,
                details={"reason": "Re-ranking disabled or no chunks"}
            ), final

        scored_chunks = []
        ranking_details_raw = []

        for chunk, sim_score, grade in graded_chunks[:min(10, len(graded_chunks))]:
            prompt = self.prompts.rerank_scorer(query, chunk.text[:500])
            result = self._call_llm(prompt)
            parsed = self._parse_json_response(result)
            deep_score = parsed.get("score", 5.0)
            reason = parsed.get("reason", "No reason provided")
            # Combine similarity score with deep relevance score
            combined_score = (sim_score * 0.3) + (deep_score / 10.0 * 0.7)
            scored_chunks.append((chunk, combined_score, deep_score, sim_score, reason))

        # Sort by combined score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Diversity filtering: avoid duplicate sources in top-K
        if self.settings.rag.rerank_diversity_enabled:
            final_chunks = []
            seen_sources = set()
            not_selected = []
            for chunk, score, deep, sim, reason in scored_chunks:
                if len(final_chunks) >= self.settings.rag.rerank_top_k:
                    not_selected.append((chunk, score, deep, sim, reason))
                    continue
                source_key = f"{chunk.source}_{chunk.section}"
                if source_key not in seen_sources or len(final_chunks) < 3:
                    final_chunks.append((chunk, score))
                    seen_sources.add(source_key)
                    ranking_details_raw.append({
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "section": chunk.section if chunk.section else "General",
                        "text_preview": chunk.text[:250],
                        "deep_score": f"{deep:.1f}",
                        "similarity_score": f"{sim:.2%}",
                        "combined_score": f"{score:.4f}",
                        "reason": reason,
                        "selected": True,
                        "selection_reason": f"Rank #{len(final_chunks)} — Selected (combined score: {score:.4f})",
                    })
                else:
                    not_selected.append((chunk, score, deep, sim, reason))

            # Record not-selected chunks
            for chunk, score, deep, sim, reason in not_selected:
                ranking_details_raw.append({
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section if chunk.section else "General",
                    "text_preview": chunk.text[:250],
                    "deep_score": f"{deep:.1f}",
                    "similarity_score": f"{sim:.2%}",
                    "combined_score": f"{score:.4f}",
                    "reason": reason,
                    "selected": False,
                    "selection_reason": "Below Top-K cutoff or diversity filter applied",
                })
        else:
            final_chunks = [(c, s) for c, s, _, _, _ in scored_chunks[:self.settings.rag.rerank_top_k]]
            for i, (chunk, score, deep, sim, reason) in enumerate(scored_chunks):
                ranking_details_raw.append({
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section if chunk.section else "General",
                    "text_preview": chunk.text[:250],
                    "deep_score": f"{deep:.1f}",
                    "similarity_score": f"{sim:.2%}",
                    "combined_score": f"{score:.4f}",
                    "reason": reason,
                    "selected": i < self.settings.rag.rerank_top_k,
                    "selection_reason": f"Rank #{i+1}" + (" — Selected" if i < self.settings.rag.rerank_top_k else " — Below cutoff"),
                })

        duration = (time.time() - start) * 1000

        return LayerResult(
            layer_number=5, layer_name="LLM Re-Ranking",
            status="executed", duration_ms=duration,
            details={
                "chunks_scored": len(scored_chunks),
                "top_k_selected": len(final_chunks),
                "top_score": f"{scored_chunks[0][2]:.1f}/10" if scored_chunks else "N/A",
                "diversity_filter": self.settings.rag.rerank_diversity_enabled,
                "ranking_details": ranking_details_raw,
            }
        ), final_chunks

    # ==========================================
    # LAYER 6: Agentic RAG (ReAct)
    # ==========================================
    def _layer6_agentic(self, query: str, ranked_chunks: List[Tuple[ChunkMetadata, float]]) -> Tuple[LayerResult, str]:
        """Layer 6: Multi-step reasoning for complex queries with detailed step tracking."""
        start = time.time()

        if not self.settings.rag.agentic_enabled:
            context = "\n\n---\n\n".join([
                f"[Source: {c.source} | Section: {c.section or 'General'}]\n{c.text}"
                for c, _ in ranked_chunks
            ])
            duration = (time.time() - start) * 1000
            return LayerResult(
                layer_number=6, layer_name="Agentic RAG (ReAct)",
                status="skipped", duration_ms=duration,
                details={"reason": "Agentic RAG disabled"}
            ), context

        # Check query complexity
        complexity_prompt = self.prompts.query_complexity_classifier(query)
        complexity_result = self._call_llm(complexity_prompt)
        complexity = self._parse_json_response(complexity_result)
        complexity_score = complexity.get("score", 0.3)
        complexity_reason = complexity.get("reason", "N/A")

        context = "\n\n---\n\n".join([
            f"[Source: {c.source} | Section: {c.section or 'General'}]\n{c.text}"
            for c, _ in ranked_chunks
        ])

        # Build chunk summary for details
        chunks_used = []
        for chunk, score in ranked_chunks:
            chunks_used.append({
                "source": chunk.source,
                "section": chunk.section if chunk.section else "General",
                "score": f"{score:.4f}",
            })

        if complexity_score < self.settings.rag.agentic_complexity_threshold:
            duration = (time.time() - start) * 1000
            return LayerResult(
                layer_number=6, layer_name="Agentic RAG (ReAct)",
                status="pass_through", duration_ms=duration,
                details={
                    "complexity": complexity.get("complexity", "simple"),
                    "complexity_score": f"{complexity_score:.2f}",
                    "complexity_reason": complexity_reason,
                    "reason": "Simple query — direct pass-through to Layer 7",
                    "chunks_used": chunks_used,
                    "max_iterations": self.settings.rag.agentic_max_iterations,
                    "iterations_used": 0,
                }
            ), context

        # Complex query: run ReAct reasoning
        iterations = 0
        enriched_context = context
        reasoning_steps = []

        while iterations < self.settings.rag.agentic_max_iterations:
            iterations += 1
            step_start = time.time()
            planner_prompt = self.prompts.agentic_planner(query, enriched_context)
            plan_result = self._call_llm(planner_prompt)
            plan = self._parse_json_response(plan_result)
            step_duration = (time.time() - step_start) * 1000

            reasoning_steps.append({
                "step": iterations,
                "action": f"ReAct iteration {iterations}: Analyze context sufficiency",
                "observation": plan.get("reasoning", "No reasoning provided")[:300],
                "status": plan.get("status", "UNKNOWN"),
                "duration_ms": f"{step_duration:.0f}",
                "missing_info": plan.get("missing_info", ""),
            })

            if plan.get("status") == "SUFFICIENT":
                enriched_context = f"{context}\n\nAgent Analysis:\n{plan.get('reasoning', '')}"
                break

        duration = (time.time() - start) * 1000
        return LayerResult(
            layer_number=6, layer_name="Agentic RAG (ReAct)",
            status="executed", duration_ms=duration,
            details={
                "complexity": complexity.get("complexity", "complex"),
                "complexity_score": f"{complexity_score:.2f}",
                "complexity_reason": complexity_reason,
                "iterations_used": iterations,
                "max_iterations": self.settings.rag.agentic_max_iterations,
                "reasoning_steps": reasoning_steps,
                "chunks_used": chunks_used,
            }
        ), enriched_context

    # ==========================================
    # LAYER 7: Response Validation
    # ==========================================
    def _layer7_validate(self, query: str, context: str, sources: List[Dict[str, str]]) -> Tuple[LayerResult, RAGResponse]:
        """Layer 7: Generate response, validate, and check for hallucinations with detailed results."""
        start = time.time()

        sources_str = ", ".join([s.get("source", "Unknown") for s in sources])
        gen_prompt = self.prompts.response_generator(query, context, sources_str)
        answer = self._call_llm(gen_prompt)

        # Hallucination check
        confidence = 0.85  # Default
        validation_status = "approved"
        hallucination_details = {}

        if self.settings.rag.hallucination_check_enabled:
            hall_prompt = self.prompts.hallucination_checker(query, answer, context)
            hall_result = self._call_llm(hall_prompt)
            hall_parsed = self._parse_json_response(hall_result)

            is_hallucinated = hall_parsed.get("is_hallucinated", False)
            confidence = hall_parsed.get("confidence", 0.85)
            unsupported = hall_parsed.get("unsupported_claims", [])
            verified = hall_parsed.get("verified_claims", [])

            hallucination_details = {
                "is_hallucinated": is_hallucinated,
                "unsupported_claims": unsupported if isinstance(unsupported, list) else [],
                "verified_claims_count": len(verified) if isinstance(verified, list) else 0,
                "verified_claims": verified if isinstance(verified, list) else [],
            }

            if is_hallucinated and isinstance(unsupported, list) and len(unsupported) > 0:
                # Proportional penalty based on ratio of unsupported vs verified claims
                total_claims = len(unsupported) + (len(verified) if isinstance(verified, list) else 0)
                if total_claims > 0:
                    unsupported_ratio = len(unsupported) / total_claims
                    # Scale penalty: 1 unsupported out of 10 = small penalty, 5/10 = large penalty
                    penalty = unsupported_ratio * 0.3  # Max penalty is 0.30
                    confidence = max(0.1, confidence - penalty)
                else:
                    confidence = max(0.1, confidence - 0.05)  # Minimal penalty if no claims parsed

        # Determine validation status
        if confidence >= self.settings.rag.validation_confidence_high:
            validation_status = "approved"
        elif confidence >= self.settings.rag.validation_confidence_medium:
            validation_status = "warning"
            answer = f"⚠️ **Note:** This answer may be incomplete. Please verify with your relationship manager.\n\n{answer}"
        else:
            validation_status = "blocked"
            answer = "🚫 The system could not generate a reliable answer for your query. Your question has been escalated to a banking specialist who will contact you shortly."

        duration = (time.time() - start) * 1000

        response = RAGResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            layer_results=[],  # Will be filled by the main pipeline
            total_duration_ms=0,
            validation_status=validation_status
        )

        layer_result = LayerResult(
            layer_number=7, layer_name="Response Validation",
            status="executed", duration_ms=duration,
            details={
                "confidence": f"{confidence:.0%}",
                "validation_status": validation_status,
                "hallucination_check": hallucination_details,
                "answer_length": len(answer),
                "sources_used": sources,
            }
        )

        return layer_result, response

    # ==========================================
    # MAIN PIPELINE
    # ==========================================
    def _layer0_orchestrate(self, query: str) -> Tuple[LayerResult, Optional[OrchestrationResult]]:
        """Layer 0 (Pre-Pipeline): Product Classification & Orchestration."""
        start = time.time()

        if not self.settings.rag.orchestrator_enabled or not self.orchestrator:
            duration = (time.time() - start) * 1000
            return LayerResult(
                layer_number=0, layer_name="Product Orchestration",
                status="skipped", duration_ms=duration,
                details={"reason": "Orchestrator disabled in config"}
            ), None

        orch_result = self.orchestrator.route(query)
        classification = orch_result.classification
        duration = (time.time() - start) * 1000

        # Build detailed orchestration info for display
        routing_reasons_display = []
        for rr in classification.routing.routing_reasons:
            routing_reasons_display.append({
                "collection": rr["collection"],
                "product": rr["product"],
                "reason": rr["reason"],
                "weight": rr["weight"],
            })

        details = {
            # Product Classification
            "product": classification.primary_product_name,
            "product_id": classification.primary_product,
            "product_confidence": f"{classification.confidence.product_confidence:.0%}",
            "classification_method": classification.classification_method,
            "is_cross_product": classification.is_cross_product,
            "secondary_products": [PRODUCT_CATALOG[s]["name"] for s in classification.secondary_products] if classification.secondary_products else [],

            # Intent Classification
            "intent": classification.intent.intent_name,
            "intent_id": classification.intent.primary_intent,
            "intent_confidence": f"{classification.intent.intent_confidence:.0%}",
            "intent_reasoning": classification.intent.reasoning,
            "secondary_intents": classification.intent.secondary_intents,
            "requires_human_handoff": classification.intent.requires_human_handoff,

            # Risk Assessment
            "risk_score": f"{classification.risk.risk_score:.0%}",
            "risk_level": classification.risk.risk_level,
            "risk_label": classification.risk.risk_label,
            "risk_color": classification.risk.risk_color,
            "risk_action": classification.risk.risk_action,
            "risk_factors": classification.risk.risk_factors,
            "risk_reasoning": classification.risk.reasoning,

            # Confidence Breakdown
            "overall_confidence": f"{classification.confidence.overall_confidence:.0%}",
            "confidence_breakdown": {
                "product": f"{classification.confidence.product_confidence:.0%}",
                "intent": f"{classification.intent.intent_confidence:.0%}",
                "keyword_match": f"{classification.confidence.keyword_match_score:.0%}",
                "semantic_alignment": f"{classification.confidence.semantic_alignment_score:.0%}",
            },
            "confidence_reasoning": classification.confidence.reasoning,

            # Routing Decision
            "routed_collections": orch_result.routed_collections,
            "routing_reasons": routing_reasons_display,
            "fallback_applied": classification.routing.fallback_applied,
            "cross_product_routing": classification.routing.cross_product_routing,

            # Human Handoff
            "should_handoff_to_human": orch_result.should_handoff_to_human,

            # Metadata
            "query_fingerprint": classification.query_fingerprint,
            "context_summary": orch_result.context_summary,
        }

        return LayerResult(
            layer_number=0, layer_name="Product Orchestration",
            status="executed", duration_ms=duration,
            details=details,
        ), orch_result

    def process_query(self, query: str, turbo_mode: bool = False) -> RAGResponse:
        """Process a query through all layers: Orchestration (Layer 0) + 7 RAG Layers.
        
        If turbo_mode=True, bypasses all heavy layers:
          FAQ Match → Simple Vector Search → 1 LLM call → Answer (~1-3s)
        """
        pipeline_start = time.time()
        layer_results = []

        # ── Langfuse: Start pipeline trace ──
        _lf_trace = None
        if self.tracer:
            try:
                _lf_trace = self.tracer.start_pipeline_trace(
                    query=query, turbo_mode=turbo_mode
                )
            except Exception:
                pass

        # ════════════════════════════════════════════════════════════════
        # TURBO MODE — Ultra-fast path for FAQ-heavy documents
        # ════════════════════════════════════════════════════════════════
        if turbo_mode:
            print(f"[TURBO] ⚡ Turbo Mode active — bypassing all heavy layers")
            response = self._turbo_pipeline(query, pipeline_start)
            # ── Langfuse: End turbo trace ──
            if self.tracer and _lf_trace:
                try:
                    self.tracer.end_pipeline_trace(_lf_trace, response)
                    self.tracer.flush()
                except Exception:
                    pass
            return response

        # ════════════════════════════════════════════════════════════════
        # FAQ SMART ROUTER — 3-Tier Routing (runs BEFORE all layers)
        # ════════════════════════════════════════════════════════════════
        # Embeds the FULL query, compares against ALL FAQ question embeddings.
        # Tier 1 (EXACT, sim≥0.85):  Return answer instantly. 0 LLM calls. ~50ms.
        # Tier 2 (FUZZY, sim 0.60-0.85): Adapt answer with 1 nano LLM call. ~1-2s.
        # Tier 3 (NOVEL, sim<0.60):  Fall through to full 7-layer pipeline.
        # ── DIAGNOSTIC: Log FAQ engine state ──
        if self.faq_engine:
            print(f"[FAQ_ROUTER] FAQ engine has {len(self.faq_engine.faq_pairs)} pairs loaded")
        else:
            print("[FAQ_ROUTER] FAQ engine is None — FAQ routing disabled")

        if self.faq_engine and self.faq_engine.faq_pairs:
            try:
                faq_start = time.time()
                faq_result = self.faq_engine.lookup(query)
                faq_duration = (time.time() - faq_start) * 1000
                print(f"[FAQ_ROUTER] Lookup took {faq_duration:.1f}ms → tier={faq_result.tier}, sim={faq_result.similarity:.4f}")

                # ── TIER 1: EXACT MATCH (sim >= 0.85) ──
                # Clear FAQ question → return exact answer, zero LLM calls
                if faq_result.tier == "exact" and faq_result.faq_pair:
                    pair = faq_result.faq_pair
                    top_matches_display = [
                        {"q": p.question_number, "sim": f"{s:.3f}", "question": p.question[:60]}
                        for s, p in faq_result.top_matches
                    ]
                    faq_layer = LayerResult(
                        layer_number=0,
                        layer_name="FAQ Smart Router",
                        status="TIER_1_EXACT",
                        duration_ms=faq_duration,
                        details={
                            "tier": "1 (EXACT)",
                            "similarity": f"{faq_result.similarity:.4f}",
                            "matched_question": pair.question,
                            "question_number": pair.question_number,
                            "source_file": pair.source_file,
                            "top_3_matches": top_matches_display,
                            "total_faq_pairs": len(self.faq_engine.faq_pairs),
                            "layers_bypassed": "All (Layer 0–7)",
                            "llm_calls": 0,
                            "estimated_time_saved": "10–20 seconds",
                        }
                    )
                    layer_results.append(faq_layer)

                    sources = [{
                        "source": pair.source_file,
                        "section": pair.question_number,
                        "relevance": f"{faq_result.similarity:.2%}"
                    }]
                    faq_response = RAGResponse(
                        answer=pair.answer,
                        confidence=min(0.99, faq_result.similarity),
                        sources=sources,
                        layer_results=layer_results,
                        total_duration_ms=(time.time() - pipeline_start) * 1000,
                        pipeline_stopped_at=0,
                        validation_status="approved",
                        orchestration_result=None,
                    )
                    if self.settings.rag.cache_enabled:
                        self.cache.store(query, faq_response)
                    total_ms = (time.time() - pipeline_start) * 1000
                    print(f"[FAQ_ROUTER] TIER 1 EXACT → {pair.question_number} "
                          f"(sim={faq_result.similarity:.4f}, {total_ms:.0f}ms, 0 LLM calls)")
                    # ── Langfuse: Trace FAQ exact match and end trace ──
                    if self.tracer and _lf_trace:
                        try:
                            self.tracer.trace_faq_routing(
                                trace=_lf_trace, tier="exact",
                                similarity=faq_result.similarity,
                                matched_question=pair.question,
                                question_number=pair.question_number,
                                llm_calls=0, duration_ms=faq_duration,
                            )
                            self.tracer.end_pipeline_trace(_lf_trace, faq_response)
                            self.tracer.flush()
                        except Exception:
                            pass
                    return faq_response

                # ── TIER 2: FUZZY MATCH (sim 0.60–0.85) ──
                # Related but rephrased/ambiguous → adapt with 1 lightweight LLM call
                elif faq_result.tier == "fuzzy" and faq_result.faq_pair:
                    pair = faq_result.faq_pair
                    adapt_start = time.time()
                    adapted_answer = self.faq_engine.adapt_answer_for_fuzzy_match(
                        query, faq_result, self.client
                    )
                    adapt_duration = (time.time() - adapt_start) * 1000
                    total_faq_duration = faq_duration + adapt_duration

                    top_matches_display = [
                        {"q": p.question_number, "sim": f"{s:.3f}", "question": p.question[:60]}
                        for s, p in faq_result.top_matches
                    ]
                    faq_layer = LayerResult(
                        layer_number=0,
                        layer_name="FAQ Smart Router",
                        status="TIER_2_FUZZY",
                        duration_ms=total_faq_duration,
                        details={
                            "tier": "2 (FUZZY)",
                            "similarity": f"{faq_result.similarity:.4f}",
                            "closest_faq_question": pair.question,
                            "question_number": pair.question_number,
                            "source_file": pair.source_file,
                            "top_3_matches": top_matches_display,
                            "total_faq_pairs": len(self.faq_engine.faq_pairs),
                            "layers_bypassed": "All (Layer 0–7)",
                            "llm_calls": 1,
                            "llm_model": self.faq_engine.fuzzy_model,
                            "adaptation_time_ms": f"{adapt_duration:.0f}",
                            "estimated_time_saved": "8–18 seconds",
                        }
                    )
                    layer_results.append(faq_layer)

                    sources = [{
                        "source": pair.source_file,
                        "section": pair.question_number,
                        "relevance": f"{faq_result.similarity:.2%}"
                    }]
                    faq_response = RAGResponse(
                        answer=adapted_answer,
                        confidence=min(0.95, faq_result.similarity + 0.1),
                        sources=sources,
                        layer_results=layer_results,
                        total_duration_ms=(time.time() - pipeline_start) * 1000,
                        pipeline_stopped_at=0,
                        validation_status="approved",
                        orchestration_result=None,
                    )
                    if self.settings.rag.cache_enabled:
                        self.cache.store(query, faq_response)
                    total_ms = (time.time() - pipeline_start) * 1000
                    print(f"[FAQ_ROUTER] TIER 2 FUZZY → {pair.question_number} "
                          f"(sim={faq_result.similarity:.4f}, {total_ms:.0f}ms, 1 LLM call)")
                    # ── Langfuse: Trace FAQ fuzzy match and end trace ──
                    if self.tracer and _lf_trace:
                        try:
                            self.tracer.trace_faq_routing(
                                trace=_lf_trace, tier="fuzzy",
                                similarity=faq_result.similarity,
                                matched_question=pair.question,
                                question_number=pair.question_number,
                                llm_calls=1, duration_ms=total_faq_duration,
                            )
                            self.tracer.end_pipeline_trace(_lf_trace, faq_response)
                            self.tracer.flush()
                        except Exception:
                            pass
                    return faq_response

                # ── TIER 3: NOVEL QUERY (sim < 0.60) ──
                # Not an FAQ question → log and fall through to full pipeline
                else:
                    faq_layer = LayerResult(
                        layer_number=0,
                        layer_name="FAQ Smart Router",
                        status="TIER_3_NOVEL",
                        duration_ms=faq_duration,
                        details={
                            "tier": "3 (NOVEL)",
                            "best_similarity": f"{faq_result.similarity:.4f}",
                            "best_match": faq_result.faq_pair.question[:80] if faq_result.faq_pair else "N/A",
                            "exact_threshold": f"{self.faq_engine.exact_threshold}",
                            "fuzzy_threshold": f"{self.faq_engine.fuzzy_threshold}",
                            "total_faq_pairs": len(self.faq_engine.faq_pairs),
                            "action": "Falling through to full 7-layer pipeline",
                        }
                    )
                    layer_results.append(faq_layer)
            except Exception as e:
                print(f"[FAQ_ROUTER] Lookup error (non-critical, continuing to full pipeline): {e}")

        # ════════════════════════════════════════════════════════════════
        # FULL PIPELINE — Only reached if FAQ fast path didn't match
        # ════════════════════════════════════════════════════════════════

        # ---- LAYER 0: Product Orchestration ----
        _lf_l0_span = self.tracer.trace_layer_start(_lf_trace, 0, "Product Orchestration", {"query": query}) if self.tracer and _lf_trace else None
        l0_result, orch_result = self._layer0_orchestrate(query)
        layer_results.append(l0_result)
        if self.tracer and _lf_l0_span:
            try: self.tracer.trace_layer_end(_lf_l0_span, l0_result)
            except Exception: pass

        # Check for human handoff (critical risk)
        if orch_result and orch_result.should_handoff_to_human:
            duration = (time.time() - pipeline_start) * 1000
            handoff_response = RAGResponse(
                answer=(
                    f"🚨 **Human Agent Handoff Required**\n\n"
                    f"Your query has been classified as **{orch_result.classification.risk.risk_label}** "
                    f"(Risk Score: {orch_result.classification.risk.risk_score:.0%}).\n\n"
                    f"**Reason:** {orch_result.classification.risk.reasoning}\n\n"
                    f"**Intent:** {orch_result.classification.intent.intent_name}\n\n"
                    f"Your request is being escalated to a banking specialist who will contact you shortly. "
                    f"Please do not share sensitive information in this chat."
                ),
                confidence=0.0,
                sources=[],
                layer_results=layer_results,
                total_duration_ms=duration,
                pipeline_stopped_at=0,
                validation_status="blocked",
                orchestration_result=orch_result,
            )
            return handoff_response

        # ---- LAYER 1: Semantic Caching ----
        _lf_l1_span = self.tracer.trace_layer_start(_lf_trace, 1, "Semantic Cache", {"query": query}) if self.tracer and _lf_trace else None
        l1_result, cached_response = self._layer1_cache(query)
        layer_results.append(l1_result)
        if self.tracer and _lf_l1_span:
            try: self.tracer.trace_layer_end(_lf_l1_span, l1_result)
            except Exception: pass

        if cached_response is not None:
            # Pipeline STOPS here
            cached_response.layer_results = layer_results
            cached_response.pipeline_stopped_at = 1
            cached_response.total_duration_ms = (time.time() - pipeline_start) * 1000
            cached_response.orchestration_result = orch_result
            # ── Langfuse: End trace on cache hit ──
            if self.tracer and _lf_trace:
                try:
                    self.tracer.end_pipeline_trace(_lf_trace, cached_response)
                    self.tracer.flush()
                except Exception: pass
            return cached_response

        # ---- LAYER 2: HyDE ----
        _lf_l2_span = self.tracer.trace_layer_start(_lf_trace, 2, "HyDE Query Transformation", {"query": query}) if self.tracer and _lf_trace else None
        l2_result, query_embedding = self._layer2_hyde(query)
        layer_results.append(l2_result)
        if self.tracer and _lf_l2_span:
            try: self.tracer.trace_layer_end(_lf_l2_span, l2_result)
            except Exception: pass

        # ---- INNOVATION: Adaptive RAG (Pre-Layer 3) ----
        innovation_results = {}
        adaptive_strategy = None
        if self.adaptive_rag:
            try:
                adapt_start = time.time()
                adaptive_result = self.adaptive_rag.classify_and_route(query)
                adaptive_strategy = adaptive_result
                adapt_duration = (time.time() - adapt_start) * 1000
                innovation_results['adaptive_rag'] = {
                    'strategy': adaptive_result.selected_strategy,
                    'complexity': adaptive_result.complexity_level,
                    'confidence': f"{adaptive_result.confidence:.0%}",
                    'reasoning': adaptive_result.reasoning,
                    'duration_ms': f"{adapt_duration:.0f}",
                }
            except Exception as e:
                innovation_results['adaptive_rag'] = {'error': str(e)}

        # ---- INNOVATION: Query Decomposition (Pre-Layer 3) ----
        sub_queries = []
        if self.query_decomposer:
            try:
                decomp_start = time.time()
                decomp_result = self.query_decomposer.decompose(query)
                sub_queries = decomp_result.sub_queries if decomp_result.is_decomposed else []
                decomp_duration = (time.time() - decomp_start) * 1000
                innovation_results['query_decomposition'] = {
                    'is_decomposed': decomp_result.is_decomposed,
                    'sub_queries': [{'query': sq.query, 'intent': sq.intent, 'priority': sq.priority} for sq in sub_queries],
                    'original_query': query,
                    'reasoning': decomp_result.reasoning,
                    'duration_ms': f"{decomp_duration:.0f}",
                }
            except Exception as e:
                innovation_results['query_decomposition'] = {'error': str(e)}

        # ---- LAYER 3: Retrieval ----
        _lf_l3_span = self.tracer.trace_layer_start(_lf_trace, 3, "Semantic Retrieval", {"top_k": self.settings.rag.retrieval_top_k}) if self.tracer and _lf_trace else None
        l3_result, retrieved_chunks = self._layer3_retrieve(query_embedding)
        layer_results.append(l3_result)
        if self.tracer and _lf_l3_span:
            try: self.tracer.trace_layer_end(_lf_l3_span, l3_result)
            except Exception: pass

        # ---- INNOVATION: Hybrid Search (Enhances Layer 3) ----
        if self.hybrid_search and retrieved_chunks:
            try:
                hybrid_start = time.time()
                hybrid_result = self.hybrid_search.search(query, top_k=self.settings.rag.retrieval_top_k)
                hybrid_duration = (time.time() - hybrid_start) * 1000
                innovation_results['hybrid_search'] = {
                    'bm25_matches': hybrid_result.bm25_count,
                    'vector_matches': len(retrieved_chunks),
                    'fused_count': hybrid_result.fused_count,
                    'rrf_k': self.settings.rag.hybrid_search_rrf_k,
                    'bm25_weight': self.settings.rag.hybrid_search_bm25_weight,
                    'vector_weight': self.settings.rag.hybrid_search_vector_weight,
                    'top_results': hybrid_result.top_results[:5] if hasattr(hybrid_result, 'top_results') else [],
                    'duration_ms': f"{hybrid_duration:.0f}",
                }
            except Exception as e:
                innovation_results['hybrid_search'] = {'error': str(e)}

        # ---- INNOVATION: GraphRAG (Enhances Layer 3) ----
        graph_context = ""
        if self.graph_rag and retrieved_chunks:
            try:
                graph_start = time.time()
                graph_result = self.graph_rag.query(query, max_hops=self.settings.rag.graph_rag_max_hops)
                graph_context = graph_result.context if graph_result else ""
                graph_duration = (time.time() - graph_start) * 1000
                innovation_results['graph_rag'] = {
                    'entities_found': graph_result.entities_found if graph_result else 0,
                    'relationships_traversed': graph_result.relationships_traversed if graph_result else 0,
                    'hops_used': graph_result.hops_used if graph_result else 0,
                    'max_hops': self.settings.rag.graph_rag_max_hops,
                    'context_added': len(graph_context) > 0,
                    'context_preview': graph_context[:200] if graph_context else "No graph context",
                    'duration_ms': f"{graph_duration:.0f}",
                }
            except Exception as e:
                innovation_results['graph_rag'] = {'error': str(e)}

        # ---- INNOVATION: RAPTOR (Enhances Layer 3) ----
        raptor_context = ""
        if self.raptor:
            try:
                raptor_start = time.time()
                raptor_result = self.raptor.search(query, top_k=3)
                raptor_context = raptor_result.context if raptor_result else ""
                raptor_duration = (time.time() - raptor_start) * 1000
                innovation_results['raptor'] = {
                    'search_level': raptor_result.search_level if raptor_result else -1,
                    'granularity': raptor_result.granularity if raptor_result else 'none',
                    'nodes_matched': len(raptor_result.matched_nodes) if raptor_result else 0,
                    'context_added': len(raptor_context) > 0,
                    'tree_stats': self.raptor.get_stats(),
                    'duration_ms': f"{raptor_duration:.0f}",
                }
            except Exception as e:
                innovation_results['raptor'] = {'error': str(e)}

        if not retrieved_chunks:
            l7_result, response = self._layer7_validate(query, "", [])
            layer_results.extend([
                LayerResult(4, "Corrective RAG (CRAG)", "skipped", 0, {"reason": "No chunks to grade"}),
                LayerResult(5, "LLM Re-Ranking", "skipped", 0, {"reason": "No chunks to rank"}),
                LayerResult(6, "Agentic RAG (ReAct)", "skipped", 0, {"reason": "No context available"}),
                l7_result
            ])
            response.layer_results = layer_results
            response.total_duration_ms = (time.time() - pipeline_start) * 1000
            response.orchestration_result = orch_result
            response.innovation_results = innovation_results
            return response

        # ---- LAYER 4: CRAG ----
        _lf_l4_span = self.tracer.trace_layer_start(_lf_trace, 4, "Corrective RAG (CRAG)", {"chunks_in": len(retrieved_chunks)}) if self.tracer and _lf_trace else None
        l4_result, graded_chunks = self._layer4_crag(query, retrieved_chunks)
        layer_results.append(l4_result)
        if self.tracer and _lf_l4_span:
            try: self.tracer.trace_layer_end(_lf_l4_span, l4_result)
            except Exception: pass

        # ---- LAYER 5: Re-Ranking ----
        _lf_l5_span = self.tracer.trace_layer_start(_lf_trace, 5, "LLM Re-Ranking", {"chunks_in": len(graded_chunks)}) if self.tracer and _lf_trace else None
        l5_result, ranked_chunks = self._layer5_rerank(query, graded_chunks)
        layer_results.append(l5_result)
        if self.tracer and _lf_l5_span:
            try: self.tracer.trace_layer_end(_lf_l5_span, l5_result)
            except Exception: pass

        # ---- LAYER 6: Agentic RAG ----
        _lf_l6_span = self.tracer.trace_layer_start(_lf_trace, 6, "Agentic RAG (ReAct)", {"chunks_in": len(ranked_chunks)}) if self.tracer and _lf_trace else None
        l6_result, enriched_context = self._layer6_agentic(query, ranked_chunks)
        layer_results.append(l6_result)
        if self.tracer and _lf_l6_span:
            try: self.tracer.trace_layer_end(_lf_l6_span, l6_result)
            except Exception: pass

        # Append Graph and RAPTOR context to enriched context
        if graph_context:
            enriched_context += f"\n\n--- Knowledge Graph Context ---\n{graph_context}"
        if raptor_context:
            enriched_context += f"\n\n--- RAPTOR Hierarchical Context ---\n{raptor_context}"

        # Prepare sources
        sources = []
        seen = set()
        for chunk, score in ranked_chunks:
            key = f"{chunk.source}_{chunk.section}"
            if key not in seen:
                sources.append({
                    "source": chunk.source,
                    "section": chunk.section,
                    "relevance": f"{score:.2%}"
                })
                seen.add(key)

        # ---- INNOVATION: Speculative RAG (Alternative to standard Layer 7) ----
        speculative_used = False
        if self.speculative_rag and ranked_chunks:
            try:
                spec_start = time.time()
                chunk_dicts = [
                    {'text': c.text, 'source': c.source, 'section': c.section, 'chunk_id': c.chunk_id}
                    for c, _ in ranked_chunks
                ]
                spec_result = self.speculative_rag.generate_and_verify(query, chunk_dicts)
                spec_duration = (time.time() - spec_start) * 1000
                if spec_result and spec_result.drafts:
                    innovation_results['speculative_rag'] = {
                        'total_drafts': spec_result.total_drafts,
                        'selected_draft_id': spec_result.selected_draft_id,
                        'improvement_over_first': f"{spec_result.improvement_over_first:.1f}%",
                        'drafts': [
                            {
                                'draft_id': d.draft_id,
                                'chunks_used': len(d.chunk_subset),
                                'answer_preview': d.answer_text[:150],
                                'generation_time_ms': f"{d.generation_time_ms:.0f}",
                            }
                            for d in spec_result.drafts
                        ],
                        'verifications': [
                            {
                                'draft_id': v.draft_id,
                                'faithfulness': f"{v.faithfulness_score:.0%}",
                                'completeness': f"{v.completeness_score:.0%}",
                                'relevance': f"{v.relevance_score:.0%}",
                                'overall': f"{v.overall_score:.0%}",
                                'is_selected': v.is_selected,
                                'reasoning': v.reasoning[:150],
                            }
                            for v in spec_result.verifications
                        ],
                        'drafting_time_ms': f"{spec_result.drafting_time_ms:.0f}",
                        'verification_time_ms': f"{spec_result.verification_time_ms:.0f}",
                        'total_time_ms': f"{spec_duration:.0f}",
                    }
                    speculative_used = True
            except Exception as e:
                innovation_results['speculative_rag'] = {'error': str(e)}

        # ---- LAYER 7: Response Validation ----
        _lf_l7_span = self.tracer.trace_layer_start(_lf_trace, 7, "Response Validation", {"context_length": len(enriched_context)}) if self.tracer and _lf_trace else None
        l7_result, response = self._layer7_validate(query, enriched_context, sources)
        layer_results.append(l7_result)
        if self.tracer and _lf_l7_span:
            try: self.tracer.trace_layer_end(_lf_l7_span, l7_result)
            except Exception: pass

        # ---- INNOVATION: Self-RAG (Post-Layer 7 self-reflection) ----
        if self.self_rag:
            try:
                self_start = time.time()
                self_result = self.self_rag.reflect_and_correct(
                    query=query,
                    response=response.answer,
                    context=enriched_context,
                    sources=[c.text[:300] for c, _ in ranked_chunks],
                )
                self_duration = (time.time() - self_start) * 1000
                if self_result:
                    innovation_results['self_rag'] = {
                        'iterations': self_result.iterations,
                        'retrieval_needed': self_result.retrieval_needed,
                        'corrections_made': self_result.corrections_made,
                        'final_confidence': f"{self_result.final_confidence:.0%}",
                        'reflection_log': self_result.reflection_log[:3] if hasattr(self_result, 'reflection_log') else [],
                        'response_improved': self_result.response_improved,
                        'duration_ms': f"{self_duration:.0f}",
                    }
                    if self_result.response_improved and self_result.corrected_response:
                        response.answer = self_result.corrected_response
            except Exception as e:
                innovation_results['self_rag'] = {'error': str(e)}

        # ---- INNOVATION: RAGAS Evaluation (Post-Layer 7 quality metrics) ----
        if self.ragas_evaluator:
            try:
                import random
                if random.random() <= self.settings.rag.ragas_sample_rate:
                    ragas_start = time.time()
                    ragas_result = self.ragas_evaluator.evaluate(
                        query=query,
                        response=response.answer,
                        context=enriched_context,
                        sources=[c.text for c, _ in ranked_chunks],
                    )
                    ragas_duration = (time.time() - ragas_start) * 1000
                    innovation_results['ragas_evaluation'] = {
                        'faithfulness': f"{ragas_result.faithfulness:.0%}",
                        'answer_relevancy': f"{ragas_result.answer_relevancy:.0%}",
                        'context_precision': f"{ragas_result.context_precision:.0%}",
                        'context_recall': f"{ragas_result.context_recall:.0%}",
                        'overall_score': f"{ragas_result.overall_score:.0%}",
                        'grade': ragas_result.grade,
                        'duration_ms': f"{ragas_duration:.0f}",
                    }
            except Exception as e:
                innovation_results['ragas_evaluation'] = {'error': str(e)}

        # ---- INNOVATION: Contextual Retrieval stats ----
        if self.contextual_retrieval:
            innovation_results['contextual_retrieval'] = {
                'enabled': True,
                'documents_enriched': self.contextual_retrieval.get_stats().get('documents_enriched', 0),
                'chunks_enriched': self.contextual_retrieval.get_stats().get('chunks_enriched', 0),
            }

        # ---- GOVERNANCE: Four-Check System (Post-Layer 7) ----
        _lf_gov_span = self.tracer.trace_layer_start(_lf_trace, 8, "AI Governance (Four-Check)", {"query": query[:200]}) if self.tracer and _lf_trace else None
        gov_result = self._run_governance(query, response.answer, enriched_context, ranked_chunks)
        if gov_result:
            # Apply governance modifications to the response
            if gov_result.final_response != response.answer:
                response.answer = gov_result.final_response
            if gov_result.overall_status == "blocked":
                response.validation_status = "blocked"
                response.answer = (
                    "🚫 This response has been blocked by the governance system. "
                    "Your query has been escalated to a banking specialist."
                )
            elif gov_result.overall_status == "escalated":
                response.validation_status = "warning"
                # Deliver the answer with a governance disclaimer instead of blocking
                response.answer = (
                    f"{response.answer}\n\n"
                    f"---\n"
                    f"⚠️ *Governance Note: This response has been flagged for additional review. "
                    f"For critical financial decisions, please verify with your relationship manager "
                    f"or contact Mashreq Bank directly.*"
                )
            elif gov_result.modifications_made:
                # Response was modified (e.g., PII redacted, compliance template applied)
                pass  # final_response already updated above

            # Add governance layer result
            gov_layer = LayerResult(
                layer_number=8,
                layer_name="AI Governance (Four-Check System)",
                status="executed",
                duration_ms=gov_result.total_duration_ms,
                details={
                    "overall_status": gov_result.overall_status,
                    "checks": [
                        {
                            "check_number": c.check_number,
                            "check_name": c.check_name,
                            "status": c.status,
                            "score": f"{c.score:.0%}",
                            "action_taken": c.action_taken,
                            "duration_ms": f"{c.duration_ms:.0f}",
                            "details": c.details,
                        }
                        for c in gov_result.checks
                    ],
                    "modifications": gov_result.modifications_made,
                    "escalated_to_human": gov_result.escalated_to_human,
                    "retry_count": gov_result.retry_count,
                }
            )
            layer_results.append(gov_layer)
            response.governance_result = gov_result

        # ── Langfuse: Trace governance results and emit scores ──
        if self.tracer and _lf_trace:
            try:
                if gov_result and getattr(self.settings.langfuse, 'trace_governance', True):
                    self.tracer.trace_governance(_lf_trace, gov_result)
                if _lf_gov_span:
                    from observability.langfuse_integration import SpanHandle
                    if isinstance(_lf_gov_span, SpanHandle):
                        gov_lr = LayerResult(8, "AI Governance", "executed",
                                             gov_result.total_duration_ms if gov_result else 0,
                                             {"status": gov_result.overall_status if gov_result else "skipped"})
                        self.tracer.trace_layer_end(_lf_gov_span, gov_lr)
            except Exception:
                pass

        response.layer_results = layer_results
        response.total_duration_ms = (time.time() - pipeline_start) * 1000
        response.orchestration_result = orch_result
        response.innovation_results = innovation_results

        # ── Langfuse: Trace innovations and end pipeline trace ──
        if self.tracer and _lf_trace:
            try:
                if innovation_results and getattr(self.settings.langfuse, 'trace_innovations', True):
                    self.tracer.trace_innovations(_lf_trace, innovation_results)
                self.tracer.end_pipeline_trace(_lf_trace, response)
                self.tracer.flush()
            except Exception:
                pass

        # ---- AUDIT TRAIL ----
        self._create_audit_trail(query, response, gov_result, orch_result, ranked_chunks)

        # Store in cache for future queries (only if governance approved)
        if response.validation_status == "approved":
            self.cache.store(query, response)

        return response

    def _turbo_pipeline(self, query: str, pipeline_start: float) -> RAGResponse:
        """Ultra-fast pipeline: FAQ match → Vector search → 1 LLM call → Answer.
        
        Bypasses: Layer 0 (Orchestration), Layer 1 (Cache), Layer 2 (HyDE),
                  Layer 4 (CRAG), Layer 5 (Re-Ranking), Layer 6 (Agentic),
                  All innovations, Governance.
        
        Uses: FAQ exact match, simple vector search (Layer 3), single LLM answer (Layer 7).
        Target: < 3 seconds, 0-1 LLM calls.
        """
        layer_results = []
        turbo_start = time.time()

        # ---- Step 1: FAQ Exact Match (0 LLM calls, ~50ms) ----
        if self.faq_engine and self.faq_engine.faq_pairs:
            try:
                faq_start = time.time()
                faq_result = self.faq_engine.lookup(query)
                faq_ms = (time.time() - faq_start) * 1000
                print(f"[TURBO] FAQ lookup: {faq_ms:.0f}ms, tier={faq_result.tier}, sim={faq_result.similarity:.4f}")

                if faq_result.tier == "exact" and faq_result.faq_pair:
                    pair = faq_result.faq_pair
                    layer_results.append(LayerResult(
                        layer_number=0, layer_name="TURBO: FAQ Exact Match",
                        status="HIT", duration_ms=faq_ms,
                        details={
                            "mode": "TURBO",
                            "tier": "1 (EXACT)",
                            "similarity": f"{faq_result.similarity:.4f}",
                            "matched_question": pair.question,
                            "question_number": pair.question_number,
                            "llm_calls": 0,
                            "layers_bypassed": "ALL (0-7 + Governance)",
                        }
                    ))
                    total_ms = (time.time() - pipeline_start) * 1000
                    print(f"[TURBO] ✅ EXACT MATCH → {pair.question_number} in {total_ms:.0f}ms (0 LLM calls)")
                    return RAGResponse(
                        answer=pair.answer,
                        confidence=min(0.99, faq_result.similarity),
                        sources=[{"source": pair.source_file, "section": pair.question_number, "relevance": f"{faq_result.similarity:.2%}"}],
                        layer_results=layer_results,
                        total_duration_ms=total_ms,
                        pipeline_stopped_at=0,
                        validation_status="approved",
                        orchestration_result=None,
                    )

                elif faq_result.tier == "fuzzy" and faq_result.faq_pair:
                    pair = faq_result.faq_pair
                    adapt_start = time.time()
                    adapted = self.faq_engine.adapt_answer_for_fuzzy_match(query, faq_result, self.client)
                    adapt_ms = (time.time() - adapt_start) * 1000
                    layer_results.append(LayerResult(
                        layer_number=0, layer_name="TURBO: FAQ Fuzzy Match",
                        status="FUZZY_HIT", duration_ms=faq_ms + adapt_ms,
                        details={
                            "mode": "TURBO",
                            "tier": "2 (FUZZY)",
                            "similarity": f"{faq_result.similarity:.4f}",
                            "closest_question": pair.question,
                            "llm_calls": 1,
                            "llm_model": self.faq_engine.fuzzy_model,
                            "layers_bypassed": "ALL (0-7 + Governance)",
                        }
                    ))
                    total_ms = (time.time() - pipeline_start) * 1000
                    print(f"[TURBO] ✅ FUZZY MATCH → {pair.question_number} in {total_ms:.0f}ms (1 LLM call)")
                    return RAGResponse(
                        answer=adapted,
                        confidence=min(0.95, faq_result.similarity + 0.1),
                        sources=[{"source": pair.source_file, "section": pair.question_number, "relevance": f"{faq_result.similarity:.2%}"}],
                        layer_results=layer_results,
                        total_duration_ms=total_ms,
                        pipeline_stopped_at=0,
                        validation_status="approved",
                        orchestration_result=None,
                    )
            except Exception as e:
                print(f"[TURBO] FAQ lookup error: {e}")

        # ---- Step 2: Simple Vector Search (0 LLM calls, ~100ms) ----
        search_start = time.time()
        try:
            query_embedding = self.indexer.embed_model.encode(query, convert_to_numpy=True)
            retrieved_chunks = self.indexer.search(query_embedding, top_k=self.settings.rag.retrieval_top_k)
            search_ms = (time.time() - search_start) * 1000
            print(f"[TURBO] Vector search: {search_ms:.0f}ms, {len(retrieved_chunks)} chunks")

            layer_results.append(LayerResult(
                layer_number=3, layer_name="TURBO: Vector Search",
                status="executed", duration_ms=search_ms,
                details={
                    "mode": "TURBO",
                    "chunks_retrieved": len(retrieved_chunks),
                    "method": "Direct cosine similarity (no HyDE)",
                    "llm_calls": 0,
                }
            ))
        except Exception as e:
            print(f"[TURBO] Vector search error: {e}")
            retrieved_chunks = []

        if not retrieved_chunks:
            total_ms = (time.time() - pipeline_start) * 1000
            return RAGResponse(
                answer="I couldn't find relevant information in the uploaded documents. Please try rephrasing your question.",
                confidence=0.0, sources=[], layer_results=layer_results,
                total_duration_ms=total_ms, pipeline_stopped_at=3,
                validation_status="approved", orchestration_result=None,
            )

        # ---- Step 3: Single LLM Answer Generation (1 LLM call) ----
        gen_start = time.time()
        context = "\n\n---\n\n".join([chunk.text for chunk, score in retrieved_chunks[:5]])
        sources = []
        seen = set()
        for chunk, score in retrieved_chunks[:5]:
            key = f"{chunk.source}_{chunk.section}"
            if key not in seen:
                sources.append({"source": chunk.source, "section": chunk.section, "relevance": f"{score:.2%}"})
                seen.add(key)

        try:
            turbo_prompt = (
                f"Answer the customer's question using ONLY the provided context. "
                f"CRITICAL: You MUST reproduce the FULL and COMPLETE answer from the context — do NOT summarize, shorten, or paraphrase. "
                f"Include ALL details, steps, conditions, amounts, and specifics exactly as stated in the document. "
                f"If the exact answer is in the context, reproduce it in full. "
                f"Do NOT add information not present in the context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
            answer = self._call_llm(turbo_prompt)
            gen_ms = (time.time() - gen_start) * 1000
            print(f"[TURBO] LLM generation: {gen_ms:.0f}ms")

            layer_results.append(LayerResult(
                layer_number=7, layer_name="TURBO: Direct Answer",
                status="executed", duration_ms=gen_ms,
                details={
                    "mode": "TURBO",
                    "context_chunks": min(5, len(retrieved_chunks)),
                    "llm_calls": 1,
                    "llm_model": self.settings.rag.llm_model,
                }
            ))
        except Exception as e:
            print(f"[TURBO] LLM generation error: {e}")
            answer = f"Error generating answer: {e}"
            gen_ms = (time.time() - gen_start) * 1000

        total_ms = (time.time() - pipeline_start) * 1000
        print(f"[TURBO] ✅ Complete in {total_ms:.0f}ms (1 LLM call)")

        return RAGResponse(
            answer=answer,
            confidence=0.85,
            sources=sources,
            layer_results=layer_results,
            total_duration_ms=total_ms,
            pipeline_stopped_at=7,
            validation_status="approved",
            orchestration_result=None,
        )

    def _run_governance(
        self, query: str, response: str, context: str,
        ranked_chunks: List[Tuple['ChunkMetadata', float]]
    ) -> Optional[GovernanceResult]:
        """Run the Four-Check Governance System on the generated response."""
        if not self.settings.rag.governance_enabled or not self.governance:
            return None

        try:
            gov_result = self.governance.run_governance_checks(
                query=query,
                response=response,
                context="\n\n---\n\n".join([c.text for c, _ in ranked_chunks]) if ranked_chunks else context,
                max_retries=self.settings.rag.governance_max_retries,
            )
            return gov_result
        except Exception as e:
            # Governance failure should not break the pipeline — log and continue
            print(f"[GOVERNANCE ERROR] {e}")
            return None

    def _create_audit_trail(
        self, query: str, response: 'RAGResponse',
        gov_result: Optional[GovernanceResult],
        orch_result: Optional[OrchestrationResult],
        ranked_chunks: List[Tuple['ChunkMetadata', float]]
    ):
        """Create and persist the 14-field audit record."""
        if not self.settings.rag.governance_audit_trail_enabled or not self.governance:
            return

        try:
            # Build orchestration summary for audit
            orch_summary = {}
            if orch_result:
                c = orch_result.classification
                orch_summary = {
                    "product": c.primary_product_name,
                    "intent": c.intent.intent_name,
                    "risk_level": c.risk.risk_level,
                    "risk_score": c.risk.risk_score,
                    "confidence": c.confidence.overall_confidence,
                    "routed_collections": orch_result.routed_collections,
                }

            # Build chunk summaries for audit
            retrieved = []
            used = []
            for chunk, score in ranked_chunks:
                chunk_info = {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section,
                    "score": round(score, 4),
                }
                retrieved.append(chunk_info)
                used.append(chunk_info)

            self.governance.create_audit_record(
                user_id="session_user",  # Will be replaced by actual user from session
                query=query,
                response_text=response.answer,
                confidence=response.confidence,
                governance_result=gov_result if gov_result else GovernanceResult(
                    overall_status="skipped", final_response=response.answer,
                    original_response=response.answer
                ),
                orchestration_result=orch_summary,
                retrieved_chunks=retrieved,
                used_chunks=used,
                model_version=f"7-Layer-RAG-v2.0 | {self.settings.llm.model_name}",
            )
        except Exception as e:
            print(f"[AUDIT ERROR] {e}")
