"""
MongoDB Document Store — Persistent storage for documents, chunks, tables, and graph data.

Collections:
  - documents: Raw document metadata (filename, upload_time, file_hash, page_count, etc.)
  - chunks: Individual text chunks with metadata (text, source, section, chunk_id, embedding_id)
  - tables: Extracted table data (markdown, page, extraction_method, source)
  - graph_entities: Knowledge graph entities (name, type, attributes, source)
  - graph_relationships: Knowledge graph relationships (source_entity, target_entity, type)
  - faq_pairs: FAQ question-answer pairs with embeddings for exact-match retrieval

Feature Toggle: mongodb.enabled in settings (default: True)
"""
import os
import time
import hashlib
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


@dataclass
class DocumentRecord:
    """A document stored in MongoDB."""
    filename: str
    file_hash: str
    file_size_bytes: int
    upload_time: float
    page_count: int = 0
    chunk_count: int = 0
    table_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    extraction_strategies: List[str] = None
    processing_time_ms: float = 0.0
    status: str = "indexed"  # indexed, pending, failed
    product_category: str = "general"
    document_summary: str = ""
    is_bilingual: bool = False

    def __post_init__(self):
        if self.extraction_strategies is None:
            self.extraction_strategies = []


@dataclass
class ChunkRecord:
    """A text chunk stored in MongoDB."""
    chunk_id: str
    text: str
    source: str  # filename
    section: str = ""
    page: int = 0
    chunk_type: str = "text"  # text, table_row, product_detail, vision
    confidence: float = 0.0
    faiss_index_id: int = -1  # Maps to FAISS vector index position
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class MongoDocumentStore:
    """MongoDB-backed persistent document and chunk store.
    
    Stores:
    - Document metadata (filename, hash, page count, extraction stats)
    - Text chunks with FAISS index mapping
    - Extracted tables as markdown
    - Knowledge graph entities and relationships
    
    All data persists across server restarts.
    """

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self.settings = settings
        self.client = None
        self.db = None
        self._connected = False

        if not PYMONGO_AVAILABLE:
            print("[MONGO] pymongo not installed. MongoDB persistence disabled.")
            return

        if not getattr(settings, 'mongodb', None) or not settings.mongodb.enabled:
            print("[MONGO] MongoDB disabled in settings.")
            return

        try:
            self.client = MongoClient(
                settings.mongodb.connection_string,
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[settings.mongodb.database_name]
            self._connected = True
            self._ensure_indexes()
            print(f"[MONGO] Connected to MongoDB: {settings.mongodb.database_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"[MONGO] Failed to connect to MongoDB: {e}")
            self._connected = False
        except Exception as e:
            print(f"[MONGO] Unexpected error: {e}")
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _ensure_indexes(self):
        """Create indexes for efficient querying."""
        if not self._connected:
            return
        try:
            # Documents collection
            self.db.documents.create_index([("filename", ASCENDING)], unique=True)
            self.db.documents.create_index([("file_hash", ASCENDING)])
            self.db.documents.create_index([("status", ASCENDING)])
            self.db.documents.create_index([("product_category", ASCENDING)])

            # Chunks collection
            self.db.chunks.create_index([("source", ASCENDING)])
            self.db.chunks.create_index([("chunk_id", ASCENDING)])
            self.db.chunks.create_index([("faiss_index_id", ASCENDING)])
            self.db.chunks.create_index([("chunk_type", ASCENDING)])

            # Tables collection
            self.db.tables.create_index([("source", ASCENDING)])
            self.db.tables.create_index([("page", ASCENDING)])

            # Graph collections
            self.db.graph_entities.create_index([("name", ASCENDING)])
            self.db.graph_entities.create_index([("source", ASCENDING)])
            self.db.graph_relationships.create_index([("source_entity", ASCENDING)])
            self.db.graph_relationships.create_index([("target_entity", ASCENDING)])

            # FAQ pairs collection
            self.db.faq_pairs.create_index([("source", ASCENDING)])
            self.db.faq_pairs.create_index([("question_number", ASCENDING)])
        except Exception as e:
            print(f"[MONGO] Error creating indexes: {e}")

    # ═══════════════════════════════════════════════════════════
    # DOCUMENT OPERATIONS
    # ═══════════════════════════════════════════════════════════

    def document_exists(self, filename: str) -> bool:
        """Check if a document is already stored."""
        if not self._connected:
            return False
        return self.db.documents.find_one({"filename": filename}) is not None

    def document_hash_exists(self, file_hash: str) -> bool:
        """Check if a document with this hash is already stored (dedup by content)."""
        if not self._connected:
            return False
        return self.db.documents.find_one({"file_hash": file_hash}) is not None

    def store_document(self, record: DocumentRecord) -> bool:
        """Store or update a document record."""
        if not self._connected:
            return False
        try:
            doc_dict = asdict(record)
            self.db.documents.update_one(
                {"filename": record.filename},
                {"$set": doc_dict},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"[MONGO] Error storing document: {e}")
            return False

    def get_document(self, filename: str) -> Optional[Dict]:
        """Get a document record by filename."""
        if not self._connected:
            return None
        doc = self.db.documents.find_one({"filename": filename}, {"_id": 0})
        return doc

    def get_all_documents(self) -> List[Dict]:
        """Get all stored document records."""
        if not self._connected:
            return []
        return list(self.db.documents.find({}, {"_id": 0}).sort("upload_time", DESCENDING))

    def delete_document(self, filename: str) -> bool:
        """Delete a document and all its associated chunks, tables, and graph data."""
        if not self._connected:
            return False
        try:
            self.db.documents.delete_one({"filename": filename})
            self.db.chunks.delete_many({"source": filename})
            self.db.tables.delete_many({"source": filename})
            self.db.graph_entities.delete_many({"source": filename})
            self.db.graph_relationships.delete_many({"source": filename})
            self.db.faq_pairs.delete_many({"source": filename})
            return True
        except Exception as e:
            print(f"[MONGO] Error deleting document: {e}")
            return False

    # ═══════════════════════════════════════════════════════════
    # CHUNK OPERATIONS
    # ═══════════════════════════════════════════════════════════

    def store_chunks(self, chunks: List[ChunkRecord]) -> int:
        """Store multiple chunks in bulk. Returns count of stored chunks."""
        if not self._connected or not chunks:
            return 0
        try:
            chunk_dicts = [asdict(c) for c in chunks]
            result = self.db.chunks.insert_many(chunk_dicts)
            return len(result.inserted_ids)
        except Exception as e:
            print(f"[MONGO] Error storing chunks: {e}")
            return 0

    def get_chunks_by_source(self, source: str) -> List[Dict]:
        """Get all chunks for a given document source."""
        if not self._connected:
            return []
        return list(self.db.chunks.find({"source": source}, {"_id": 0}))

    def get_all_chunks(self) -> List[Dict]:
        """Get all stored chunks across all documents."""
        if not self._connected:
            return []
        return list(self.db.chunks.find({}, {"_id": 0}))

    def get_chunk_by_faiss_id(self, faiss_id: int) -> Optional[Dict]:
        """Get a chunk by its FAISS index ID."""
        if not self._connected:
            return None
        return self.db.chunks.find_one({"faiss_index_id": faiss_id}, {"_id": 0})

    def get_total_chunk_count(self) -> int:
        """Get total number of chunks across all documents."""
        if not self._connected:
            return 0
        return self.db.chunks.count_documents({})

    # ═══════════════════════════════════════════════════════════
    # TABLE OPERATIONS
    # ═══════════════════════════════════════════════════════════

    def store_tables(self, tables: List[Dict]) -> int:
        """Store extracted tables. Each table dict should have: markdown, page, source, extraction_method."""
        if not self._connected or not tables:
            return 0
        try:
            result = self.db.tables.insert_many(tables)
            return len(result.inserted_ids)
        except Exception as e:
            print(f"[MONGO] Error storing tables: {e}")
            return 0

    def get_tables_by_source(self, source: str) -> List[Dict]:
        """Get all tables for a given document."""
        if not self._connected:
            return []
        return list(self.db.tables.find({"source": source}, {"_id": 0}))

    # ═══════════════════════════════════════════════════════════
    # GRAPH OPERATIONS
    # ═══════════════════════════════════════════════════════════

    def store_graph_entities(self, entities: List[Dict]) -> int:
        """Store knowledge graph entities."""
        if not self._connected or not entities:
            return 0
        try:
            result = self.db.graph_entities.insert_many(entities)
            return len(result.inserted_ids)
        except Exception as e:
            print(f"[MONGO] Error storing entities: {e}")
            return 0

    def store_graph_relationships(self, relationships: List[Dict]) -> int:
        """Store knowledge graph relationships."""
        if not self._connected or not relationships:
            return 0
        try:
            result = self.db.graph_relationships.insert_many(relationships)
            return len(result.inserted_ids)
        except Exception as e:
            print(f"[MONGO] Error storing relationships: {e}")
            return 0

    def get_all_graph_entities(self) -> List[Dict]:
        """Get all graph entities."""
        if not self._connected:
            return []
        return list(self.db.graph_entities.find({}, {"_id": 0}))

    def get_all_graph_relationships(self) -> List[Dict]:
        """Get all graph relationships."""
        if not self._connected:
            return []
        return list(self.db.graph_relationships.find({}, {"_id": 0}))

    # ═══════════════════════════════════════════════════════════
    # FAQ PAIR OPERATIONS
    # ═══════════════════════════════════════════════════════════

    def store_faq_pairs(self, faq_pairs: List[Dict]) -> int:
        """Store FAQ question-answer pairs extracted from documents.
        Each dict should have: question, answer, question_number, source, embedding (as list).
        """
        if not self._connected or not faq_pairs:
            return 0
        try:
            # Remove existing FAQ pairs for the same source to avoid duplicates
            sources = set(p.get('source', '') for p in faq_pairs)
            for src in sources:
                self.db.faq_pairs.delete_many({"source": src})
            result = self.db.faq_pairs.insert_many(faq_pairs)
            return len(result.inserted_ids)
        except Exception as e:
            print(f"[MONGO] Error storing FAQ pairs: {e}")
            return 0

    def get_faq_pairs_by_source(self, source: str) -> List[Dict]:
        """Get all FAQ pairs for a given document source."""
        if not self._connected:
            return []
        return list(self.db.faq_pairs.find({"source": source}, {"_id": 0}))

    def get_all_faq_pairs(self) -> List[Dict]:
        """Get all stored FAQ pairs across all documents."""
        if not self._connected:
            return []
        return list(self.db.faq_pairs.find({}, {"_id": 0}))

    def delete_faq_pairs_by_source(self, source: str) -> int:
        """Delete all FAQ pairs for a given document source."""
        if not self._connected:
            return 0
        result = self.db.faq_pairs.delete_many({"source": source})
        return result.deleted_count

    # ═══════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Get overall store statistics."""
        if not self._connected:
            return {"connected": False, "documents": 0, "chunks": 0, "tables": 0, "entities": 0, "relationships": 0}
        return {
            "connected": True,
            "documents": self.db.documents.count_documents({}),
            "chunks": self.db.chunks.count_documents({}),
            "tables": self.db.tables.count_documents({}),
            "entities": self.db.graph_entities.count_documents({}),
            "relationships": self.db.graph_relationships.count_documents({}),
            "faq_pairs": self.db.faq_pairs.count_documents({}),
        }

    def clear_all(self):
        """Clear all collections. Use with caution."""
        if not self._connected:
            return
        self.db.documents.delete_many({})
        self.db.chunks.delete_many({})
        self.db.tables.delete_many({})
        self.db.graph_entities.delete_many({})
        self.db.graph_relationships.delete_many({})
        self.db.faq_pairs.delete_many({})
        print("[MONGO] All collections cleared.")

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        """Compute SHA-256 hash of file content for deduplication."""
        return hashlib.sha256(file_bytes).hexdigest()
