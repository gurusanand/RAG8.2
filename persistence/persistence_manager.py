"""
Persistence Manager — Bridges MongoDB + FAISS with the existing RAG pipeline.

This module provides a unified interface for:
  1. Checking if a document is already indexed (skip re-upload)
  2. Storing document metadata + chunks in MongoDB after extraction
  3. Storing embeddings in FAISS after embedding
  4. Loading all persisted chunks + embeddings on server startup (no re-embedding)
  5. Searching via FAISS instead of in-memory numpy arrays

Feature Toggle: mongodb.enabled in settings (default: True)
When disabled, falls back to the original in-memory behavior.
"""
import os
import time
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from persistence.mongo_store import MongoDocumentStore, DocumentRecord, ChunkRecord
from persistence.faiss_store import FAISSVectorStore


class PersistenceManager:
    """Unified persistence layer for the RAG pipeline.
    
    Manages the lifecycle of documents from upload to retrieval:
    
    Upload flow:
      1. Check if document already exists (by filename or content hash)
      2. If new: extract, chunk, embed, store in MongoDB + FAISS
      3. If exists: skip extraction and embedding, return existing stats
    
    Startup flow:
      1. Load FAISS index from disk (embeddings already persisted)
      2. Load chunk metadata from MongoDB (text, source, section, etc.)
      3. Populate the in-memory DocumentIndexer for backward compatibility
    
    Query flow:
      1. Embed query using local model
      2. Search FAISS for top-K nearest neighbors
      3. Retrieve full chunk metadata from FAISS metadata (or MongoDB fallback)
    """

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self.settings = settings
        self._enabled = getattr(settings, 'mongodb', None) and settings.mongodb.enabled

        self.mongo = None
        self.faiss = None

        if self._enabled:
            try:
                self.mongo = MongoDocumentStore(settings)
                # Use 384 dimensions for all-MiniLM-L6-v2
                self.faiss = FAISSVectorStore(settings, dimension=384)
                print(f"[PERSISTENCE] Initialized: MongoDB={self.mongo.is_connected}, FAISS={self.faiss.is_initialized}")
            except Exception as e:
                print(f"[PERSISTENCE] Initialization error: {e}")
                self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.mongo is not None and self.faiss is not None

    @property
    def is_connected(self) -> bool:
        return self.is_enabled and self.mongo.is_connected and self.faiss.is_initialized

    # ═══════════════════════════════════════════════════════════
    # DOCUMENT LIFECYCLE
    # ═══════════════════════════════════════════════════════════

    def document_exists(self, filename: str, file_bytes: bytes = None) -> bool:
        """Check if a document is already stored (by filename or content hash).
        
        Returns True if the document exists and is fully indexed.
        """
        if not self.is_connected:
            return False

        # Check by filename
        if self.mongo.document_exists(filename):
            doc = self.mongo.get_document(filename)
            if doc and doc.get('status') == 'indexed':
                return True

        # Check by content hash (same file, different name)
        if file_bytes:
            file_hash = MongoDocumentStore.compute_file_hash(file_bytes)
            if self.mongo.document_hash_exists(file_hash):
                return True

        return False

    def store_document_with_chunks(
        self,
        filename: str,
        file_bytes: bytes,
        chunks: List[Dict],
        embeddings: np.ndarray,
        extraction_result: Dict = None
    ) -> Dict:
        """Store a fully processed document: metadata in MongoDB, embeddings in FAISS.
        
        Args:
            filename: Document filename
            file_bytes: Raw file bytes (for hash computation)
            chunks: List of chunk dicts with keys: text, source, section, chunk_id, chunk_type, confidence
            embeddings: numpy array of shape (n_chunks, dimension)
            extraction_result: Optional dict from the extraction pipeline with stats
        
        Returns:
            Dict with storage stats
        """
        if not self.is_connected:
            return {"stored": False, "reason": "persistence not connected"}

        start_time = time.time()
        file_hash = MongoDocumentStore.compute_file_hash(file_bytes)

        # 1. Store embeddings in FAISS with metadata
        metadata_list = []
        for i, chunk in enumerate(chunks):
            metadata_list.append({
                'chunk_id': chunk.get('chunk_id', f'{filename}_chunk_{i}'),
                'source': filename,
                'text': chunk.get('text', ''),
                'section': chunk.get('section', ''),
                'chunk_type': chunk.get('chunk_type', 'text'),
                'confidence': chunk.get('confidence', 0.0),
                'page': chunk.get('page', 0),
            })

        faiss_ids = self.faiss.add_vectors(embeddings, metadata_list)

        # 2. Store chunks in MongoDB with FAISS ID mapping
        chunk_records = []
        for i, chunk in enumerate(chunks):
            chunk_records.append(ChunkRecord(
                chunk_id=chunk.get('chunk_id', f'{filename}_chunk_{i}'),
                text=chunk.get('text', ''),
                source=filename,
                section=chunk.get('section', ''),
                page=chunk.get('page', 0),
                chunk_type=chunk.get('chunk_type', 'text'),
                confidence=chunk.get('confidence', 0.0),
                faiss_index_id=faiss_ids[i] if i < len(faiss_ids) else -1,
            ))
        stored_count = self.mongo.store_chunks(chunk_records)

        # 3. Store tables if available
        table_count = 0
        if extraction_result and extraction_result.get('tables_extracted'):
            tables = []
            for t in extraction_result.get('tables_data', []):
                tables.append({
                    'markdown': t.get('markdown', ''),
                    'page': t.get('page', 0),
                    'source': filename,
                    'extraction_method': t.get('extraction_method', 'unknown'),
                })
            if tables:
                table_count = self.mongo.store_tables(tables)

        # 4. Store graph data if available
        entity_count = 0
        relationship_count = 0
        if extraction_result:
            if extraction_result.get('graph_entities_data'):
                entities = [
                    {**e, 'source': filename}
                    for e in extraction_result.get('graph_entities_data', [])
                ]
                entity_count = self.mongo.store_graph_entities(entities)
            if extraction_result.get('graph_relationships_data'):
                rels = [
                    {**r, 'source': filename}
                    for r in extraction_result.get('graph_relationships_data', [])
                ]
                relationship_count = self.mongo.store_graph_relationships(rels)

        # 5. Store document metadata
        ext_result = extraction_result or {}
        doc_record = DocumentRecord(
            filename=filename,
            file_hash=file_hash,
            file_size_bytes=len(file_bytes),
            upload_time=time.time(),
            page_count=ext_result.get('profile', {}).get('page_count', 0),
            chunk_count=stored_count,
            table_count=table_count,
            entity_count=entity_count,
            relationship_count=relationship_count,
            extraction_strategies=ext_result.get('strategies_used', []),
            processing_time_ms=ext_result.get('processing_time_ms', 0.0),
            status='indexed',
            product_category=ext_result.get('product_category', 'general'),
            document_summary=ext_result.get('document_summary', ''),
            is_bilingual=ext_result.get('profile', {}).get('is_bilingual', False),
        )
        self.mongo.store_document(doc_record)

        elapsed = (time.time() - start_time) * 1000
        return {
            "stored": True,
            "chunks_stored": stored_count,
            "faiss_vectors_added": len(faiss_ids),
            "tables_stored": table_count,
            "entities_stored": entity_count,
            "relationships_stored": relationship_count,
            "persistence_time_ms": elapsed
        }

    # ═══════════════════════════════════════════════════════════
    # STARTUP — LOAD PERSISTED DATA
    # ═══════════════════════════════════════════════════════════

    def load_into_indexer(self, indexer) -> int:
        """Load all persisted chunks and embeddings into the in-memory DocumentIndexer.
        
        This is called on server startup to restore the state without re-embedding.
        The FAISS index is already loaded from disk; we just need to populate
        the DocumentIndexer.chunks list for backward compatibility with the
        existing retrieval pipeline.
        
        Args:
            indexer: DocumentIndexer instance to populate
        
        Returns:
            Number of chunks loaded
        """
        if not self.is_connected:
            return 0

        # Load all chunks from MongoDB
        all_chunks = self.mongo.get_all_chunks()
        if not all_chunks:
            return 0

        from rag_engine.seven_layer_rag import ChunkMetadata

        loaded = 0
        for chunk_data in all_chunks:
            chunk = ChunkMetadata(
                text=chunk_data.get('text', ''),
                source=chunk_data.get('source', ''),
                page=chunk_data.get('page', 0),
                section=chunk_data.get('section', ''),
                chunk_id=chunk_data.get('chunk_id', ''),
            )
            indexer.chunks.append(chunk)
            loaded += 1

        # Load embeddings from FAISS into the indexer's numpy array
        # CRITICAL: Sync FAISS vectors with MongoDB chunks to prevent index-out-of-range errors
        if self.faiss.get_vector_count() > 0:
            faiss_count = self.faiss.get_vector_count()
            chunk_count = len(indexer.chunks)
            
            if faiss_count == chunk_count:
                # Perfect sync — load all vectors directly
                vectors = []
                for i in range(faiss_count):
                    try:
                        vec = self.faiss.index.reconstruct(i)
                        vectors.append(vec)
                    except:
                        pass
                if vectors:
                    indexer.embeddings = np.array(vectors, dtype=np.float32)
                print(f"[PERSISTENCE] Loaded {loaded} chunks and {faiss_count} vectors from persistent storage")
            else:
                # MISMATCH: FAISS has different count than MongoDB chunks
                # This happens when documents were deleted from MongoDB but FAISS wasn't cleaned up,
                # or when the app crashed during indexing.
                print(f"[PERSISTENCE] WARNING: FAISS/chunk mismatch — {faiss_count} vectors vs {chunk_count} chunks")
                
                # Strategy: Use FAISS metadata to match vectors to chunks by chunk_id
                chunk_id_set = set(c.chunk_id for c in indexer.chunks)
                chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(indexer.chunks)}
                
                if self.faiss.metadata:
                    # Build ordered vectors matching the chunk order
                    matched_vectors = [None] * chunk_count
                    matched_count = 0
                    
                    for faiss_idx, meta in enumerate(self.faiss.metadata):
                        cid = meta.get('chunk_id', '')
                        if cid in chunk_id_to_idx:
                            try:
                                vec = self.faiss.index.reconstruct(faiss_idx)
                                chunk_pos = chunk_id_to_idx[cid]
                                matched_vectors[chunk_pos] = vec
                                matched_count += 1
                            except:
                                pass
                    
                    if matched_count > 0:
                        # For any unmatched chunks, re-embed them
                        unmatched_indices = [i for i, v in enumerate(matched_vectors) if v is None]
                        if unmatched_indices and hasattr(indexer, 'embed_model') and indexer.embed_model:
                            print(f"[PERSISTENCE] Re-embedding {len(unmatched_indices)} unmatched chunks...")
                            for idx in unmatched_indices:
                                try:
                                    vec = indexer.embed_model.encode(indexer.chunks[idx].text, convert_to_numpy=True)
                                    matched_vectors[idx] = vec
                                    matched_count += 1
                                except:
                                    # Use zero vector as fallback
                                    matched_vectors[idx] = np.zeros(self.faiss.dimension, dtype=np.float32)
                        else:
                            # Fill unmatched with zero vectors
                            for idx in unmatched_indices:
                                matched_vectors[idx] = np.zeros(self.faiss.dimension, dtype=np.float32)
                        
                        indexer.embeddings = np.array(matched_vectors, dtype=np.float32)
                        print(f"[PERSISTENCE] Synced: {matched_count}/{chunk_count} chunks matched with FAISS vectors")
                    else:
                        # No matches found — re-embed all chunks
                        print(f"[PERSISTENCE] No chunk_id matches found. Re-embedding {chunk_count} chunks...")
                        if hasattr(indexer, 'embed_model') and indexer.embed_model:
                            texts = [c.text for c in indexer.chunks]
                            indexer.embeddings = indexer.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
                            print(f"[PERSISTENCE] Re-embedded {chunk_count} chunks")
                else:
                    # No FAISS metadata — truncate vectors to match chunk count
                    print(f"[PERSISTENCE] No FAISS metadata available. Truncating vectors to {chunk_count}.")
                    vectors = []
                    for i in range(min(faiss_count, chunk_count)):
                        try:
                            vec = self.faiss.index.reconstruct(i)
                            vectors.append(vec)
                        except:
                            pass
                    if vectors:
                        indexer.embeddings = np.array(vectors, dtype=np.float32)
                
                # Rebuild FAISS index to match the synced state
                if indexer.embeddings is not None and len(indexer.embeddings) == chunk_count:
                    try:
                        import faiss as faiss_lib
                        self.faiss.index = faiss_lib.IndexFlatIP(self.faiss.dimension)
                        normalized = indexer.embeddings.copy()
                        faiss_lib.normalize_L2(normalized)
                        self.faiss.index.add(normalized)
                        # Rebuild metadata
                        self.faiss.metadata = []
                        for i, chunk in enumerate(indexer.chunks):
                            self.faiss.metadata.append({
                                'chunk_id': chunk.chunk_id,
                                'source': chunk.source,
                                'section': chunk.section,
                                'faiss_index_id': i,
                            })
                        self.faiss._save_to_disk()
                        print(f"[PERSISTENCE] Rebuilt FAISS index: {self.faiss.index.ntotal} vectors (synced with {chunk_count} chunks)")
                    except Exception as rebuild_err:
                        print(f"[PERSISTENCE] Error rebuilding FAISS index: {rebuild_err}")
        else:
            print(f"[PERSISTENCE] Loaded {loaded} chunks, no FAISS vectors found")

        return loaded

    def search_faiss(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[Dict, float]]:
        """Search FAISS directly for nearest neighbors.
        
        Returns list of (metadata_dict, score) tuples.
        Each metadata_dict contains: chunk_id, source, text, section, chunk_type, confidence, faiss_index_id
        """
        if not self.is_connected or not self.faiss:
            return []
        return self.faiss.search(query_embedding, top_k)

    # ═══════════════════════════════════════════════════════════
    # STATISTICS & MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Get combined persistence statistics."""
        stats = {
            "enabled": self._enabled,
            "connected": self.is_connected,
        }
        if self.mongo:
            stats["mongodb"] = self.mongo.get_stats()
        if self.faiss:
            stats["faiss"] = self.faiss.get_stats()
        return stats

    def get_all_documents(self) -> List[Dict]:
        """Get all stored document records."""
        if not self.is_connected:
            return []
        return self.mongo.get_all_documents()

    def delete_document(self, filename: str) -> bool:
        """Delete a document from both MongoDB and FAISS."""
        if not self.is_connected:
            return False
        # Remove from FAISS
        self.faiss.remove_document(filename)
        # Remove from MongoDB
        return self.mongo.delete_document(filename)

    def clear_all(self):
        """Clear all persistent data. Use with caution."""
        if self.mongo:
            self.mongo.clear_all()
        if self.faiss:
            self.faiss.clear_all()
