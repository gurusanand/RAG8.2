"""
FAISS Persistent Vector Store — Saves and loads FAISS indexes to/from disk.

Provides:
  - add_vectors(): Add new embeddings to the index with metadata mapping
  - search(): Find top-K nearest neighbors
  - save(): Persist the FAISS index and metadata to disk
  - load(): Restore the FAISS index and metadata from disk
  - get_vector_count(): Total vectors in the index

Feature Toggle: mongodb.enabled in settings (shared with MongoDB — both are part of persistence layer)
"""
import os
import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSVectorStore:
    """FAISS-backed persistent vector store.
    
    Stores embeddings in a FAISS index (IndexFlatIP for cosine similarity)
    and maintains a metadata mapping (faiss_id -> chunk_id, source, text_preview).
    
    Both the FAISS index and metadata are saved to disk and can be restored
    on server restart without re-embedding.
    """

    def __init__(self, settings=None, dimension: int = 384):
        """Initialize the FAISS vector store.
        
        Args:
            settings: Application settings
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2, 1536 for OpenAI)
        """
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self.settings = settings
        self.dimension = dimension

        # Paths for persistence
        self.index_dir = settings.paths.get_abs_path(settings.paths.faiss_index_dir)
        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path = os.path.join(self.index_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.index_dir, "faiss_metadata.json")

        # FAISS index (IndexFlatIP = Inner Product, used with normalized vectors for cosine similarity)
        self.index = None
        self.metadata: List[Dict[str, Any]] = []  # faiss_id -> {chunk_id, source, text, section, ...}
        self._initialized = False

        if not FAISS_AVAILABLE:
            print("[FAISS] faiss-cpu not installed. FAISS persistence disabled.")
            return

        # Try to load existing index from disk
        if self._load_from_disk():
            print(f"[FAISS] Loaded existing index: {self.index.ntotal} vectors, {len(self.metadata)} metadata entries")
            self._initialized = True
        else:
            # Create fresh index
            self.index = faiss.IndexFlatIP(self.dimension)
            self._initialized = True
            print(f"[FAISS] Created new index with dimension {self.dimension}")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict]) -> List[int]:
        """Add vectors to the FAISS index with associated metadata.
        
        Args:
            embeddings: numpy array of shape (n, dimension), L2-normalized
            metadata_list: List of dicts with keys: chunk_id, source, text, section, chunk_type, etc.
        
        Returns:
            List of FAISS index IDs assigned to the new vectors
        """
        if not self._initialized or self.index is None:
            return []

        if len(embeddings) == 0:
            return []

        # Ensure embeddings are float32 and normalized for cosine similarity
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        # Get starting ID
        start_id = self.index.ntotal

        # Add to FAISS
        self.index.add(embeddings)

        # Store metadata with FAISS IDs
        faiss_ids = []
        for i, meta in enumerate(metadata_list):
            faiss_id = start_id + i
            meta['faiss_index_id'] = faiss_id
            self.metadata.append(meta)
            faiss_ids.append(faiss_id)

        # Auto-save to disk after adding
        self._save_to_disk()

        return faiss_ids

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[Dict, float]]:
        """Search for nearest neighbors.
        
        Args:
            query_embedding: numpy array of shape (dimension,)
            top_k: Number of results to return
        
        Returns:
            List of (metadata_dict, similarity_score) tuples
        """
        if not self._initialized or self.index is None or self.index.ntotal == 0:
            return []

        # Normalize query for cosine similarity
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # Search
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for i in range(k):
            idx = int(indices[0][i])
            score = float(scores[0][i])
            if 0 <= idx < len(self.metadata):
                results.append((self.metadata[idx], score))

        return results

    def get_vector_count(self) -> int:
        """Get total number of vectors in the index."""
        if not self._initialized or self.index is None:
            return 0
        return self.index.ntotal

    def get_sources(self) -> List[str]:
        """Get list of unique document sources in the index."""
        return list(set(m.get('source', '') for m in self.metadata))

    def get_document_count(self) -> int:
        """Get number of unique documents in the index."""
        return len(self.get_sources())

    def has_document(self, source: str) -> bool:
        """Check if a document is already indexed."""
        return any(m.get('source') == source for m in self.metadata)

    def remove_document(self, source: str) -> bool:
        """Remove all vectors for a document. 
        
        Note: FAISS IndexFlatIP doesn't support removal, so we rebuild the index.
        """
        if not self._initialized or self.index is None:
            return False

        # Filter out the document's metadata and vectors
        keep_indices = [i for i, m in enumerate(self.metadata) if m.get('source') != source]
        
        if len(keep_indices) == len(self.metadata):
            return False  # Nothing to remove

        if len(keep_indices) == 0:
            # All vectors removed, reset
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self._save_to_disk()
            return True

        # Reconstruct vectors for kept indices
        kept_vectors = np.array([self.index.reconstruct(i) for i in keep_indices], dtype=np.float32)
        kept_metadata = [self.metadata[i] for i in keep_indices]

        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(kept_vectors)

        # Update metadata with new FAISS IDs
        self.metadata = []
        for i, meta in enumerate(kept_metadata):
            meta['faiss_index_id'] = i
            self.metadata.append(meta)

        self._save_to_disk()
        return True

    def _save_to_disk(self):
        """Persist the FAISS index and metadata to disk."""
        if not self._initialized or self.index is None:
            return
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump({
                    'dimension': self.dimension,
                    'total_vectors': self.index.ntotal,
                    'metadata': self.metadata,
                    'saved_at': time.time()
                }, f, indent=2, default=str)
        except Exception as e:
            print(f"[FAISS] Error saving to disk: {e}")

    def _load_from_disk(self) -> bool:
        """Load the FAISS index and metadata from disk."""
        if not FAISS_AVAILABLE:
            return False
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            return False
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
            self.metadata = data.get('metadata', [])
            self.dimension = data.get('dimension', self.dimension)

            # Validate consistency
            if self.index.ntotal != len(self.metadata):
                print(f"[FAISS] Warning: index has {self.index.ntotal} vectors but {len(self.metadata)} metadata entries. Resetting.")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []
                return False

            return True
        except Exception as e:
            print(f"[FAISS] Error loading from disk: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return {
            "initialized": self._initialized,
            "total_vectors": self.get_vector_count(),
            "total_documents": self.get_document_count(),
            "dimension": self.dimension,
            "index_path": self.index_path,
            "index_exists_on_disk": os.path.exists(self.index_path),
            "sources": self.get_sources()
        }

    def clear_all(self):
        """Clear the entire index and metadata. Use with caution."""
        if not self._initialized:
            return
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self._save_to_disk()
        print("[FAISS] Index cleared.")
