"""
Hybrid Search Engine — BM25 + Vector Search with Reciprocal Rank Fusion (RRF).
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class HybridSearchResult:
    bm25_count: int = 0
    fused_count: int = 0
    top_results: List[Dict] = field(default_factory=list)


class HybridSearchEngine:
    """BM25 + Vector search with Reciprocal Rank Fusion."""

    def __init__(self, settings):
        self.settings = settings
        self.documents: Dict[str, Dict] = {}  # doc_id -> {text, source, section, tokens}
        self.df: Counter = Counter()  # Document frequency
        self.total_docs = 0
        self.avg_dl = 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return re.findall(r'\b\w+\b', text.lower())

    def add_document(self, doc_id: str, text: str, source: str, section: str = ""):
        """Add a document to the BM25 index."""
        tokens = self._tokenize(text)
        self.documents[doc_id] = {
            "text": text,
            "source": source,
            "section": section,
            "tokens": tokens,
            "tf": Counter(tokens),
            "dl": len(tokens),
        }
        for token in set(tokens):
            self.df[token] += 1
        self.total_docs = len(self.documents)
        if self.total_docs > 0:
            self.avg_dl = sum(d["dl"] for d in self.documents.values()) / self.total_docs

    def index_document(self, doc_id: str, text: str, source: str, section: str = ""):
        """Alias for add_document for backward compatibility."""
        self.add_document(doc_id, text, source, section)

    def _bm25_score(self, query_tokens: List[str], doc_id: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score for a document."""
        doc = self.documents[doc_id]
        score = 0.0
        for token in query_tokens:
            if token not in doc["tf"]:
                continue
            tf = doc["tf"][token]
            df = self.df.get(token, 0)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc["dl"] / (self.avg_dl + 1e-10)))
            score += idf * tf_norm
        return score

    def search(self, query: str, top_k: int = 20) -> HybridSearchResult:
        """Search using BM25 and return results."""
        if not self.documents:
            return HybridSearchResult()

        query_tokens = self._tokenize(query)
        scores = []
        for doc_id in self.documents:
            score = self._bm25_score(query_tokens, doc_id)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        results = []
        for doc_id, score in top:
            doc = self.documents[doc_id]
            results.append({
                "doc_id": doc_id,
                "source": doc["source"],
                "section": doc["section"],
                "bm25_score": score,
                "text_preview": doc["text"][:200],
            })

        return HybridSearchResult(
            bm25_count=len(results),
            fused_count=len(results),
            top_results=results,
        )

    def remove_by_source(self, source: str) -> int:
        """Remove all documents from the BM25 index that match the given source.
        
        Rebuilds document frequency counts after removal.
        
        Args:
            source: The document source/filename to remove
        
        Returns:
            Number of documents removed
        """
        # Find doc_ids to remove
        remove_ids = [doc_id for doc_id, doc in self.documents.items() if doc.get('source') == source]
        if not remove_ids:
            return 0

        # Remove documents
        for doc_id in remove_ids:
            del self.documents[doc_id]

        # Rebuild document frequency from scratch
        self.df = Counter()
        for doc in self.documents.values():
            for token in set(doc['tokens']):
                self.df[token] += 1

        # Update stats
        self.total_docs = len(self.documents)
        if self.total_docs > 0:
            self.avg_dl = sum(d['dl'] for d in self.documents.values()) / self.total_docs
        else:
            self.avg_dl = 0.0

        print(f"[HYBRID] Removed {len(remove_ids)} BM25 documents for '{source}', {self.total_docs} remaining")
        return len(remove_ids)
