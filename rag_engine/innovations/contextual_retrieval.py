"""
Contextual Retrieval Engine — Anthropic's context-prefix technique.
Adds document-level context to each chunk before indexing.
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from openai import OpenAI


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    context_prefix: str
    source: str
    section: str = ""


@dataclass
class EnrichmentResult:
    enriched_chunks: List[EnrichedChunk] = field(default_factory=list)
    document_summary: str = ""


class ContextualRetrievalEngine:
    """Adds contextual prefixes to chunks for improved retrieval."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings
        self._stats = {"documents_enriched": 0, "chunks_enriched": 0}

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def _generate_document_summary(self, text: str) -> str:
        """Generate a brief document summary."""
        prompt = f"""Summarize this banking document in 2-3 sentences:

{text[:2000]}

Respond with ONLY the summary text."""
        return self._call_llm(prompt)

    def enrich_chunks(self, text: str, source: str) -> EnrichmentResult:
        """Enrich document chunks with contextual prefixes."""
        summary = self._generate_document_summary(text)

        # Simple chunking for enrichment
        sentences = text.split(". ")
        chunks = []
        current = []
        for s in sentences:
            current.append(s)
            if len(". ".join(current)) > 500:
                chunk_text = ". ".join(current)
                prefix = f"[From {source}] {summary[:100]}... "
                chunks.append(EnrichedChunk(
                    original_text=chunk_text,
                    enriched_text=prefix + chunk_text,
                    context_prefix=prefix,
                    source=source,
                ))
                current = []

        if current:
            chunk_text = ". ".join(current)
            prefix = f"[From {source}] {summary[:100]}... "
            chunks.append(EnrichedChunk(
                original_text=chunk_text,
                enriched_text=prefix + chunk_text,
                context_prefix=prefix,
                source=source,
            ))

        self._stats["documents_enriched"] += 1
        self._stats["chunks_enriched"] += len(chunks)

        return EnrichmentResult(enriched_chunks=chunks, document_summary=summary)

    def get_stats(self) -> Dict:
        return self._stats
