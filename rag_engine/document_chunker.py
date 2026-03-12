"""
Universal Document Chunker — Semantic-aware chunking for banking documents.
Supports TXT, PDF, and MD with section detection and overlap.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChunkResult:
    """A single chunk result from the chunker."""
    text: str
    section: str = ""
    chunk_id: str = ""
    start_char: int = 0
    end_char: int = 0


class UniversalDocumentChunker:
    """Chunks documents with semantic awareness, section detection, and configurable overlap."""

    def __init__(self, chunk_size_min: int = 200, chunk_size_max: int = 800, chunk_overlap: int = 50):
        self.min_size = chunk_size_min
        self.max_size = chunk_size_max
        self.overlap = chunk_overlap

    def _detect_sections(self, text: str) -> List[tuple]:
        """Detect sections in the text based on common heading patterns."""
        sections = []
        current_section = "General"
        current_text = []

        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            is_heading = False
            if stripped and len(stripped) < 100:
                if stripped.isupper() and len(stripped) > 3:
                    is_heading = True
                elif re.match(r'^\d+[\.\)]\s+\S', stripped):
                    is_heading = True
                elif re.match(r'^#{1,3}\s+\S', stripped):
                    is_heading = True
                elif re.match(r'^[A-Z][A-Za-z\s]+:$', stripped):
                    is_heading = True

            if is_heading:
                if current_text:
                    sections.append((current_section, "\n".join(current_text)))
                current_section = stripped.strip("#").strip(":").strip()
                current_text = []
            else:
                current_text.append(line)

        if current_text:
            sections.append((current_section, "\n".join(current_text)))

        return sections if sections else [("General", text)]

    def chunk_document(self, text: str, source: str = "unknown") -> List[ChunkResult]:
        """Chunk text into overlapping segments with section awareness.
        
        Args:
            text: The full document text.
            source: The source filename.
            
        Returns:
            List of ChunkResult objects.
        """
        if not text or not text.strip():
            return []

        sections = self._detect_sections(text)
        chunks = []
        chunk_index = 0

        for section_name, section_text in sections:
            sentences = re.split(r'(?<=[.!?])\s+', section_text)
            current_chunk = []
            current_size = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_len = len(sentence)

                if current_size + sentence_len > self.max_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) >= self.min_size:
                        chunks.append(ChunkResult(
                            text=chunk_text,
                            section=section_name,
                            chunk_id=f"{source}_{chunk_index}",
                        ))
                        chunk_index += 1

                    # Keep overlap
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= self.overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_size = overlap_size

                current_chunk.append(sentence)
                current_size += sentence_len

            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_size // 2:
                    chunks.append(ChunkResult(
                        text=chunk_text,
                        section=section_name,
                        chunk_id=f"{source}_{chunk_index}",
                    ))
                    chunk_index += 1

        return chunks
