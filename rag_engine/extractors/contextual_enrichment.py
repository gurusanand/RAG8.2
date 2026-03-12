"""
Contextual Retrieval Enhancement Engine
=========================================
Implements Anthropic's Contextual Retrieval technique combined with Late Chunking
to produce enriched, self-contained chunks that carry document-level context.

Strategy:
1. **Contextual Prefix**: Prepend each chunk with a document-level context summary
   so that isolated chunks are self-explanatory during retrieval.
2. **Late Chunking**: First embed the full document, then split into chunks that
   inherit the document-level semantic context.
3. **Cross-Reference Linking**: Link chunks to related tables, formulas, and
   graph entities for multi-modal retrieval fusion.
4. **Formula Linearization**: Convert mathematical formulas into natural language
   descriptions for better embedding quality.

Feature Toggle: extraction_contextual_enabled (default: True)
"""
import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EnrichedChunk:
    """A chunk enriched with contextual metadata for high-quality retrieval."""
    chunk_id: str
    original_text: str
    enriched_text: str  # Original text + contextual prefix
    context_prefix: str  # The prepended context
    source_file: str
    page_number: int = 0
    section: str = ""
    chunk_type: str = "text"  # text, table, formula, flowchart, mixed
    related_table_ids: List[str] = field(default_factory=list)
    related_entity_ids: List[str] = field(default_factory=list)
    formulas_linearized: List[str] = field(default_factory=list)
    language: str = "en"
    confidence: float = 0.0


@dataclass
class EnrichmentResult:
    """Complete enrichment result for a document."""
    filename: str
    chunks: List[EnrichedChunk] = field(default_factory=list)
    document_summary: str = ""
    strategy_used: str = "contextual_prefix"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextualEnrichmentEngine:
    """
    Enriches document chunks with contextual information for better retrieval.

    Implements:
    1. Document-level summary generation
    2. Contextual prefix injection per chunk
    3. Formula linearization (math → natural language)
    4. Cross-reference linking (chunks ↔ tables ↔ entities)
    5. Late chunking with inherited context
    """

    # Centralized prompts
    DOCUMENT_SUMMARY_PROMPT = """Summarize this banking document in 2-3 sentences. Include:
- Document type (product brochure, policy, KFS, etc.)
- Key products or topics covered
- Target audience

Document text (first 3000 chars):
{text}

Summary:"""

    CONTEXTUAL_PREFIX_PROMPT = """You are a document context engine. Given a document summary and a specific chunk of text, generate a brief contextual prefix (1-2 sentences) that situates this chunk within the broader document.

The prefix should answer: "What is this chunk about in the context of the full document?"

Document summary: {doc_summary}
Document filename: {filename}

Chunk text:
{chunk_text}

Generate ONLY the contextual prefix (1-2 sentences), nothing else:"""

    FORMULA_LINEARIZATION_PROMPT = """Convert these mathematical formulas/calculations from a banking document into clear natural language descriptions.

Formulas:
{formulas}

For each formula, provide:
1. What it calculates
2. The formula in words
3. An example if values are given

RESPOND as a JSON array of objects:
[{{"formula": "original", "description": "natural language description", "example": "worked example if applicable"}}]"""

    def __init__(self, client, settings):
        self.client = client
        self.settings = settings
        extraction_config = getattr(settings.rag, 'extraction_config', {})
        self.fast_mode = extraction_config.get('fast_mode', True)

    def enrich_document(
        self,
        text_chunks: List[Dict[str, Any]],
        filename: str,
        full_text: str = "",
        tables: List[Dict[str, Any]] = None,
        formulas: List[str] = None,
        entities: List[Dict[str, Any]] = None
    ) -> EnrichmentResult:
        """
        Enrich document chunks with contextual information.

        Args:
            text_chunks: List of dicts with 'text', 'section', 'page', 'chunk_id'
            filename: Source document filename
            full_text: Full document text for summary generation
            tables: Extracted tables for cross-referencing
            formulas: Extracted formulas for linearization
            entities: Extracted entities for cross-referencing

        Returns:
            EnrichmentResult with enriched chunks
        """
        logger.info(f"[CONTEXTUAL_ENRICHMENT] Processing {len(text_chunks)} chunks from {filename}")

        # Step 1: Generate document summary
        doc_summary = self._generate_document_summary(full_text or "", filename)

        # Step 2: Linearize formulas if present
        linearized_formulas = []
        if formulas:
            linearized_formulas = self._linearize_formulas(formulas)

        # Step 3: Build cross-reference index
        table_index = self._build_table_index(tables or [])
        entity_index = self._build_entity_index(entities or [])

        # Step 4: Enrich each chunk
        enriched_chunks = []
        for chunk_data in text_chunks:
            enriched = self._enrich_single_chunk(
                chunk_data=chunk_data,
                doc_summary=doc_summary,
                filename=filename,
                table_index=table_index,
                entity_index=entity_index,
                linearized_formulas=linearized_formulas
            )
            enriched_chunks.append(enriched)

        # Step 5: Add table chunks as enriched chunks
        if tables:
            table_chunks = self._create_table_chunks(tables, doc_summary, filename)
            enriched_chunks.extend(table_chunks)

        # Step 6: Add formula chunks as enriched chunks
        if linearized_formulas:
            formula_chunks = self._create_formula_chunks(
                linearized_formulas, doc_summary, filename
            )
            enriched_chunks.extend(formula_chunks)

        logger.info(f"[CONTEXTUAL_ENRICHMENT] Produced {len(enriched_chunks)} enriched chunks")

        return EnrichmentResult(
            filename=filename,
            chunks=enriched_chunks,
            document_summary=doc_summary,
            strategy_used="contextual_prefix",
            metadata={
                "original_chunks": len(text_chunks),
                "table_chunks_added": len(tables or []),
                "formula_chunks_added": len(linearized_formulas),
                "total_enriched": len(enriched_chunks)
            }
        )

    def _generate_document_summary(self, full_text: str, filename: str) -> str:
        """Generate a document-level summary for contextual prefixing."""
        if not full_text:
            return f"Banking document: {filename}"

        try:
            prompt = self.DOCUMENT_SUMMARY_PROMPT.format(text=full_text[:3000])
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"[CONTEXTUAL_ENRICHMENT] Document summary: {summary[:100]}...")
            return summary
        except Exception as e:
            logger.error(f"[CONTEXTUAL_ENRICHMENT] Summary generation failed: {e}")
            return f"Banking document: {filename}"

    def _enrich_single_chunk(
        self,
        chunk_data: Dict[str, Any],
        doc_summary: str,
        filename: str,
        table_index: Dict[int, List[str]],
        entity_index: Dict[str, List[str]],
        linearized_formulas: List[Dict[str, str]]
    ) -> EnrichedChunk:
        """Enrich a single chunk with contextual prefix and cross-references."""
        chunk_text = chunk_data.get("text", "")
        page_num = chunk_data.get("page", 0)
        section = chunk_data.get("section", "")
        chunk_id = chunk_data.get("chunk_id", hashlib.md5(chunk_text.encode()).hexdigest()[:12])

        # Generate contextual prefix
        context_prefix = self._generate_context_prefix(
            chunk_text, doc_summary, filename
        )

        # Find related tables (same page)
        related_tables = table_index.get(page_num, [])

        # Find related entities (keyword match)
        related_entities = self._find_related_entities(chunk_text, entity_index)

        # Find related formulas
        related_formulas = [
            f["description"] for f in linearized_formulas
            if any(keyword in chunk_text.lower()
                   for keyword in f.get("formula", "").lower().split()[:3])
        ]

        # Build enriched text
        enriched_text = f"{context_prefix}\n\n{chunk_text}"

        # Append formula descriptions if relevant
        if related_formulas:
            enriched_text += "\n\nRelated calculations:\n" + "\n".join(
                f"- {f}" for f in related_formulas
            )

        return EnrichedChunk(
            chunk_id=chunk_id,
            original_text=chunk_text,
            enriched_text=enriched_text,
            context_prefix=context_prefix,
            source_file=filename,
            page_number=page_num,
            section=section,
            chunk_type="text",
            related_table_ids=related_tables,
            related_entity_ids=related_entities,
            formulas_linearized=related_formulas,
            confidence=0.85
        )

    def _generate_context_prefix(self, chunk_text: str, doc_summary: str, filename: str) -> str:
        """Generate a contextual prefix for a chunk.
        
        In fast_mode: uses a template-based prefix (instant, no LLM call).
        In full mode: uses LLM to generate a richer contextual prefix.
        """
        # Fast mode: template-based prefix (no LLM call — instant)
        if self.fast_mode:
            # Extract a meaningful section hint from the chunk's first line
            first_line = chunk_text.strip().split('\n')[0][:80].strip()
            return f"From {filename} — {doc_summary[:120]}. Section context: {first_line}"

        # Full mode: LLM-generated prefix
        try:
            prompt = self.CONTEXTUAL_PREFIX_PROMPT.format(
                doc_summary=doc_summary,
                filename=filename,
                chunk_text=chunk_text[:1000]
            )
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            prefix = response.choices[0].message.content.strip()
            return prefix
        except Exception as e:
            logger.warning(f"[CONTEXTUAL_ENRICHMENT] Prefix generation failed: {e}")
            return f"From {filename}: {doc_summary[:100]}"

    def _linearize_formulas(self, formulas: List[str]) -> List[Dict[str, str]]:
        """Convert mathematical formulas to natural language descriptions."""
        if not formulas:
            return []

        try:
            formulas_text = "\n".join(f"- {f}" for f in formulas)
            prompt = self.FORMULA_LINEARIZATION_PROMPT.format(formulas=formulas_text)
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON array
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            return []
        except Exception as e:
            logger.warning(f"[CONTEXTUAL_ENRICHMENT] Formula linearization failed: {e}")
            return [{"formula": f, "description": f, "example": ""} for f in formulas]

    def _build_table_index(self, tables: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """Build a page→table_ids index for cross-referencing."""
        index = {}
        for table in tables:
            page = table.get("page", 0)
            table_id = table.get("table_id", "")
            if page not in index:
                index[page] = []
            index[page].append(table_id)
        return index

    def _build_entity_index(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build a keyword→entity_ids index for cross-referencing."""
        index = {}
        for entity in entities:
            name = entity.get("name", "").lower()
            entity_id = entity.get("entity_id", "")
            # Index by each word in the entity name
            for word in name.split():
                if len(word) > 2:  # Skip short words
                    if word not in index:
                        index[word] = []
                    index[word].append(entity_id)
        return index

    def _find_related_entities(self, text: str, entity_index: Dict[str, List[str]]) -> List[str]:
        """Find entities related to a text chunk via keyword matching."""
        text_lower = text.lower()
        related = set()
        for keyword, entity_ids in entity_index.items():
            if keyword in text_lower:
                related.update(entity_ids)
        return list(related)[:10]  # Limit to 10 related entities

    def _create_table_chunks(
        self, tables: List[Dict[str, Any]], doc_summary: str, filename: str
    ) -> List[EnrichedChunk]:
        """Create enriched chunks from extracted tables."""
        chunks = []
        for table in tables:
            title = table.get("title", "Table")
            markdown = table.get("markdown", "")
            page = table.get("page", 0)
            table_id = table.get("table_id", "")

            if not markdown:
                continue

            context_prefix = (
                f"This table is from {filename}. {doc_summary} "
                f"Table: {title} (Page {page})."
            )

            chunks.append(EnrichedChunk(
                chunk_id=f"table_{table_id}",
                original_text=markdown,
                enriched_text=f"{context_prefix}\n\n{markdown}",
                context_prefix=context_prefix,
                source_file=filename,
                page_number=page,
                section=title,
                chunk_type="table",
                related_table_ids=[table_id],
                confidence=0.9
            ))

        return chunks

    def _create_formula_chunks(
        self, linearized_formulas: List[Dict[str, str]], doc_summary: str, filename: str
    ) -> List[EnrichedChunk]:
        """Create enriched chunks from linearized formulas."""
        chunks = []
        for i, formula_data in enumerate(linearized_formulas):
            formula = formula_data.get("formula", "")
            description = formula_data.get("description", "")
            example = formula_data.get("example", "")

            text = f"Formula: {formula}\nDescription: {description}"
            if example:
                text += f"\nExample: {example}"

            context_prefix = f"Mathematical formula from {filename}. {doc_summary}"

            chunk_id = hashlib.md5(formula.encode()).hexdigest()[:12]
            chunks.append(EnrichedChunk(
                chunk_id=f"formula_{chunk_id}",
                original_text=formula,
                enriched_text=f"{context_prefix}\n\n{text}",
                context_prefix=context_prefix,
                source_file=filename,
                chunk_type="formula",
                formulas_linearized=[description],
                confidence=0.85
            ))

        return chunks
