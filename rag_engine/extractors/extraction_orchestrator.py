"""
Multi-Strategy Extraction Orchestrator
========================================
Unifies all extraction engines (Vision, Table, Knowledge Graph, Contextual Enrichment)
into a single pipeline that processes any document through the optimal combination
of strategies based on document characteristics.

Architecture:
    PDF Upload → Document Analysis → Strategy Selection → Parallel Extraction →
    Fusion & Dedup → Enriched Chunks → Index into RAG Engine

Feature Toggle: extraction_orchestrator_enabled (default: True)
All sub-strategies are individually toggleable.

Human-in-the-Loop: If all extraction strategies fail for a document,
the orchestrator emits a critical failure event and flags the document
for human review.
"""
import os
import io
import json
import time
import hashlib
import logging
import tempfile
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import extraction engines
try:
    from rag_engine.extractors.vision_extractor import VisionExtractor, VisionExtractionResult
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("[ORCHESTRATOR] VisionExtractor not available")

try:
    from rag_engine.extractors.table_extractor import TableExtractor, TableExtractionResult
    TABLE_AVAILABLE = True
except ImportError:
    TABLE_AVAILABLE = False
    logger.warning("[ORCHESTRATOR] TableExtractor not available")

try:
    from rag_engine.extractors.knowledge_graph_builder import KnowledgeGraphBuilder
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logger.warning("[ORCHESTRATOR] KnowledgeGraphBuilder not available")

try:
    from rag_engine.extractors.contextual_enrichment import ContextualEnrichmentEngine, EnrichmentResult
    ENRICHMENT_AVAILABLE = True
except ImportError:
    ENRICHMENT_AVAILABLE = False
    logger.warning("[ORCHESTRATOR] ContextualEnrichmentEngine not available")


@dataclass
class DocumentProfile:
    """Profile of a document's characteristics for strategy selection."""
    filename: str
    file_type: str  # pdf, txt, md, docx
    file_size_bytes: int
    page_count: int = 0
    has_tables: bool = False
    has_images: bool = False
    has_formulas: bool = False
    has_flowcharts: bool = False
    is_bilingual: bool = False
    is_scanned: bool = False  # Scanned PDF (image-only, no text layer)
    complexity_score: float = 0.0  # 0.0 (simple text) to 1.0 (complex multi-modal)


@dataclass
class ExtractionStrategy:
    """A selected extraction strategy with its configuration."""
    name: str
    enabled: bool
    priority: int  # Lower = higher priority
    reason: str


@dataclass
class UnifiedExtractionResult:
    """Complete extraction result from the orchestrator."""
    filename: str
    profile: DocumentProfile
    strategies_used: List[str]
    enriched_chunks: List[Dict[str, Any]]  # Ready for indexing
    tables_extracted: List[Dict[str, Any]]
    formulas_extracted: List[Dict[str, Any]]
    flowcharts_extracted: List[Dict[str, Any]]
    graph_entities: int
    graph_relationships: int
    document_summary: str
    total_chunks: int
    processing_time_ms: float
    events: List[Dict[str, Any]] = field(default_factory=list)  # Event log
    human_review_required: bool = False
    human_review_reason: str = ""


class ExtractionOrchestrator:
    """
    Unified extraction orchestrator that coordinates all extraction strategies.

    Workflow:
    1. Analyze document → build DocumentProfile
    2. Select optimal strategies based on profile + config toggles
    3. Execute strategies (vision, table, graph, enrichment)
    4. Fuse results → deduplicate → produce enriched chunks
    5. Emit events for monitoring and human-in-the-loop fallback
    """

    def __init__(self, client, settings):
        self.client = client
        self.settings = settings
        self.events: List[Dict[str, Any]] = []

        # Initialize sub-engines based on config toggles
        self.vision_extractor = None
        self.table_extractor = None
        self.graph_builder = None
        self.enrichment_engine = None

        extraction_config = getattr(settings.rag, 'extraction_config', {})

        if VISION_AVAILABLE and extraction_config.get('vision_enabled', True):
            self.vision_extractor = VisionExtractor(client, settings)

        if TABLE_AVAILABLE and extraction_config.get('table_enabled', True):
            self.table_extractor = TableExtractor(settings)

        if GRAPH_AVAILABLE and extraction_config.get('graph_enabled', True):
            self.graph_builder = KnowledgeGraphBuilder(client, settings)

        if ENRICHMENT_AVAILABLE and extraction_config.get('contextual_enabled', True):
            self.enrichment_engine = ContextualEnrichmentEngine(client, settings)

    def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        file_type: str = None,
        progress_callback=None
    ) -> UnifiedExtractionResult:
        """
        Process a document through the full extraction pipeline.

        Args:
            file_bytes: Raw file content
            filename: Document filename
            file_type: File type override (auto-detected if None)
            progress_callback: Optional callable(stage_name, progress_pct) for UI updates

        Returns:
            UnifiedExtractionResult with all extracted and enriched content
        """
        start_time = time.time()
        self.events = []
        self._progress = progress_callback or (lambda stage, pct: None)

        self._emit_event("extraction_started", {"filename": filename})

        # Step 1: Determine file type
        if file_type is None:
            file_type = self._detect_file_type(filename)

        # Step 2: Save to temp file for processing
        temp_path = self._save_temp_file(file_bytes, filename)

        try:
            # Step 3: Analyze document profile
            profile = self._analyze_document(temp_path, filename, file_type, len(file_bytes))
            self._emit_event("profile_analyzed", {
                "complexity": profile.complexity_score,
                "has_tables": profile.has_tables,
                "has_images": profile.has_images,
                "page_count": profile.page_count
            })

            self._progress("Analyzing document...", 0.1)

            # Step 4: Select strategies (respect fast_mode)
            strategies = self._select_strategies(profile)
            strategy_names = [s.name for s in strategies if s.enabled]
            self._emit_event("strategies_selected", {"strategies": strategy_names})
            self._progress(f"Strategies: {', '.join(strategy_names)}", 0.15)

            # Step 5: Execute extraction strategies
            vision_result = None
            table_result = None
            text_chunks = []
            all_tables = []
            all_formulas = []
            all_flowcharts = []
            full_text = ""

            # Strategy A: Vision Extraction (for PDFs)
            if "vision" in strategy_names and self.vision_extractor and file_type == "pdf":
                self._progress("Vision extraction (LLM page analysis)...", 0.2)
                try:
                    max_pages = getattr(self.settings.rag, 'extraction_config', {}).get('vision_max_pages', 3)
                    vision_result = self.vision_extractor.extract(temp_path, filename, max_pages=max_pages)
                    full_text = vision_result.get_full_text()
                    all_tables.extend(vision_result.get_all_tables())
                    all_formulas.extend(vision_result.get_all_formulas())
                    all_flowcharts.extend(vision_result.get_all_flowcharts())

                    # Create text chunks from vision results
                    for pr in vision_result.page_results:
                        if pr.extracted_text:
                            text_chunks.append({
                                "text": pr.extracted_text,
                                "page": pr.page_number,
                                "section": f"Page {pr.page_number}",
                                "chunk_id": hashlib.md5(
                                    pr.extracted_text.encode()
                                ).hexdigest()[:12]
                            })

                    self._emit_event("vision_completed", {
                        "pages": vision_result.total_pages,
                        "tables": len(all_tables),
                        "formulas": len(all_formulas)
                    })
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Vision extraction failed: {e}")
                    self._emit_event("vision_failed", {"error": str(e)})

            # Strategy B: Table Extraction (for PDFs)
            if "table" in strategy_names and self.table_extractor and file_type == "pdf":
                self._progress("Table extraction (pdfplumber)...", 0.45)
                try:
                    table_result = self.table_extractor.extract(temp_path, filename)

                    # Merge tables (dedup by content hash)
                    existing_hashes = {
                        hashlib.md5(json.dumps(t, sort_keys=True).encode()).hexdigest()
                        for t in all_tables
                    }
                    for table in table_result.tables:
                        table_dict = {
                            "table_id": table.table_id,
                            "title": table.title,
                            "headers": table.headers,
                            "rows": table.rows,
                            "markdown": table.markdown,
                            "page": table.page_number,
                            "source_method": table.source_method,
                            "confidence": table.confidence
                        }
                        table_hash = hashlib.md5(
                            json.dumps(table_dict, sort_keys=True).encode()
                        ).hexdigest()
                        if table_hash not in existing_hashes:
                            all_tables.append(table_dict)
                            existing_hashes.add(table_hash)

                    # Use table extractor's full text if vision didn't produce any
                    if not full_text and table_result.full_text:
                        full_text = table_result.full_text

                    self._emit_event("table_completed", {
                        "tables_found": len(table_result.tables),
                        "backend": table_result.strategy_used
                    })
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Table extraction failed: {e}")
                    self._emit_event("table_failed", {"error": str(e)})

            # Targeted Vision: Even in Fast Mode, use Vision API for specific pages
            # that contain product comparison tables with misaligned coordinate data.
            # This is a single API call (~10s) instead of full vision extraction (~10min).
            if (
                file_type == "pdf" and
                self.vision_extractor is not None and
                VISION_AVAILABLE
            ):
                try:
                    self._progress("Targeted vision analysis for product tables...", 0.45)
                    # Check if any table has multi-product cells (misaligned data)
                    import re
                    product_regex = re.compile(
                        r'(?:Platinum\s*(?:Elite|Plus))|(?:Solitaire)|(?:Cashback)|(?:Gold)',
                        re.IGNORECASE
                    )
                    needs_vision_pages = set()
                    for table in all_tables:
                        for row in table.get('rows', []):
                            for val in row.values():
                                products = product_regex.findall(str(val))
                                if len(products) >= 2:
                                    needs_vision_pages.add(table.get('page', 1))

                    if needs_vision_pages:
                        logger.info(f"[ORCHESTRATOR] Targeted vision for pages: {needs_vision_pages}")
                        # Use vision to extract just these specific pages
                        import fitz
                        doc = fitz.open(temp_path)
                        for page_num in needs_vision_pages:
                            if page_num <= len(doc):
                                page = doc[page_num - 1]
                                pix = page.get_pixmap(dpi=200)
                                img_path = os.path.join(
                                    self.settings.paths.base_dir, "data", "temp",
                                    f"vision_page_{page_num}.png"
                                )
                                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                                pix.save(img_path)

                                # Call vision extractor for just this page
                                vision_result = self.vision_extractor._extract_page_with_vision(
                                    img_path, page_num
                                )
                                if vision_result and vision_result.tables:
                                    for vt in vision_result.tables:
                                        vt_dict = {
                                            "table_id": f"vision_p{page_num}_{vt.get('title', 'table')[:20]}",
                                            "title": vt.get("title", f"Vision Table Page {page_num}"),
                                            "headers": vt.get("headers", []),
                                            "rows": vt.get("rows", []),
                                            "markdown": "",
                                            "page": page_num,
                                            "source_method": "targeted_vision",
                                            "confidence": 0.95
                                        }
                                        # Build markdown
                                        if vt_dict["headers"] and vt_dict["rows"]:
                                            md = "| " + " | ".join(vt_dict["headers"]) + " |\n"
                                            md += "| " + " | ".join(["---"] * len(vt_dict["headers"])) + " |\n"
                                            for r in vt_dict["rows"]:
                                                cells = [str(r.get(h, "")) for h in vt_dict["headers"]]
                                                md += "| " + " | ".join(cells) + " |\n"
                                            vt_dict["markdown"] = md
                                        all_tables.append(vt_dict)

                                    logger.info(f"[ORCHESTRATOR] Vision found {len(vision_result.tables)} tables on page {page_num}")
                                    self._emit_event("targeted_vision_completed", {
                                        "page": page_num,
                                        "tables_found": len(vision_result.tables)
                                    })

                                # Clean up temp image
                                try:
                                    os.remove(img_path)
                                except Exception:
                                    pass
                        doc.close()
                except Exception as e:
                    logger.warning(f"[ORCHESTRATOR] Targeted vision failed (non-critical): {e}")

            # PyMuPDF Text Extraction Fallback: For PDFs when Vision is disabled (Fast Mode)
            # This ensures text_chunks are always populated for PDF documents
            if file_type == "pdf" and not text_chunks:
                self._progress("Extracting text (PyMuPDF)...", 0.5)
                try:
                    import fitz
                    doc = fitz.open(temp_path)
                    page_texts = []
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        page_text = page.get_text("text")
                        if page_text and page_text.strip():
                            page_texts.append((page_num + 1, page_text.strip()))
                    doc.close()

                    # Build full_text if not already set
                    if not full_text:
                        full_text = "\n\n".join([t for _, t in page_texts])

                    # Create text chunks per page
                    for page_num, page_text in page_texts:
                        # Split long pages into paragraph-level chunks
                        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                        for i, para in enumerate(paragraphs):
                            if len(para) >= 30:  # Lower threshold for FAQ-style short answers
                                text_chunks.append({
                                    "text": para,
                                    "page": page_num,
                                    "section": f"Page {page_num}, Para {i + 1}",
                                    "chunk_id": hashlib.md5(
                                        f"{filename}_p{page_num}_para{i}_{para[:50]}".encode()
                                    ).hexdigest()[:12]
                                })

                    logger.info(f"[ORCHESTRATOR] PyMuPDF text extraction: {len(text_chunks)} chunks from {len(page_texts)} pages")
                    self._emit_event("pymupdf_text_extracted", {
                        "chunks": len(text_chunks),
                        "pages": len(page_texts),
                        "full_text_length": len(full_text)
                    })
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] PyMuPDF text extraction failed: {e}")
                    self._emit_event("pymupdf_failed", {"error": str(e)})

            # ═══════════════════════════════════════════════════════════════
            # SCANNED PDF FALLBACK: Vision API extraction for image-only PDFs
            # Even in Fast Mode, scanned PDFs MUST use Vision API because
            # there is no text layer to extract. This is the only way to
            # get content from image-based documents.
            # ═══════════════════════════════════════════════════════════════
            if file_type == "pdf" and not text_chunks and not full_text:
                if self.vision_extractor and VISION_AVAILABLE:
                    self._progress("Scanned PDF detected — using Vision API...", 0.5)
                    logger.info(f"[ORCHESTRATOR] Scanned PDF detected for {filename} — activating Vision API fallback")
                    try:
                        # Use Vision API for ALL pages of scanned PDFs
                        vision_result = self.vision_extractor.extract(temp_path, filename)
                        full_text = vision_result.get_full_text()
                        all_tables.extend(vision_result.get_all_tables())
                        all_formulas.extend(vision_result.get_all_formulas())
                        all_flowcharts.extend(vision_result.get_all_flowcharts())

                        # Create text chunks from vision results
                        for pr in vision_result.page_results:
                            if pr.extracted_text:
                                text_chunks.append({
                                    "text": pr.extracted_text,
                                    "page": pr.page_number,
                                    "section": f"Page {pr.page_number}",
                                    "chunk_id": hashlib.md5(
                                        pr.extracted_text.encode()
                                    ).hexdigest()[:12]
                                })

                        self._emit_event("scanned_pdf_vision_fallback", {
                            "pages": vision_result.total_pages,
                            "chunks": len(text_chunks),
                            "tables": len(all_tables),
                            "full_text_length": len(full_text)
                        })
                        logger.info(f"[ORCHESTRATOR] Scanned PDF Vision fallback: {len(text_chunks)} chunks from {vision_result.total_pages} pages")
                    except Exception as e:
                        logger.error(f"[ORCHESTRATOR] Scanned PDF Vision fallback failed: {e}")
                        self._emit_event("scanned_pdf_vision_failed", {"error": str(e)})
                else:
                    logger.warning(f"[ORCHESTRATOR] Scanned PDF detected but Vision extractor not available")
                    self._emit_event("scanned_pdf_no_vision", {
                        "reason": "Vision extractor not initialized"
                    })

            # For text files: read directly
            if file_type in ("txt", "md"):
                try:
                    full_text = file_bytes.decode("utf-8", errors="replace")
                    # Split into paragraph chunks
                    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
                    for i, para in enumerate(paragraphs):
                        if len(para) >= 50:
                            text_chunks.append({
                                "text": para,
                                "page": 0,
                                "section": f"Paragraph {i + 1}",
                                "chunk_id": hashlib.md5(para.encode()).hexdigest()[:12]
                            })
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Text file reading failed: {e}")

            # Strategy C: Knowledge Graph Construction
            if "graph" in strategy_names and self.graph_builder:
                self._progress("Knowledge graph construction...", 0.6)
                try:
                    # Build graph from text chunks
                    total_entities = 0
                    max_graph_chunks = getattr(self.settings.rag, 'extraction_config', {}).get('graph_max_chunks', 5)
                    for chunk in text_chunks[:max_graph_chunks]:  # Limit to avoid excessive LLM calls
                        count = self.graph_builder.build_from_text(
                            chunk["text"], filename, chunk.get("page", 0)
                        )
                        total_entities += count

                    # Build graph from tables
                    if all_tables:
                        table_entities = self.graph_builder.build_from_tables(
                            all_tables, filename
                        )
                        total_entities += table_entities

                    self._emit_event("graph_completed", {
                        "entities": total_entities,
                        "relationships": len(self.graph_builder.relationships)
                    })
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Graph construction failed: {e}")
                    self._emit_event("graph_failed", {"error": str(e)})

            # Strategy D: Contextual Enrichment
            enriched_chunks = []
            document_summary = ""

            if "enrichment" in strategy_names and self.enrichment_engine:
                self._progress("Contextual enrichment...", 0.75)
                try:
                    # Prepare entity data for cross-referencing
                    entity_data = []
                    if self.graph_builder:
                        entity_data = [
                            {"entity_id": e.entity_id, "name": e.name, "type": e.entity_type}
                            for e in self.graph_builder.entities.values()
                        ]

                    # Prepare formula strings
                    formula_strings = [f.get("formula", "") for f in all_formulas if f.get("formula")]

                    enrichment_result = self.enrichment_engine.enrich_document(
                        text_chunks=text_chunks,
                        filename=filename,
                        full_text=full_text,
                        tables=all_tables,
                        formulas=formula_strings,
                        entities=entity_data
                    )

                    document_summary = enrichment_result.document_summary

                    # Convert EnrichedChunks to dicts for indexing
                    for ec in enrichment_result.chunks:
                        enriched_chunks.append({
                            "chunk_id": ec.chunk_id,
                            "text": ec.enriched_text,
                            "original_text": ec.original_text,
                            "context_prefix": ec.context_prefix,
                            "source": filename,
                            "page": ec.page_number,
                            "section": ec.section,
                            "chunk_type": ec.chunk_type,
                            "related_tables": ec.related_table_ids,
                            "related_entities": ec.related_entity_ids,
                            "confidence": ec.confidence
                        })

                    self._emit_event("enrichment_completed", {
                        "enriched_chunks": len(enriched_chunks),
                        "summary_length": len(document_summary)
                    })
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] Contextual enrichment failed: {e}")
                    self._emit_event("enrichment_failed", {"error": str(e)})

            # Fallback: If enrichment didn't run, use raw text chunks
            if not enriched_chunks and text_chunks:
                for chunk in text_chunks:
                    enriched_chunks.append({
                        "chunk_id": chunk.get("chunk_id", ""),
                        "text": chunk["text"],
                        "original_text": chunk["text"],
                        "context_prefix": "",
                        "source": filename,
                        "page": chunk.get("page", 0),
                        "section": chunk.get("section", ""),
                        "chunk_type": "text",
                        "related_tables": [],
                        "related_entities": [],
                        "confidence": 0.7
                    })

            # Add table-specific chunks for direct table retrieval
            existing_ids = {c["chunk_id"] for c in enriched_chunks}
            for table in all_tables:
                # Chunk 1: Full table overview (markdown)
                table_chunk = {
                    "chunk_id": f"table_{table.get('table_id', '')}",
                    "text": f"Table: {table.get('title', 'Unknown')}\n{table.get('markdown', '')}",
                    "original_text": table.get("markdown", ""),
                    "context_prefix": f"Table from {filename}",
                    "source": filename,
                    "page": table.get("page", 0),
                    "section": table.get("title", "Table"),
                    "chunk_type": "table",
                    "related_tables": [table.get("table_id", "")],
                    "related_entities": [],
                    "confidence": table.get("confidence", 0.8)
                }
                if table_chunk["chunk_id"] not in existing_ids:
                    enriched_chunks.append(table_chunk)
                    existing_ids.add(table_chunk["chunk_id"])

                # Chunk 2+: Per-row chunks for precise retrieval
                # This ensures queries like "annual fee for PlatinumPlus" match directly
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                table_title = table.get("title", "Table")
                for row_idx, row in enumerate(rows):
                    row_text = f"From table '{table_title}' (Page {table.get('page', 0)}, Source: {filename}):\n"
                    for header in headers:
                        value = row.get(header, "")
                        if value and str(value).strip():
                            row_text += f"  {header}: {value}\n"
                    row_text = row_text.strip()
                    if len(row_text) > 50:  # Skip trivially small rows
                        row_chunk_id = f"table_row_{table.get('table_id', '')}_{row_idx}"
                        if row_chunk_id not in existing_ids:
                            enriched_chunks.append({
                                "chunk_id": row_chunk_id,
                                "text": row_text,
                                "original_text": row_text,
                                "context_prefix": f"Row {row_idx+1} from table '{table_title}'",
                                "source": filename,
                                "page": table.get("page", 0),
                                "section": table_title,
                                "chunk_type": "table_row",
                                "related_tables": [table.get("table_id", "")],
                                "related_entities": [],
                                "confidence": table.get("confidence", 0.8)
                            })
                            existing_ids.add(row_chunk_id)

            # Step 5b: Smart card/product fee chunk generation
            # Parse tables that contain product names + fee data and create per-product chunks
            enriched_chunks, existing_ids = self._generate_product_fee_chunks(
                all_tables, enriched_chunks, existing_ids, filename
            )

            # Step 6: Check for critical failure
            human_review = False
            human_review_reason = ""
            if not enriched_chunks:
                human_review = True
                human_review_reason = (
                    f"All extraction strategies failed for {filename}. "
                    "No content could be extracted. Document requires human review."
                )
                self._emit_event("critical_failure", {
                    "filename": filename,
                    "reason": human_review_reason
                })

            self._progress("Finalizing...", 0.95)
            processing_time = (time.time() - start_time) * 1000

            self._emit_event("extraction_completed", {
                "total_chunks": len(enriched_chunks),
                "processing_time_ms": processing_time,
                "human_review": human_review
            })

            return UnifiedExtractionResult(
                filename=filename,
                profile=profile,
                strategies_used=strategy_names,
                enriched_chunks=enriched_chunks,
                tables_extracted=all_tables,
                formulas_extracted=all_formulas,
                flowcharts_extracted=all_flowcharts,
                graph_entities=len(self.graph_builder.entities) if self.graph_builder else 0,
                graph_relationships=len(self.graph_builder.relationships) if self.graph_builder else 0,
                document_summary=document_summary,
                total_chunks=len(enriched_chunks),
                processing_time_ms=processing_time,
                events=self.events,
                human_review_required=human_review,
                human_review_reason=human_review_reason
            )

        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    def get_graph_builder(self) -> Optional['KnowledgeGraphBuilder']:
        """Get the knowledge graph builder for query-time graph traversal."""
        return self.graph_builder

    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from filename extension."""
        ext = os.path.splitext(filename)[1].lower()
        type_map = {
            ".pdf": "pdf",
            ".txt": "txt",
            ".md": "md",
            ".docx": "docx",
            ".doc": "doc",
        }
        return type_map.get(ext, "txt")

    def _save_temp_file(self, file_bytes: bytes, filename: str) -> str:
        """Save file bytes to a temporary file for processing."""
        ext = os.path.splitext(filename)[1]
        temp_dir = os.path.join(self.settings.paths.base_dir, "data", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"extract_{hashlib.md5(filename.encode()).hexdigest()[:8]}{ext}")
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        return temp_path

    def _analyze_document(
        self, file_path: str, filename: str, file_type: str, file_size: int
    ) -> DocumentProfile:
        """Analyze document characteristics to guide strategy selection."""
        profile = DocumentProfile(
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size
        )

        if file_type == "pdf":
            try:
                import fitz
                doc = fitz.open(file_path)
                profile.page_count = len(doc)

                # Analyze first few pages for characteristics
                has_text = False
                has_images = False
                sample_text = ""

                for page_num in range(min(3, len(doc))):
                    page = doc[page_num]
                    text = page.get_text("text")
                    if text and len(text.strip()) > 50:
                        has_text = True
                        sample_text += text

                    # Check for images
                    image_list = page.get_images(full=True)
                    if image_list:
                        has_images = True
                        profile.has_images = True

                doc.close()

                # Detect if scanned (no text layer)
                if not has_text and has_images:
                    profile.is_scanned = True

                # Detect tables (look for table-like patterns)
                if sample_text:
                    import re
                    # Check for table indicators
                    table_indicators = [
                        r'\t.*\t',  # Tab-separated values
                        r'\|.*\|',  # Pipe-separated values
                        r'(?:AED|USD|%)\s+\d',  # Currency/percentage values
                        r'\d+\.\d+%',  # Percentage values
                    ]
                    for pattern in table_indicators:
                        if re.search(pattern, sample_text):
                            profile.has_tables = True
                            break

                    # Detect formulas
                    formula_indicators = [
                        r'[=×÷±]',
                        r'\d+\s*[x×]\s*\d+',
                        r'formula|calculation|computed',
                    ]
                    for pattern in formula_indicators:
                        if re.search(pattern, sample_text, re.IGNORECASE):
                            profile.has_formulas = True
                            break

                    # Detect bilingual (Arabic)
                    arabic_pattern = r'[\u0600-\u06FF]'
                    if re.search(arabic_pattern, sample_text):
                        profile.is_bilingual = True

                # Calculate complexity score
                complexity = 0.0
                if profile.has_tables: complexity += 0.25
                if profile.has_images: complexity += 0.2
                if profile.has_formulas: complexity += 0.15
                if profile.is_bilingual: complexity += 0.15
                if profile.is_scanned: complexity += 0.25
                if profile.page_count > 10: complexity += 0.1
                profile.complexity_score = min(complexity, 1.0)

            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Document analysis failed: {e}")
                profile.complexity_score = 0.5  # Assume moderate complexity

        elif file_type in ("txt", "md"):
            profile.page_count = 1
            profile.complexity_score = 0.1

        return profile

    def _select_strategies(self, profile: DocumentProfile) -> List[ExtractionStrategy]:
        """Select optimal extraction strategies based on document profile."""
        strategies = []

        extraction_config = getattr(self.settings.rag, 'extraction_config', {})
        fast_mode = extraction_config.get('fast_mode', True)

        # Vision: Always for PDFs with images, scanned docs, or complex layouts
        # In fast_mode, vision is disabled (it's the slowest strategy)
        vision_enabled = (
            not fast_mode and
            extraction_config.get('vision_enabled', True) and
            VISION_AVAILABLE and
            self.vision_extractor is not None and
            profile.file_type == "pdf"
        )
        strategies.append(ExtractionStrategy(
            name="vision",
            enabled=vision_enabled,
            priority=1,
            reason="PDF with visual content" if vision_enabled else "Not a PDF or vision disabled"
        ))

        # Table: For documents with detected tables
        table_enabled = (
            extraction_config.get('table_enabled', True) and
            TABLE_AVAILABLE and
            self.table_extractor is not None and
            profile.file_type == "pdf"
        )
        strategies.append(ExtractionStrategy(
            name="table",
            enabled=table_enabled,
            priority=2,
            reason="PDF with tables detected" if table_enabled else "No tables or table extraction disabled"
        ))

        # Graph: For complex documents with multiple entities
        # In fast_mode, graph is disabled (many LLM calls)
        graph_enabled = (
            not fast_mode and
            extraction_config.get('graph_enabled', True) and
            GRAPH_AVAILABLE and
            self.graph_builder is not None and
            profile.complexity_score >= 0.2
        )
        strategies.append(ExtractionStrategy(
            name="graph",
            enabled=graph_enabled,
            priority=3,
            reason="Complex document with multiple entities" if graph_enabled else "Simple document or graph disabled"
        ))

        # Enrichment: Always enabled for better retrieval quality
        enrichment_enabled = (
            extraction_config.get('contextual_enabled', True) and
            ENRICHMENT_AVAILABLE and
            self.enrichment_engine is not None
        )
        strategies.append(ExtractionStrategy(
            name="enrichment",
            enabled=enrichment_enabled,
            priority=4,
            reason="Contextual enrichment for better retrieval" if enrichment_enabled else "Enrichment disabled"
        ))

        # Sort by priority
        strategies.sort(key=lambda s: s.priority)
        return strategies

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event for monitoring and logging."""
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        self.events.append(event)
        logger.info(f"[ORCHESTRATOR] Event: {event_type} — {json.dumps(data)}")

    def _generate_product_fee_chunks(
        self,
        all_tables: List[Dict[str, Any]],
        enriched_chunks: List[Dict[str, Any]],
        existing_ids: set,
        filename: str
    ) -> tuple:
        """
        Parse tables that contain product/card names alongside fee data
        and generate clean, self-contained per-product chunks.
        
        This solves the problem where coordinate-based extraction puts
        multiple card names in one cell and fees in another, making
        retrieval impossible for specific card queries.
        """
        import re

        # Known banking product patterns
        product_patterns = [
            r'(?:Platinum\s*(?:Elite|Plus))',
            r'(?:Solitaire)',
            r'(?:Cashback)',
            r'(?:Gold)',
            r'(?:Silver)',
            r'(?:Classic)',
            r'(?:Titanium)',
            r'(?:World\s*Elite)',
            r'(?:Infinite)',
            r'(?:Signature)',
        ]
        product_regex = re.compile('|'.join(product_patterns), re.IGNORECASE)

        # Fee/amount patterns
        fee_regex = re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\b')

        product_chunks_created = 0

        for table in all_tables:
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            page = table.get("page", 0)

            # Scan all cell values for product names
            all_cell_text = ""
            for row in rows:
                for val in row.values():
                    all_cell_text += " " + str(val)
            for header in headers:
                all_cell_text += " " + str(header)

            products_found = product_regex.findall(all_cell_text)
            if not products_found:
                continue

            # This table contains product names — try to extract per-product data
            # Strategy: collect all text from the table and use LLM-free heuristic parsing

            # Collect all fee-like values from the table
            fee_values = fee_regex.findall(all_cell_text)
            fee_values = [f for f in fee_values if float(f.replace(',', '')) > 0]

            # Collect all header keywords that indicate fee types
            fee_headers = []
            for h in headers:
                h_lower = str(h).lower()
                if any(kw in h_lower for kw in ['fee', 'annual', 'supplementary', 'recurring', 'charge', 'rate']):
                    fee_headers.append(h)

            # Try to match products with fees by scanning rows
            # Look for cells that contain product names
            for row_idx, row in enumerate(rows):
                row_values = list(row.values())
                row_text = " ".join(str(v) for v in row_values)

                # Find products in this row
                row_products = product_regex.findall(row_text)
                if not row_products:
                    continue

                # Find fees in this row
                row_fees = fee_regex.findall(row_text)
                row_fees = [f for f in row_fees if float(f.replace(',', '')) >= 1]

                # Also check if "Free for life" is in this row
                free_for_life = bool(re.search(r'free\s+for\s+life', row_text, re.IGNORECASE))

                # Create a chunk for each product found in this row
                for product in row_products:
                    product_clean = product.strip()
                    chunk_text = f"Product: {product_clean}\n"
                    chunk_text += f"Source: {filename} (Page {page})\n"

                    # Add fee information
                    if fee_headers:
                        for i, fh in enumerate(fee_headers):
                            val = row.get(fh, "")
                            if val and str(val).strip():
                                chunk_text += f"{fh}: {val}\n"

                    # Add all header-value pairs from this row
                    for header in headers:
                        val = row.get(header, "")
                        if val and str(val).strip() and len(str(val)) < 200:
                            if header not in fee_headers:
                                chunk_text += f"{header}: {val}\n"

                    # Add fee amounts found
                    if row_fees:
                        chunk_text += f"Fee amounts found in this row: {', '.join(row_fees)}\n"

                    if free_for_life:
                        chunk_text += "Note: This product may be 'Free for life' (no annual fee).\n"

                    # Also scan adjacent rows for related data
                    for adj_idx in [row_idx - 1, row_idx + 1]:
                        if 0 <= adj_idx < len(rows):
                            adj_row = rows[adj_idx]
                            adj_text = " ".join(str(v) for v in adj_row.values())
                            adj_fees = fee_regex.findall(adj_text)
                            adj_fees = [f for f in adj_fees if float(f.replace(',', '')) >= 1]
                            if adj_fees and not product_regex.search(adj_text):
                                chunk_text += f"Adjacent row data: {adj_text[:200]}\n"

                    chunk_id = f"product_{product_clean.lower().replace(' ', '_')}_{page}_{row_idx}"
                    if chunk_id not in existing_ids:
                        enriched_chunks.append({
                            "chunk_id": chunk_id,
                            "text": chunk_text.strip(),
                            "original_text": chunk_text.strip(),
                            "context_prefix": f"Product details for {product_clean} from {filename}",
                            "source": filename,
                            "page": page,
                            "section": f"Product: {product_clean}",
                            "chunk_type": "product_detail",
                            "related_tables": [table.get("table_id", "")],
                            "related_entities": [],
                            "confidence": 0.85
                        })
                        existing_ids.add(chunk_id)
                        product_chunks_created += 1

            # Also try to parse multi-product cells (e.g., "Platinum Elite Platinum Plus Solitaire")
            for row_idx, row in enumerate(rows):
                for cell_val in row.values():
                    cell_str = str(cell_val)
                    cell_products = product_regex.findall(cell_str)
                    if len(cell_products) >= 2:
                        # Multiple products in one cell — this is a comparison row
                        # Collect all fees from the entire table for this row context
                        all_row_data = []
                        for r in rows:
                            r_text = " | ".join(f"{k}: {v}" for k, v in r.items() if str(v).strip())
                            all_row_data.append(r_text)

                        comparison_text = f"Card Comparison from {filename} (Page {page}):\n"
                        comparison_text += f"Cards compared: {', '.join(cell_products)}\n"
                        comparison_text += f"Table headers: {', '.join(headers)}\n\n"
                        for r_text in all_row_data:
                            comparison_text += f"  {r_text}\n"

                        chunk_id = f"comparison_{page}_{row_idx}"
                        if chunk_id not in existing_ids:
                            enriched_chunks.append({
                                "chunk_id": chunk_id,
                                "text": comparison_text.strip(),
                                "original_text": comparison_text.strip(),
                                "context_prefix": f"Card comparison from {filename}",
                                "source": filename,
                                "page": page,
                                "section": "Card Comparison",
                                "chunk_type": "product_comparison",
                                "related_tables": [table.get("table_id", "")],
                                "related_entities": [],
                                "confidence": 0.85
                            })
                            existing_ids.add(chunk_id)
                            product_chunks_created += 1

        if product_chunks_created > 0:
            logger.info(f"[ORCHESTRATOR] Generated {product_chunks_created} product-specific chunks")
            self._emit_event("product_chunks_generated", {"count": product_chunks_created})

        return enriched_chunks, existing_ids
