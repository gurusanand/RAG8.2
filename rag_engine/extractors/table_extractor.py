"""
Structured Table Extraction Engine (TabRAG)
============================================
Implements table-aware document parsing that preserves row-column relationships.

Strategy: Uses multiple extraction backends (Docling, pdfplumber, PyMuPDF) to
detect and extract tables from PDFs, converting them to structured Markdown/JSON
format that preserves the relational integrity of tabular data.

Falls back through backends in order of capability:
1. Docling (best: layout-aware, handles merged cells)
2. pdfplumber (good: precise table detection)
3. PyMuPDF text extraction (fallback: regex-based table detection)

Feature Toggle: extraction_table_enabled (default: True)
"""
import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTable:
    """A single extracted table with full structural metadata."""
    table_id: str
    page_number: int
    title: str
    headers: List[str]
    rows: List[Dict[str, str]]
    markdown: str
    row_count: int
    col_count: int
    source_method: str  # "docling", "pdfplumber", "pymupdf_regex"
    confidence: float = 0.0

    def to_searchable_chunks(self) -> List[Dict[str, str]]:
        """Convert table to searchable text chunks — one per row for precise retrieval."""
        chunks = []
        # Chunk 1: Table overview (headers + summary)
        overview = f"Table: {self.title}\nColumns: {', '.join(self.headers)}\n"
        overview += f"This table has {self.row_count} rows and {self.col_count} columns.\n"
        overview += self.markdown
        chunks.append({
            "text": overview,
            "chunk_type": "table_overview",
            "table_id": self.table_id,
            "page": self.page_number
        })

        # Chunk 2+: Individual rows for precise retrieval
        for i, row in enumerate(self.rows):
            row_text = f"From table '{self.title}' (Page {self.page_number}):\n"
            for header, value in row.items():
                row_text += f"  {header}: {value}\n"
            chunks.append({
                "text": row_text.strip(),
                "chunk_type": "table_row",
                "table_id": self.table_id,
                "row_index": i,
                "page": self.page_number
            })

        return chunks


@dataclass
class TableExtractionResult:
    """Complete table extraction result for a document."""
    filename: str
    total_pages: int
    tables: List[ExtractedTable] = field(default_factory=list)
    full_text: str = ""  # Non-table text extracted alongside
    strategy_used: str = ""
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_table_chunks(self) -> List[Dict[str, str]]:
        """Get all searchable chunks from all tables."""
        chunks = []
        for table in self.tables:
            chunks.extend(table.to_searchable_chunks())
        return chunks

    def get_table_markdown(self) -> str:
        """Get all tables as Markdown."""
        parts = []
        for table in self.tables:
            parts.append(f"### {table.title} (Page {table.page_number})\n\n{table.markdown}")
        return "\n\n".join(parts)


class TableExtractor:
    """
    Multi-backend table extraction engine.

    Tries extraction backends in order of capability:
    1. Docling — best for complex layouts, merged cells, multi-page tables
    2. pdfplumber — precise geometric table detection
    3. PyMuPDF regex — fallback pattern-based extraction
    """

    def __init__(self, settings):
        self.settings = settings
        self._docling_available = self._check_docling()
        self._pdfplumber_available = self._check_pdfplumber()

    def _check_docling(self) -> bool:
        try:
            from docling.document_converter import DocumentConverter
            return True
        except ImportError:
            logger.info("[TABLE_EXTRACTOR] Docling not available")
            return False

    def _check_pdfplumber(self) -> bool:
        try:
            import pdfplumber
            return True
        except ImportError:
            logger.info("[TABLE_EXTRACTOR] pdfplumber not available")
            return False

    def extract(self, pdf_path: str, filename: str = None) -> TableExtractionResult:
        """
        Extract tables from a PDF using the best available backend.

        Args:
            pdf_path: Path to the PDF file
            filename: Display name for the document

        Returns:
            TableExtractionResult with all extracted tables
        """
        if filename is None:
            filename = os.path.basename(pdf_path)

        logger.info(f"[TABLE_EXTRACTOR] Processing: {filename}")

        all_tables = []
        full_text = ""
        total_pages = 0
        strategies_used = []

        # Backend 1: Docling (best for complex layouts)
        if self._docling_available:
            try:
                result = self._extract_with_docling(pdf_path, filename)
                if result and result.tables:
                    logger.info(f"[TABLE_EXTRACTOR] Docling extracted {len(result.tables)} tables")
                    all_tables.extend(result.tables)
                    full_text = result.full_text
                    total_pages = result.total_pages
                    strategies_used.append("docling")
            except Exception as e:
                logger.warning(f"[TABLE_EXTRACTOR] Docling failed: {e}")

        # Backend 2: pdfplumber (precise geometric detection)
        if self._pdfplumber_available:
            try:
                result = self._extract_with_pdfplumber(pdf_path, filename)
                if result and result.tables:
                    logger.info(f"[TABLE_EXTRACTOR] pdfplumber extracted {len(result.tables)} tables")
                    all_tables.extend(result.tables)
                    if not full_text:
                        full_text = result.full_text
                    total_pages = max(total_pages, result.total_pages)
                    strategies_used.append("pdfplumber")
            except Exception as e:
                logger.warning(f"[TABLE_EXTRACTOR] pdfplumber failed: {e}")

        # Backend 3: PyMuPDF coordinate-based (ALWAYS runs to catch visual tables)
        # This catches tables that pdfplumber misses (e.g., card fee tables with images)
        try:
            result = self._extract_with_pymupdf(pdf_path, filename)
            if result and result.tables:
                logger.info(f"[TABLE_EXTRACTOR] PyMuPDF coordinate extracted {len(result.tables)} tables")
                all_tables.extend(result.tables)
                if not full_text:
                    full_text = result.full_text
                total_pages = max(total_pages, result.total_pages)
                strategies_used.append("pymupdf_coordinate")
        except Exception as e:
            logger.error(f"[TABLE_EXTRACTOR] PyMuPDF failed: {e}")

        if not all_tables:
            return TableExtractionResult(
                filename=filename, total_pages=total_pages, strategy_used="failed",
                extraction_metadata={"error": "All backends failed"}
            )

        # Deduplicate tables by content similarity
        unique_tables = self._deduplicate_tables(all_tables)

        return TableExtractionResult(
            filename=filename,
            total_pages=total_pages,
            tables=unique_tables,
            full_text=full_text,
            strategy_used="+".join(strategies_used),
            extraction_metadata={
                "backends": strategies_used,
                "tables_before_dedup": len(all_tables),
                "tables_after_dedup": len(unique_tables)
            }
        )

    def _extract_with_docling(self, pdf_path: str, filename: str) -> TableExtractionResult:
        """Extract tables using Docling's layout-aware parser."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        doc_result = converter.convert(pdf_path)

        tables = []
        table_idx = 0

        # Extract tables from Docling's structured output
        doc = doc_result.document
        full_text_parts = []

        for item in doc.iterate_items():
            element = item[1] if isinstance(item, tuple) else item
            # Check if element is a table
            element_type = type(element).__name__

            if element_type == "TableItem" or hasattr(element, 'export_to_dataframe'):
                try:
                    df = element.export_to_dataframe()
                    headers = [str(h) for h in df.columns.tolist()]
                    rows = []
                    for _, row in df.iterrows():
                        row_dict = {}
                        for h in headers:
                            row_dict[h] = str(row[h]) if row[h] is not None else ""
                        rows.append(row_dict)

                    markdown = self._dataframe_to_markdown(headers, rows)
                    table_id = f"table_{filename}_{table_idx}"

                    tables.append(ExtractedTable(
                        table_id=table_id,
                        page_number=getattr(element, 'page_no', 0) or 0,
                        title=getattr(element, 'caption', f"Table {table_idx + 1}") or f"Table {table_idx + 1}",
                        headers=headers,
                        rows=rows,
                        markdown=markdown,
                        row_count=len(rows),
                        col_count=len(headers),
                        source_method="docling",
                        confidence=0.9
                    ))
                    table_idx += 1
                except Exception as e:
                    logger.warning(f"[TABLE_EXTRACTOR] Docling table export failed: {e}")

            elif hasattr(element, 'text') and element.text:
                full_text_parts.append(element.text)

        # Also get full markdown export
        full_text = doc_result.document.export_to_markdown()

        return TableExtractionResult(
            filename=filename,
            total_pages=getattr(doc, 'num_pages', len(tables)),
            tables=tables,
            full_text=full_text,
            strategy_used="docling",
            extraction_metadata={"backend": "docling", "tables_found": len(tables)}
        )

    def _extract_with_pdfplumber(self, pdf_path: str, filename: str) -> TableExtractionResult:
        """Extract tables using pdfplumber's geometric detection."""
        import pdfplumber

        tables = []
        full_text_parts = []
        table_idx = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables
                page_tables = page.extract_tables()
                for raw_table in page_tables:
                    if not raw_table or len(raw_table) < 2:
                        continue

                    # First row as headers
                    headers = [str(h).strip() if h else f"Col_{i}" for i, h in enumerate(raw_table[0])]
                    # Clean empty headers
                    headers = [h if h else f"Col_{i}" for i, h in enumerate(headers)]

                    rows = []
                    for row_data in raw_table[1:]:
                        if not row_data:
                            continue
                        row_dict = {}
                        for i, cell in enumerate(row_data):
                            header = headers[i] if i < len(headers) else f"Col_{i}"
                            row_dict[header] = str(cell).strip() if cell else ""
                        # Skip completely empty rows
                        if any(v for v in row_dict.values()):
                            rows.append(row_dict)

                    if rows:
                        markdown = self._dataframe_to_markdown(headers, rows)
                        table_id = f"table_{filename}_{table_idx}"

                        tables.append(ExtractedTable(
                            table_id=table_id,
                            page_number=page_num,
                            title=self._infer_table_title(headers, page_num),
                            headers=headers,
                            rows=rows,
                            markdown=markdown,
                            row_count=len(rows),
                            col_count=len(headers),
                            source_method="pdfplumber",
                            confidence=0.85
                        ))
                        table_idx += 1

                # Extract non-table text
                text = page.extract_text()
                if text:
                    full_text_parts.append(f"--- Page {page_num} ---\n{text}")

        return TableExtractionResult(
            filename=filename,
            total_pages=len(full_text_parts),
            tables=tables,
            full_text="\n\n".join(full_text_parts),
            strategy_used="pdfplumber",
            extraction_metadata={"backend": "pdfplumber", "tables_found": len(tables)}
        )

    def _extract_with_pymupdf(self, pdf_path: str, filename: str) -> TableExtractionResult:
        """Fallback: Extract text with PyMuPDF using coordinate-based table reconstruction.
        
        Uses block bounding boxes to group text into rows (by Y-coordinate) and
        columns (by X-coordinate), reconstructing visual table structure that
        standard text extraction destroys.
        """
        import fitz

        tables = []
        full_text_parts = []
        table_idx = 0

        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            full_text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

            # Method 1: Coordinate-based visual table reconstruction
            coord_tables = self._extract_tables_by_coordinates(page, page_num + 1, filename, table_idx)
            if coord_tables:
                tables.extend(coord_tables)
                table_idx += len(coord_tables)
            else:
                # Method 2: Regex-based fallback
                detected_tables = self._detect_tables_from_text(text, page_num + 1, filename, table_idx)
                tables.extend(detected_tables)
                table_idx += len(detected_tables)

        doc.close()

        return TableExtractionResult(
            filename=filename,
            total_pages=len(full_text_parts),
            tables=tables,
            full_text="\n\n".join(full_text_parts),
            strategy_used="pymupdf_coordinate",
            extraction_metadata={"backend": "pymupdf_coordinate", "tables_found": len(tables)}
        )

    def _extract_tables_by_coordinates(self, page, page_num: int,
                                        filename: str, start_idx: int) -> List[ExtractedTable]:
        """Extract tables using coordinate-based block grouping.
        
        Groups text blocks by Y-coordinate proximity into rows, then by
        X-coordinate into columns. This preserves the visual table structure
        that standard text extraction destroys.
        """
        import fitz

        # Get all text blocks with positions
        blocks_data = page.get_text('dict')['blocks']
        text_blocks = []
        for block in blocks_data:
            if block['type'] != 0:  # skip image blocks
                continue
            text = ''
            for line in block['lines']:
                for span in line['spans']:
                    text += span['text'] + ' '
            text = text.strip()
            if text and len(text) > 1:
                bbox = block['bbox']
                text_blocks.append({
                    'text': text,
                    'x0': bbox[0], 'y0': bbox[1],
                    'x1': bbox[2], 'y1': bbox[3],
                    'y_center': (bbox[1] + bbox[3]) / 2,
                    'x_center': (bbox[0] + bbox[2]) / 2,
                    'height': bbox[3] - bbox[1],
                })

        if len(text_blocks) < 4:
            return []

        # Group blocks into rows by Y-coordinate proximity
        sorted_blocks = sorted(text_blocks, key=lambda b: b['y_center'])
        rows = []
        current_row = [sorted_blocks[0]]
        y_tolerance = 20  # pixels

        for block in sorted_blocks[1:]:
            if abs(block['y_center'] - current_row[0]['y_center']) <= y_tolerance:
                current_row.append(block)
            else:
                rows.append(sorted(current_row, key=lambda b: b['x0']))
                current_row = [block]
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b['x0']))

        # Find rows with multiple columns (potential table rows)
        multi_col_rows = [(i, row) for i, row in enumerate(rows) if len(row) >= 2]
        if len(multi_col_rows) < 2:
            return []

        # Detect contiguous table regions (consecutive multi-column rows)
        table_regions = []
        current_region = [multi_col_rows[0]]
        for j in range(1, len(multi_col_rows)):
            prev_idx = multi_col_rows[j-1][0]
            curr_idx = multi_col_rows[j][0]
            # Allow up to 2 single-column rows between table rows
            if curr_idx - prev_idx <= 3:
                current_region.append(multi_col_rows[j])
            else:
                if len(current_region) >= 2:
                    table_regions.append(current_region)
                current_region = [multi_col_rows[j]]
        if len(current_region) >= 2:
            table_regions.append(current_region)

        # Convert each table region to an ExtractedTable
        extracted_tables = []
        for region_idx, region in enumerate(table_regions):
            region_rows = [row for _, row in region]
            
            # Determine column boundaries from the most common X positions
            all_x_starts = []
            for row in region_rows:
                for block in row:
                    all_x_starts.append(round(block['x0'] / 30) * 30)  # Quantize to 30px grid
            
            # Build table data
            table_data = []
            for row in region_rows:
                row_texts = [block['text'] for block in row]
                table_data.append(row_texts)

            if not table_data:
                continue

            # Normalize column count
            max_cols = max(len(row) for row in table_data)
            
            # Use first row as headers if it looks like a header
            first_row = table_data[0]
            headers = []
            for i, cell in enumerate(first_row):
                headers.append(cell if cell else f"Col_{i}")
            # Pad headers if needed
            while len(headers) < max_cols:
                headers.append(f"Col_{len(headers)}")

            # Build row dicts
            rows_dicts = []
            for row_data in table_data[1:]:
                row_dict = {}
                for i, header in enumerate(headers):
                    row_dict[header] = row_data[i] if i < len(row_data) else ""
                if any(v.strip() for v in row_dict.values()):
                    rows_dicts.append(row_dict)

            if rows_dicts:
                markdown = self._dataframe_to_markdown(headers, rows_dicts)
                table_id = f"table_{filename}_{start_idx + region_idx}"

                extracted_tables.append(ExtractedTable(
                    table_id=table_id,
                    page_number=page_num,
                    title=self._infer_table_title(headers, page_num),
                    headers=headers,
                    rows=rows_dicts,
                    markdown=markdown,
                    row_count=len(rows_dicts),
                    col_count=len(headers),
                    source_method="pymupdf_coordinate",
                    confidence=0.75
                ))

        return extracted_tables

    def _detect_tables_from_text(self, text: str, page_num: int,
                                  filename: str, start_idx: int) -> List[ExtractedTable]:
        """Detect table-like structures from plain text using pattern matching."""
        tables = []
        lines = text.split('\n')

        # Look for lines with consistent delimiters (tabs, multiple spaces, pipes)
        potential_table_lines = []
        current_table = []

        for line in lines:
            # Check if line looks like a table row (has multiple tab/space-separated values)
            cells = re.split(r'\t|  {2,}|\|', line.strip())
            cells = [c.strip() for c in cells if c.strip()]

            if len(cells) >= 2:
                current_table.append(cells)
            else:
                if len(current_table) >= 2:
                    potential_table_lines.append(current_table)
                current_table = []

        if len(current_table) >= 2:
            potential_table_lines.append(current_table)

        # Convert detected tables
        for i, raw_table in enumerate(potential_table_lines):
            # Normalize column count
            max_cols = max(len(row) for row in raw_table)
            headers = raw_table[0] + [f"Col_{j}" for j in range(len(raw_table[0]), max_cols)]

            rows = []
            for row_data in raw_table[1:]:
                row_dict = {}
                for j, header in enumerate(headers):
                    row_dict[header] = row_data[j] if j < len(row_data) else ""
                rows.append(row_dict)

            if rows:
                markdown = self._dataframe_to_markdown(headers, rows)
                table_id = f"table_{filename}_{start_idx + i}"

                tables.append(ExtractedTable(
                    table_id=table_id,
                    page_number=page_num,
                    title=self._infer_table_title(headers, page_num),
                    headers=headers,
                    rows=rows,
                    markdown=markdown,
                    row_count=len(rows),
                    col_count=len(headers),
                    source_method="pymupdf_regex",
                    confidence=0.6
                ))

        return tables

    def _dataframe_to_markdown(self, headers: List[str], rows: List[Dict[str, str]]) -> str:
        """Convert table data to Markdown format."""
        if not headers or not rows:
            return ""

        # Header row
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        # Data rows
        for row in rows:
            cells = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
            md += "| " + " | ".join(cells) + " |\n"

        return md

    def _infer_table_title(self, headers: List[str], page_num: int) -> str:
        """Infer a descriptive title from table headers."""
        header_text = ", ".join(h for h in headers if h and not h.startswith("Col_"))
        if header_text:
            return f"Table (Page {page_num}): {header_text[:80]}"
        return f"Table on Page {page_num}"

    def _deduplicate_tables(self, tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """Deduplicate tables by comparing content overlap.
        
        Tables from different backends may extract the same data.
        We keep the higher-confidence version when duplicates are found.
        """
        if len(tables) <= 1:
            return tables

        unique = []
        seen_content_hashes = set()

        for table in tables:
            # Create a content fingerprint from the first few cell values
            fingerprint_parts = []
            for row in table.rows[:3]:
                for val in list(row.values())[:3]:
                    clean = re.sub(r'\s+', ' ', str(val).strip())[:50]
                    if clean and len(clean) > 2:
                        fingerprint_parts.append(clean)

            fingerprint = "|".join(sorted(fingerprint_parts))
            content_hash = hashlib.md5(fingerprint.encode()).hexdigest()

            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                unique.append(table)
            else:
                # Keep the higher-confidence version
                for i, existing in enumerate(unique):
                    existing_fp = []
                    for row in existing.rows[:3]:
                        for val in list(row.values())[:3]:
                            clean = re.sub(r'\s+', ' ', str(val).strip())[:50]
                            if clean and len(clean) > 2:
                                existing_fp.append(clean)
                    existing_hash = hashlib.md5("|".join(sorted(existing_fp)).encode()).hexdigest()
                    if existing_hash == content_hash and table.confidence > existing.confidence:
                        unique[i] = table
                        break

        logger.info(f"[TABLE_EXTRACTOR] Dedup: {len(tables)} -> {len(unique)} tables")
        return unique
