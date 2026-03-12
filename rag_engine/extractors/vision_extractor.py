"""
Vision-Based Document Extraction Engine
========================================
Implements ColPali-style page-level visual understanding for complex PDFs.

Strategy: Convert each PDF page to a high-resolution image, then use a multimodal
LLM (GPT-4.1-mini with vision) to extract structured content including tables,
flowcharts, mathematical formulas, and embedded images — preserving layout context
that text-only extraction destroys.

Falls back to Tesseract OCR when LLM vision is unavailable.

Feature Toggle: extraction_vision_enabled (default: True)
"""
import os
import io
import json
import base64
import hashlib
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PageVisionResult:
    """Result of vision-based extraction for a single page."""
    page_number: int
    page_image_path: str
    extracted_text: str
    tables: List[Dict[str, Any]] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    flowcharts: List[Dict[str, Any]] = field(default_factory=list)
    images_described: List[str] = field(default_factory=list)
    language: str = "en"
    confidence: float = 0.0
    raw_llm_response: str = ""


@dataclass
class VisionExtractionResult:
    """Complete vision extraction result for an entire document."""
    filename: str
    total_pages: int
    page_results: List[PageVisionResult] = field(default_factory=list)
    strategy_used: str = "llm_vision"
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n---PAGE BREAK---\n\n".join(
            pr.extracted_text for pr in self.page_results if pr.extracted_text
        )

    def get_all_tables(self) -> List[Dict[str, Any]]:
        """Get all tables across all pages."""
        tables = []
        for pr in self.page_results:
            for t in pr.tables:
                t["page"] = pr.page_number
                tables.append(t)
        return tables

    def get_all_formulas(self) -> List[Dict[str, Any]]:
        """Get all formulas across all pages."""
        formulas = []
        for pr in self.page_results:
            for f in pr.formulas:
                formulas.append({"page": pr.page_number, "formula": f})
        return formulas

    def get_all_flowcharts(self) -> List[Dict[str, Any]]:
        """Get all flowcharts across all pages."""
        flowcharts = []
        for pr in self.page_results:
            for fc in pr.flowcharts:
                fc["page"] = pr.page_number
                flowcharts.append(fc)
        return flowcharts


class VisionExtractor:
    """
    Vision-based document extraction engine.

    Uses multimodal LLM to "see" PDF pages as images and extract:
    - Structured text with layout awareness
    - Tables with row-column preservation
    - Mathematical formulas
    - Flowchart/diagram descriptions
    - Embedded image descriptions
    - Language detection (EN/AR)
    """

    # Centralized prompt for vision extraction
    VISION_EXTRACTION_PROMPT = """You are a document intelligence expert. Analyze this PDF page image and extract ALL content with perfect structural fidelity.

EXTRACTION RULES:
1. **Text**: Extract all visible text preserving paragraph structure and hierarchy.
2. **Tables**: Extract every table as a JSON array of objects. Each object represents a row with column headers as keys. Preserve ALL data — do not summarize or skip rows.
3. **Mathematical Formulas**: Extract any formulas, calculations, or equations exactly as shown. Use plain text notation (e.g., "AED 1,000 x (3.85% x 12 / 365) x 57 days = AED 72.15").
4. **Flowcharts/Diagrams**: Describe any visual process flows as step-by-step sequences with arrows (→).
5. **Images**: Describe any embedded images (logos, card images, icons) and their context.
6. **Language**: Detect the primary language of this page (en/ar/mixed).

RESPOND IN THIS EXACT JSON FORMAT:
{
    "text": "Full extracted text preserving structure...",
    "tables": [
        {
            "title": "Table title or description",
            "headers": ["Col1", "Col2", "Col3"],
            "rows": [
                {"Col1": "val1", "Col2": "val2", "Col3": "val3"}
            ]
        }
    ],
    "formulas": ["formula1 as plain text", "formula2"],
    "flowcharts": [
        {
            "title": "Process name",
            "steps": ["Step 1 → Step 2 → Step 3"]
        }
    ],
    "images_described": ["Description of image 1", "Description of image 2"],
    "language": "en",
    "confidence": 0.95
}

CRITICAL: Extract EVERY piece of data from tables. Do not skip rows or columns. For comparison tables with multiple products, extract ALL products."""

    def __init__(self, client, settings, output_dir: str = None):
        """
        Initialize VisionExtractor.

        Args:
            client: OpenAI client instance
            settings: Application settings
            output_dir: Directory to save page images (default: data/page_images/)
        """
        self.client = client
        self.settings = settings
        self.output_dir = output_dir or os.path.join(
            settings.paths.base_dir, "data", "page_images"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def extract(self, pdf_path: str, filename: str = None, max_pages: int = None) -> VisionExtractionResult:
        """
        Extract content from a PDF using vision-based analysis.

        Args:
            pdf_path: Path to the PDF file
            filename: Display name for the document
            max_pages: Max pages to analyze (samples first, middle, last). None = all pages.

        Returns:
            VisionExtractionResult with all extracted content
        """
        if filename is None:
            filename = os.path.basename(pdf_path)

        logger.info(f"[VISION_EXTRACTOR] Processing: {filename}")

        # Step 1: Convert PDF pages to images
        page_images = self._pdf_to_images(pdf_path, filename)

        # Sample pages if max_pages is set and we have more pages
        if max_pages and len(page_images) > max_pages:
            indices = set()
            indices.add(0)  # First page
            indices.add(len(page_images) - 1)  # Last page
            # Add evenly spaced middle pages
            step = len(page_images) / max_pages
            for i in range(1, max_pages - 1):
                indices.add(min(int(i * step), len(page_images) - 1))
            sampled = sorted(indices)
            page_images = [page_images[i] for i in sampled]
            logger.info(f"[VISION_EXTRACTOR] Sampled {len(page_images)} of {len(page_images)} pages")

        if not page_images:
            logger.warning(f"[VISION_EXTRACTOR] No page images generated for {filename}")
            return VisionExtractionResult(
                filename=filename, total_pages=0,
                strategy_used="failed", extraction_metadata={"error": "No pages extracted"}
            )

        # Step 2: Process each page with vision LLM
        page_results = []
        for page_num, image_path in page_images:
            try:
                result = self._extract_page_with_vision(image_path, page_num)
                page_results.append(result)
                logger.info(f"[VISION_EXTRACTOR] Page {page_num}: "
                          f"{len(result.tables)} tables, "
                          f"{len(result.formulas)} formulas, "
                          f"lang={result.language}")
            except Exception as e:
                logger.error(f"[VISION_EXTRACTOR] Page {page_num} LLM vision failed: {e}")
                # Fallback to OCR
                try:
                    result = self._extract_page_with_ocr(image_path, page_num)
                    result.confidence = 0.5  # Lower confidence for OCR
                    page_results.append(result)
                    logger.info(f"[VISION_EXTRACTOR] Page {page_num}: OCR fallback used")
                except Exception as ocr_e:
                    logger.error(f"[VISION_EXTRACTOR] Page {page_num} OCR also failed: {ocr_e}")
                    page_results.append(PageVisionResult(
                        page_number=page_num, page_image_path=image_path,
                        extracted_text="", confidence=0.0
                    ))

        return VisionExtractionResult(
            filename=filename,
            total_pages=len(page_images),
            page_results=page_results,
            strategy_used="llm_vision",
            extraction_metadata={
                "output_dir": self.output_dir,
                "pages_processed": len(page_results),
                "pages_with_tables": sum(1 for pr in page_results if pr.tables),
                "pages_with_formulas": sum(1 for pr in page_results if pr.formulas),
                "pages_with_flowcharts": sum(1 for pr in page_results if pr.flowcharts),
            }
        )

    def _pdf_to_images(self, pdf_path: str, filename: str) -> List[tuple]:
        """Convert PDF pages to high-resolution images using PyMuPDF."""
        page_images = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            doc_hash = hashlib.md5(filename.encode()).hexdigest()[:8]

            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render at 2x resolution for better OCR/vision quality
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)

                image_filename = f"{doc_hash}_page_{page_num + 1}.png"
                image_path = os.path.join(self.output_dir, image_filename)
                pix.save(image_path)

                page_images.append((page_num + 1, image_path))

            doc.close()
            logger.info(f"[VISION_EXTRACTOR] Rendered {len(page_images)} pages from {filename}")

        except Exception as e:
            logger.error(f"[VISION_EXTRACTOR] PyMuPDF rendering failed: {e}")
            # Fallback to pdf2image (poppler)
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path, dpi=200)
                doc_hash = hashlib.md5(filename.encode()).hexdigest()[:8]

                for i, img in enumerate(images):
                    image_filename = f"{doc_hash}_page_{i + 1}.png"
                    image_path = os.path.join(self.output_dir, image_filename)
                    img.save(image_path, "PNG")
                    page_images.append((i + 1, image_path))

                logger.info(f"[VISION_EXTRACTOR] pdf2image rendered {len(page_images)} pages")
            except Exception as e2:
                logger.error(f"[VISION_EXTRACTOR] pdf2image also failed: {e2}")

        return page_images

    def _extract_page_with_vision(self, image_path: str, page_number: int) -> PageVisionResult:
        """Extract content from a page image using multimodal LLM."""
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Call multimodal LLM
        response = self.client.chat.completions.create(
            model=self.settings.llm.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.VISION_EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=4096
        )

        raw_response = response.choices[0].message.content.strip()

        # Parse JSON response
        parsed = self._parse_vision_response(raw_response)

        return PageVisionResult(
            page_number=page_number,
            page_image_path=image_path,
            extracted_text=parsed.get("text", ""),
            tables=parsed.get("tables", []),
            formulas=parsed.get("formulas", []),
            flowcharts=parsed.get("flowcharts", []),
            images_described=parsed.get("images_described", []),
            language=parsed.get("language", "en"),
            confidence=parsed.get("confidence", 0.8),
            raw_llm_response=raw_response
        )

    def _extract_page_with_ocr(self, image_path: str, page_number: int) -> PageVisionResult:
        """Fallback: Extract text using Tesseract OCR."""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            # Try English + Arabic OCR
            text = pytesseract.image_to_string(img, lang='eng')

            return PageVisionResult(
                page_number=page_number,
                page_image_path=image_path,
                extracted_text=text,
                language="en",
                confidence=0.5
            )
        except Exception as e:
            logger.error(f"[VISION_EXTRACTOR] OCR failed for page {page_number}: {e}")
            return PageVisionResult(
                page_number=page_number,
                page_image_path=image_path,
                extracted_text="",
                confidence=0.0
            )

    def _parse_vision_response(self, raw_response: str) -> Dict:
        """Parse the JSON response from the vision LLM."""
        # Strip markdown code blocks if present
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

            # Last resort: return raw text as extracted text
            logger.warning("[VISION_EXTRACTOR] Could not parse JSON, using raw text")
            return {
                "text": raw_response,
                "tables": [],
                "formulas": [],
                "flowcharts": [],
                "images_described": [],
                "language": "en",
                "confidence": 0.5
            }
