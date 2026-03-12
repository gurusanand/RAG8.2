"""
FAQ Test Runner — Zero-LLM-Cost Automated Test Pack for Banking RAG App
========================================================================
Tests the quality of FAQ extraction and storage by comparing stored FAQ
pairs against ground-truth answers from the original PDF document.

**ZERO LLM TOKEN COST** — All comparisons use local metrics only:
  - Cosine Similarity (local embedding model, no API calls)
  - Token Overlap Score (keyword/token-level match)
  - Key Phrase Coverage (numbers, proper nouns, banking terms)
  - Answer Completeness (length ratio check)

The test does NOT send queries through the RAG pipeline. Instead, it:
  1. Extracts ground-truth Q&A pairs from the original PDF
  2. Looks up the corresponding stored FAQ pair by question number
  3. Compares the stored answer against the ground-truth answer
  4. Reports similarity scores and pass/fail status

This approach validates that:
  - FAQ extraction is correct (no truncation, no wrong splits)
  - FAQ storage preserves the full answer
  - Question matching is accurate

Supports random sampling: user selects how many questions to test.

Author: Optimum AI Lab
"""

import os
import re
import sys
import time
import json
import random
import hashlib
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class GroundTruthQA:
    """A ground-truth Q&A pair extracted from the original PDF."""
    question_number: str       # e.g., "Q1", "Q25"
    question: str              # Full question text
    answer: str                # Full original answer text
    source_file: str           # PDF filename
    page: int = 0              # Page number (if available)


@dataclass
class TestResult:
    """Result of testing a single FAQ question."""
    question_number: str
    question: str
    original_answer: str       # Ground truth from PDF
    stored_answer: str         # Answer stored in FAQ engine

    # Similarity metrics
    cosine_similarity: float = 0.0        # Semantic similarity (0-1)
    token_overlap_score: float = 0.0      # Token-level overlap (0-1)
    key_phrase_coverage: float = 0.0      # Critical phrase coverage (0-1)
    answer_completeness: float = 0.0      # Length ratio (stored/original)
    composite_score: float = 0.0          # Weighted composite (0-1)

    # Match info
    match_type: str = ""                  # exact_qnum / fuzzy_question / not_found
    stored_question: str = ""             # The question as stored in FAQ engine
    question_similarity: float = 0.0      # Similarity between ground-truth Q and stored Q

    # Test verdict
    status: str = "PENDING"               # PASS / FAIL / WARNING / ERROR / NOT_FOUND
    failure_reason: str = ""              # Reason for failure (if any)

    # Timing
    comparison_time_ms: float = 0.0       # Time for comparison (local, no API)


@dataclass
class TestReport:
    """Complete test report for all FAQ questions."""
    report_id: str
    timestamp: str
    source_file: str
    test_mode: str = "zero_cost_local"    # Always zero-cost
    sampling_method: str = "all"          # "all" or "random_N"
    total_ground_truth: int = 0           # Total Q&A in PDF
    total_tested: int = 0                 # Number actually tested
    total_stored: int = 0                 # Total FAQ pairs in engine
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    errors: int = 0
    not_found: int = 0
    avg_cosine_similarity: float = 0.0
    avg_token_overlap: float = 0.0
    avg_key_phrase_coverage: float = 0.0
    avg_answer_completeness: float = 0.0
    avg_composite_score: float = 0.0
    total_test_duration_s: float = 0.0
    results: List[TestResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# FAQ TEST RUNNER (ZERO LLM COST)
# ═══════════════════════════════════════════════════════════

class FAQTestRunner:
    """
    Zero-LLM-cost FAQ test runner that compares stored FAQ pairs
    against ground-truth answers from the original PDF document.

    No API calls are made. All comparisons use:
      - Local embedding model (all-MiniLM-L6-v2) for cosine similarity
      - Token overlap (Jaccard-like) for keyword matching
      - Key phrase extraction for critical content coverage
      - Answer completeness (length ratio)
    """

    # Thresholds for pass/fail/warning
    PASS_THRESHOLD = 0.70       # composite_score >= 0.70 → PASS
    WARNING_THRESHOLD = 0.50    # composite_score >= 0.50 → WARNING
    # Below WARNING_THRESHOLD → FAIL

    # Weights for composite score
    COSINE_WEIGHT = 0.40        # Semantic similarity weight
    TOKEN_WEIGHT = 0.20         # Token overlap weight
    KEYPHRASE_WEIGHT = 0.20     # Key phrase coverage weight
    COMPLETENESS_WEIGHT = 0.20  # Answer completeness weight

    def __init__(self, rag_engine, embedding_model=None):
        """
        Initialize the test runner.

        Args:
            rag_engine: SevenLayerRAG instance (used to access stored FAQ pairs)
            embedding_model: SentenceTransformer model for cosine similarity
                           (uses the engine's indexer model if not provided)
        """
        self.engine = rag_engine
        self.embed_model = embedding_model
        if not self.embed_model and hasattr(rag_engine, 'indexer') and hasattr(rag_engine.indexer, 'embed_model'):
            self.embed_model = rag_engine.indexer.embed_model

        # Build a lookup of stored FAQ pairs by question number
        self._stored_pairs_by_qnum = {}
        self._stored_pairs_list = []
        self._build_stored_pairs_index()

    def _build_stored_pairs_index(self):
        """Build an index of stored FAQ pairs for fast lookup."""
        self._stored_pairs_by_qnum = {}
        self._stored_pairs_list = []

        if hasattr(self.engine, 'faq_engine') and self.engine.faq_engine:
            for pair in self.engine.faq_engine.faq_pairs:
                qnum = pair.question_number.strip().upper()
                self._stored_pairs_by_qnum[qnum] = pair
                self._stored_pairs_list.append(pair)

        logger.info(f"[FAQ_TEST] Built index of {len(self._stored_pairs_by_qnum)} stored FAQ pairs")

    def get_stored_pair_count(self) -> int:
        """Return the number of stored FAQ pairs."""
        return len(self._stored_pairs_by_qnum)

    # ─────────────────────────────────────────────────────
    # STEP 1: Extract Ground-Truth Q&A from PDF
    # ─────────────────────────────────────────────────────

    def extract_ground_truth_from_pdf(self, pdf_path: str) -> List[GroundTruthQA]:
        """
        Extract all Q&A pairs from a PDF file as ground truth.
        Uses PyMuPDF (fitz) for text extraction and regex for Q&A detection.
        """
        import fitz

        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                full_text += page_text + "\n"
        doc.close()

        if not full_text.strip():
            logger.warning(f"No text extracted from {pdf_path} — may be a scanned PDF")
            return []

        return self._parse_qa_pairs(full_text, os.path.basename(pdf_path))

    def extract_ground_truth_from_text(self, text: str, source_file: str = "uploaded_document") -> List[GroundTruthQA]:
        """Extract Q&A pairs from raw text."""
        return self._parse_qa_pairs(text, source_file)

    def _parse_qa_pairs(self, text: str, source_file: str) -> List[GroundTruthQA]:
        """
        Parse Q&A pairs from text using regex patterns.
        Handles multi-part questions (multiple ? marks) correctly.
        """
        pairs = []

        # Find all Q-markers: Q1, Q2, ..., Q999
        q_pattern = re.compile(r'Q(\d+)[.\s:)\-]+', re.IGNORECASE)
        q_positions = [(m.start(), m.group(), int(m.group(1))) for m in q_pattern.finditer(text)]

        if not q_positions:
            logger.warning(f"No Q-markers found in {source_file}")
            return pairs

        for i, (pos, marker, q_num) in enumerate(q_positions):
            start = pos
            end = q_positions[i + 1][0] if i + 1 < len(q_positions) else len(text)
            qa_block = text[start:end].strip()

            # Remove the Q-marker prefix
            qa_block = re.sub(r'^Q\d+[.\s:)\-]+', '', qa_block, flags=re.IGNORECASE).strip()

            if len(qa_block) < 10:
                continue

            # Split question from answer using smart splitting
            question, answer = self._smart_split_qa(qa_block)

            if question and answer and len(answer.strip()) > 5:
                pairs.append(GroundTruthQA(
                    question_number=f"Q{q_num}",
                    question=question.strip(),
                    answer=answer.strip(),
                    source_file=source_file
                ))

        logger.info(f"[FAQ_TEST] Extracted {len(pairs)} ground-truth Q&A pairs from {source_file}")
        return pairs

    def _smart_split_qa(self, qa_block: str) -> Tuple[str, str]:
        """
        Smart split of Q&A block into question and answer.
        Handles multi-part questions with multiple ? marks.
        """
        q_mark_positions = [i for i, c in enumerate(qa_block) if c == '?']

        if not q_mark_positions:
            newline_pos = qa_block.find('\n')
            if newline_pos > 0:
                return qa_block[:newline_pos], qa_block[newline_pos + 1:]
            return qa_block, ""

        answer_starters = {
            'you', 'the', 'a', 'an', 'to', 'yes', 'no', 'it', 'this', 'that',
            'we', 'our', 'your', 'if', 'for', 'in', 'on', 'at', 'by', 'with',
            'please', 'kindly', 'currently', 'as', 'all', 'any', 'each',
            'there', 'here', 'when', 'once', 'after', 'before', 'during',
            'customers', 'cardholders', 'members', 'users', 'clients',
            'mashreq', 'bank', 'uae', 'aed', 'funds', 'transfer',
            'minimum', 'maximum', 'total', 'cashback', 'interest', 'fee',
            'based', 'subject', 'according', 'following', 'below',
            'simply', 'just', 'first', 'step', 'visit', 'call', 'contact',
            'both', 'either', 'neither', 'every', 'most', 'some',
            'initially', 'generally', 'typically', 'usually', 'normally',
        }

        question_words = {
            'also', 'what', 'how', 'why', 'when', 'where', 'which', 'who',
            'is', 'are', 'can', 'do', 'does', 'will', 'would', 'should',
            'could', 'has', 'have', 'had', 'was', 'were', 'am',
            'and', 'or', 'but', 'if',
        }

        best_split = q_mark_positions[0]

        for pos in q_mark_positions:
            after_text = qa_block[pos + 1:].lstrip()
            if not after_text:
                continue

            # Skip URL query parameters
            pre_context = qa_block[max(0, pos - 80):pos]
            if re.search(r'https?://', pre_context) or re.search(r'\.(html|php|asp|jsp|com|net|org)\s*$', pre_context):
                continue
            char_before = qa_block[pos - 1] if pos > 0 else ''
            if char_before in ('/', '=', '&', '#'):
                continue

            first_word = re.split(r'[\s,;:]+', after_text)[0].lower().rstrip('.')

            if first_word in answer_starters and first_word not in question_words:
                best_split = pos
                break
            elif first_word not in question_words:
                best_split = pos

        question = qa_block[:best_split + 1].strip()
        answer = qa_block[best_split + 1:].strip()

        return question, answer

    # ─────────────────────────────────────────────────────
    # STEP 2: Find Stored FAQ Pair for a Ground-Truth Q
    # ─────────────────────────────────────────────────────

    def _find_stored_pair(self, ground_truth: GroundTruthQA) -> Tuple[Optional[Any], str, float]:
        """
        Find the stored FAQ pair that matches the ground-truth question.

        Returns:
            (stored_pair, match_type, question_similarity)
            match_type: "exact_qnum" | "fuzzy_question" | "not_found"
        """
        # Method 1: Exact match by question number (Q1, Q25, etc.)
        qnum = ground_truth.question_number.strip().upper()
        if qnum in self._stored_pairs_by_qnum:
            pair = self._stored_pairs_by_qnum[qnum]
            q_sim = self._cosine_similarity(ground_truth.question, pair.question)
            return pair, "exact_qnum", q_sim

        # Method 2: Fuzzy match by question text similarity
        best_pair = None
        best_sim = 0.0

        for pair in self._stored_pairs_list:
            sim = self._cosine_similarity(ground_truth.question, pair.question)
            if sim > best_sim:
                best_sim = sim
                best_pair = pair

        if best_pair and best_sim >= 0.80:
            return best_pair, "fuzzy_question", best_sim

        return None, "not_found", 0.0

    # ─────────────────────────────────────────────────────
    # STEP 3: Compare Stored Answer vs Ground Truth
    # ─────────────────────────────────────────────────────

    def run_single_test(self, ground_truth: GroundTruthQA) -> TestResult:
        """
        Compare a single ground-truth Q&A pair against the stored FAQ pair.
        ZERO LLM COST — all comparisons are local.
        """
        result = TestResult(
            question_number=ground_truth.question_number,
            question=ground_truth.question,
            original_answer=ground_truth.answer,
            stored_answer="",
        )

        try:
            start_time = time.time()

            # Find the stored pair
            stored_pair, match_type, q_sim = self._find_stored_pair(ground_truth)
            result.match_type = match_type
            result.question_similarity = q_sim

            if stored_pair is None:
                result.status = "NOT_FOUND"
                result.failure_reason = f"No matching FAQ pair found in stored index for {ground_truth.question_number}"
                result.comparison_time_ms = (time.time() - start_time) * 1000
                return result

            result.stored_answer = stored_pair.answer
            result.stored_question = stored_pair.question

            # Calculate similarity metrics (all local, no API calls)
            result.cosine_similarity = self._cosine_similarity(
                ground_truth.answer, stored_pair.answer
            )
            result.token_overlap_score = self._token_overlap(
                ground_truth.answer, stored_pair.answer
            )
            result.key_phrase_coverage = self._key_phrase_coverage(
                ground_truth.answer, stored_pair.answer
            )
            result.answer_completeness = self._answer_completeness(
                ground_truth.answer, stored_pair.answer
            )

            # Composite score
            result.composite_score = (
                self.COSINE_WEIGHT * result.cosine_similarity +
                self.TOKEN_WEIGHT * result.token_overlap_score +
                self.KEYPHRASE_WEIGHT * result.key_phrase_coverage +
                self.COMPLETENESS_WEIGHT * result.answer_completeness
            )

            # Determine pass/fail
            if result.composite_score >= self.PASS_THRESHOLD:
                result.status = "PASS"
            elif result.composite_score >= self.WARNING_THRESHOLD:
                result.status = "WARNING"
                result.failure_reason = f"Composite score {result.composite_score:.2f} below PASS threshold {self.PASS_THRESHOLD}"
            else:
                result.status = "FAIL"
                result.failure_reason = f"Composite score {result.composite_score:.2f} below threshold"

            # Additional checks for obvious problems
            if not stored_pair.answer or len(stored_pair.answer.strip()) < 10:
                result.status = "FAIL"
                result.failure_reason = "Stored answer is empty or near-empty"
            elif result.answer_completeness < 0.30:
                if result.status == "PASS":
                    result.status = "WARNING"
                result.failure_reason = f"Stored answer is significantly shorter than original ({result.answer_completeness:.0%} completeness)"

            result.comparison_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            result.status = "ERROR"
            result.failure_reason = f"Exception: {str(e)}"
            logger.error(f"[FAQ_TEST] Error testing {ground_truth.question_number}: {e}")

        return result

    def run_all_tests(
        self,
        ground_truth_pairs: List[GroundTruthQA],
        sample_count: int = 0,
        random_seed: Optional[int] = None,
        progress_callback=None
    ) -> TestReport:
        """
        Run FAQ comparison tests. ZERO LLM COST.

        Args:
            ground_truth_pairs: List of ground-truth Q&A pairs from PDF
            sample_count: Number of random questions to test (0 = all)
            random_seed: Seed for random sampling (for reproducibility)
            progress_callback: Optional callback(current, total, result) for progress
        """
        total_ground_truth = len(ground_truth_pairs)

        # Random sampling
        if sample_count > 0 and sample_count < total_ground_truth:
            if random_seed is not None:
                random.seed(random_seed)
            test_pairs = random.sample(ground_truth_pairs, sample_count)
            sampling_method = f"random_{sample_count}"
        else:
            test_pairs = ground_truth_pairs
            sampling_method = "all"

        report = TestReport(
            report_id=hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_file=ground_truth_pairs[0].source_file if ground_truth_pairs else "unknown",
            test_mode="zero_cost_local",
            sampling_method=sampling_method,
            total_ground_truth=total_ground_truth,
            total_tested=len(test_pairs),
            total_stored=self.get_stored_pair_count(),
        )

        start_time = time.time()

        for idx, gt in enumerate(test_pairs):
            result = self.run_single_test(gt)
            report.results.append(result)

            # Update counters
            if result.status == "PASS":
                report.passed += 1
            elif result.status == "FAIL":
                report.failed += 1
            elif result.status == "WARNING":
                report.warnings += 1
            elif result.status == "NOT_FOUND":
                report.not_found += 1
            elif result.status == "ERROR":
                report.errors += 1

            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, len(test_pairs), result)

        report.total_test_duration_s = time.time() - start_time

        # Calculate averages (exclude NOT_FOUND and ERROR)
        valid_results = [r for r in report.results if r.status in ("PASS", "FAIL", "WARNING")]
        if valid_results:
            report.avg_cosine_similarity = float(np.mean([r.cosine_similarity for r in valid_results]))
            report.avg_token_overlap = float(np.mean([r.token_overlap_score for r in valid_results]))
            report.avg_key_phrase_coverage = float(np.mean([r.key_phrase_coverage for r in valid_results]))
            report.avg_answer_completeness = float(np.mean([r.answer_completeness for r in valid_results]))
            report.avg_composite_score = float(np.mean([r.composite_score for r in valid_results]))

        return report

    # ─────────────────────────────────────────────────────
    # SIMILARITY METRICS (ALL LOCAL — ZERO API COST)
    # ─────────────────────────────────────────────────────

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate cosine similarity using local embedding model. No API calls."""
        if not text_a or not text_b or not self.embed_model:
            return 0.0

        try:
            text_a_trunc = text_a[:2000]
            text_b_trunc = text_b[:2000]

            emb_a = self.embed_model.encode([text_a_trunc])[0]
            emb_b = self.embed_model.encode([text_b_trunc])[0]

            dot = np.dot(emb_a, emb_b)
            norm = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)

            if norm == 0:
                return 0.0

            return float(max(0.0, min(1.0, dot / norm)))
        except Exception as e:
            logger.error(f"[FAQ_TEST] Cosine similarity error: {e}")
            return 0.0

    def _token_overlap(self, original: str, stored: str) -> float:
        """
        Calculate token overlap score (recall-based) between original and stored answer.
        Focuses on meaningful tokens (removes stopwords).
        """
        if not original or not stored:
            return 0.0

        def tokenize(text):
            tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
            stopwords = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                'as', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'between', 'out', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'not',
                'so', 'yet', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too',
                'very', 'just', 'because', 'if', 'when', 'where', 'how',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                'those', 'it', 'its', 'you', 'your', 'we', 'our', 'they',
                'their', 'he', 'she', 'him', 'her', 'his', 'my', 'me',
            }
            return set(t for t in tokens if t not in stopwords and len(t) > 1)

        original_tokens = tokenize(original)
        stored_tokens = tokenize(stored)

        if not original_tokens:
            return 0.0

        overlap = original_tokens & stored_tokens
        recall = len(overlap) / len(original_tokens)

        return float(recall)

    def _key_phrase_coverage(self, original: str, stored: str) -> float:
        """
        Extract key phrases from original answer and check coverage in stored answer.
        Key phrases: numbers/amounts, proper nouns, banking terms.
        """
        if not original or not stored:
            return 0.0

        stored_lower = stored.lower()
        key_phrases = []

        # 1. Numbers and amounts (AED 10,000, 5%, etc.)
        numbers = re.findall(r'(?:AED\s*)?[\d,]+(?:\.\d+)?(?:\s*%)?', original)
        for num in numbers:
            num_clean = num.strip()
            if len(num_clean) > 0 and any(c.isdigit() for c in num_clean):
                key_phrases.append(num_clean)

        # 2. Proper nouns (capitalized words > 3 chars, excluding sentence starters)
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', original)
        sentence_starters = {
            'This', 'That', 'These', 'Those', 'Your', 'Please', 'However',
            'Also', 'Both', 'Each', 'Every', 'Some', 'Once', 'After', 'Before',
            'During', 'When', 'Where', 'What', 'Which', 'Enter', 'Create',
            'Follow', 'Click', 'Visit', 'Call', 'Contact', 'Make', 'Note'
        }
        for pn in proper_nouns:
            if pn not in sentence_starters:
                key_phrases.append(pn)

        # 3. Banking/product terms
        banking_terms = re.findall(
            r'\b(?:credit card|debit card|savings account|current account|'
            r'cheque book|online banking|mobile banking|fund transfer|'
            r'standing order|direct debit|fixed deposit|'
            r'Fitness First|sMiles|Solitaire|Platinum|Mashreq|'
            r'Emirates ID|trade license|CVV|OTP|PIN)\b',
            original, re.IGNORECASE
        )
        key_phrases.extend(banking_terms)

        if not key_phrases:
            return self._token_overlap(original, stored)

        found = 0
        for phrase in key_phrases:
            if phrase.lower() in stored_lower:
                found += 1

        return found / len(key_phrases) if key_phrases else 0.0

    def _answer_completeness(self, original: str, stored: str) -> float:
        """
        Calculate answer completeness as a ratio of stored answer length
        to original answer length. Capped at 1.0.

        A score of 1.0 means the stored answer is at least as long as the original.
        A score of 0.5 means the stored answer is half the length (likely truncated).
        """
        if not original or not stored:
            return 0.0

        original_len = len(original.strip())
        stored_len = len(stored.strip())

        if original_len == 0:
            return 0.0

        ratio = stored_len / original_len
        return float(min(1.0, ratio))

    # ─────────────────────────────────────────────────────
    # REPORT GENERATION
    # ─────────────────────────────────────────────────────

    def generate_csv_report(self, report: TestReport) -> str:
        """Generate a CSV report string from the test report."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'Q#', 'Question', 'Status', 'Match Type', 'Composite Score',
            'Cosine Similarity', 'Token Overlap', 'Key Phrase Coverage',
            'Answer Completeness', 'Question Similarity',
            'Comparison Time (ms)', 'Failure Reason',
            'Original Answer (first 300 chars)', 'Stored Answer (first 300 chars)'
        ])

        # Data rows
        for r in report.results:
            writer.writerow([
                r.question_number,
                r.question[:100],
                r.status,
                r.match_type,
                f"{r.composite_score:.3f}",
                f"{r.cosine_similarity:.3f}",
                f"{r.token_overlap_score:.3f}",
                f"{r.key_phrase_coverage:.3f}",
                f"{r.answer_completeness:.3f}",
                f"{r.question_similarity:.3f}",
                f"{r.comparison_time_ms:.1f}",
                r.failure_reason,
                r.original_answer[:300].replace('\n', ' '),
                r.stored_answer[:300].replace('\n', ' ')
            ])

        # Summary
        writer.writerow([])
        writer.writerow(['SUMMARY'])
        writer.writerow(['Total Ground Truth Q&A', report.total_ground_truth])
        writer.writerow(['Total Tested', report.total_tested])
        writer.writerow(['Total Stored FAQ Pairs', report.total_stored])
        writer.writerow(['Sampling Method', report.sampling_method])
        writer.writerow(['Passed', report.passed])
        writer.writerow(['Failed', report.failed])
        writer.writerow(['Warnings', report.warnings])
        writer.writerow(['Not Found', report.not_found])
        writer.writerow(['Errors', report.errors])
        writer.writerow(['Pass Rate', f"{(report.passed / report.total_tested * 100):.1f}%" if report.total_tested > 0 else "N/A"])
        writer.writerow(['Avg Composite Score', f"{report.avg_composite_score:.3f}"])
        writer.writerow(['Avg Cosine Similarity', f"{report.avg_cosine_similarity:.3f}"])
        writer.writerow(['Avg Token Overlap', f"{report.avg_token_overlap:.3f}"])
        writer.writerow(['Avg Key Phrase Coverage', f"{report.avg_key_phrase_coverage:.3f}"])
        writer.writerow(['Avg Answer Completeness', f"{report.avg_answer_completeness:.3f}"])
        writer.writerow(['Total Test Duration (s)', f"{report.total_test_duration_s:.1f}"])
        writer.writerow(['Test Mode', report.test_mode])
        writer.writerow(['LLM Token Cost', '$0.00 (zero cost)'])
        writer.writerow(['Report ID', report.report_id])
        writer.writerow(['Timestamp', report.timestamp])

        return output.getvalue()

    def generate_json_report(self, report: TestReport) -> str:
        """Generate a JSON report string from the test report."""
        report_dict = {
            'report_id': report.report_id,
            'timestamp': report.timestamp,
            'source_file': report.source_file,
            'test_mode': report.test_mode,
            'llm_token_cost': '$0.00 (zero cost)',
            'sampling_method': report.sampling_method,
            'summary': {
                'total_ground_truth': report.total_ground_truth,
                'total_tested': report.total_tested,
                'total_stored_faq_pairs': report.total_stored,
                'passed': report.passed,
                'failed': report.failed,
                'warnings': report.warnings,
                'not_found': report.not_found,
                'errors': report.errors,
                'pass_rate': f"{(report.passed / report.total_tested * 100):.1f}%" if report.total_tested > 0 else "N/A",
                'avg_composite_score': round(report.avg_composite_score, 3),
                'avg_cosine_similarity': round(report.avg_cosine_similarity, 3),
                'avg_token_overlap': round(report.avg_token_overlap, 3),
                'avg_key_phrase_coverage': round(report.avg_key_phrase_coverage, 3),
                'avg_answer_completeness': round(report.avg_answer_completeness, 3),
                'total_test_duration_s': round(report.total_test_duration_s, 1),
            },
            'results': []
        }

        for r in report.results:
            report_dict['results'].append({
                'question_number': r.question_number,
                'question': r.question,
                'status': r.status,
                'match_type': r.match_type,
                'composite_score': round(r.composite_score, 3),
                'cosine_similarity': round(r.cosine_similarity, 3),
                'token_overlap_score': round(r.token_overlap_score, 3),
                'key_phrase_coverage': round(r.key_phrase_coverage, 3),
                'answer_completeness': round(r.answer_completeness, 3),
                'question_similarity': round(r.question_similarity, 3),
                'comparison_time_ms': round(r.comparison_time_ms, 1),
                'failure_reason': r.failure_reason,
                'original_answer': r.original_answer,
                'stored_answer': r.stored_answer,
            })

        return json.dumps(report_dict, indent=2, ensure_ascii=False)

