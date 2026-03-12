"""
FAQ Smart Router — 3-Tier Query Routing via Embedding Similarity
================================================================
Embeds the user's FULL query and compares it against ALL stored FAQ question
embeddings using cosine similarity. Routes to one of three tiers:

  Tier 1 (EXACT):   similarity >= 0.85
    → Return the exact FAQ answer instantly. Zero LLM calls. ~50ms.

  Tier 2 (FUZZY):   similarity 0.60–0.85
    → The query is related but rephrased/ambiguous. Use ONE lightweight
      LLM call (gpt-4.1-nano) to adapt the closest FAQ answer to the query.
      ~1-2 seconds instead of 10-20 seconds.

  Tier 3 (NOVEL):   similarity < 0.60
    → Not an FAQ question. Fall through to the full 7-layer RAG pipeline.

Architecture:
  - During document upload: Detects Q&A pairs via regex, stores each as an
    atomic {question, answer, embedding} unit.
  - During query: Embeds the full query, computes cosine similarity against
    ALL stored FAQ question embeddings in a single vectorized operation.
  - Returns top-N matches for context (not just top-1) to handle multi-part
    questions that span multiple FAQ entries.

Feature Toggle: faq_exact_match_enabled (default: True)
"""
import re
import hashlib
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FAQPair:
    """A single FAQ question-answer pair."""
    faq_id: str
    question: str
    answer: str
    question_number: str  # e.g., "Q5", "Q42"
    source_file: str
    page: int = 0
    section: str = ""
    question_embedding: Optional[np.ndarray] = None


@dataclass
class FAQMatchResult:
    """Result of FAQ smart routing lookup."""
    matched: bool
    tier: str = "none"           # "exact", "fuzzy", "novel"
    faq_pair: Optional[FAQPair] = None
    similarity: float = 0.0
    match_type: str = "none"     # backward compat: "exact", "near_exact", "fuzzy", "none"
    top_matches: List[Tuple[float, 'FAQPair']] = field(default_factory=list)
    adapted_answer: Optional[str] = None  # For fuzzy tier: LLM-adapted answer


class FAQExactMatchEngine:
    """
    FAQ-aware extraction and 3-tier smart routing engine.

    Extraction Phase:
        - Parses FAQ documents to detect Q&A pairs using regex patterns
        - Stores each pair as an atomic unit with its own embedding

    Query Phase (3-Tier Routing):
        - Embeds the FULL user query
        - Computes cosine similarity against ALL stored FAQ question embeddings
        - Tier 1 (sim >= exact_threshold):  Return exact answer, 0 LLM calls
        - Tier 2 (sim >= fuzzy_threshold):  Adapt answer with 1 nano LLM call
        - Tier 3 (sim < fuzzy_threshold):   Fall through to full pipeline
    """

    # Regex patterns to detect FAQ question markers
    FAQ_QUESTION_PATTERNS = [
        r'(?:^|\n)\s*Q[\.\s]*(\d+)\s*[\.\:\)]\s*',
        r'(?:^|\n)\s*Question\s+(\d+)\s*[\.\:\)]\s*',
    ]

    def __init__(self, embed_model, settings):
        """
        Args:
            embed_model: Sentence transformer model for computing embeddings
            settings: Application settings
        """
        self.embed_model = embed_model
        self.settings = settings
        self.faq_pairs: List[FAQPair] = []
        self.question_embeddings: Optional[np.ndarray] = None

        # 3-Tier thresholds from config
        self.exact_threshold = getattr(settings.rag, 'faq_exact_threshold', 0.85)
        self.fuzzy_threshold = getattr(settings.rag, 'faq_fuzzy_threshold', 0.60)
        self.fuzzy_model = getattr(settings.rag, 'faq_fuzzy_model', 'gpt-4.1-nano')

        # Backward compat alias
        self.near_exact_threshold = self.exact_threshold

        logger.info(
            f"[FAQ_SMART_ROUTER] Initialized — "
            f"Exact≥{self.exact_threshold}, Fuzzy≥{self.fuzzy_threshold}, "
            f"FuzzyModel={self.fuzzy_model}"
        )

    # ═══════════════════════════════════════════════════════════════
    # EXTRACTION PHASE — Detect and store FAQ Q&A pairs
    # ═══════════════════════════════════════════════════════════════

    def extract_faq_pairs(self, text: str, source_file: str, page: int = 0) -> List[FAQPair]:
        """
        Extract FAQ Q&A pairs from document text.

        Detects patterns like:
            Q5. How do I earn cashback on my SmartSaver Credit Card?
            You will save on all purchases made on the SmartSaver Credit Card...

        Returns list of extracted FAQPair objects with pre-computed embeddings.
        """
        pairs = []

        # Strategy 1: Q-number pattern (Q1. Q2. Q.5 etc.)
        q_pattern = re.compile(
            r'(?:^|\n)\s*Q[\.\s]*(\d+)\s*[\.\:\)]\s*',
            re.MULTILINE
        )
        matches = list(q_pattern.finditer(text))

        if matches:
            logger.info(f"[FAQ_SMART_ROUTER] Found {len(matches)} Q-markers in {source_file}")
            for i, match in enumerate(matches):
                q_num = match.group(1)
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                qa_block = text[start:end].strip()

                if not qa_block:
                    continue

                question, answer = self._split_question_answer(qa_block)

                if question and answer and len(answer) >= 20:
                    faq_id = hashlib.md5(
                        f"{source_file}_Q{q_num}_{question[:50]}".encode()
                    ).hexdigest()[:12]
                    pair = FAQPair(
                        faq_id=faq_id,
                        question=question.strip(),
                        answer=answer.strip(),
                        question_number=f"Q{q_num}",
                        source_file=source_file,
                        page=page,
                        section=f"Q{q_num}"
                    )
                    pairs.append(pair)

        # Strategy 2: "Question N:" pattern
        if not pairs:
            q_pattern2 = re.compile(
                r'(?:^|\n)\s*Question\s+(\d+)\s*[\.\:\)]\s*',
                re.MULTILINE | re.IGNORECASE
            )
            matches2 = list(q_pattern2.finditer(text))

            if matches2:
                logger.info(f"[FAQ_SMART_ROUTER] Found {len(matches2)} 'Question N' markers in {source_file}")
                for i, match in enumerate(matches2):
                    q_num = match.group(1)
                    start = match.end()
                    end = matches2[i + 1].start() if i + 1 < len(matches2) else len(text)
                    qa_block = text[start:end].strip()

                    if not qa_block:
                        continue

                    question, answer = self._split_question_answer(qa_block)

                    if question and answer and len(answer) >= 20:
                        faq_id = hashlib.md5(
                            f"{source_file}_Q{q_num}_{question[:50]}".encode()
                        ).hexdigest()[:12]
                        pair = FAQPair(
                            faq_id=faq_id,
                            question=question.strip(),
                            answer=answer.strip(),
                            question_number=f"Q{q_num}",
                            source_file=source_file,
                            page=page,
                            section=f"Q{q_num}"
                        )
                        pairs.append(pair)

        if pairs:
            logger.info(f"[FAQ_SMART_ROUTER] Extracted {len(pairs)} FAQ pairs from {source_file}")
            self._embed_pairs(pairs)
        else:
            logger.info(f"[FAQ_SMART_ROUTER] No FAQ patterns detected in {source_file}")

        return pairs

    def _split_question_answer(self, qa_block: str) -> Tuple[str, str]:
        """
        Split a Q&A block into question and answer parts.

        CRITICAL: Many FAQ questions are multi-part, containing multiple '?'
        marks (e.g., "How do I get X? Also, is Y applicable for Z as well?").
        We must find the correct split point — the last '?' that ends the
        question portion, after which the actual answer begins.

        Strategy:
        1. Find ALL '?' positions in the QA block
        2. For each '?' (from last to first), check if the text AFTER it
           looks like an answer (starts with a declarative/answer sentence)
           rather than another question continuation or a URL parameter
        3. Use the best split point found
        """
        # Find all '?' positions
        q_positions = [m.start() for m in re.finditer(r'\?', qa_block)]

        if q_positions:
            # Try each '?' from LAST to FIRST to find the best split
            best_split = None
            for q_pos in reversed(q_positions):
                # Skip '?' that are too far into the block (likely in answer text)
                if q_pos > 400:
                    continue

                # Skip '?' that appear inside URLs
                # Check 1: preceded by URL-typical characters
                if q_pos > 0 and qa_block[q_pos - 1] in ('/', '=', '&', '#'):
                    continue
                # Check 2: preceded by URL file extensions (.html? .php? .asp? .com?)
                pre_5 = qa_block[max(0, q_pos - 5):q_pos].lower()
                if re.search(r'\.(html|php|asp|aspx|jsp|com|net|org)$', pre_5):
                    continue
                # Check 3: URL pattern in broader context (up to 200 chars back)
                pre_context = qa_block[max(0, q_pos - 200):q_pos]
                # If there's a URL in the same line (no newline between URL and ?)
                url_match = re.search(r'https?://\S+$', pre_context)
                if url_match:
                    # This '?' is part of a URL query string
                    continue

                question_part = qa_block[:q_pos + 1].strip()
                answer_part = qa_block[q_pos + 1:].strip()

                if not question_part or not answer_part:
                    continue

                # Check if answer_part starts like an actual answer
                # (not another question word like "Also", "Is", "How", "What", etc.)
                # Answer-like starts: "You", "The", "This", "Yes", "No", "It",
                # "Cashback", "Total", digits, etc.
                answer_first_word = answer_part.lstrip('\n ').split()[0] if answer_part.strip() else ''

                # Question continuation words (these suggest the answer hasn't started yet)
                question_words = {
                    'also', 'is', 'are', 'how', 'what', 'which', 'where', 'when',
                    'why', 'who', 'can', 'could', 'will', 'would', 'do', 'does',
                    'did', 'if', 'shall', 'should', 'has', 'have', 'may', 'might'
                }

                first_word_lower = answer_first_word.lower().rstrip(',.;:')

                if first_word_lower not in question_words:
                    # This looks like an answer start
                    best_split = (question_part, answer_part)
                    break
                else:
                    # This '?' is followed by another question part — keep looking
                    # But save it as a fallback in case no better split is found
                    if best_split is None:
                        best_split = (question_part, answer_part)

            if best_split:
                question, raw_answer = best_split
                answer = self._extract_direct_answer(raw_answer)
                if question and answer:
                    return question, answer

        # Strategy 2: Split on first newline if first line is short
        lines = qa_block.split('\n', 1)
        if len(lines) == 2 and len(lines[0].strip()) < 200:
            question = lines[0].strip()
            raw_answer = lines[1].strip()
            if question and raw_answer:
                answer = self._extract_direct_answer(raw_answer)
                return question, answer

        # Strategy 3: Split on first sentence boundary
        sent_match = re.match(r'(.+?[.!?])\s+(.+)', qa_block, re.DOTALL)
        if sent_match:
            raw_answer = sent_match.group(2).strip()
            return sent_match.group(1).strip(), self._extract_direct_answer(raw_answer)

        # Fallback: first 100 chars as question, rest as answer
        if len(qa_block) > 100:
            return qa_block[:100], self._extract_direct_answer(qa_block[100:])

        return qa_block, ""

    def _extract_direct_answer(self, raw_answer: str) -> str:
        """
        Return the FULL answer text between Q-markers.

        The raw_answer is already the text between two consecutive Q-markers
        (e.g., between Q25 and Q26), so it IS the complete answer.

        We only do minimal cleanup:
        1. Strip leading/trailing whitespace
        2. Normalize excessive whitespace within the text
        3. No character limit — FAQ answers must be returned in full
        """
        if not raw_answer:
            return raw_answer

        # Minimal cleanup: normalize excessive internal whitespace
        # Replace 3+ newlines with 2 (paragraph break), but preserve structure
        cleaned = re.sub(r'\n{3,}', '\n\n', raw_answer)

        # Collapse runs of spaces (but not newlines) into single space
        cleaned = re.sub(r'[^\S\n]+', ' ', cleaned)

        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()

        return cleaned

    def _embed_pairs(self, pairs: List[FAQPair]):
        """Compute and store embeddings for FAQ question texts (batch operation)."""
        questions = [p.question for p in pairs]
        embeddings = self.embed_model.encode(
            questions, convert_to_numpy=True, show_progress_bar=False
        )
        for i, pair in enumerate(pairs):
            pair.question_embedding = embeddings[i]

    def add_pairs(self, pairs: List[FAQPair]):
        """Add extracted FAQ pairs to the engine's store and rebuild the embedding index."""
        self.faq_pairs.extend(pairs)
        self._rebuild_index()
        logger.info(f"[FAQ_SMART_ROUTER] Total FAQ pairs stored: {len(self.faq_pairs)}")

    def _rebuild_index(self):
        """Rebuild the combined question embedding matrix for fast vectorized similarity.
        Pre-normalizes embeddings so lookup only needs a single dot product (no per-query normalization).
        """
        if not self.faq_pairs:
            self.question_embeddings = None
            self._normalized_embeddings = None
            return

        embeddings = []
        for pair in self.faq_pairs:
            if pair.question_embedding is not None:
                embeddings.append(pair.question_embedding)
            else:
                emb = self.embed_model.encode(pair.question, convert_to_numpy=True)
                pair.question_embedding = emb
                embeddings.append(emb)

        self.question_embeddings = np.vstack(embeddings)
        # Pre-normalize once — saves ~1ms per query by avoiding repeated normalization
        norms = np.linalg.norm(self.question_embeddings, axis=1, keepdims=True) + 1e-10
        self._normalized_embeddings = self.question_embeddings / norms
        print(f"[FAQ_SMART_ROUTER] Index rebuilt: {len(self.faq_pairs)} pairs, embeddings pre-normalized")

    # ═══════════════════════════════════════════════════════════════
    # QUERY PHASE — 3-Tier Smart Routing
    # ═══════════════════════════════════════════════════════════════

    def lookup(self, query: str) -> FAQMatchResult:
        """
        Embed the full query and compare against ALL stored FAQ question embeddings.

        Returns FAQMatchResult with:
        - tier: "exact" (≥0.85), "fuzzy" (0.60-0.85), or "novel" (<0.60)
        - matched: True for exact and fuzzy tiers
        - top_matches: Top-3 closest FAQ pairs with similarity scores
        - The best matching FAQ pair and similarity score

        Performance: ~20-50ms total (embedding ~15-30ms + similarity ~0.1ms)
        """
        import time as _t
        if not self.faq_pairs or self._normalized_embeddings is None:
            return FAQMatchResult(matched=False, tier="novel")

        # Step 1: Embed the FULL user query (~15-30ms after model warm-up)
        t0 = _t.time()
        query_embedding = self.embed_model.encode(query, convert_to_numpy=True)
        embed_ms = (_t.time() - t0) * 1000

        # Step 2: Vectorized cosine similarity — single matrix dot product (~0.1ms)
        # FAQ embeddings are PRE-NORMALIZED during _rebuild_index(), so we only
        # normalize the query vector here.
        t1 = _t.time()
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        similarities = np.dot(self._normalized_embeddings, query_norm)
        sim_ms = (_t.time() - t1) * 1000

        print(f"[FAQ_ROUTER_PERF] embed={embed_ms:.1f}ms, similarity={sim_ms:.2f}ms, total={embed_ms+sim_ms:.1f}ms")

        # Get top-3 matches for context
        top_indices = np.argsort(similarities)[::-1][:3]
        top_matches = [
            (float(similarities[idx]), self.faq_pairs[idx])
            for idx in top_indices
        ]

        best_idx = top_indices[0]
        best_sim = float(similarities[best_idx])
        best_pair = self.faq_pairs[best_idx]

        logger.info(
            f"[FAQ_SMART_ROUTER] Query: '{query[:60]}...' → "
            f"Best: {best_sim:.4f} ({best_pair.question_number}), "
            f"Top3: [{', '.join(f'{s:.3f}' for s, _ in top_matches)}]"
        )

        # ── TIER 1: EXACT MATCH (sim >= 0.85) ──
        # The question is clearly asking what a specific FAQ answers.
        # Return the exact FAQ answer. Zero LLM calls.
        if best_sim >= self.exact_threshold:
            logger.info(f"[FAQ_SMART_ROUTER] TIER 1 (EXACT) — {best_pair.question_number}")
            return FAQMatchResult(
                matched=True,
                tier="exact",
                faq_pair=best_pair,
                similarity=best_sim,
                match_type="exact",
                top_matches=top_matches,
            )

        # ── TIER 2: FUZZY MATCH (sim >= 0.60) ──
        # The question is related but rephrased, partial, or slightly ambiguous.
        # We have a good FAQ candidate — adapt it with 1 lightweight LLM call.
        elif best_sim >= self.fuzzy_threshold:
            logger.info(f"[FAQ_SMART_ROUTER] TIER 2 (FUZZY) — {best_pair.question_number}")
            return FAQMatchResult(
                matched=True,
                tier="fuzzy",
                faq_pair=best_pair,
                similarity=best_sim,
                match_type="fuzzy",
                top_matches=top_matches,
            )

        # ── TIER 3: NOVEL QUERY (sim < 0.60) ──
        # Not an FAQ question. Fall through to the full 7-layer pipeline.
        else:
            logger.info(f"[FAQ_SMART_ROUTER] TIER 3 (NOVEL) — best sim={best_sim:.4f}")
            return FAQMatchResult(
                matched=False,
                tier="novel",
                faq_pair=best_pair,  # Include best match for debugging
                similarity=best_sim,
                match_type="none",
                top_matches=top_matches,
            )

    def adapt_answer_for_fuzzy_match(
        self, query: str, faq_result: FAQMatchResult, llm_client
    ) -> str:
        """
        For Tier 2 (fuzzy) matches: Use ONE lightweight LLM call to adapt
        the closest FAQ answer to fit the user's actual query.

        This uses gpt-4.1-nano (fastest, cheapest model) with a tight prompt
        that provides the FAQ Q&A as context and asks the model to answer
        the user's specific question using ONLY that FAQ content.

        Args:
            query: The user's original query
            faq_result: The FAQMatchResult from lookup()
            llm_client: OpenAI client instance

        Returns:
            Adapted answer string
        """
        pair = faq_result.faq_pair
        if not pair:
            return "I couldn't find a matching answer in the FAQ."

        # Build context from top matches (not just top-1)
        context_parts = []
        for sim, match_pair in faq_result.top_matches[:3]:
            if sim >= self.fuzzy_threshold:
                context_parts.append(
                    f"[{match_pair.question_number}] Q: {match_pair.question}\n"
                    f"A: {match_pair.answer}"
                )

        faq_context = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"You are a banking FAQ assistant. A customer asked a question that is similar "
            f"to existing FAQ entries. Answer their question using ONLY the FAQ content below.\n\n"
            f"RULES:\n"
            f"- Use ONLY information from the FAQ entries provided\n"
            f"- If the FAQ doesn't cover their exact question, say what the FAQ does cover\n"
            f"- Do NOT invent information not in the FAQ\n"
            f"- CRITICAL: You MUST reproduce the FULL and COMPLETE answer text from the FAQ entries — do NOT summarize, shorten, or paraphrase\n"
            f"- Include ALL details, conditions, amounts, fees, and specifics exactly as stated in the FAQ\n"
            f"- If the customer's question maps to a specific FAQ entry, reproduce that entry's ENTIRE answer verbatim\n"
            f"- Include the source FAQ number(s) in parentheses at the end, e.g. (Q29)\n\n"
            f"FAQ ENTRIES:\n{faq_context}\n\n"
            f"CUSTOMER QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        try:
            response = llm_client.chat.completions.create(
                model=self.fuzzy_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            adapted = response.choices[0].message.content.strip()
            logger.info(f"[FAQ_SMART_ROUTER] Fuzzy adaptation complete ({len(adapted)} chars)")
            return adapted
        except Exception as e:
            logger.error(f"[FAQ_SMART_ROUTER] Fuzzy adaptation LLM error: {e}")
            # Fallback: return the raw FAQ answer with a note
            return (
                f"{pair.answer}\n\n"
                f"(Source: {pair.source_file} | {pair.question_number})"
            )

    # ═══════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════

    def get_faq_chunks(self) -> List[Dict]:
        """
        Convert FAQ pairs into chunk-format dicts for indexing alongside regular chunks.
        Each FAQ pair becomes its own atomic chunk with the question as context prefix.
        """
        chunks = []
        for pair in self.faq_pairs:
            chunk_text = (
                f"{pair.question_number}. {pair.question}\n\n"
                f"{pair.answer}"
            )
            chunks.append({
                "chunk_id": f"faq_{pair.faq_id}",
                "text": chunk_text,
                "original_text": chunk_text,
                "context_prefix": f"FAQ {pair.question_number} from {pair.source_file}",
                "source": pair.source_file,
                "page": pair.page,
                "section": pair.question_number,
                "chunk_type": "faq",
                "related_tables": [],
                "related_entities": [],
                "confidence": 0.98
            })
        return chunks

    def get_stats(self) -> Dict:
        """Return statistics about stored FAQ pairs."""
        sources = set(p.source_file for p in self.faq_pairs)
        return {
            "total_pairs": len(self.faq_pairs),
            "sources": list(sources),
            "source_count": len(sources),
            "exact_threshold": self.exact_threshold,
            "fuzzy_threshold": self.fuzzy_threshold,
            "fuzzy_model": self.fuzzy_model,
        }

    def clear(self):
        """Clear all stored FAQ pairs."""
        self.faq_pairs.clear()
        self.question_embeddings = None

    def remove_by_source(self, source: str) -> int:
        """Remove all FAQ pairs for a given document source.
        
        Rebuilds the embedding index after removal.
        
        Args:
            source: The document source/filename to remove
        
        Returns:
            Number of FAQ pairs removed
        """
        original_count = len(self.faq_pairs)
        self.faq_pairs = [p for p in self.faq_pairs if p.source_file != source]
        removed_count = original_count - len(self.faq_pairs)

        if removed_count > 0:
            self._rebuild_index()
            print(f"[FAQ_SMART_ROUTER] Removed {removed_count} FAQ pairs for '{source}', {len(self.faq_pairs)} remaining")

        return removed_count
