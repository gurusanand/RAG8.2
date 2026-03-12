"""
Speculative RAG Engine — Parallel draft generation + verification.
Generates multiple answer drafts from different chunk subsets and selects the best.
"""
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict
from openai import OpenAI


@dataclass
class DraftResult:
    draft_id: int
    chunk_subset: List[Dict]
    answer_text: str
    generation_time_ms: float = 0.0


@dataclass
class VerificationResult:
    draft_id: int
    faithfulness_score: float = 0.0
    completeness_score: float = 0.0
    relevance_score: float = 0.0
    overall_score: float = 0.0
    is_selected: bool = False
    reasoning: str = ""


@dataclass
class SpeculativeResult:
    total_drafts: int = 0
    selected_draft_id: int = 0
    improvement_over_first: float = 0.0
    drafts: List[DraftResult] = field(default_factory=list)
    verifications: List[VerificationResult] = field(default_factory=list)
    drafting_time_ms: float = 0.0
    verification_time_ms: float = 0.0


class SpeculativeRAGEngine:
    """Generates multiple answer drafts and selects the best one."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings
        self.num_drafts = settings.rag.speculative_rag_num_drafts

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Slightly higher for diversity
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except:
                    pass
            return {}

    def generate_and_verify(self, query: str, chunks: List[Dict]) -> SpeculativeResult:
        """Generate multiple drafts from different chunk subsets and verify."""
        if not chunks:
            return SpeculativeResult()

        drafts = []
        draft_start = time.time()

        # Generate drafts from different chunk subsets
        for i in range(min(self.num_drafts, len(chunks))):
            # Each draft uses a different subset of chunks
            start_idx = i
            subset = chunks[start_idx::self.num_drafts]  # Interleaved subsets
            if not subset:
                subset = chunks[:3]

            context = "\n\n".join([c.get("text", "")[:500] for c in subset])
            gen_start = time.time()

            prompt = f"""Answer this banking query using the provided context.
Be specific and cite sources when possible.

Query: {query}
Context: {context}

Answer:"""

            answer = self._call_llm(prompt)
            gen_time = (time.time() - gen_start) * 1000

            drafts.append(DraftResult(
                draft_id=i + 1,
                chunk_subset=subset,
                answer_text=answer,
                generation_time_ms=gen_time,
            ))

        drafting_time = (time.time() - draft_start) * 1000

        # Verify each draft
        verify_start = time.time()
        verifications = []
        full_context = "\n\n".join([c.get("text", "")[:300] for c in chunks])

        for draft in drafts:
            prompt = f"""Evaluate this answer draft for a banking query.

Query: {query}
Draft Answer: {draft.answer_text[:1000]}
Full Context: {full_context[:1500]}

Score on 3 dimensions (0.0-1.0):
{{
    "faithfulness_score": "Is the answer grounded in the context?",
    "completeness_score": "Does it fully answer the query?",
    "relevance_score": "Is it directly relevant to the query?",
    "reasoning": "Brief explanation"
}}"""

            result = self._call_llm(prompt)
            parsed = self._parse_json(result)

            f = parsed.get("faithfulness_score", 0.7)
            c = parsed.get("completeness_score", 0.7)
            r = parsed.get("relevance_score", 0.7)
            overall = (f + c + r) / 3

            verifications.append(VerificationResult(
                draft_id=draft.draft_id,
                faithfulness_score=f if isinstance(f, (int, float)) else 0.7,
                completeness_score=c if isinstance(c, (int, float)) else 0.7,
                relevance_score=r if isinstance(r, (int, float)) else 0.7,
                overall_score=overall if isinstance(overall, (int, float)) else 0.7,
                reasoning=parsed.get("reasoning", ""),
            ))

        verification_time = (time.time() - verify_start) * 1000

        # Select best draft
        if verifications:
            best_idx = max(range(len(verifications)), key=lambda i: verifications[i].overall_score)
            verifications[best_idx].is_selected = True
            selected_id = verifications[best_idx].draft_id

            first_score = verifications[0].overall_score if verifications[0].overall_score else 0.7
            best_score = verifications[best_idx].overall_score if verifications[best_idx].overall_score else 0.7
            improvement = ((best_score - first_score) / (first_score + 1e-10)) * 100
        else:
            selected_id = 1
            improvement = 0.0

        return SpeculativeResult(
            total_drafts=len(drafts),
            selected_draft_id=selected_id,
            improvement_over_first=improvement,
            drafts=drafts,
            verifications=verifications,
            drafting_time_ms=drafting_time,
            verification_time_ms=verification_time,
        )
