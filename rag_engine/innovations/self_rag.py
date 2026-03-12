"""
Self-RAG Engine — Self-reflective retrieval with correction loops.
Evaluates its own response and corrects if needed.
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional
from openai import OpenAI


@dataclass
class SelfRAGResult:
    iterations: int = 0
    retrieval_needed: bool = False
    corrections_made: int = 0
    final_confidence: float = 0.8
    response_improved: bool = False
    corrected_response: Optional[str] = None
    reflection_log: List[dict] = field(default_factory=list)


class SelfRAGEngine:
    """Self-reflective RAG with iterative correction."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings
        self.max_iterations = settings.rag.self_rag_max_iterations

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "{}"

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

    def reflect_and_correct(
        self, query: str, response: str, context: str, sources: List[str]
    ) -> SelfRAGResult:
        """Reflect on the response and correct if needed."""
        reflection_log = []
        current_response = response
        corrections = 0

        for i in range(self.max_iterations):
            prompt = f"""You are a self-reflective AI evaluating a banking RAG response.

Query: {query}
Current Response: {current_response[:1500]}
Source Context: {context[:1500]}

Evaluate:
1. Is the response faithful to the context?
2. Does it fully answer the query?
3. Are there any inaccuracies or missing information?

Respond in JSON:
{{
    "needs_correction": true|false,
    "confidence": 0.0-1.0,
    "issues": ["list of issues found"],
    "corrected_response": "improved response text (only if needs_correction is true)",
    "reasoning": "explanation"
}}"""

            result = self._call_llm(prompt)
            parsed = self._parse_json(result)

            reflection_log.append({
                "iteration": i + 1,
                "needs_correction": parsed.get("needs_correction", False),
                "confidence": parsed.get("confidence", 0.8),
                "issues": parsed.get("issues", []),
            })

            if not parsed.get("needs_correction", False):
                return SelfRAGResult(
                    iterations=i + 1,
                    retrieval_needed=False,
                    corrections_made=corrections,
                    final_confidence=parsed.get("confidence", 0.85),
                    response_improved=corrections > 0,
                    corrected_response=current_response if corrections > 0 else None,
                    reflection_log=reflection_log,
                )

            corrected = parsed.get("corrected_response", "")
            if corrected and len(corrected) > 50:
                current_response = corrected
                corrections += 1

        return SelfRAGResult(
            iterations=self.max_iterations,
            retrieval_needed=False,
            corrections_made=corrections,
            final_confidence=0.7,
            response_improved=corrections > 0,
            corrected_response=current_response if corrections > 0 else None,
            reflection_log=reflection_log,
        )
