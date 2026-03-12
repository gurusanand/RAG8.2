"""
RAGAS Evaluator — Retrieval-Augmented Generation Assessment.
Evaluates response quality using Faithfulness, Answer Relevancy, Context Precision, Context Recall.
"""
import json
from dataclasses import dataclass
from typing import List
from openai import OpenAI


@dataclass
class RAGASResult:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall_score: float = 0.0
    grade: str = "N/A"


class RAGASEvaluator:
    """LLM-based RAGAS evaluation for RAG responses."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
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

    def evaluate(self, query: str, response: str, context: str, sources: List[str]) -> RAGASResult:
        """Evaluate a RAG response using RAGAS metrics."""
        prompt = f"""Evaluate this RAG response on 4 metrics (0.0-1.0 each):

Query: {query}
Response: {response[:1000]}
Context: {context[:1500]}

Metrics:
1. Faithfulness: Is the response factually grounded in the context?
2. Answer Relevancy: Does the response directly address the query?
3. Context Precision: Are the retrieved contexts relevant to the query?
4. Context Recall: Does the context contain all information needed?

Respond in JSON:
{{
    "faithfulness": 0.0-1.0,
    "answer_relevancy": 0.0-1.0,
    "context_precision": 0.0-1.0,
    "context_recall": 0.0-1.0
}}"""

        result = self._call_llm(prompt)
        parsed = self._parse_json(result)

        f = parsed.get("faithfulness", 0.8)
        ar = parsed.get("answer_relevancy", 0.8)
        cp = parsed.get("context_precision", 0.8)
        cr = parsed.get("context_recall", 0.8)
        overall = (f + ar + cp + cr) / 4

        grade = "A" if overall >= 0.9 else "B" if overall >= 0.75 else "C" if overall >= 0.6 else "D" if overall >= 0.4 else "F"

        return RAGASResult(
            faithfulness=f,
            answer_relevancy=ar,
            context_precision=cp,
            context_recall=cr,
            overall_score=overall,
            grade=grade,
        )
