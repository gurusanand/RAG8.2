"""
Query Decomposition Engine — Breaks complex multi-part queries into sub-queries.
"""
import json
from dataclasses import dataclass, field
from typing import List
from openai import OpenAI


@dataclass
class SubQuery:
    query: str
    intent: str = ""
    priority: int = 1


@dataclass
class DecompositionResult:
    is_decomposed: bool = False
    sub_queries: List[SubQuery] = field(default_factory=list)
    original_query: str = ""
    reasoning: str = ""


class QueryDecompositionEngine:
    """Decomposes complex queries into simpler sub-queries."""

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

    def decompose(self, query: str) -> DecompositionResult:
        """Decompose a complex query into sub-queries."""
        prompt = f"""Analyze if this banking query should be decomposed into simpler sub-queries.

Query: {query}

If the query is simple (single topic, single question), respond:
{{"is_decomposed": false, "reasoning": "Simple query, no decomposition needed"}}

If complex (multiple topics, comparisons, multi-step), respond:
{{
    "is_decomposed": true,
    "sub_queries": [
        {{"query": "sub-query 1", "intent": "intent description", "priority": 1}},
        {{"query": "sub-query 2", "intent": "intent description", "priority": 2}}
    ],
    "reasoning": "Explanation of decomposition"
}}

Max {self.settings.rag.query_decomposition_max_sub_queries} sub-queries."""

        result = self._call_llm(prompt)
        try:
            text = result.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            parsed = json.loads(text) if text.startswith("{") else json.loads(text[text.find("{"):text.rfind("}")+1])
        except:
            parsed = {}

        sub_queries = []
        for sq in parsed.get("sub_queries", []):
            sub_queries.append(SubQuery(
                query=sq.get("query", ""),
                intent=sq.get("intent", ""),
                priority=sq.get("priority", 1),
            ))

        return DecompositionResult(
            is_decomposed=parsed.get("is_decomposed", False),
            sub_queries=sub_queries,
            original_query=query,
            reasoning=parsed.get("reasoning", ""),
        )
