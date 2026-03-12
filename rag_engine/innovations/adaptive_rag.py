"""
Adaptive RAG Router — Dynamic strategy selection based on query complexity.
Routes queries to the optimal retrieval strategy.
"""
import json
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class AdaptiveResult:
    selected_strategy: str = "standard"
    complexity_level: str = "simple"
    confidence: float = 0.8
    reasoning: str = ""


class AdaptiveRAGRouter:
    """Dynamically selects retrieval strategy based on query analysis."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "{}"

    def classify_and_route(self, query: str) -> AdaptiveResult:
        """Classify query complexity and select optimal strategy."""
        prompt = f"""Analyze this banking query and recommend the optimal retrieval strategy.

Query: {query}

Strategies:
1. "standard" — Simple vector search (for straightforward factual queries)
2. "enhanced" — Vector + BM25 hybrid (for queries needing keyword precision)
3. "graph" — Knowledge graph traversal (for cross-product comparisons)
4. "decomposed" — Multi-step decomposition (for complex multi-part queries)
5. "agentic" — Full agentic reasoning (for queries requiring multi-hop reasoning)

Respond in JSON:
{{
    "selected_strategy": "standard|enhanced|graph|decomposed|agentic",
    "complexity_level": "simple|moderate|complex",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}"""

        result = self._call_llm(prompt)
        try:
            text = result.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            parsed = json.loads(text) if text.startswith("{") else json.loads(text[text.find("{"):text.rfind("}")+1])
        except:
            parsed = {}

        return AdaptiveResult(
            selected_strategy=parsed.get("selected_strategy", "standard"),
            complexity_level=parsed.get("complexity_level", "simple"),
            confidence=parsed.get("confidence", 0.8),
            reasoning=parsed.get("reasoning", "Default strategy selected"),
        )
