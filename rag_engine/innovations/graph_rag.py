"""
Knowledge Graph RAG — Entity extraction and graph-based reasoning.
Enables cross-product comparisons and multi-hop queries.
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from openai import OpenAI

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


@dataclass
class GraphQueryResult:
    context: str = ""
    entities_found: int = 0
    relationships_traversed: int = 0
    hops_used: int = 0


class KnowledgeGraphRAG:
    """Knowledge graph for entity-relationship reasoning."""

    def __init__(self, client: OpenAI, settings):
        self.client = client
        self.settings = settings
        self.graph = nx.DiGraph() if NX_AVAILABLE else None

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

    def extract_and_add(self, text: str, source: str, section: str = ""):
        """Extract entities and relationships from text and add to graph."""
        if not self.graph:
            return

        from prompts.prompt_manager import PromptManager
        pm = PromptManager()
        result = self._call_llm(pm.graph_entity_extractor(text))
        parsed = self._parse_json(result)

        for entity in parsed.get("entities", []):
            name = entity.get("name", "")
            if name:
                safe_attrs = {k: v for k, v in entity.items() if k not in ("name",) and isinstance(v, (str, int, float, bool))}
                safe_attrs["entity_type"] = entity.get("type", "unknown")
                safe_attrs["source"] = source
                self.graph.add_node(name, **safe_attrs)

        for rel in parsed.get("relationships", []):
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src and tgt:
                safe_attrs = {k: v for k, v in rel.items() if k not in ("source", "target") and isinstance(v, (str, int, float, bool))}
                safe_attrs["rel_type"] = rel.get("type", "related_to")
                self.graph.add_edge(src, tgt, **safe_attrs)

    def query(self, query: str, max_hops: int = 2) -> GraphQueryResult:
        """Query the knowledge graph for relevant context."""
        if not self.graph or len(self.graph.nodes) == 0:
            return GraphQueryResult()

        # Find relevant nodes by keyword matching
        query_lower = query.lower()
        relevant_nodes = []
        for node in self.graph.nodes:
            if node.lower() in query_lower or any(w in node.lower() for w in query_lower.split()):
                relevant_nodes.append(node)

        if not relevant_nodes:
            # Fallback: return top nodes by degree
            sorted_nodes = sorted(self.graph.nodes, key=lambda n: self.graph.degree(n), reverse=True)
            relevant_nodes = sorted_nodes[:3]

        # BFS traversal
        visited = set()
        context_parts = []
        for start_node in relevant_nodes[:5]:
            queue = [(start_node, 0)]
            while queue:
                node, depth = queue.pop(0)
                if node in visited or depth > max_hops:
                    continue
                visited.add(node)

                node_data = self.graph.nodes[node]
                node_type = node_data.get("entity_type", "unknown")
                context_parts.append(f"{node} ({node_type})")

                for _, neighbor, edge_data in self.graph.edges(node, data=True):
                    rel_type = edge_data.get("rel_type", "related_to")
                    context_parts.append(f"  → {rel_type} → {neighbor}")
                    if depth + 1 <= max_hops:
                        queue.append((neighbor, depth + 1))

        context = "\n".join(context_parts)
        return GraphQueryResult(
            context=context,
            entities_found=len(visited),
            relationships_traversed=len([p for p in context_parts if "→" in p]),
            hops_used=max_hops,
        )
