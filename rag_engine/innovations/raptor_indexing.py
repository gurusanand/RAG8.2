"""
RAPTOR Indexer — Recursive Abstractive Processing for Tree-Organized Retrieval.
Builds hierarchical summaries of document chunks for multi-level retrieval.
"""
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from openai import OpenAI


@dataclass
class TreeNode:
    node_id: str
    level: int  # 0 = leaf (original chunk), 1+ = summary levels
    text: str
    source: str
    children: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


@dataclass
class RAPTORResult:
    tree_depth: int = 0
    total_nodes: int = 0
    summaries_generated: int = 0
    context: str = ""


class RAPTORIndexer:
    """Builds hierarchical summary trees for multi-level retrieval."""

    def __init__(self, client: OpenAI, settings, embed_model):
        self.client = client
        self.settings = settings
        self.embed_model = embed_model
        self.trees: Dict[str, List[TreeNode]] = {}  # source -> nodes
        self.all_nodes: List[TreeNode] = []

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
            return ""

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embed_model.encode(text, convert_to_numpy=True)

    def _summarize_group(self, texts: List[str]) -> str:
        """Generate a summary for a group of chunks."""
        combined = "\n\n".join(texts[:5])  # Limit to 5 chunks per summary
        prompt = f"""Summarize the following banking document sections into a concise paragraph.
Preserve key facts, numbers, product names, and policy details.

Sections:
{combined[:3000]}

Summary:"""
        return self._call_llm(prompt)

    def build_tree(self, chunks: List[Dict], source: str, group_size: int = 3):
        """Build a RAPTOR tree from document chunks."""
        # Level 0: Leaf nodes
        nodes = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if not text:
                continue
            node = TreeNode(
                node_id=f"{source}_L0_{i}",
                level=0,
                text=text,
                source=source,
                embedding=self._get_embedding(text),
            )
            nodes.append(node)

        if not nodes:
            return

        # Build higher levels
        current_level = nodes
        level_num = 1
        max_levels = 3

        while len(current_level) > 1 and level_num <= max_levels:
            next_level = []
            for i in range(0, len(current_level), group_size):
                group = current_level[i:i + group_size]
                group_texts = [n.text for n in group]
                summary = self._summarize_group(group_texts)

                if summary:
                    parent = TreeNode(
                        node_id=f"{source}_L{level_num}_{i // group_size}",
                        level=level_num,
                        text=summary,
                        source=source,
                        children=[n.node_id for n in group],
                        embedding=self._get_embedding(summary),
                    )
                    next_level.append(parent)
                    nodes.append(parent)

            current_level = next_level
            level_num += 1

        self.trees[source] = nodes
        self.all_nodes.extend(nodes)

    def query(self, query: str, top_k: int = 5) -> RAPTORResult:
        """Query the RAPTOR tree at multiple levels."""
        if not self.all_nodes:
            return RAPTORResult()

        query_emb = self._get_embedding(query)

        # Search across all levels
        scored = []
        for node in self.all_nodes:
            if node.embedding is not None:
                sim = float(np.dot(query_emb, node.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-10
                ))
                scored.append((node, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        context_parts = []
        for node, score in top:
            level_label = "Summary" if node.level > 0 else "Detail"
            context_parts.append(f"[{level_label} L{node.level}] {node.text}")

        max_depth = max(n.level for n, _ in top) if top else 0
        summaries = sum(1 for n, _ in top if n.level > 0)

        return RAPTORResult(
            tree_depth=max_depth,
            total_nodes=len(self.all_nodes),
            summaries_generated=summaries,
            context="\n\n".join(context_parts),
        )
