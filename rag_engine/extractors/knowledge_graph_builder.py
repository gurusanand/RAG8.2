"""
Knowledge Graph Construction Engine (GraphRAG)
===============================================
Builds a structured knowledge graph from extracted document content,
enabling entity-relationship reasoning for complex multi-hop queries.

Strategy: Uses LLM-based entity extraction to identify banking entities
(products, fees, rates, conditions) and their relationships, then stores
them in a NetworkX graph for traversal during retrieval.

Feature Toggle: extraction_graph_enabled (default: True)
"""
import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("[KNOWLEDGE_GRAPH] NetworkX not available")


@dataclass
class Entity:
    """A named entity extracted from a document."""
    entity_id: str
    name: str
    entity_type: str  # product, fee, rate, condition, process, regulation
    attributes: Dict[str, str] = field(default_factory=dict)
    source_page: int = 0
    source_file: str = ""


@dataclass
class Relationship:
    """A relationship between two entities."""
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # has_fee, offers_benefit, requires_condition, etc.
    attributes: Dict[str, str] = field(default_factory=dict)
    source_page: int = 0


@dataclass
class GraphQueryResult:
    """Result of querying the knowledge graph."""
    entities: List[Entity]
    relationships: List[Relationship]
    subgraph_text: str  # Human-readable representation of the subgraph
    confidence: float = 0.0


class KnowledgeGraphBuilder:
    """
    Builds and queries a knowledge graph from banking document content.

    The graph captures:
    - Products (cards, accounts) as central nodes
    - Fees, rates, limits as attribute nodes
    - Benefits, loyalty programs as feature nodes
    - Conditions, requirements as constraint nodes
    - Processes (chargeback, card blocking) as workflow nodes
    """

    # Centralized prompt for entity extraction
    ENTITY_EXTRACTION_PROMPT = """You are a banking document analyst. Extract ALL entities and relationships from this text.

ENTITY TYPES:
- product: Banking products (credit cards, accounts, savings plans)
- fee: Any fee or charge (annual fee, late payment fee, cash advance fee)
- rate: Interest rates, exchange rates, reward rates
- benefit: Loyalty programs, cashback, rewards, perks
- limit: Credit limits, withdrawal limits, transaction limits
- condition: Requirements, eligibility criteria, terms
- process: Procedures (chargeback, card blocking, dispute filing)
- regulation: Regulatory references (CBUAE, consumer protection)

RELATIONSHIP TYPES:
- has_fee: Product → Fee
- has_rate: Product → Rate
- offers_benefit: Product → Benefit
- has_limit: Product → Limit
- requires_condition: Product/Process → Condition
- follows_process: Action → Process steps
- regulated_by: Product → Regulation
- compared_with: Product → Product (when explicitly compared)
- earns_reward: Spend Category → Reward Rate

RESPOND IN THIS EXACT JSON FORMAT:
{
    "entities": [
        {
            "name": "Solitaire Credit Card",
            "type": "product",
            "attributes": {"issuer": "Mashreq", "category": "premium"}
        }
    ],
    "relationships": [
        {
            "source": "Solitaire Credit Card",
            "target": "AED 1,575 Annual Fee",
            "type": "has_fee",
            "attributes": {"vat_inclusive": "yes"}
        }
    ]
}

CRITICAL: Extract EVERY entity and relationship. Do not skip any fees, rates, or product features. Be exhaustive."""

    GRAPH_QUERY_PROMPT = """You are a knowledge graph query expert. Given the user's question and the available entities and relationships in the graph, identify which entities and relationship types are most relevant.

Available entity types: {entity_types}
Available relationship types: {relationship_types}
Sample entities: {sample_entities}

User question: {query}

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "target_entity_types": ["product", "fee"],
    "target_relationship_types": ["has_fee"],
    "target_entity_names": ["Solitaire", "annual fee"],
    "traversal_depth": 2,
    "reasoning": "The user is asking about fees for a specific card"
}}"""

    def __init__(self, client, settings):
        self.client = client
        self.settings = settings
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []

        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None

        self._entity_types: Set[str] = set()
        self._relationship_types: Set[str] = set()

    def build_from_text(self, text: str, source_file: str, page_number: int = 0) -> int:
        """
        Extract entities and relationships from text and add to the graph.

        Args:
            text: Text content to extract from
            source_file: Source document filename
            page_number: Page number in the source document

        Returns:
            Number of new entities added
        """
        if not text or len(text.strip()) < 50:
            return 0

        # Truncate very long text to avoid token limits
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars]

        try:
            prompt = f"{self.ENTITY_EXTRACTION_PROMPT}\n\nTEXT TO ANALYZE:\n{text}"
            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096
            )
            raw = response.choices[0].message.content.strip()
            parsed = self._parse_json(raw)

            if not parsed:
                return 0

            new_count = 0

            # Process entities
            for ent_data in parsed.get("entities", []):
                entity = self._create_entity(ent_data, source_file, page_number)
                if entity and entity.entity_id not in self.entities:
                    self.entities[entity.entity_id] = entity
                    self._entity_types.add(entity.entity_type)
                    new_count += 1

                    # Add to NetworkX graph
                    if self.graph is not None:
                        # Note: 'type' is reserved in some networkx contexts;
                        # use 'entity_type' to avoid conflicts with attributes dict
                        safe_attrs = {k: v for k, v in entity.attributes.items() if k != 'type'}
                        self.graph.add_node(
                            entity.entity_id,
                            name=entity.name,
                            entity_type=entity.entity_type,
                            **safe_attrs
                        )

            # Process relationships
            for rel_data in parsed.get("relationships", []):
                rel = self._create_relationship(rel_data, page_number)
                if rel:
                    self.relationships.append(rel)
                    self._relationship_types.add(rel.relationship_type)

                    # Add edge to NetworkX graph
                    if self.graph is not None:
                        safe_rel_attrs = {k: v for k, v in rel.attributes.items() if k != 'type'}
                        self.graph.add_edge(
                            rel.source_entity_id,
                            rel.target_entity_id,
                            rel_type=rel.relationship_type,
                            **safe_rel_attrs
                        )

            logger.info(f"[KNOWLEDGE_GRAPH] Extracted {new_count} entities, "
                       f"{len(parsed.get('relationships', []))} relationships from page {page_number}")
            return new_count

        except Exception as e:
            logger.error(f"[KNOWLEDGE_GRAPH] Entity extraction failed: {e}")
            return 0

    def build_from_tables(self, tables: List[Dict[str, Any]], source_file: str) -> int:
        """
        Build graph from extracted table data (more structured than free text).

        Args:
            tables: List of table dicts with headers, rows, title
            source_file: Source document filename

        Returns:
            Number of new entities added
        """
        total_new = 0

        for table in tables:
            # Convert table to a structured text representation for LLM
            table_text = f"Table: {table.get('title', 'Unknown')}\n"
            headers = table.get("headers", [])
            rows = table.get("rows", [])

            if headers:
                table_text += f"Columns: {', '.join(headers)}\n\n"

            for row in rows:
                if isinstance(row, dict):
                    row_parts = [f"{k}: {v}" for k, v in row.items() if v]
                    table_text += " | ".join(row_parts) + "\n"

            new_count = self.build_from_text(
                table_text, source_file,
                page_number=table.get("page", 0)
            )
            total_new += new_count

        return total_new

    def query(self, user_query: str, max_hops: int = 2) -> GraphQueryResult:
        """
        Query the knowledge graph to find relevant entities and relationships.

        Args:
            user_query: The user's natural language question
            max_hops: Maximum traversal depth in the graph

        Returns:
            GraphQueryResult with relevant entities and relationships
        """
        if not self.entities:
            return GraphQueryResult(
                entities=[], relationships=[],
                subgraph_text="Knowledge graph is empty.",
                confidence=0.0
            )

        # Step 1: Use LLM to identify target entities/relationships
        try:
            sample_entities = list(self.entities.values())[:10]
            sample_names = [e.name for e in sample_entities]

            prompt = self.GRAPH_QUERY_PROMPT.format(
                entity_types=list(self._entity_types),
                relationship_types=list(self._relationship_types),
                sample_entities=sample_names,
                query=user_query
            )

            response = self.client.chat.completions.create(
                model=self.settings.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            raw = response.choices[0].message.content.strip()
            query_plan = self._parse_json(raw)

        except Exception as e:
            logger.error(f"[KNOWLEDGE_GRAPH] Query planning failed: {e}")
            query_plan = {}

        # Step 2: Find matching entities
        target_names = query_plan.get("target_entity_names", [])
        target_types = query_plan.get("target_entity_types", [])
        matched_entities = self._find_matching_entities(target_names, target_types)

        # Step 3: Traverse graph from matched entities
        if self.graph is not None and matched_entities:
            expanded_entities, expanded_rels = self._traverse_graph(
                [e.entity_id for e in matched_entities],
                max_hops=min(max_hops, query_plan.get("traversal_depth", 2))
            )
        else:
            expanded_entities = matched_entities
            expanded_rels = [r for r in self.relationships
                          if r.source_entity_id in {e.entity_id for e in matched_entities}
                          or r.target_entity_id in {e.entity_id for e in matched_entities}]

        # Step 4: Generate human-readable subgraph text
        subgraph_text = self._generate_subgraph_text(expanded_entities, expanded_rels)

        return GraphQueryResult(
            entities=expanded_entities,
            relationships=expanded_rels,
            subgraph_text=subgraph_text,
            confidence=0.8 if expanded_entities else 0.2
        )

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": dict(self._count_by_type()),
            "relationship_types": list(self._relationship_types),
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
        }

    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for e in self.entities.values():
            counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
        return counts

    def _create_entity(self, data: Dict, source_file: str, page_number: int) -> Optional[Entity]:
        """Create an Entity from parsed LLM output."""
        name = data.get("name", "").strip()
        entity_type = data.get("type", "unknown").strip().lower()
        if not name:
            return None

        entity_id = hashlib.md5(f"{name}_{entity_type}".encode()).hexdigest()[:12]
        return Entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=data.get("attributes", {}),
            source_page=page_number,
            source_file=source_file
        )

    def _create_relationship(self, data: Dict, page_number: int) -> Optional[Relationship]:
        """Create a Relationship from parsed LLM output."""
        source_name = data.get("source", "").strip()
        target_name = data.get("target", "").strip()
        rel_type = data.get("type", "related_to").strip()

        if not source_name or not target_name:
            return None

        # Find or create entity IDs
        source_id = self._find_entity_id_by_name(source_name)
        target_id = self._find_entity_id_by_name(target_name)

        if not source_id or not target_id:
            return None

        return Relationship(
            source_entity_id=source_id,
            target_entity_id=target_id,
            relationship_type=rel_type,
            attributes=data.get("attributes", {}),
            source_page=page_number
        )

    def _find_entity_id_by_name(self, name: str) -> Optional[str]:
        """Find an entity ID by name (fuzzy match)."""
        name_lower = name.lower().strip()
        for eid, entity in self.entities.items():
            if entity.name.lower() == name_lower:
                return eid
            if name_lower in entity.name.lower() or entity.name.lower() in name_lower:
                return eid
        # Create a temporary ID for unmatched entities
        return hashlib.md5(f"{name}_unknown".encode()).hexdigest()[:12]

    def _find_matching_entities(self, names: List[str], types: List[str]) -> List[Entity]:
        """Find entities matching given names and/or types."""
        matched = []
        for entity in self.entities.values():
            # Match by type
            if entity.entity_type in types:
                matched.append(entity)
                continue
            # Match by name (fuzzy)
            for name in names:
                if (name.lower() in entity.name.lower() or
                    entity.name.lower() in name.lower()):
                    matched.append(entity)
                    break
        return matched

    def _traverse_graph(self, start_ids: List[str], max_hops: int = 2) -> Tuple[List[Entity], List[Relationship]]:
        """Traverse the graph from start nodes up to max_hops."""
        if not self.graph:
            return [], []

        visited_nodes = set()
        visited_edges = set()

        for start_id in start_ids:
            if start_id not in self.graph:
                continue

            # BFS traversal
            queue = [(start_id, 0)]
            while queue:
                node_id, depth = queue.pop(0)
                if node_id in visited_nodes or depth > max_hops:
                    continue
                visited_nodes.add(node_id)

                # Get neighbors
                for neighbor in self.graph.successors(node_id):
                    edge_data = self.graph.get_edge_data(node_id, neighbor)
                    visited_edges.add((node_id, neighbor, edge_data.get("rel_type", edge_data.get("type", "related"))))
                    if depth + 1 <= max_hops:
                        queue.append((neighbor, depth + 1))

                for neighbor in self.graph.predecessors(node_id):
                    edge_data = self.graph.get_edge_data(neighbor, node_id)
                    visited_edges.add((neighbor, node_id, edge_data.get("rel_type", edge_data.get("type", "related"))))
                    if depth + 1 <= max_hops:
                        queue.append((neighbor, depth + 1))

        # Collect entities and relationships
        entities = [self.entities[nid] for nid in visited_nodes if nid in self.entities]
        relationships = []
        for src, tgt, rel_type in visited_edges:
            relationships.append(Relationship(
                source_entity_id=src,
                target_entity_id=tgt,
                relationship_type=rel_type
            ))

        return entities, relationships

    def _generate_subgraph_text(self, entities: List[Entity], relationships: List[Relationship]) -> str:
        """Generate a human-readable text representation of the subgraph."""
        if not entities:
            return "No relevant entities found in the knowledge graph."

        parts = ["**Knowledge Graph Context:**\n"]

        # Group entities by type
        by_type: Dict[str, List[Entity]] = {}
        for e in entities:
            by_type.setdefault(e.entity_type, []).append(e)

        for etype, ents in by_type.items():
            parts.append(f"\n**{etype.title()}s:**")
            for e in ents:
                attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items()) if e.attributes else ""
                parts.append(f"  - {e.name}" + (f" ({attrs})" if attrs else ""))

        # Relationships
        if relationships:
            parts.append("\n**Relationships:**")
            entity_names = {e.entity_id: e.name for e in entities}
            for r in relationships:
                src_name = entity_names.get(r.source_entity_id, r.source_entity_id)
                tgt_name = entity_names.get(r.target_entity_id, r.target_entity_id)
                parts.append(f"  - {src_name} --[{r.relationship_type}]--> {tgt_name}")

        return "\n".join(parts)

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {}

    def export_graph_json(self) -> Dict:
        """Export the entire graph as JSON for persistence."""
        return {
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "type": e.entity_type,
                    "attributes": e.attributes,
                    "source_page": e.source_page,
                    "source_file": e.source_file
                }
                for e in self.entities.values()
            ],
            "relationships": [
                {
                    "source": r.source_entity_id,
                    "target": r.target_entity_id,
                    "type": r.relationship_type,
                    "attributes": r.attributes,
                    "source_page": r.source_page
                }
                for r in self.relationships
            ]
        }

    def import_graph_json(self, data: Dict):
        """Import a graph from JSON (for persistence reload)."""
        for ent_data in data.get("entities", []):
            entity = Entity(
                entity_id=ent_data["entity_id"],
                name=ent_data["name"],
                entity_type=ent_data["type"],
                attributes=ent_data.get("attributes", {}),
                source_page=ent_data.get("source_page", 0),
                source_file=ent_data.get("source_file", "")
            )
            self.entities[entity.entity_id] = entity
            self._entity_types.add(entity.entity_type)

            if self.graph is not None:
                safe_attrs = {k: v for k, v in entity.attributes.items() if k != 'type'}
                self.graph.add_node(
                    entity.entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    **safe_attrs
                )

        for rel_data in data.get("relationships", []):
            rel = Relationship(
                source_entity_id=rel_data["source"],
                target_entity_id=rel_data["target"],
                relationship_type=rel_data["type"],
                attributes=rel_data.get("attributes", {}),
                source_page=rel_data.get("source_page", 0)
            )
            self.relationships.append(rel)
            self._relationship_types.add(rel.relationship_type)

            if self.graph is not None:
                safe_rel_attrs = {k: v for k, v in rel.attributes.items() if k != 'type'}
                self.graph.add_edge(
                    rel.source_entity_id,
                    rel.target_entity_id,
                    rel_type=rel.relationship_type,
                    **safe_rel_attrs
                )
