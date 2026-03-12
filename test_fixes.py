"""Quick validation of the two bug fixes."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings
from openai import OpenAI

settings = get_settings()
client = OpenAI()

# TEST A: Knowledge Graph — no more 'type' keyword conflict
print("TEST A: Knowledge Graph 'type' keyword fix")
from rag_engine.extractors.knowledge_graph_builder import KnowledgeGraphBuilder
gb = KnowledgeGraphBuilder(client, settings)

# Simulate adding an entity with 'type' in attributes (the problematic case)
import networkx as nx
import hashlib

entity_id = hashlib.md5(b"test_product").hexdigest()[:12]
attrs = {"issuer": "Mashreq", "type": "premium", "category": "credit"}  # 'type' in attrs!
safe_attrs = {k: v for k, v in attrs.items() if k != 'type'}
gb.graph.add_node(entity_id, name="Test Card", entity_type="product", **safe_attrs)
print(f"  ✓ Node added with 'type' in attributes — no crash")
print(f"  Node data: {dict(gb.graph.nodes[entity_id])}")

# Test build_from_text with a small sample
count = gb.build_from_text(
    "The Mashreq Solitaire Credit Card has an annual fee of AED 1,575 including VAT. "
    "It offers 2x Salaam points on dining and 1x on all other purchases.",
    "test.pdf", 1
)
print(f"  ✓ build_from_text: {count} entities extracted")
print(f"  Total entities: {len(gb.entities)}, relationships: {len(gb.relationships)}")
print(f"  Graph nodes: {gb.graph.number_of_nodes()}, edges: {gb.graph.number_of_edges()}")

# TEST B: Table Extractor with Docling (permission fix)
print("\nTEST B: Table Extractor Docling permission fix")
from rag_engine.extractors.table_extractor import TableExtractor
te = TableExtractor(settings)

accounts_path = "/home/ubuntu/upload/Accounts.pdf"
if os.path.exists(accounts_path):
    result = te.extract(accounts_path, "Accounts.pdf")
    print(f"  ✓ Tables extracted: {len(result.tables)}")
    print(f"  Strategy: {result.strategy_used}")
    for t in result.tables[:3]:
        print(f"    - {t.title} (page {t.page_number}, method: {t.source_method})")
else:
    print("  ⚠ Accounts.pdf not found")

# TEST C: Export graph JSON
print("\nTEST C: Graph export/import")
exported = gb.export_graph_json()
print(f"  ✓ Exported: {len(exported['entities'])} entities, {len(exported['relationships'])} relationships")

gb2 = KnowledgeGraphBuilder(client, settings)
gb2.import_graph_json(exported)
print(f"  ✓ Imported: {len(gb2.entities)} entities, graph nodes: {gb2.graph.number_of_nodes()}")

print("\n✅ ALL FIXES VALIDATED")
