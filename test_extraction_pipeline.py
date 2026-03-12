"""
End-to-End Test: Multi-Strategy Extraction Pipeline
=====================================================
Tests the full extraction pipeline with both banking PDFs:
1. Accounts.pdf (complex tables, multi-product, bilingual)
2. Mashreq-Cards-KFS-Final-EngArb.pdf (bilingual, tables, card images)
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings
from openai import OpenAI

settings = get_settings()
client = OpenAI()

# ========================================
# TEST 1: Individual Extractors
# ========================================
print("=" * 70)
print("TEST 1: Individual Extractor Imports")
print("=" * 70)

try:
    from rag_engine.extractors.vision_extractor import VisionExtractor
    print("  ✓ VisionExtractor imported")
except Exception as e:
    print(f"  ✗ VisionExtractor: {e}")

try:
    from rag_engine.extractors.table_extractor import TableExtractor
    print("  ✓ TableExtractor imported")
except Exception as e:
    print(f"  ✗ TableExtractor: {e}")

try:
    from rag_engine.extractors.knowledge_graph_builder import KnowledgeGraphBuilder
    print("  ✓ KnowledgeGraphBuilder imported")
except Exception as e:
    print(f"  ✗ KnowledgeGraphBuilder: {e}")

try:
    from rag_engine.extractors.contextual_enrichment import ContextualEnrichmentEngine
    print("  ✓ ContextualEnrichmentEngine imported")
except Exception as e:
    print(f"  ✗ ContextualEnrichmentEngine: {e}")

try:
    from rag_engine.extractors.extraction_orchestrator import ExtractionOrchestrator
    print("  ✓ ExtractionOrchestrator imported")
except Exception as e:
    print(f"  ✗ ExtractionOrchestrator: {e}")

# ========================================
# TEST 2: Orchestrator Initialization
# ========================================
print("\n" + "=" * 70)
print("TEST 2: Orchestrator Initialization")
print("=" * 70)

try:
    orchestrator = ExtractionOrchestrator(client, settings)
    print(f"  ✓ Orchestrator created")
    print(f"    Vision: {'enabled' if orchestrator.vision_extractor else 'disabled'}")
    print(f"    Table:  {'enabled' if orchestrator.table_extractor else 'disabled'}")
    print(f"    Graph:  {'enabled' if orchestrator.graph_builder else 'disabled'}")
    print(f"    Enrich: {'enabled' if orchestrator.enrichment_engine else 'disabled'}")
except Exception as e:
    print(f"  ✗ Orchestrator init failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========================================
# TEST 3: Process Accounts.pdf
# ========================================
print("\n" + "=" * 70)
print("TEST 3: Process Accounts.pdf")
print("=" * 70)

accounts_path = "/home/ubuntu/upload/Accounts.pdf"
if os.path.exists(accounts_path):
    with open(accounts_path, "rb") as f:
        file_bytes = f.read()
    print(f"  File size: {len(file_bytes):,} bytes")

    start = time.time()
    try:
        result = orchestrator.process_document(file_bytes, "Accounts.pdf")
        elapsed = time.time() - start
        print(f"  ✓ Processing completed in {elapsed:.1f}s")
        print(f"    Strategies used: {result.strategies_used}")
        print(f"    Total chunks: {result.total_chunks}")
        print(f"    Tables extracted: {len(result.tables_extracted)}")
        print(f"    Formulas extracted: {len(result.formulas_extracted)}")
        print(f"    Graph entities: {result.graph_entities}")
        print(f"    Graph relationships: {result.graph_relationships}")
        print(f"    Document summary: {result.document_summary[:150]}...")
        print(f"    Human review required: {result.human_review_required}")

        # Show profile
        p = result.profile
        print(f"\n    Document Profile:")
        print(f"      Pages: {p.page_count}")
        print(f"      Has tables: {p.has_tables}")
        print(f"      Has images: {p.has_images}")
        print(f"      Has formulas: {p.has_formulas}")
        print(f"      Is bilingual: {p.is_bilingual}")
        print(f"      Complexity: {p.complexity_score:.2f}")

        # Show first 3 chunks
        print(f"\n    Sample enriched chunks:")
        for i, chunk in enumerate(result.enriched_chunks[:3]):
            text_preview = chunk.get('text', '')[:120].replace('\n', ' ')
            print(f"      [{i+1}] type={chunk.get('chunk_type', '?')}, "
                  f"page={chunk.get('page', '?')}: {text_preview}...")

        # Show tables
        if result.tables_extracted:
            print(f"\n    Extracted Tables:")
            for t in result.tables_extracted[:3]:
                print(f"      - {t.get('title', 'Unknown')} (page {t.get('page', '?')}, "
                      f"confidence: {t.get('confidence', 0):.2f})")

    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ⚠ File not found: {accounts_path}")

# ========================================
# TEST 4: Process Cards KFS PDF
# ========================================
print("\n" + "=" * 70)
print("TEST 4: Process Mashreq-Cards-KFS-Final-EngArb.pdf")
print("=" * 70)

cards_path = "/home/ubuntu/upload/Mashreq-Cards-KFS-Final-EngArb.pdf"
if os.path.exists(cards_path):
    with open(cards_path, "rb") as f:
        file_bytes = f.read()
    print(f"  File size: {len(file_bytes):,} bytes")

    start = time.time()
    try:
        result2 = orchestrator.process_document(file_bytes, "Mashreq-Cards-KFS-Final-EngArb.pdf")
        elapsed = time.time() - start
        print(f"  ✓ Processing completed in {elapsed:.1f}s")
        print(f"    Strategies used: {result2.strategies_used}")
        print(f"    Total chunks: {result2.total_chunks}")
        print(f"    Tables extracted: {len(result2.tables_extracted)}")
        print(f"    Graph entities: {result2.graph_entities}")
        print(f"    Graph relationships: {result2.graph_relationships}")
        print(f"    Is bilingual: {result2.profile.is_bilingual}")
        print(f"    Complexity: {result2.profile.complexity_score:.2f}")
        print(f"    Document summary: {result2.document_summary[:150]}...")

    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ⚠ File not found: {cards_path}")

# ========================================
# TEST 5: Full RAG Engine Integration
# ========================================
print("\n" + "=" * 70)
print("TEST 5: Full RAG Engine Integration")
print("=" * 70)

try:
    from rag_engine.seven_layer_rag import SevenLayerRAG
    engine = SevenLayerRAG()
    print(f"  ✓ SevenLayerRAG created")
    print(f"    Extraction orchestrator: {'enabled' if engine.extraction_orchestrator else 'disabled'}")

    # Initialize with sample docs
    engine.initialize()
    print(f"  ✓ Engine initialized with sample docs ({engine.indexer.get_chunk_count()} chunks)")

    # Index Accounts.pdf via advanced extraction
    if os.path.exists(accounts_path):
        with open(accounts_path, "rb") as f:
            file_bytes = f.read()

        # Extract text for fallback
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n\n".join(page.get_text("text") for page in doc if page.get_text("text").strip())
        doc.close()

        start = time.time()
        count = engine.index_uploaded_document(text, "Accounts.pdf", file_bytes=file_bytes)
        elapsed = time.time() - start
        print(f"  ✓ Accounts.pdf indexed: {count} chunks in {elapsed:.1f}s")
        print(f"    Total chunks in engine: {engine.indexer.get_chunk_count()}")

    # Test a query
    print("\n  Testing query: 'What are the fees for Mashreq Neo savings account?'")
    start = time.time()
    response = engine.process_query("What are the fees for Mashreq Neo savings account?")
    elapsed = time.time() - start
    print(f"  ✓ Query processed in {elapsed:.1f}s")
    print(f"    Confidence: {response.confidence:.2f}")
    print(f"    Answer preview: {response.answer[:200]}...")
    print(f"    Sources: {response.sources}")

except Exception as e:
    print(f"  ✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
