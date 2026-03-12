"""
End-to-end test for MongoDB + FAISS persistence.
Tests: upload → store → restart → query without re-upload.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persistence.mongo_store import MongoDocumentStore
from persistence.faiss_store import FAISSVectorStore
from persistence.persistence_manager import PersistenceManager

def test_full_persistence():
    print("=" * 60)
    print("TEST 1: MongoDB + FAISS Persistence End-to-End")
    print("=" * 60)
    
    # Clean up any previous test data
    pm = PersistenceManager()
    pm.mongo.db.documents.drop()
    pm.mongo.db.chunks.drop()
    pm.mongo.db.tables.drop()
    pm.mongo.db.entities.drop()
    pm.mongo.db.relationships.drop()
    
    # Reset FAISS completely
    pm.faiss.clear_all()
    
    print("\n--- Step 1 & 2: Store document with chunks and embeddings ---")
    chunks = [
        {
            'chunk_id': 'chunk_001',
            'text': 'The annual fee for Platinum Plus card is AED 313.95 (VAT inclusive). Supplementary card is free for life.',
            'source': 'test_cards_kfs.pdf',
            'chunk_type': 'table_row',
            'page': 1,
            'confidence': 0.95,
        },
        {
            'chunk_id': 'chunk_002',
            'text': 'The annual fee for Solitaire card is AED 1,575 (VAT inclusive). Supplementary card fee is AED 630.',
            'source': 'test_cards_kfs.pdf',
            'chunk_type': 'table_row',
            'page': 1,
            'confidence': 0.95,
        },
        {
            'chunk_id': 'chunk_003',
            'text': 'Cashback card has no annual fee - it is free for life. Earn 5% cashback on all purchases.',
            'source': 'test_cards_kfs.pdf',
            'chunk_type': 'table_row',
            'page': 1,
            'confidence': 0.90,
        },
    ]
    
    # Generate fake embeddings (384-dim, normalized)
    embeddings = []
    for chunk in chunks:
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Fake file bytes for hash
    file_bytes = b'fake pdf content for testing'
    
    result = pm.store_document_with_chunks(
        filename='test_cards_kfs.pdf',
        file_bytes=file_bytes,
        chunks=chunks,
        embeddings=embeddings_array,
        extraction_result={
            'strategies_used': ['table', 'enrichment', 'vision'],
            'profile': {'page_count': 12, 'is_bilingual': True},
            'processing_time_ms': 5000.0,
        }
    )
    print(f"  Store result: {result}")
    
    # FAISS auto-saves to disk in add_vectors()
    print("  FAISS index auto-saved to disk")
    
    # Check stats
    stats = pm.get_stats()
    print(f"\n  MongoDB: {stats['mongodb']['documents']} docs, {stats['mongodb']['chunks']} chunks")
    print(f"  FAISS: {stats['faiss']['total_vectors']} vectors")
    
    assert stats['mongodb']['documents'] == 1, f"Expected 1 doc, got {stats['mongodb']['documents']}"
    assert stats['mongodb']['chunks'] == 3, f"Expected 3 chunks, got {stats['mongodb']['chunks']}"
    assert stats['faiss']['total_vectors'] == 3, f"Expected 3 vectors, got {stats['faiss']['total_vectors']}"
    print("  ✅ Step 2 PASSED")
    
    print("\n--- Step 3: Check document exists (skip re-upload) ---")
    exists = pm.document_exists('test_cards_kfs.pdf')
    print(f"  Document exists: {exists}")
    assert exists, "Document should exist"
    print("  ✅ Step 3 PASSED")
    
    print("\n--- Step 4: Simulate restart (create new PersistenceManager) ---")
    pm2 = PersistenceManager()
    
    # FAISS auto-loads from disk in __init__()
    
    stats2 = pm2.get_stats()
    print(f"  MongoDB (after restart): {stats2['mongodb']['documents']} docs, {stats2['mongodb']['chunks']} chunks")
    print(f"  FAISS (after restart): {stats2['faiss']['total_vectors']} vectors")
    
    assert stats2['mongodb']['documents'] == 1, f"Expected 1 doc after restart, got {stats2['mongodb']['documents']}"
    assert stats2['mongodb']['chunks'] == 3, f"Expected 3 chunks after restart, got {stats2['mongodb']['chunks']}"
    assert stats2['faiss']['total_vectors'] == 3, f"Expected 3 vectors after restart, got {stats2['faiss']['total_vectors']}"
    print("  ✅ Step 4 PASSED — Data persists across restart!")
    
    print("\n--- Step 5: Search FAISS for 'Platinum Plus annual fee' ---")
    # Create a query embedding (use the first chunk's embedding as a proxy)
    query_emb = embeddings[0]
    results = pm2.faiss.search(query_emb, top_k=3)
    print(f"  Found {len(results)} results")
    for meta, score in results:
        print(f"    Score: {score:.4f} | Source: {meta['source']} | Text: {meta['text'][:80]}...")
    
    assert len(results) > 0, "Should find results"
    top_meta, top_score = results[0]
    assert 'Platinum Plus' in top_meta['text'], f"Top result should mention Platinum Plus, got: {top_meta['text'][:80]}"
    print("  ✅ Step 5 PASSED — FAISS search works after restart!")
    
    print("\n--- Step 6: Load chunks from MongoDB ---")
    all_chunks = pm2.mongo.get_chunks_by_source('test_cards_kfs.pdf')
    print(f"  Loaded {len(all_chunks)} chunks from MongoDB")
    for c in all_chunks:
        print(f"    {c['chunk_id']}: {c['text'][:60]}...")
    
    assert len(all_chunks) == 3, f"Expected 3 chunks, got {len(all_chunks)}"
    print("  ✅ Step 6 PASSED")
    
    print("\n--- Step 7: Get all documents ---")
    all_docs = pm2.get_all_documents()
    print(f"  Found {len(all_docs)} documents")
    for d in all_docs:
        print(f"    {d['filename']} — {d.get('chunk_count', 0)} chunks, strategies: {d.get('extraction_strategies', [])}")
    
    assert len(all_docs) == 1, f"Expected 1 document, got {len(all_docs)}"
    print("  ✅ Step 7 PASSED")
    
    print("\n--- Step 8: Delete document ---")
    pm2.delete_document('test_cards_kfs.pdf')
    stats3 = pm2.get_stats()
    print(f"  After delete: {stats3['mongodb']['documents']} docs, {stats3['mongodb']['chunks']} chunks")
    assert stats3['mongodb']['documents'] == 0, "Should have 0 docs after delete"
    assert stats3['mongodb']['chunks'] == 0, "Should have 0 chunks after delete"
    print("  ✅ Step 8 PASSED — Delete works!")
    
    print("\n" + "=" * 60)
    print("ALL 8 TESTS PASSED ✅")
    print("=" * 60)

if __name__ == "__main__":
    test_full_persistence()
