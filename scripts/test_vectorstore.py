# import asyncio
# from app.rag_pipeline.loader import load_and_chunk_pdf
# from app.rag_pipeline.vector_store import build_or_load_vector_store, merge_adjacent_chunks

# url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"  # replace

# async def run():
#     chunks = await load_and_chunk_pdf(url)
#     retriever = build_or_load_vector_store(chunks)

#     query = "What is the grace period for premium payment?"
#     # results = retriever.get_relevant_documents(query)
#     results = merge_adjacent_chunks(retriever.invoke(query))

#     print("🔍 Top Matching Clauses:\n")
#     for i, doc in enumerate(results):
#         print(f"--- Chunk #{i+1} ---")
#         print(doc.page_content[:400], "\n")

# asyncio.run(run())

import asyncio
from app.rag_pipeline.loader import load_and_chunk_pdf
from app.rag_pipeline.vector_store import build_or_load_vector_store, search_with_context

url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

async def run():
    print("📄 Loading and chunking PDF...")
    chunks = await load_and_chunk_pdf(url)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Debug: Show chunk sizes
    print("\n📊 Chunk size distribution:")
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    print(f"Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)//len(chunk_sizes)}")
    
    # Show oversized chunks
    oversized = [(i, len(chunk.page_content)) for i, chunk in enumerate(chunks) if len(chunk.page_content) > 6000]
    if oversized:
        print(f"⚠️  Oversized chunks: {oversized}")
    
    # Debug: Show chunks that contain "Grace Period"
    print("\n🔍 Chunks containing 'Grace Period':")
    for i, chunk in enumerate(chunks):
        if "grace period" in chunk.page_content.lower():
            print(f"\n--- Debug Chunk #{i} ---")
            print(f"Length: {len(chunk.page_content)} chars")
            print(f"Content preview: {chunk.page_content[:500]}...")
    
    print("\n🏗️ Building vector store...")
    retriever = build_or_load_vector_store(chunks, persist=False)  # Don't persist for testing
    
    query = "What is the grace period for premium payment?"
    print(f"\n❓ Query: {query}")
    
    # Use enhanced search
    results = search_with_context(retriever, query)
    
    print("\n🔍 Top Matching Clauses:\n")
    for i, doc in enumerate(results):
        print(f"--- Chunk #{i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content ({len(doc.page_content)} chars):")
        print(doc.page_content[:600])
        print("\n" + "="*50 + "\n")

asyncio.run(run())