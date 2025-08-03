# import asyncio
# import os
# import traceback
# import logging
# from typing import List
# from langchain.schema import Document

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Your imports (uncomment these)
# from app.rag_pipeline.loader import load_and_chunk_pdf
# from app.rag_pipeline.vector_store import build_or_load_vector_store, search_with_context
# from app.llm.structured_qa import ask_llm_structured

# URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# QUESTIONS = [
#     "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
#     "What is the waiting period for pre-existing diseases (PED) to be covered?",
#     "Does this policy cover maternity expenses, and what are the conditions?",
# ]

# async def debug_step_by_step():
#     """Debug each step individually with timeouts"""
    
#     try:
#         # STEP 1: Test PDF Loading with timeout
#         logger.info("🔍 STEP 1: Testing PDF loading...")
#         logger.info(f"📄 Loading from URL: {URL[:100]}...")
        
#         chunks = await asyncio.wait_for(
#             load_and_chunk_pdf(URL), 
#             timeout=120.0  # 2 minutes max
#         )
        
#         logger.info(f"✅ STEP 1 SUCCESS: Loaded {len(chunks)} chunks")
#         logger.info(f"📊 Sample chunk: {chunks[0].page_content[:100] if chunks else 'No chunks'}...")
        
#         # STEP 2: Test Vector Store Building
#         logger.info("🔍 STEP 2: Testing vector store building...")
        
#         # Try with just a few chunks first
#         test_chunks = chunks[:5] if len(chunks) > 5 else chunks
#         logger.info(f"🧪 Testing with {len(test_chunks)} chunks first...")
        
#         retriever = await asyncio.get_event_loop().run_in_executor(
#             None, 
#             lambda: build_or_load_vector_store(test_chunks, persist=False)
#         )
        
#         logger.info("✅ STEP 2 SUCCESS: Vector store built")
        
#         # STEP 3: Test Search Functionality
#         logger.info("🔍 STEP 3: Testing search functionality...")
        
#         test_question = QUESTIONS[0]
#         logger.info(f"🔍 Testing search with: {test_question}")
        
#         search_results = await asyncio.get_event_loop().run_in_executor(
#             None,
#             lambda: search_with_context(retriever, test_question)
#         )
        
#         logger.info(f"✅ STEP 3 SUCCESS: Found {len(search_results)} results")
#         logger.info(f"📋 Sample result: {search_results[0].page_content[:100] if search_results else 'No results'}...")
        
#         # STEP 4: Test LLM with minimal data
#         logger.info("🔍 STEP 4: Testing LLM with minimal data...")
        
#         # Use only first question and first result
#         test_questions = [QUESTIONS[0]]
#         test_docs = search_results[:1] if search_results else []
        
#         response = await asyncio.get_event_loop().run_in_executor(
#             None,
#             lambda: ask_llm_structured(test_questions, test_docs)
#         )
        
#         logger.info("✅ STEP 4 SUCCESS: LLM responded")
#         logger.info(f"📝 Response: {response}")
        
#         return True
        
#     except asyncio.TimeoutError as e:
#         logger.error(f"⏰ TIMEOUT: Step timed out - {e}")
#         return False
#     except Exception as e:
#         logger.error(f"💥 ERROR: Step failed - {e}")
#         logger.error(f"💥 Traceback: {traceback.format_exc()}")
#         return False

# async def debug_individual_components():
#     """Test each component individually"""
    
#     # Test 1: Just PDF loading
#     logger.info("🧪 TEST 1: PDF Loading only...")
#     try:
#         chunks = await asyncio.wait_for(load_and_chunk_pdf(URL), timeout=60.0)
#         logger.info(f"✅ PDF loaded: {len(chunks)} chunks")
#     except Exception as e:
#         logger.error(f"❌ PDF loading failed: {e}")
#         return
    
#     # Test 2: Vector store with minimal data
#     logger.info("🧪 TEST 2: Vector store with 1 chunk...")
#     try:
#         minimal_chunks = chunks[:1]  # Just one chunk
#         retriever = build_or_load_vector_store(minimal_chunks, persist=False)
#         logger.info("✅ Vector store built with 1 chunk")
#     except Exception as e:
#         logger.error(f"❌ Vector store failed: {e}")
#         return
    
#     # Test 3: Search with minimal data
#     logger.info("🧪 TEST 3: Search with minimal setup...")
#     try:
#         results = search_with_context(retriever, "test query")
#         logger.info(f"✅ Search completed: {len(results)} results")
#     except Exception as e:
#         logger.error(f"❌ Search failed: {e}")
#         return
    
#     logger.info("🎉 All individual components work!")

# async def main():
#     """Main debug function"""
#     logger.info("🚀 Starting debug session...")
    
#     # Test individual components first
#     await debug_individual_components()
    
#     # Then test step by step
#     success = await debug_step_by_step()
    
#     if success:
#         logger.info("🎉 All steps completed successfully!")
#     else:
#         logger.error("💥 Debug session failed at some step")

# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio
import logging
from app.rag_pipeline.loader import load_and_chunk_pdf
from app.rag_pipeline.vector_store import build_or_load_vector_store, search_with_context
from app.llm.structured_qa import ask_llm_structured

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

async def run():
    try:
        logger.info("🚀 Starting RAG pipeline with local embeddings...")
        
        # Step 1: Load PDF
        logger.info("📖 Loading PDF...")
        chunks = await load_and_chunk_pdf(URL)
        logger.info(f"✅ Loaded {len(chunks)} chunks")
        
        # Step 2: Build vector store WITH LOCAL EMBEDDINGS
        logger.info("🏗️ Building vector store with local embeddings...")
        retriever = build_or_load_vector_store(
            chunks, 
            persist=False,  # Don't persist for testing
            # collection_name="policy_clauses_1",
            use_local_embeddings=True  # KEY CHANGE: Use local embeddings
        )
        logger.info("✅ Vector store ready")
        
        # Step 3: Search for relevant documents
        logger.info("🔍 Searching for relevant documents...")
        all_retrieved = []
        
        for i, question in enumerate(QUESTIONS):
            logger.info(f"   Question {i+1}: {question[:50]}...")
            top_docs = search_with_context(retriever, question)
            all_retrieved.extend(top_docs)
            logger.info(f"   Found {len(top_docs)} docs")

        # Deduplicate
        unique_docs = list({doc.page_content: doc for doc in all_retrieved}.values())
        logger.info(f"📋 Total unique documents: {len(unique_docs)}")
        
        # Step 4: Query LLM
        # for doc in unique_docs:
        #     if "maternity" in doc.page_content.lower():
        #         print("✅ Found a maternity-related chunk:")
        #         print(doc.page_content[:500])  # Preview first 500 characters

        logger.info("🤖 Querying LLM...")
        response_json = ask_llm_structured(QUESTIONS, unique_docs)
        
        logger.info("🎯 Pipeline completed successfully!")
        print("\n🎯 Final JSON Response:")
        print(response_json)
        
        return response_json
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        import traceback
        logger.error(f"💥 Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(run())