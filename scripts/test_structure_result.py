import asyncio
from app.rag_pipeline.loader import load_and_chunk_pdf
from app.rag_pipeline.vector_store import build_or_load_vector_store, search_with_context
from app.llm.structured_qa import ask_llm_structured
from langchain_core.documents import Document
import json

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
    chunks = await load_and_chunk_pdf(URL)
    retriever = build_or_load_vector_store(chunks, persist=False, use_local_embeddings=False)
    
    all_retrieved = []
    for question in QUESTIONS:
        top_docs = search_with_context(retriever, question)
        all_retrieved.extend(top_docs)

    # Deduplicate
    unique_docs = {doc.page_content: doc for doc in all_retrieved}.values()

    
    print("🛠 Calling ask_llm_structured()...")
    response_json = ask_llm_structured(QUESTIONS, list(unique_docs))
    
    print("\n🎯 Final JSON Response:\n")
    print(json.dumps(response_json, indent=4))

asyncio.run(run())
