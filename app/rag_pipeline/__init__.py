"""
End-to-end orchestration for the /hackrx/run endpoint.

Steps
1. Download + chunk PDF                                → loader.load_and_chunk_pdf
2. Build / load FAISS store & embed                    → vector_store.build_or_load_vector_store
3. Retrieve contextually relevant clauses              → vector_store.search_with_context
   (or .similarity_search_combined  for many Qs)
4. Ask the LLM for structured JSON answers             → structured_qa.ask_llm_structured
5. Return the { "answers": [...] } dict.
"""

import asyncio
from typing import Dict, List

from app.rag_pipeline.loader import load_and_chunk_pdf
from app.rag_pipeline.vector_store import (
    Document,
    build_or_load_vector_store,
    search_with_context,  # optional helper
)
from app.llm.structured_qa import ask_llm_structured, ask_llm_structured_parallel


async def handle(req) -> List[str]:
    """
    Main entry called by FastAPI - expects a RunRequest Pydantic obj
    having fields:  req.documents (str URL)  and  req.questions (List[str])
    """
    pdf_url: str = req.documents
    questions: List[str] = req.questions

    # 1️⃣  Chunk the PDF
    chunks = await load_and_chunk_pdf(pdf_url)

    # 2️⃣  Build / load FAISS vector store
    store = build_or_load_vector_store(chunks)

    # # 3️⃣  Retrieve clauses (choose ONE approach)
    # # --- Option A: single broad query (fast, simple) ----
    # # retrieved = search_with_context(store, " ".join(questions))
    # # --- Option B: multi-query combined (higher recall) --
    retrieved = store.similarity_search_combined(
        queries=questions,
        k_per_query=5,
        max_total=20,
    )

    # 4️⃣  Ask LLM for structured answers
    result: Dict = await ask_llm_structured(questions, retrieved)
    # result: Dict = await ask_llm_structured_parallel(questions, retrieved)

    # 5️⃣  Return answers list (RunResponse expects it)
    return result["answers"]
    # return answers
