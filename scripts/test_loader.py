import asyncio
from app.rag_pipeline.loader import load_and_chunk_pdf

url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

async def run():
    chunks = await load_and_chunk_pdf(url)
    print(f"Total chunks: {len(chunks)}")
    print("First chunk:\n", chunks[0].page_content[:500])

asyncio.run(run())
