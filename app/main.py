# app/main.py
from fastapi import FastAPI, Header, HTTPException
from app.models import RunRequest, RunResponse
from app.config import settings
from app.rag_pipeline import handle as pipeline_handle

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI(title="HackRx Retrieval System", version="1.0")

@app.get("/")
async def health_check():
    return "API is running"
@app.post("/hackrx/run", response_model=RunResponse)
@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def hackrx_run(payload: RunRequest, authorization: str = Header(...)):
    if authorization != f"Bearer {settings.TEAM_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid team token")

    answers = await pipeline_handle(payload)
    # documents = []
    # for url in payload.documents:
    #     documents.append(url)

    # answers = await pipeline_handle(documents, payload.questions)
    return RunResponse(answers=answers)
