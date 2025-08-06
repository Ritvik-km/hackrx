# app/models.py
from pydantic import BaseModel, Field
from typing import List

class RunRequest(BaseModel):
    documents: str = Field(..., description="Public PDF URL")
    # documents: List[str] = Field(..., description="Public PDF URLs")
    questions: List[str] = Field(..., min_length=1)

class RunResponse(BaseModel):
    answers: List[str]
