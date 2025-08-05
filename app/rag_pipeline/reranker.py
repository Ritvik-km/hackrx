"""Reranking utilities for retrieved documents."""

from __future__ import annotations

from typing import List

try:
    from langchain.schema import Document
except ModuleNotFoundError:  # pragma: no cover - fallback for tests
    from dataclasses import dataclass

    @dataclass
    class Document:  # type: ignore
        page_content: str
        metadata: dict | None = None

from functools import lru_cache

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - library optional during some tests
    CrossEncoder = None  # type: ignore

from typing import Union, TYPE_CHECKING


_MODEL_NAME = "BAAI/bge-reranker-large"
_TOP_N = 10

@lru_cache(maxsize=1)
def _get_model() -> object | None:
    """Lazily load the CrossEncoder model."""
    if CrossEncoder is None:  # pragma: no cover
        return None
    return CrossEncoder(_MODEL_NAME)

def rerank(query: str, docs: List[Document]) -> List[Document]:
    """Rerank *docs* for *query* using a cross-encoder model.

    Parameters
    ----------
    query: str
        The search query used to retrieve the documents.
    docs: List[Document]
        The candidate documents to reorder.

    Returns
    -------
    List[Document]
        Documents ordered by descending relevance score; only the top N are
        returned where N is a small constant (_TOP_N).
    """
    if not docs:
        return []

    model = _get_model()
    if model is None:  # pragma: no cover - if sentence_transformers missing
        return docs[:_TOP_N]

    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:_TOP_N]]
