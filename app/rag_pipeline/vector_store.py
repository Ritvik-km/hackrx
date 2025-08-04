# app/rag_pipeline/vector_store.py
"""
FAISS-based Vector Store for high-speed semantic retrieval.
Drop-in replacement for the original Chroma implementation.
"""

from __future__ import annotations

import os
import re
import pickle
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss

from app.config import settings

from app.rag_pipeline.embedder import GoogleEmbeddingModel, GitHubEmbeddingModel

try:
    from langchain.schema import Document
except ModuleNotFoundError:
    from dataclasses import dataclass

    @dataclass
    class Document:  # type: ignore
        page_content: str
        metadata: Dict = None  # noqa: D401 – simple stub


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


#                               Helper functions                              #

def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Normalise vectors in-place for cosine similarity (dot product)."""
    faiss.normalize_L2(vectors)
    return vectors


def sanitize_metadata(meta: Dict, *, max_val_len: int = 500) -> Dict:
    """
    Remove non-JSON-serialisable or overly large fields from metadata so they
    can be safely persisted.
    """
    serialisable = {}
    for k, v in (meta or {}).items():
        try:
            if isinstance(v, (str, int, float, bool)) and len(str(v)) <= max_val_len:
                serialisable[k] = v
            else:
                serialisable[k] = str(v)[:max_val_len]
        except Exception:  # pragma: no cover
            serialisable[k] = "<unserialisable>"
    return serialisable


def _is_rare_query(query: str, min_terms: int = 4) -> bool:
    """Heuristically determine if a query is "rare" and may need a wider search."""
    terms = re.findall(r"\w+", query)
    if len(set(t.lower() for t in terms)) < min_terms:
        return True
    if re.search(r"\b[A-Z]{2,}\b", query):
        # Abbreviation-style question, often sparse
        return True
    return False


def _generate_alternate_queries(query: str) -> List[str]:
    """Extract simple alternates like abbreviations or their expansions."""
    variants: List[str] = []
    match = re.search(r"(.+?)\((.+?)\)", query)
    if match:
        before, inside = match.group(1).strip(), match.group(2).strip()
        if before:
            variants.append(before)
        if inside:
            variants.append(inside)
    return variants


#                               Vector Store Class                            #

class FAISSVectorStore:
    """
    Minimal yet production-oriented wrapper around FAISS for document retrieval.
    """

    def __init__(
        self,
        dim: int = 3072,
        store_dir: Path | None = None,
        use_google_embeddings: bool = True,
    ) -> None:
        self.dim = dim
        self.store_dir: Path = store_dir or Path(settings.DATA_DIR) / "faiss_store"
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.store_dir / "faiss.index"
        self.docs_path = self.store_dir / "docs.pkl"

        self.index: faiss.IndexFlatIP | None = None
        self.id_to_doc: List[Tuple[str, Dict]] = []
        self.doc_hashes: set[str] = set()

        # Embedding model
        if use_google_embeddings:
            self.embedder = GoogleEmbeddingModel()
            self._doc_task = "RETRIEVAL_DOCUMENT"
            self._query_task = "RETRIEVAL_QUERY"
            print("🔵 Using GoogleEmbeddingModel (Gemini)")
        else:
            self.embedder = GitHubEmbeddingModel()
            self._doc_task = self._query_task = None  # GitHub model has no task-type switch
            print("🟢 Using GitHubEmbeddingModel (openai/text-embedding-3-large)")

        # Initialise / load existing store
        self._load_or_init()

    # ----------------------------- Persistence ----------------------------- #

    def _load_or_init(self) -> None:
        if self.index_path.exists() and self.docs_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.docs_path, "rb") as f:
                # self.id_to_doc = pickle.load(f)
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.id_to_doc = data.get("id_to_doc", [])
                    self.doc_hashes = set(data.get("doc_hashes", []))
                else:  # Backwards compatibility
                    self.id_to_doc = data
                    self.doc_hashes = {
                        hashlib.sha256(text.encode("utf-8")).hexdigest()
                        for text, _ in self.id_to_doc
                    }
            logger.info("Loaded FAISS index with %d vectors.", self.index.ntotal)
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # Cosine (after L2 normalise)
            logger.info("Created new FAISS index (dim=%d).", self.dim)

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with open(self.docs_path, "wb") as f:
            # pickle.dump(self.id_to_doc, f)
            pickle.dump(
                {"id_to_doc": self.id_to_doc, "doc_hashes": list(self.doc_hashes)},
                f,
            )

    # ----------------------------- API methods ----------------------------- #

    def add_documents(self, docs: List[Document]) -> None:
        """
        Embed & add documents to the index.  Persists automatically.
        """
        if not docs:
            return
        
        new_docs: List[Document] = []
        new_hashes: List[str] = []
        for d in docs:
            h = hashlib.sha256(d.page_content.encode("utf-8")).hexdigest()
            if h in self.doc_hashes:
                continue
            new_docs.append(d)
            new_hashes.append(h)

        if not new_docs:
            logger.info("No new documents to add; all were duplicates.")
            return

        # Sanitize metadata & gather texts
        # texts = [d.page_content for d in docs]
        # sanitized_meta = [sanitize_metadata(d.metadata) for d in docs]
        texts = [d.page_content for d in new_docs]
        sanitized_meta = [sanitize_metadata(d.metadata) for d in new_docs]

        # Embeddings
        if self._doc_task:
            embeds = self.embedder.embed_documents(
                texts, output_dim=self.dim, task_type=self._doc_task  # type: ignore[arg-type]
            )
        else:
            embeds = self.embedder.embed_documents(texts)

        vecs = _l2_normalise(np.array(embeds, dtype="float32"))
        self.index.add(vecs)

        # Persist mapping
        self.id_to_doc.extend(list(zip(texts, sanitized_meta)))
        self.doc_hashes.update(new_hashes)
        self._save()
        # logger.info("Added %d documents (store now has %d).", len(docs), self.index.ntotal)
        logger.info(
            "Added %d documents (store now has %d).",
            len(new_docs),
            self.index.ntotal,
        )

    # ................................................................. #

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        include_scores: bool = False,
    ) -> List[Document] | List[Tuple[Document, float]]:
        """
        Standard top-k cosine similarity search.
        """
        if self.index.ntotal == 0:
            return []

        q_embed = (
            self.embedder.embed_query(query, output_dim=self.dim, task_type=self._query_task)  # type: ignore[arg-type]
            if self._query_task
            else self.embedder.embed_query(query)
        )
        q_vec = _l2_normalise(np.array([q_embed], dtype="float32"))
        D, I = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:  # FAISS returns -1 when not enough neighbours
                continue
            text, meta = self.id_to_doc[idx]
            doc = Document(page_content=text, metadata=meta | {"score": float(score)})
            results.append((doc, float(score)) if include_scores else doc)
        return results

    # ................................................................. #

    def similarity_search_combined(
        self,
        queries: List[str],
        k_per_query: int = 5,
        max_total: int = 15,
    ) -> List[Document]:
        """
        Runs similarity search for each query separately and returns a de-duplicated,
        score-weighted union, capped at *max_total* results.
        """
        aggregate: Dict[str, Tuple[Document, float]] = {}

        for q in queries:
            for doc, score in self.similarity_search(q, k=k_per_query, include_scores=True):
                # Use content as unique key
                key = doc.page_content[:256]
                # Retain the *highest* score if doc appears via multiple queries
                if key not in aggregate or aggregate[key][1] < score:
                    aggregate[key] = (doc, score)

        # Sort by score desc and limit
        sorted_docs = sorted(aggregate.values(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in sorted_docs[:max_total]]

    # ................................................................. #

    def delete_all_documents(self) -> None:
        """
        Clears the FAISS index and document store from disk AND memory.
        """
        if self.index_path.exists():
            self.index_path.unlink()
        if self.docs_path.exists():
            self.docs_path.unlink()
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_to_doc = []
        self.doc_hashes = set()
        logger.warning("All documents deleted; empty index initialised.")


#                     Convenience builder for the existing code               #
def build_or_load_vector_store(
    docs: List[Document],
    persist: bool = True,
    collection_name: str = "policy_clauses",  # kept for interface parity
    use_local_embeddings: bool = True,
) -> FAISSVectorStore:
    """
    Replaces the old Chroma util.  If *docs* is non-empty, they will be added
    to the store (idempotent: FAISS ignores duplicate vectors).
    """
    store_dir = Path(settings.DATA_DIR) / collection_name
    store = FAISSVectorStore(
        dim=3072 if use_local_embeddings else 3072,
        store_dir=store_dir,
        use_google_embeddings=use_local_embeddings,
    )
    if docs:
        store.add_documents(docs)
    return store


def merge_adjacent_chunks(docs: List[Document]) -> List[Document]:
    """Merge adjacent chunks and prioritize definition chunks"""
    if not docs:
        return docs

    definition_keywords = ["means", "refers to", "includes", "shall be", "defined as"]

    def sort_key(doc):
        content_lower = doc.page_content.lower()
        has_definition = any(keyword in content_lower for keyword in definition_keywords)
        return (0 if has_definition else 1, doc.metadata.get("chunk_index", 999))

    sorted_docs = sorted(docs, key=sort_key)

    merged = []
    skip_next = False

    for i in range(len(sorted_docs)):
        if skip_next:
            skip_next = False
            continue

        current = sorted_docs[i]
        if i + 1 < len(sorted_docs):
            next_doc = sorted_docs[i + 1]
            current_idx = current.metadata.get("chunk_index", -1)
            next_idx = next_doc.metadata.get("chunk_index", -1)

            if abs(current_idx - next_idx) == 1:
                combined_text = current.page_content + " " + next_doc.page_content
                merged_doc = Document(
                    page_content=combined_text,
                    metadata={**current.metadata, "merged_with": next_idx, "combined_length": len(combined_text)}
                )
                merged.append(merged_doc)
                skip_next = True
                continue

        merged.append(current)

    return merged[:10]


def search_with_context(retriever: FAISSVectorStore, query: str) -> List[Document]:
    """Enhanced search that includes context expansion"""
    results = retriever.similarity_search(query, k=10)

    if any(word in query.lower() for word in ["what is", "define", "definition", "meaning"]):
        key_terms = re.findall(r'\b[a-zA-Z\s]+\b', query.lower())
        for term in key_terms:
            if len(term.strip()) > 3:
                alt_query = f"{term.strip()} means definition"
                alt_results = retriever.similarity_search(alt_query, k=5)
                for alt_doc in alt_results:
                    if not any(alt_doc.page_content == doc.page_content for doc in results):
                        results.append(alt_doc)

    return merge_adjacent_chunks(results)

