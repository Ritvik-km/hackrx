# app/utils/gemini_cache.py
import hashlib
import sqlite3
import json
from typing import Optional, List

class GeminiEmbeddingCache:
    def __init__(self, db_path: str = "gemini_embeddings_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                text TEXT,
                embedding TEXT
            )
        """)
        self.conn.commit()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        key = self._hash_text(text)
        cursor = self.conn.execute("SELECT embedding FROM embeddings WHERE id = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, text: str, embedding: List[float]):
        key = self._hash_text(text)
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (id, text, embedding) VALUES (?, ?, ?)",
            (key, text, json.dumps(embedding))
        )
        self.conn.commit()
