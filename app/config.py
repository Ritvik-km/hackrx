from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # ---- API Auth ----
    TEAM_TOKEN: str
    GITHUB_TOKEN: str  # used for GitHub-hosted embeddings
    GOOGLE_API_KEY: str

    # ---- Embeddings Config ----
    EMBEDDING_MODEL: str = "openai/text-embedding-3-large"
    EMBEDDING_PROVIDER: str = "github"  # only 'github' supported now
    EMBEDDING_ENDPOINT: str = "https://models.github.ai/inference"

    # ---- Chunking ----
    CHUNK_SIZE: int = 1400
    CHUNK_OVERLAP: int = 200

    # ---- Local Paths ----
    DATA_DIR: Path = Path("data")

    # ---- Pydantic meta ----
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
