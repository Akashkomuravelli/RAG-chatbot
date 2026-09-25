"""Centralized application configuration and environment validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by ingestion and the Streamlit app."""

    groq_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "openai/gpt-oss-20b"
    pinecone_namespace: str = "documents-v1"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load required values without ever logging their contents."""
        missing = [
            name
            for name in ("GROQ_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")
            if not os.getenv(name)
        ]
        if missing:
            raise ConfigurationError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        return cls(
            groq_api_key=os.environ["GROQ_API_KEY"],
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
            pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            llm_model=os.getenv("GROQ_MODEL", cls.llm_model),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", cls.pinecone_namespace),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            retrieval_k=int(os.getenv("RETRIEVAL_K", cls.retrieval_k)),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", cls.pinecone_cloud),
            pinecone_region=os.getenv("PINECONE_REGION", cls.pinecone_region),
        )


DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
