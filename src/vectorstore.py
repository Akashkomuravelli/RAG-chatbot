"""Pinecone connection and vector-store helpers."""

from __future__ import annotations

import hashlib
from typing import Iterable

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import Settings

EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension


def get_pinecone_client(settings: Settings) -> Pinecone:
    """Create a Pinecone client using the environment-provided API key."""
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index(settings: Settings) -> None:
    """Create the configured index once if it does not already exist."""
    client = get_pinecone_client(settings)
    if not client.has_index(settings.pinecone_index_name):
        client.create_index(
            name=settings.pinecone_index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )


def create_vector_store(settings: Settings, embeddings) -> PineconeVectorStore:
    """Connect to the existing configured Pinecone index."""
    index = get_pinecone_client(settings).Index(settings.pinecone_index_name)
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=settings.pinecone_namespace,
    )


def document_ids(documents: Iterable[Document]) -> list[str]:
    """Return stable IDs so repeated ingestion upserts rather than duplicates."""
    ids = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", 0)
        start = document.metadata.get("start_index", 0)
        digest = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:16]
        ids.append(f"{source}:{page}:{start}:{digest}")
    return ids
