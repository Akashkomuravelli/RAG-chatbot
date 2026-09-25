"""Hugging Face embedding model construction."""

from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings


def create_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Create local sentence-transformer embeddings.

    Embeddings convert text into numerical vectors that can be searched for
    semantic similarity in Pinecone.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
