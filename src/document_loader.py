"""PDF loading and chunking utilities."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


def load_and_split_documents(
    documents_dir: Path, settings: Settings
) -> list[Document]:
    """Load all PDFs and split them into overlapping chunks."""
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")
    if not list(documents_dir.glob("*.pdf")):
        raise FileNotFoundError(f"No PDF documents found in {documents_dir}")

    try:
        documents = PyPDFDirectoryLoader(str(documents_dir), recursive=False).load()
    except Exception as exc:
        raise RuntimeError("Could not load one or more PDF documents.") from exc

    if not documents:
        raise RuntimeError("PDF loading returned no document pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        source = Path(str(chunk.metadata.get("source", "unknown")))
        chunk.metadata["source"] = source.name
        if "page" in chunk.metadata:
            chunk.metadata["page"] = int(chunk.metadata["page"]) + 1
    return chunks
