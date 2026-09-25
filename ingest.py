"""Ingest PDF files from data/documents into Pinecone."""

from __future__ import annotations

from src.config import DOCUMENTS_DIR, ConfigurationError, Settings
from src.document_loader import load_and_split_documents
from src.embeddings import create_embeddings
from src.vectorstore import create_vector_store, document_ids, ensure_index


def main() -> None:
    """Run the PDF -> chunks -> embeddings -> Pinecone ingestion flow."""
    try:
        settings = Settings.from_environment()
        chunks = load_and_split_documents(DOCUMENTS_DIR, settings)
        embeddings = create_embeddings(settings)
        ensure_index(settings)
        vector_store = create_vector_store(settings, embeddings)
        vector_store.add_documents(documents=chunks, ids=document_ids(chunks))
        print(f"Ingested {len(chunks)} chunks into Pinecone.")
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
