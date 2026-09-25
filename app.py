"""Streamlit interface for the Pinecone-backed RAG chatbot."""

from __future__ import annotations

import streamlit as st

from src.config import DOCUMENTS_DIR, ConfigurationError, Settings
from src.document_loader import load_and_split_documents
from src.embeddings import create_embeddings
from src.rag import RAGPipeline
from src.vectorstore import create_vector_store, document_ids, ensure_index

st.set_page_config(page_title="Knowledge Base Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Connecting to the knowledge base...")
def initialize_pipeline() -> tuple[Settings, RAGPipeline]:
    """Initialize expensive models once per Streamlit process."""
    settings = Settings.from_environment()
    embeddings = create_embeddings(settings)
    vector_store = create_vector_store(settings, embeddings)
    return settings, RAGPipeline(settings, vector_store)


def render_sources(sources) -> None:
    """Render unique source/page citations for retrieved chunks."""
    if not sources:
        st.caption("No source chunks were retrieved.")
        return
    seen = set()
    for document in sources:
        source = str(document.metadata.get("source", "Unknown document"))
        page = document.metadata.get("page")
        label = f"{source} - Page {page}" if page else source
        if label not in seen:
            st.markdown(f"- {label}")
            seen.add(label)


def ingest_uploaded_files(uploaded_files, settings: Settings) -> int:
    """Save uploaded PDFs and upsert their chunks into Pinecone."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        (DOCUMENTS_DIR / uploaded_file.name).write_bytes(uploaded_file.getvalue())

    chunks = load_and_split_documents(DOCUMENTS_DIR, settings)
    embeddings = create_embeddings(settings)
    ensure_index(settings)
    vector_store = create_vector_store(settings, embeddings)
    vector_store.add_documents(documents=chunks, ids=document_ids(chunks))
    return len(chunks)


def main() -> None:
    st.title("MyRAG chatbot")
    st.write("Ask questions about the PDFs in your Pinecone-backed knowledge base.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("System status")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        try:
            settings, pipeline = initialize_pipeline()
            st.success("Pinecone connected")
            st.caption(f"Retrieves up to {settings.retrieval_k} chunks per question")
            st.caption(f"Embeddings: `{settings.embedding_model}`")
            st.caption(f"LLM: `{settings.llm_model}`")

            uploaded_files = st.file_uploader(
                "Upload PDF documents",
                type=["pdf"],
                accept_multiple_files=True,
            )
            if uploaded_files and st.button("Ingest uploaded PDFs", use_container_width=True):
                with st.spinner("Saving, embedding, and indexing PDFs..."):
                    try:
                        chunk_count = ingest_uploaded_files(uploaded_files, settings)
                        st.success(f"Indexed {chunk_count} chunks.")
                    except Exception:
                        st.error("Could not ingest the uploaded PDFs. Check the files and Pinecone configuration.")
        except ConfigurationError as exc:
            st.error(str(exc))
            st.info("Add the required values to .env, then restart Streamlit.")
            return
        except Exception:
            st.error("Could not connect to the knowledge base. Check Pinecone and ingestion setup.")
            return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    render_sources(message["sources"])

    question = st.chat_input("Ask a question about your documents")
    if not question or not question.strip():
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                result = pipeline.ask(question)
                st.markdown(result.answer)
                with st.expander("Sources"):
                    render_sources(result.sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                    }
                )
            except ValueError as exc:
                st.error(str(exc))
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The request failed ({type(exc).__name__}). Restart Streamlit and check your API configuration.")


if __name__ == "__main__":
    main()
