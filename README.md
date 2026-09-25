MyRAG Chatbot

A production-oriented Retrieval-Augmented Generation chatbot built with Streamlit, LangChain, Hugging Face sentence-transformer embeddings, Pinecone, and Groq. It answers from indexed PDF documents and cites the retrieved document pages instead of behaving like an ungrounded general-purpose chatbot.

## Features

- PDF ingestion from `data/documents/` using LangChain's PyPDF loader
- Recursive chunking with configurable 1,000-character chunks and 200-character overlap
- Local Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Pinecone cloud vector search with cosine similarity
- Groq chat generation through LangChain
- Grounded prompt that refuses to invent unsupported answers
- Stable chunk IDs and a namespace for repeatable, non-duplicating ingestion
- Streamlit chat history, clear-chat control, system status, and source citations
- Friendly handling for missing configuration, missing PDFs, connection failures, and model errors

## Architecture

### Query flow

```text
USER
  |
  v
STREAMLIT UI
  |
  v
LANGCHAIN RAG PIPELINE
  |
  v
Hugging Face embedding of the question
  |
  v
Pinecone similarity search
  |
  v
Relevant document chunks
  |
  v
Grounded prompt + Groq LLM
  |
  v
Answer and source citations in Streamlit
```

### Ingestion flow

```text
PDF files
  |
  v
PyPDFDirectoryLoader
  |
  v
RecursiveCharacterTextSplitter
  |
  v
Hugging Face embeddings
  |
  v
Pinecone upsert with source/page metadata
```

## Tech stack

- Python 3.11+
- Streamlit
- LangChain and LangChain Community
- `langchain-huggingface`
- `langchain-groq`
- `langchain-pinecone`
- Sentence Transformers
- Pinecone serverless
- Groq API
- PyPDF
- python-dotenv

## Project structure

```text
rag-chatbot/
├── app.py
├── ingest.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── documents/
│       └── .gitkeep
└── src/
    ├── __init__.py
    ├── config.py
    ├── document_loader.py
    ├── embeddings.py
    ├── rag.py
    └── vectorstore.py
```

Add your own PDFs to `data/documents/`. The directory is intentionally empty in the repository so no sample binary document needs to be committed.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` with the values described below.

## Environment variables

Required:

```dotenv
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
```

Optional settings include `GROQ_MODEL`, `EMBEDDING_MODEL`, `PINECONE_NAMESPACE`, `PINECONE_CLOUD`, `PINECONE_REGION`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `RETRIEVAL_K`. Never commit `.env` or display its values.

## Pinecone setup

1. Create an account at [Pinecone](https://www.pinecone.io/).
2. Create an API key and put it in `PINECONE_API_KEY`.
3. Choose an unused index name and put it in `PINECONE_INDEX_NAME`.
4. The application creates the index automatically if it does not exist.
5. The index is created with dimension **384**, which matches `all-MiniLM-L6-v2`, and cosine distance. The default serverless region is `aws` / `us-east-1`; change the optional variables if needed.

Do not manually create an index with a different dimension while using the default embedding model.

## Groq setup

1. Create an account and API key at [Groq Console](https://console.groq.com/).
2. Put the key in `GROQ_API_KEY`.
3. The default model is `openai/gpt-oss-20b`. Set `GROQ_MODEL` to another currently available Groq model if required by your account.

## Hugging Face embeddings

`HuggingFaceEmbeddings` downloads `sentence-transformers/all-MiniLM-L6-v2` the first time the application or ingestion script runs. It runs locally through Sentence Transformers; no Hugging Face API key is required for this public model. Embeddings turn each document chunk and user question into a 384-dimensional numerical vector, allowing Pinecone to find semantically related text even when exact words differ.

## Add documents and ingest

Place one or more PDF files in `data/documents/`, then run:

```powershell
python ingest.py
```

The script loads every PDF, splits pages into overlapping chunks, embeds them, and upserts them into Pinecone with filename, page, and chunk metadata. IDs include the source, page, character offset, and content hash, so rerunning ingestion updates the same chunks instead of inserting duplicates.

## Run the application

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit. The sidebar reports Pinecone status, retrieval count, embedding model, and Groq model. Each answer includes the distinct source filename and page numbers returned by retrieval.

## RAG workflow

1. The user submits a non-empty question.
2. The same Hugging Face embedding model converts the question into a vector.
3. Pinecone performs similarity search over the indexed PDF chunk vectors.
4. LangChain formats the retrieved chunks as context.
5. The context and question are sent to Groq with a grounding system instruction.
6. Groq generates an answer constrained by the retrieved context.
7. Streamlit displays the answer and the source pages used.

A normal LLM chatbot answers from its trained model knowledge. This application adds an external retrieval step and explicitly instructs the model to say when the provided documents do not contain the answer.

## Troubleshooting

- **Missing environment variables:** verify `.env` exists at the project root and contains all three required values.
- **No PDFs found:** add `.pdf` files under `data/documents/`, then rerun `python ingest.py`.
- **Dimension mismatch:** keep the default embedding model with a 384-dimensional Pinecone index, or recreate the index using the dimension of any replacement model.
- **Pinecone connection failure:** confirm the API key, index name, cloud, and region, and check Pinecone service status.
- **Groq failure:** verify the API key and that the selected model is available to your account.
- **Slow first run:** model weights are downloaded locally on first use.
- **Stale vectors:** rerun ingestion after changing PDFs. For a complete reset, delete the configured Pinecone index and let the ingestion script recreate it.

## How to Explain This Project in an Interview

- **What is RAG?** Retrieval-Augmented Generation retrieves relevant external knowledge and gives it to an LLM before generation.
- **Why embeddings?** They represent meaning as vectors, enabling semantic similarity search.
- **Why Hugging Face?** It provides a strong, open, lightweight sentence-transformer model that can run locally.
- **Why Pinecone?** It is a managed vector database designed for fast similarity search and production scaling.
- **Why LangChain?** It provides standard interfaces for loaders, splitters, embeddings, vector stores, prompts, retrievers, and models.
- **Why Groq?** It provides fast hosted inference for supported open models.
- **Why Streamlit?** It turns the Python pipeline into a usable chat interface quickly.
- **What happens when a question arrives?** The question is embedded, matched against Pinecone, combined with the retrieved chunks, and sent to Groq.
- **LLM chatbot versus RAG chatbot:** An LLM chatbot may rely only on model training, while a RAG chatbot grounds answers in a changing document collection.
- **What happens during ingestion?** PDFs are loaded, split, embedded, and upserted with metadata and stable IDs.
- **What happens during retrieval?** The question vector is compared with chunk vectors and the highest-similarity chunks are returned.
- **What happens during generation?** LangChain injects those chunks into a grounded prompt and Groq generates the final response.

## Future improvements

- Add authentication and per-user document namespaces.
- Add document deletion and an ingestion manifest.
- Add reranking and hybrid keyword/vector search.
- Add automated evaluation for retrieval recall and answer faithfulness.
- Add observability with LangSmith and deployment configuration.
