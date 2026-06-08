# RAG Document Q&A Service

A local RAG (Retrieval-Augmented Generation) API that indexes plain-text documents and answers questions from them using TF-IDF retrieval and extractive answering. No external LLM APIs required.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```

Server starts at `http://localhost:8000`.

## Usage

### 1. Index documents

```bash
curl -X POST http://localhost:8000/index
```

Response:
```json
{"status": "indexed", "documents": 5, "chunks": 23}
```

### 2. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the pricing for the Business plan?"}'
```

Response:
```json
{
  "answer": "Business Plan ($18/user/month): ...",
  "sources": [
    {"source": "pricing.txt", "text": "...", "score": 0.52}
  ],
  "confidence": "answered_from_docs"
}
```

### 3. Question with no relevant answer

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the weather today?"}'
```

Response:
```json
{
  "answer": "I could not find sufficient information in the documents to answer this question.",
  "sources": [],
  "confidence": "insufficient_context"
}
```

## How It Works

1. **Indexing**: Reads `.txt` files from `docs/`, splits into paragraph-based chunks (~200 words max), builds a TF-IDF matrix.
2. **Retrieval**: Converts the question to a TF-IDF vector, finds the top-3 most similar chunks via cosine similarity.
3. **Answer extraction**: Ranks sentences within the top chunks by keyword overlap with the question, returns the most relevant sentences.
4. **Confidence**: If the best chunk similarity is below 0.1, returns `"insufficient_context"` instead of guessing.

## Tradeoffs and Improvements

- **TF-IDF vs embeddings**: TF-IDF is simple and fast but misses semantic similarity (e.g., "cost" vs "pricing"). With more time, sentence-transformers would improve retrieval quality.
- **Extractive vs generative**: The current approach returns relevant sentences verbatim. A local LLM (e.g., via Ollama) would produce more natural answers.
- **Chunking**: Simple paragraph-based splitting. Overlapping chunks or sliding windows would improve context continuity.
- **Persistence**: The index is in-memory and lost on restart. Adding a vector store (FAISS, ChromaDB) would fix this.
