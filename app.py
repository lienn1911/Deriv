from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_engine import RAGEngine

app = FastAPI(title="RAG Document Q&A Service")
engine = RAGEngine()

DOCS_PATH = "docs"


class AskRequest(BaseModel):
    question: str


@app.post("/index")
def index_documents():
    try:
        stats = engine.index_documents(DOCS_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "indexed", **stats}


@app.post("/ask")
def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = engine.ask(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
