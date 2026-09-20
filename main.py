"""
FastAPI wrapper for the Document Intelligence App.

This exposes the same pipeline that the Streamlit UI app.py uses (process_document ->
chunk_text -> embed_and_store -> retrieve_relevant_chunks, plus OpenAI Q&A,
summarisation, and A/B testing) as an HTTP API.

Run with:
    uvicorn main:app --reload
Then test at http://127.0.0.1:8000/docs
"""

import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


from src.document_processor import process_document
from src.rag_processor import chunk_text, embed_and_store, retrieve_relevant_chunks
from src.ab_testing import run_ab_test


app = FastAPI(
    title="Document Intelligence API",
    description="OCR + RAG document question-answering service",
    version="0.1.0",
)

# In-memory store: document_id -> {filename, text}
# Mirrors what st.session_state was doing in the Streamlit app, just
# keyed by document_id instead of living in the session.
DOCUMENT_STORE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# OpenAI helper functions — pulled straight from app.py, unchanged in logic.
# ---------------------------------------------------------------------------

def answer_question(extracted_text: str, question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about documents. Only answer based on the document content provided. If the answer is not in the document, say so.",
                },
                {
                    "role": "user",
                    "content": f"Document content:\n{extracted_text}\n\nQuestion: {question}",
                },
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't get an answer from OpenAI: {str(e)}"


def summarise_document(extracted_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarises documents. Provide a clear and concise summary with the key points highlighted.",
                },
                {
                    "role": "user",
                    "content": f"Please summarise the following document:\n\n{extracted_text}",
                },
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't generate a summary: {str(e)}"


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class QueryRequest(BaseModel):
    document_id: str
    question: str
    use_rag: bool = True


class QueryResponse(BaseModel):
    document_id: str
    question: str
    answer: str


class SummariseResponse(BaseModel):
    document_id: str
    summary: str


class ABTestRequest(BaseModel):
    document_id: str
    question: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a file, runs it through process_document (OCR/extraction),
    then chunks and embeds it via the RAG pipeline.
    """
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}",
        )

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    try:
        text = process_document(file_bytes, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {exc}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from this file")

    try:
        chunks = chunk_text(text)
        embed_and_store(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chunking/embedding failed: {exc}")

    document_id = str(uuid.uuid4())
    DOCUMENT_STORE[document_id] = {"filename": file.filename, "text": text}

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunk_count=len(chunks),
    )


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Answers a question about a previously uploaded document.
    Uses RAG (retrieve_relevant_chunks) by default, same as the toggle
    in the Streamlit app; set use_rag=false to pass the full document text
    instead.
    """
    doc = DOCUMENT_STORE.get(request.document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        if request.use_rag:
            context = retrieve_relevant_chunks(request.question)
            answer = answer_question(context, request.question)
        else:
            answer = answer_question(doc["text"], request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    return QueryResponse(
        document_id=request.document_id,
        question=request.question,
        answer=answer,
    )


@app.post("/documents/{document_id}/summarise", response_model=SummariseResponse)
async def summarise(document_id: str):
    """Summarises a previously uploaded document."""
    doc = DOCUMENT_STORE.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    summary = summarise_document(doc["text"])
    return SummariseResponse(document_id=document_id, summary=summary)


@app.post("/ab-test")
async def ab_test(request: ABTestRequest):
    """
    Runs the same A/B test as the Streamlit app: retrieves relevant chunks
    for the question, then compares gpt-4o-mini, gpt-3.5-turbo, and local
    DistilGPT-2 on the same context.
    """
    try:
        context = retrieve_relevant_chunks(request.question)
        results = run_ab_test(request.question, context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"A/B test failed: {exc}")

    return results


@app.get("/health")
async def health_check():
    return {"status": "ok"}
