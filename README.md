# 📄 Document Intelligence App

A multimodal document intelligence application built with Python. Upload PDFs or images, extract text automatically using OCR, and use AI to ask questions, generate summaries, and compare model performance — all through both a browser UI and a REST API.

🔗 **[Try the live app](https://doc-intelligence-app-production.up.railway.app/)**
> Password protected — contact me for access

## Features
- Upload PDFs or images (including scanned documents)
- Automatic text extraction using OCR (Tesseract) and PyMuPDF
- AI-powered Q&A using GPT-4o-mini with chat history
- Document summarisation
- RAG (Retrieval Augmented Generation) pipeline for large documents
- A/B testing framework comparing GPT-4o-mini, GPT-3.5-turbo, and a local HuggingFace model
- Multi-document support — upload and query across several documents simultaneously
- REST API layer built with FastAPI — exposable to other systems and services
- Experiment tracking and model comparison logged with MLflow
- Clean browser-based UI built with Streamlit

## Technical Highlights
- **RAG Pipeline**: Documents are chunked, embedded, and stored in ChromaDB for semantic search — avoiding context window limits and reducing API costs
- **A/B Testing Framework**: Built with MLflow to compare GPT-4o-mini, GPT-3.5-turbo, and a locally-run HuggingFace model (DistilGPT-2) across response time, token usage, and answer quality
- **REST API**: FastAPI wrapper exposing the full pipeline as HTTP endpoints — meaning the document intelligence functionality can be integrated into other applications, not just used via the browser UI
- **Multi-Document Support**: Upload and query across multiple documents simultaneously
- **Containerised with Docker**: Fully portable deployment using a custom Dockerfile
- **Deployed on Railway**: Live, publicly accessible application
- **Robust Error Handling**: Gracefully handles oversized files, corrupted documents, API failures, and context window limits

## What is the REST API?
REST API stands for Representational State Transfer — a set of conventions for how APIs should be designed. It describes using URLs to identify resources, HTTP methods to describe actions (GET, POST, DELETE), and JSON to exchange data.

FastAPI is the Python library used here to build an API that follows those conventions. Think of REST as the rules of the road, and FastAPI as the car.

Having both a Streamlit UI and a REST API means this app can be used in two ways:
- **By a human** — through the browser interface at the live URL above
- **By other software** — by sending HTTP requests to the API endpoints below, allowing the document intelligence pipeline to be integrated into other systems

## API Endpoints
Run locally with `uvicorn main:app --reload`, then visit `http://127.0.0.1:8000/docs` for the interactive documentation.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents` | Upload a document — returns a `document_id` |
| POST | `/query` | Ask a question about an uploaded document |
| POST | `/documents/{document_id}/summarise` | Summarise an uploaded document |
| POST | `/ab-test` | Run A/B test across models on a question |
| GET | `/health` | Health check |

## Tech Stack
- Python 3.11
- Streamlit
- FastAPI + Uvicorn
- OpenAI API (GPT-4o-mini, GPT-3.5-turbo)
- LangChain + ChromaDB (RAG pipeline)
- PyMuPDF (fitz) + Tesseract OCR / pytesseract
- Pillow
- HuggingFace Transformers (DistilGPT-2)
- MLflow (experiment tracking)
- Docker
- Railway (deployment)

## Getting Started

### Prerequisites
- Python 3.11
- Tesseract OCR installed ([download here](https://github.com/UB-Mannheim/tesseract/wiki))
- OpenAI API key

### Installation

1. Clone the repository

git clone https://github.com/tfoley3/doc-intelligence-app.git
cd doc-intelligence-app

2. Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Add your OpenAI API key to a `.env` file

OPENAI_API_KEY=your_key_here
APP_PASSWORD=your_password_here

5. Run the app

streamlit run app.py

6. Or run the FastAPI server

uvicorn main:app --reload


## Project Structure

doc-intelligence-app/
├── src/
│ ├── document_processor.py # OCR and text extraction engine
│ ├── rag_processor.py # RAG pipeline — chunking, embedding, retrieval
│ ├── ab_testing.py # A/B testing framework with MLflow logging
│ └── huggingface_model.py # Local HuggingFace model integration
├── app.py # Streamlit frontend
├── main.py # FastAPI REST API
├── Dockerfile # Container definition
├── requirements.txt # Python dependencies
├── .env # API keys (not committed to Git)
└── .gitignore


## Author
Tim Foley — Data Engineer transitioning into Software/AI Engineering
[GitHub](https://github.com/tfoley3) | [Live App](https://doc-intelligence-app-production.up.railway.app/)
