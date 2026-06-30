# 📄 Document Intelligence App

A multimodal document analysis app built with Python that extracts text from PDFs and images, and uses AI to answer questions about the content.

🔗 **[Try the live app](https://doc-intelligence-app-production.up.railway.app/)**

## Features
- Upload PDFs or images
- Automatic text extraction using OCR (Tesseract) and PyMuPDF
- AI-powered Q&A using GPT-4o-mini
- Clean browser-based UI built with Streamlit

## Technical Highlights
- **RAG Pipeline**: Documents are chunked, embedded, and stored in ChromaDB for semantic search — avoiding context window limits and reducing API costs
- **A/B Testing Framework**: Built with MLflow to compare GPT-4o-mini, GPT-3.5-turbo, and a locally-run HuggingFace model (DistilGPT-2) across response time, token usage, and answer quality
- **Multi-Document Support**: Upload and query across multiple documents simultaneously
- **Containerised with Docker**: Fully portable deployment using a custom Dockerfile
- **Deployed on Railway**: Live, publicly accessible application
- **Robust Error Handling**: Gracefully handles oversized files, corrupted documents, and API failures

## Tech Stack
- Python 3.11
- Streamlit
- OpenAI API (GPT-4o-mini)
- PyMuPDF (fitz)
- Tesseract OCR / pytesseract
- Pillow

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

pip install streamlit fastapi uvicorn pymupdf pytesseract pillow langchain langchain-openai openai chromadb pandas python-dotenv

4. Add your OpenAI API key to a `.env` file

OPENAI_API_KEY=your_key_here

5. Run the app

streamlit run app.py




## Project Structure
doc-intelligence-app/
├── src/
│   └── document_processor.py  # document processing engine
├── app.py                      # Streamlit frontend
├── .env                        # API keys (not committed to Git)
└── .gitignore

## Author
Tim Foley — Data Engineer transitioning into ML/AI Engineering

