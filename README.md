cat > README.md << 'EOF'
# Finance RAG System — Equity Analysis

A Retrieval-Augmented Generation (RAG) system built to answer questions
about stock market fundamentals and equity analysis using local LLMs.

## Tech Stack
- Python 3.11
- LangChain (LCEL)
- FAISS (vector database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Ollama + Llama 3 (local LLM)

## Knowledge Base (7 PDFs)
Documents from NSE, SEBI, and CFA Institute covering:
- NSE Basics of Financial Markets
- SEBI Financial Education Booklet
- SEBI Beginner's Guide to Capital Markets
- SEBI Securities Market Booklet
- CFA Equity Valuation
- SEBI Financial Education Part A
- SEBI College Students Booklet

## Project Structure

finance-rag/
├── pdfs/          ← Add PDFs here (see Download section)
├── faiss_index/   ← Auto-created by ingest.py
├── ingest.py      ← Run once to build vector index
├── rag.py         ← Run to chat with your documents
├── .env           ← Config (model names, paths)
└── requirements.txt

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/finance-rag.git
cd finance-rag
```

### 2. Create virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download PDFs
Create a `pdfs/` folder and download the 7 documents listed above
from NSE (nseindia.com), SEBI (investor.sebi.gov.in), and CFA Institute.

### 5. Install and start Ollama
```bash
brew install ollama
ollama pull llama3
ollama serve
```

### 6. Run ingestion (once)
```bash
python ingest.py
```

### 7. Start chatting
```bash
python rag.py
```

## Sample Questions
- What is fundamental analysis?
- What is the difference between primary and secondary market?
- How is P/E ratio calculated?
- What is DCF valuation?
- What is the role of SEBI in Indian securities market?
EOF
<img width="2876" height="1316" alt="image" src="https://github.com/user-attachments/assets/e0e8d75f-4d33-4701-b175-fe653566a88e" />
<img width="2876" height="1316" alt="image" src="https://github.com/user-attachments/assets/18a0fdc8-d40a-4d48-9412-74e647b93240" />



