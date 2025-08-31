# RAG Demo (LangChain-Free Minimal)

A minimal Retrieval-Augmented Generation demo:
- Ingest local docs
- Embed or TF-IDF index
- Retrieve top-k passages
- Ask LLM to answer with citations

## Why Minimal?
To keep dependencies light and easy to run. 

## Stack
- Python 3.10+
- scikit-learn (TF-IDF fallback)
- OpenAI (for embeddings & answer) OR just TF-IDF + heuristic answer

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # add your OpenAI or Azure OpenAI details
streamlit run app.py
