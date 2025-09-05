# 🦜🔗 RAG Chatbot

A modular Retrieval-Augmented Generation (RAG) chatbot using LangChain, OpenAI, and FAISS/Chroma.

## Features
- Document ingestion from PDF, TXT, or URL
- Text splitting and embedding
- Vector database with FAISS or Chroma
- Retrieval + Generation pipeline
- Streamlit web interface
- Modular codebase

## Setup
1. **Clone the repo and enter the directory**
2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
3. **Configure API keys**
   - Copy `.env.example` to `.env` and add your OpenAI API key.

## Usage
### Ingest Documents
```bash
python ingestion.py --input path/to/file.pdf --db_path vector_db --faiss
```
- Use `--faiss` for FAISS (default), omit for Chroma.

### Run the Chatbot (Streamlit)
```bash
streamlit run rag_chat.py
```

## File Structure
- `ingestion.py` – Document loading, splitting, embedding, and ingestion
- `rag_chat.py` – RAG chatbot interface and pipeline
- `.env.example` – Example for environment variables
- `requirements.txt` – Python dependencies

## Notes
- Extend `ingestion.py` to support more file types or embedding models.
- The code is modular and well-commented for easy extension.
