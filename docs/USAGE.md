# RAG Chatbot Usage Guide

## Overview
This project includes RAGFlow components with simplified standalone scripts for document ingestion and chatbot interaction.

**🆕 NOW CONFIGURED TO USE LLAMA MODELS BY DEFAULT!**

The chatbot now uses local Llama models via Ollama, which means:
- ✅ Free to use (no API costs)
- ✅ Privacy-friendly (runs locally)
- ✅ Fast and efficient
- ✅ Multiple model options

See [LLAMA_SETUP.md](LLAMA_SETUP.md) for detailed setup instructions.

## What's Included

### Core Files
- **`ingestion.py`** - Document ingestion script (PDF, TXT, URL support)
- **`rag_chat.py`** - Streamlit-based chat interface
- **`rag/`** - RAGFlow core modules (LLM, NLP, utilities)
- **`deepdoc/`** - Document parsing modules from RAGFlow

### Removed (Heavy Components)
- API server infrastructure
- Database models and services
- Docker configurations
- Benchmarking tools
- Specialized document parsers (audio, email, presentations, etc.)

## Setup

### 1. Install Ollama (for Llama models)
Download and install from: https://ollama.com/download

Then download a model:
```bash
ollama pull llama3
```

See [LLAMA_SETUP.md](LLAMA_SETUP.md) for detailed instructions.

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys (Optional)
Only needed if using OpenAI or Google instead of Llama:

Create a `.env` file:
```env
# Ollama (optional, default is http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Google (optional)
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### Ingest Documents

**Ingest a PDF file (using Llama - default):**
```bash
python ingestion.py --input data/documents/mydoc.pdf --db_path data/vector_db --faiss
```

**Ingest a text file:**
```bash
python ingestion.py --input data/documents/myfile.txt --db_path data/vector_db --faiss
```

**Ingest from URL:**
```bash
python ingestion.py --input https://example.com/article --db_path data/vector_db --faiss
```

**Use HuggingFace embeddings (faster, recommended):**
```bash
python ingestion.py --input mydoc.pdf --db_path data/vector_db --faiss --embedding huggingface
```

**Use specific Llama model for embeddings:**
```bash
python ingestion.py --input mydoc.pdf --db_path data/vector_db --faiss --embedding llama --embedding_model llama3
```

**Use OpenAI embeddings:**
```bash
python ingestion.py --input mydoc.pdf --db_path data/vector_db --faiss --embedding openai
```

**Use Chroma instead of FAISS:**
```bash
python ingestion.py --input mydoc.pdf --db_path data/vector_db --embedding huggingface
```

**Custom chunk settings:**
```bash
python ingestion.py --input mydoc.pdf --db_path data/vector_db --faiss --chunk_size 500 --chunk_overlap 100
```

### Run the Chatbot

**Start the Streamlit interface:**
```bash
streamlit run rag_chat.py
```

This will open a web browser with the chat interface where you can:
- Configure LLM provider (Llama, OpenAI, or Google)
- Select model (Llama2, Llama3, Mistral, GPT-3.5, GPT-4, Gemini, etc.)
- Configure embedding provider (Llama, HuggingFace, OpenAI, or Google)
- Adjust temperature and retrieval settings
- Chat with your documents
- View source documents for each answer

## Configuration Options

### Ingestion.py Arguments
- `--input`: Path to document or URL (required)
- `--db_path`: Vector database save path (default: `data/vector_db`)
- `--faiss`: Use FAISS (if omitted, uses Chroma)
- `--chunk_size`: Text chunk size (default: 1000)
- `--chunk_overlap`: Overlap between chunks (default: 200)
- `--embedding`: Embedding provider - `llama`, `huggingface`, `openai`, or `google` (default: llama)
- `--embedding_model`: Specific model name (optional)

### RAG Chat UI Settings
- **Vector Database Path**: Where your vector DB is stored
- **Use FAISS**: Toggle between FAISS and Chroma
- **LLM Provider**: Llama (default), OpenAI, or Google
- **Model**: Specific model to use (llama2, llama3, mistral, gpt-3.5-turbo, etc.)
- **Embedding Provider**: Llama, HuggingFace (recommended), OpenAI, or Google
- **Embedding Model**: Custom embedding model (optional)
- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative)
- **Top K Documents**: Number of relevant chunks to retrieve

## Example Workflow (Using Llama - Free & Local!)

1. **Setup Ollama:**
```bash
# Download from https://ollama.com/download and install
# Then pull a model
ollama pull llama3
```

2. **Ingest your healthcare documents:**
```bash
# Recommended: Use HuggingFace for fast embeddings
for file in data/documents/*.pdf; do
    python ingestion.py --input "$file" --db_path data/vector_db --faiss --embedding huggingface
done
```

3. **Run the chatbot:**
```bash
streamlit run rag_chat.py
```

4. **Configure in the UI:**
   - LLM Provider: Select "llama" (default)
   - Model: Select "llama3"
   - Embedding Provider: Select "huggingface"
   - Start chatting!

5. **Test with your existing queries:**
   - Use queries from `data/test_queries.json`
   - View retrieved sources and answers
   - Everything runs locally - no API costs!

## Advanced: Using RAGFlow Components

The `rag/` and `deepdoc/` folders contain RAGFlow's core modules that can be imported in custom scripts:

```python
# Example: Using RAGFlow's document parser
from deepdoc.parser import PdfParser

# Example: Using RAGFlow's LLM integrations
from rag.llm import ChatOpenAI
```

## Troubleshooting

### Import Errors
If you get import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Vector Database Not Found
Make sure you've run ingestion before starting the chat:
```bash
python ingestion.py --input your_document.pdf --db_path data/vector_db --faiss
```

### API Key Errors
Verify your `.env` file has the correct API keys and they're not expired.

## Next Steps

- Add more documents to your vector database
- Experiment with different LLM models
- Adjust chunk size and overlap for better retrieval
- Try different embedding models (OpenAI vs Google)
- Integrate RAGFlow components into custom workflows
