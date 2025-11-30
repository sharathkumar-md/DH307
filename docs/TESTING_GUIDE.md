# Testing Guide

## Test Results

### System Status
- Python: 3.11.0 ✅
- Ollama: 0.13.0 ✅  
- Available Models: llama3.2:1b, llama3.1:8b ✅
- PyTorch: 2.8.0 (CPU) ✅

### Completed Tests
1. ✅ Document ingestion with HuggingFace embeddings
2. ✅ Vector database creation (FAISS)
3. ✅ Embeddings generation
4. ✅ Document retrieval from vector DB

### Ready to Use
Your RAG pipeline is ready! Use the existing vector database at `data/vector_db/`

## Launch the Chatbot

```bash
streamlit run rag_chat.py
```

### Recommended Configuration

In the Streamlit UI:
- **Vector Database Path**: `data/vector_db`
- **Use FAISS**: ✓ (checked)
- **LLM Provider**: `llama`
- **Model**: `llama3.2:1b` (faster) or `llama3.1:8b` (better quality)
- **Embedding Provider**: `huggingface` (recommended)
- **Temperature**: 0.7
- **Top K**: 4

### Alternative: Use Without Llama

If Ollama has issues, you can use OpenAI/Google:
1. Add API key to `.env`
2. Select provider in UI
3. Works the same way!

## Test Queries

Try these questions from your test_queries.json:
- "What is the formula for calculating EDD based on LMP?"
- "What is Sarcopenia in elderly health?"
- "What is the WIFS programme composition?"
