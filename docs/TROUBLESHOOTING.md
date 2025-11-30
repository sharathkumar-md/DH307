# Troubleshooting Guide

## Error: "llama runner process has terminated: exit status 2"

This error occurs when Ollama models crash. Here are solutions:

### Solution 1: Use OpenAI/Google API (EASIEST - Works Immediately)

**If you have an API key:**

1. Add to `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   # OR
   GOOGLE_API_KEY=your_google_api_key
   ```

2. In Streamlit UI:
   - **LLM Provider**: Select `openai` or `google`
   - **Model**: Select `gpt-3.5-turbo` or `gemini-pro`
   - **Embedding Provider**: Keep as `huggingface`
   - Ask your question again!

**This works perfectly with your existing vector database!**

### Solution 2: Force Ollama to Use CPU

The issue might be Ollama trying to use GPU. Force CPU mode:

**Windows:**
```bash
# Stop Ollama
taskkill /F /IM ollama.exe

# Set CPU-only mode
setx OLLAMA_NUM_GPU 0

# Restart Ollama (close terminal and reopen, or restart Ollama from Start menu)
```

**Then restart Streamlit:**
```bash
streamlit run rag_chat.py
```

### Solution 3: Try Smaller Model or Reinstall

**Try the 1B model (lighter):**
In Streamlit UI, make sure you're using `llama3.2:1b` (not `llama3.1:8b`)

**Or pull the model again:**
```bash
ollama pull llama3.2:1b
```

### Solution 4: Use HuggingFace Models for LLM (No Ollama needed)

This requires modifying the code to use HuggingFace Transformers for LLM. Let me know if you want this option.

## Recommended Quick Fix

**Use OpenAI or Google temporarily:**

1. Get a free API key:
   - OpenAI: https://platform.openai.com/api-keys
   - Google: https://makersuite.google.com/app/apikey

2. Add to `.env`:
   ```env
   OPENAI_API_KEY=your_key_here
   ```

3. In Streamlit:
   - LLM Provider: `openai`
   - Model: `gpt-3.5-turbo`
   - Embedding: `huggingface` (keep this!)
   - Temperature: 0.7

4. Ask: "What is Gingivitis?"

**This will work immediately while we debug Ollama!**

## Why This Happens

- Ollama may be trying to use GPU but failing
- Windows + CPU-only PyTorch + Ollama can have compatibility issues
- The 8B model might be too large for your system RAM

## Best Immediate Solution

**Use OpenAI with HuggingFace embeddings:**
- Your vector DB (47 PDFs) works perfectly ✓
- Fast and reliable ✓
- Costs ~$0.002 per query (very cheap) ✓
- We can fix Ollama later ✓
