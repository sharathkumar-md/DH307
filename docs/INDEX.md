# RAG Chatbot Documentation

Welcome to the RAG Chatbot documentation! This folder contains all guides and documentation for the project.

## 📚 Documentation Index

### Getting Started
1. **[Llama Setup Guide](./LLAMA_SETUP.md)** - Complete guide to setting up Ollama and Llama models
2. **[Usage Guide](./USAGE.md)** - How to use ingestion and chat scripts
3. **[GPU Setup Guide](./GPU_SETUP.md)** - GPU acceleration for faster performance
4. **[Project README](./README.md)** - Original project overview

### Quick Links
- **Installation**: See [LLAMA_SETUP.md](./LLAMA_SETUP.md#installation)
- **First Time Setup**: See [USAGE.md](./USAGE.md#setup)
- **Troubleshooting**: See [LLAMA_SETUP.md](./LLAMA_SETUP.md#troubleshooting)

## 🚀 Quick Start

### 1. Install Ollama
```bash
# Download from: https://ollama.com/download
ollama pull llama3
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Chatbot
```bash
streamlit run rag_chat.py
```

## 🎯 Features

- ✅ **Local Llama Models** - Free, private, and fast
- ✅ **GPU Acceleration** - Automatic GPU detection and usage
- ✅ **Multiple LLM Providers** - Llama, OpenAI, Google
- ✅ **Flexible Embeddings** - Llama, HuggingFace, OpenAI, Google
- ✅ **PDF Support** - Process healthcare documents
- ✅ **Conversation Memory** - Context-aware responses
- ✅ **Source Citations** - See where answers come from

## 💡 Best Practices

### For Speed:
- Use HuggingFace embeddings with GPU
- Use smaller Llama models: `phi3` or `mistral`

### For Quality:
- Use `llama3.1` models
- Increase `top_k` documents

### For Privacy:
- Use Llama models (everything runs locally)

## 📖 Detailed Guides

- **[LLAMA_SETUP.md](./LLAMA_SETUP.md)** - Ollama and Llama models setup
- **[GPU_SETUP.md](./GPU_SETUP.md)** - GPU acceleration setup
- **[USAGE.md](./USAGE.md)** - Complete usage guide with examples
