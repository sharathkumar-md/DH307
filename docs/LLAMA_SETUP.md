# Llama Model Setup Guide

This guide will help you set up and use Llama models locally with Ollama for your RAG chatbot.

## What is Ollama?

Ollama is a lightweight, easy-to-use tool that lets you run large language models (like Llama) locally on your computer. It's free and works on Windows, Mac, and Linux.

## Installation

### Step 1: Install Ollama

**Windows:**
1. Download Ollama from: https://ollama.com/download
2. Run the installer
3. Ollama will start automatically as a background service

**Mac/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Verify Installation

Open a terminal and run:
```bash
ollama --version
```

You should see the version number if installed correctly.

## Download Llama Models

### Available Models

Ollama supports many models. Here are the recommended ones:

**For General Use:**
- `llama2` (7B) - Good balance of speed and quality
- `llama3` (8B) - Latest Llama, better performance
- `llama3.1` (8B) - Most recent, best quality

**For Faster Performance:**
- `mistral` (7B) - Fast and efficient
- `phi3` (3.8B) - Very fast, good for quick responses

**For Better Quality:**
- `llama3:70b` - Best quality (requires ~40GB RAM)
- `mixtral` (8x7B) - Excellent quality (requires ~26GB RAM)

### Download a Model

```bash
# Download Llama 2 (recommended to start)
ollama pull llama2

# Or download Llama 3 (better quality)
ollama pull llama3

# Or download Mistral (faster)
ollama pull mistral
```

This will download the model (~4-7GB depending on the model).

### Test the Model

```bash
ollama run llama2
```

You can chat with the model directly. Type `/bye` to exit.

## Using Llama with Your RAG Chatbot

### Option 1: Default Setup (Llama is now default!)

The scripts are already configured to use Llama by default. Just run:

**1. Ingest documents:**
```bash
python ingestion.py --input data/documents/mydoc.pdf --db_path data/vector_db --faiss
```

**2. Run the chatbot:**
```bash
streamlit run rag_chat.py
```

The chatbot will automatically use Llama2 via Ollama!

### Option 2: Specify a Different Model

**For ingestion:**
```bash
# Use Llama3 for embeddings
python ingestion.py --input mydoc.pdf --db_path data/vector_db --faiss --embedding llama --embedding_model llama3
```

**For the chatbot:**
- Open the Streamlit UI
- Select "llama" as LLM Provider
- Choose your preferred model (llama2, llama3, mistral, etc.)

### Option 3: Use HuggingFace Embeddings (No Ollama needed for embeddings)

If Ollama is slow for embeddings, you can use HuggingFace:

```bash
python ingestion.py --input mydoc.pdf --db_path data/vector_db --faiss --embedding huggingface
```

This uses the `all-MiniLM-L6-v2` model which downloads automatically.

## Configuration

### Environment Variables

Create or update your `.env` file:

```env
# Ollama Configuration (optional)
OLLAMA_BASE_URL=http://localhost:11434

# Only needed if using OpenAI or Google as fallback
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

The default Ollama URL is `http://localhost:11434`, so you only need to set this if you're running Ollama on a different port or server.

## Performance Tips

### For Faster Embeddings:
Use HuggingFace instead of Ollama for embeddings:
```bash
python ingestion.py --input mydoc.pdf --faiss --embedding huggingface
```

### For Better Quality:
Use Llama3:
```bash
# Download Llama3
ollama pull llama3

# Then select it in the Streamlit UI
```

### For Multiple Documents:
Ingest documents one by one or in a loop:
```bash
for file in data/documents/*.pdf; do
    python ingestion.py --input "$file" --db_path data/vector_db --faiss --embedding huggingface
done
```

## Troubleshooting

### "Connection refused" or "Cannot connect to Ollama"

**Solution 1:** Check if Ollama is running
```bash
# Windows
# Check Task Manager for "ollama" process

# Mac/Linux
ps aux | grep ollama
```

**Solution 2:** Start Ollama manually
```bash
ollama serve
```

**Solution 3:** Verify the URL
Make sure Ollama is running on `http://localhost:11434`

### Model Not Found

```bash
# List downloaded models
ollama list

# Pull the model if not found
ollama pull llama2
```

### Out of Memory Errors

- Try a smaller model (phi3, mistral instead of llama3:70b)
- Close other applications
- For embeddings, use HuggingFace instead of Ollama

### Slow Performance

- Use a smaller model (phi3 is fastest)
- Use HuggingFace for embeddings
- Reduce `chunk_size` in ingestion
- Lower the `top_k` documents in the UI

## Model Comparison

| Model | Size | RAM Required | Speed | Quality |
|-------|------|--------------|-------|---------|
| phi3 | 3.8B | 4GB | ⚡⚡⚡ | ⭐⭐ |
| llama2 | 7B | 8GB | ⚡⚡ | ⭐⭐⭐ |
| mistral | 7B | 8GB | ⚡⚡ | ⭐⭐⭐ |
| llama3 | 8B | 8GB | ⚡⚡ | ⭐⭐⭐⭐ |
| llama3.1 | 8B | 8GB | ⚡⚡ | ⭐⭐⭐⭐ |
| mixtral | 8x7B | 26GB | ⚡ | ⭐⭐⭐⭐⭐ |

## Example Workflow with Llama

### Complete Setup from Scratch

1. **Install Ollama:**
   ```bash
   # Download from https://ollama.com/download
   ```

2. **Download Llama model:**
   ```bash
   ollama pull llama3
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ingest your documents:**
   ```bash
   # Using HuggingFace for fast embeddings
   python ingestion.py --input data/documents/mydoc.pdf --db_path data/vector_db --faiss --embedding huggingface
   ```

5. **Run the chatbot:**
   ```bash
   streamlit run rag_chat.py
   ```

6. **In the Streamlit UI:**
   - LLM Provider: Select "llama"
   - Model: Select "llama3"
   - Embedding Provider: Select "huggingface"
   - Start chatting!

## Advanced: Custom Ollama Models

You can create custom models with specific prompts:

```bash
# Create a Modelfile
cat > Modelfile << EOF
FROM llama3
PARAMETER temperature 0.7
SYSTEM You are a helpful medical assistant specializing in healthcare information.
EOF

# Create the custom model
ollama create medical-assistant -f Modelfile

# Use it
ollama run medical-assistant
```

Then select "medical-assistant" in the Streamlit UI.

## Resources

- Ollama Documentation: https://github.com/ollama/ollama
- Available Models: https://ollama.com/library
- Llama Information: https://ai.meta.com/llama/

## Need Help?

- Check Ollama logs: `ollama logs` (Mac/Linux) or Event Viewer (Windows)
- Ollama community: https://discord.gg/ollama
- File an issue in the project repo
