# RAG Chatbot for Document Q&A

This is my Retrieval-Augmented Generation (RAG) chatbot that lets me ask questions about my PDF documents and get accurate answers with source citations.

## What It Does

- Processes PDF documents and creates a searchable vector database
- Answers questions based on the document content
- Shows source citations for transparency
- Runs locally with Llama models

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Process Your Documents
```bash
python pipeline/run_full_pipeline.py
```

This will:
- Parse all PDFs in `data/documents/`
- Create chunks with proper metadata
- Build a FAISS vector database
- Generate a chunk mapping file for reference

### 3. Run the Chatbot
```bash
streamlit run app/rag_chat.py
```

Open the browser and start asking questions about your documents!

## Project Structure

```
├── app/
│   └── rag_chat.py              # Streamlit chatbot interface
├── pipeline/
│   ├── ingestion.py             # Simple document ingestion (deprecated)
│   ├── parse_pdfs_to_json.py    # Step 1: Extract PDF content
│   ├── create_chunks_from_json.py # Step 2: Create chunks + vector DB
│   ├── create_chunks_semantic.py  # Semantic chunking option
│   ├── run_full_pipeline.py     # Run complete pipeline
│   └── README.md                # Pipeline documentation
├── evaluation/
│   └── evaluate_answer_generation.py # Answer quality evaluation
├── PIPELINE_SUMMARY.md          # Pipeline design decisions
└── data/
    ├── documents/               # Your PDFs go here
    ├── parsed_documents.json    # Intermediate parsed data
    └── vector_db/              # FAISS database + metadata
```

## How the Pipeline Works

### Two-Stage Approach

I designed the pipeline in two stages for better control and debugging:

**Stage 1: Parse PDFs → JSON**
```bash
python pipeline/parse_pdfs_to_json.py --pdf_dir data/documents --output data/parsed_documents.json
```

**Stage 2: JSON → Chunks → Vector DB**
```bash
python pipeline/create_chunks_from_json.py --json data/parsed_documents.json --output data/vector_db
```

Or run both together:
```bash
python pipeline/run_full_pipeline.py
```

### Why Two Stages?

- **Debuggable**: Can inspect raw extracted text before chunking
- **Flexible**: Re-chunk with different parameters without re-parsing
- **Traceable**: chunk_mapping.json shows exactly where each chunk came from

## Configuration

### Using Local Llama Models (Free!)

The chatbot uses Ollama for local LLM inference:

1. Install Ollama from https://ollama.com/download
2. Pull a model: `ollama pull llama3`
3. Run the chatbot - it will use Llama automatically

### Using OpenAI or Google

Add your API key to `.env`:
```env
OPENAI_API_KEY=your_key_here
# or
GOOGLE_API_KEY=your_key_here
```

Then select the provider in the Streamlit UI.

### Chunking Parameters

Edit in `pipeline/run_full_pipeline.py`:
- `chunk_size`: 1000 tokens (default)
- `chunk_overlap`: 200 tokens (default)
- Embedding model: `sentence-transformers/all-mpnet-base-v2`

## Evaluation

I've built an evaluation tool to measure answer generation quality:

```bash
python evaluation/evaluate_answer_generation.py
```

This tests the quality of generated answers using metrics like relevance, faithfulness, and coherence.

## Documentation

- **README.md** (this file) - Project overview and quick start
- **PIPELINE_SUMMARY.md** - Pipeline design decisions and learnings
- **pipeline/README.md** - Detailed pipeline usage guide

## Tips

**For Better Results:**
- Use clear, specific questions
- Increase `top_k` in the UI to retrieve more chunks
- Try different chunk sizes (800-1200 works well)

**For Faster Performance:**
- Use HuggingFace embeddings instead of Llama
- Enable GPU acceleration if available
- Use smaller Llama models (llama3.2:1b vs llama3.1:8b)

**For Development:**
- Check `chunk_mapping.json` to understand the database structure
- Review `parsed_documents.json` to debug parsing issues
- Check `metadata.json` in vector_db for database info

## Tech Stack

- **LangChain** - RAG framework
- **FAISS** - Vector database
- **Sentence Transformers** - Embeddings
- **Streamlit** - Web interface
- **Ollama** - Local LLM inference
- **PyPDF** - PDF parsing

## License

See LICENSE file for details.
