# Pipeline Scripts

This folder contains all the scripts I use to process documents and build the vector database.

## Quick Start

Just run this to process all your PDFs:
```bash
python pipeline/run_full_pipeline.py
```

That's it! It will parse PDFs, create chunks, and build the vector database.

## The Scripts

### `run_full_pipeline.py` - Main Orchestrator

Runs the complete pipeline from PDFs to vector database.

**Basic usage:**
```bash
python pipeline/run_full_pipeline.py
```

**With custom options:**
```bash
python pipeline/run_full_pipeline.py \
  --pdf_dir data/documents \
  --json_output data/parsed.json \
  --db_output data/vector_db \
  --chunk_size 800 \
  --chunk_overlap 150
```

**Skip parsing (re-chunk only):**
```bash
python pipeline/run_full_pipeline.py --skip_parsing
```

This is useful when you want to try different chunk sizes without re-parsing all the PDFs.

### `parse_pdfs_to_json.py` - Stage 1

Extracts text from PDFs and saves to JSON format.

**Usage:**
```bash
python pipeline/parse_pdfs_to_json.py \
  --pdf_dir data/documents \
  --output data/parsed_documents.json
```

**What it does:**
- Reads all PDFs in the directory
- Extracts text page by page
- Saves to structured JSON format
- Preserves page numbers and metadata

**Output:** `data/parsed_documents.json` containing all extracted text.

### `create_chunks_from_json.py` - Stage 2

Creates chunks from JSON and builds the vector database.

**Usage:**
```bash
python pipeline/create_chunks_from_json.py \
  --json data/parsed_documents.json \
  --output data/vector_db \
  --chunk_size 1000 \
  --chunk_overlap 200
```

**What it does:**
- Reads the JSON file
- Splits text into chunks
- Generates embeddings
- Builds FAISS vector database
- Creates chunk_mapping.json for reference

**Outputs:**
- `data/vector_db/index.faiss` - Vector index
- `data/vector_db/index.pkl` - Document store
- `data/vector_db/metadata.json` - Database info
- `data/vector_db/chunk_mapping.json` - Chunk index mapping

### `create_chunks_semantic.py` - Semantic Chunking

Alternative chunking strategy using semantic similarity.

**Usage:**
```bash
python pipeline/create_chunks_semantic.py \
  --json data/parsed_documents.json \
  --output data/vector_db_semantic
```

This creates chunks based on semantic breaks rather than fixed token counts. Experimental.

### `ingestion.py` - Legacy Single-Step Ingestion

The old single-step approach. Still works but deprecated in favor of the two-stage pipeline.

**Usage:**
```bash
python pipeline/ingestion.py \
  --input path/to/file.pdf \
  --db_path data/vector_db \
  --faiss
```

I don't use this anymore but keeping it for reference.

## Understanding the Output Files

### parsed_documents.json

Intermediate file with raw extracted text:
```json
{
  "created_at": "2025-11-30T...",
  "num_documents": 45,
  "total_pages": 2588,
  "documents": [...]
}
```

Good for debugging parsing issues.

### chunk_mapping.json

Critical file for creating test queries:
```json
[
  {
    "chunk_index": 0,
    "chunk_id": "document.pdf_chunk_0",
    "source_file": "document.pdf",
    "page": 0,
    "page_label": "1",
    "content_preview": "First 100 chars..."
  }
]
```

Use this to find which chunk_index contains specific content.

### metadata.json

Database statistics:
```json
{
  "num_chunks": 3964,
  "num_documents": 45,
  "embedding_model": "sentence-transformers/all-mpnet-base-v2",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "created_at": "2025-11-30T..."
}
```

## Common Workflows

### Initial Setup
```bash
# 1. Place PDFs in data/documents/
# 2. Run full pipeline
python pipeline/run_full_pipeline.py

# 3. Check the results
cat data/vector_db/metadata.json
```

### Experiment with Chunk Sizes
```bash
# Try 800 tokens
python pipeline/run_full_pipeline.py --skip_parsing --chunk_size 800

# Try 1200 tokens
python pipeline/run_full_pipeline.py --skip_parsing --chunk_size 1200

# Compare results in the chatbot
```

### Add New Documents
```bash
# 1. Add new PDFs to data/documents/
# 2. Re-run full pipeline (will process everything)
python pipeline/run_full_pipeline.py
```

### Debug Parsing Issues
```bash
# 1. Run stage 1 only
python pipeline/parse_pdfs_to_json.py

# 2. Check parsed_documents.json
cat data/parsed_documents.json | grep "problematic_doc.pdf" -A 20

# 3. Fix the issue, then run stage 2
python pipeline/create_chunks_from_json.py
```

## Configuration Options

### Chunking Parameters

- **chunk_size**: Number of tokens per chunk (default: 1000)
  - Smaller (500-800): Better precision, more chunks
  - Larger (1200-1500): More context, fewer chunks

- **chunk_overlap**: Tokens shared between chunks (default: 200)
  - Helps preserve context across chunk boundaries
  - Usually 10-20% of chunk_size

### Embedding Model

Currently using: `sentence-transformers/all-mpnet-base-v2`

To change, edit in `create_chunks_from_json.py`:
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Faster, smaller
```

## Troubleshooting

### "No PDF files found"
- Check the path: `ls data/documents/`
- Make sure PDFs are directly in the directory, not in subdirectories

### "No documents were created"
- Check if parsed_documents.json exists
- Verify it has non-empty content: `cat data/parsed_documents.json | head`

### Pipeline runs but chatbot retrieval is poor
- Check chunk_mapping.json to see how documents were split
- Try different chunk sizes
- Verify embeddings match between pipeline and chatbot

### Out of memory
- Process fewer documents at once
- Use a smaller embedding model
- Reduce chunk size

## Tips

**For Better Results:**
- Use chunk size 800-1200 (sweet spot for most documents)
- Keep overlap at 15-20% of chunk size
- Review chunk_mapping.json after processing

**For Faster Processing:**
- Use the two-stage approach (can skip parsing on re-runs)
- Enable GPU if available
- Process documents in batches if you have many

**For Debugging:**
- Always check parsed_documents.json first
- Use chunk_mapping.json to verify chunking
- Test with a small set of documents first

## Next Steps After Running Pipeline

1. Check `data/vector_db/metadata.json` - verify counts
2. Review `data/vector_db/chunk_mapping.json` - understand structure
3. Run the chatbot: `streamlit run app/rag_chat.py`
4. Test with queries from `data/test_queries.json`
5. Run evaluation: `python evaluation/evaluate_retrieval.py`
