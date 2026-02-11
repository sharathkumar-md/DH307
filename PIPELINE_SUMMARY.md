# Pipeline Design Summary

## Why I Built a Two-Stage Pipeline

When I started this project, I was using a single-step approach where PDFs went directly into chunks and then into the vector database. This made debugging hard and created issues with evaluation because I couldn't see the intermediate data or track where chunks came from.

So I redesigned it as a two-stage pipeline.

## The Problem with Single-Stage

**Old Approach:**
```
PDFs → Chunks → Vector DB
```

Problems I faced:
- Couldn't inspect raw extracted text
- Hard to debug parsing errors
- No way to re-chunk without re-parsing
- Difficult to create accurate test queries (didn't know chunk IDs)
- Slow iteration when experimenting with chunk sizes

## My Two-Stage Solution

**New Approach:**
```
Stage 1: PDFs → JSON (raw extracted text)
Stage 2: JSON → Chunks → Vector DB
```

### Stage 1: PDF Parsing
Script: `pipeline/parse_pdfs_to_json.py`

Extracts all content from PDFs and saves to `data/parsed_documents.json`:
```json
{
  "created_at": "2025-11-30T...",
  "num_documents": 45,
  "total_pages": 2588,
  "documents": [
    {
      "filename": "document.pdf",
      "total_pages": 80,
      "pages": [
        {
          "page_number": 0,
          "page_label": "1",
          "content": "Full page text...",
          "metadata": {...}
        }
      ]
    }
  ]
}
```

### Stage 2: Chunking and Vector DB Creation
Script: `pipeline/create_chunks_from_json.py`

Reads the JSON, creates chunks, builds vector database, and generates the critical `chunk_mapping.json` file:
```json
[
  {
    "chunk_index": 0,
    "chunk_id": "document.pdf_chunk_0",
    "source_file": "document.pdf",
    "page": 0,
    "page_label": "1",
    "content_preview": "First 100 characters of chunk..."
  }
]
```

This mapping file is essential for creating test queries with correct chunk indices.

## Running the Pipeline

### Option 1: Run Complete Pipeline
```bash
python pipeline/run_full_pipeline.py
```

### Option 2: Run Stages Separately
```bash
# Stage 1: Parse PDFs
python pipeline/parse_pdfs_to_json.py --pdf_dir data/documents --output data/parsed_documents.json

# Stage 2: Create chunks and database
python pipeline/create_chunks_from_json.py --json data/parsed_documents.json --output data/vector_db
```

### Option 3: Skip Parsing (Re-chunk Only)
```bash
python pipeline/run_full_pipeline.py --skip_parsing --chunk_size 800
```

## Why This Approach Works Better

### 1. Debuggable
I can open `parsed_documents.json` and see exactly what was extracted from each PDF before any chunking happens.

### 2. Flexible
Want to try different chunk sizes? Just re-run stage 2:
```bash
python pipeline/create_chunks_from_json.py --chunk_size 800 --chunk_overlap 150
```
No need to re-parse the PDFs!

### 3. Traceable
The `chunk_mapping.json` file shows me:
- Which chunk contains what content
- Which PDF page each chunk came from
- The exact chunk index for evaluation

### 4. Faster Iteration
When experimenting with chunking strategies, I only run stage 2 which is much faster than re-parsing 45 PDFs.

## Files Generated

```
data/
├── documents/              # Input PDFs
├── parsed_documents.json   # Stage 1 output (intermediate)
└── vector_db/
    ├── index.faiss         # FAISS vector index
    ├── index.pkl           # Pickled document store
    ├── metadata.json       # Database metadata
    └── chunk_mapping.json  # Chunk index reference (IMPORTANT!)
```

## Current Configuration

- **Embedding Model:** `sentence-transformers/all-mpnet-base-v2`
- **Chunk Size:** 1000 tokens
- **Chunk Overlap:** 200 tokens
- **Splitting Strategy:** Recursive character text splitter

## Creating Test Queries

The chunk_mapping.json file makes this easy:

1. Open `chunk_mapping.json`
2. Search for content you want to test
3. Note the `chunk_index` value
4. Add to `test_queries.json`:

```json
{
  "query_id": 1,
  "query": "What is the EDD formula?",
  "relevant_docs": ["21", "22"],  // Use actual chunk_index values
  "query_type": "factual"
}
```