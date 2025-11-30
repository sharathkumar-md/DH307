# New Data Pipeline Summary

## What Changed

### Old Approach (Single Step)
```
PDFs → Direct Chunking → Vector DB
```
- Hard to debug
- No visibility into raw data
- Difficult to create accurate test queries

### New Approach (Two Steps)
```
Step 1: PDFs → JSON (raw data)
Step 2: JSON → Chunks → Vector DB
```
- Inspectable raw data
- Reproducible chunking
- Easy to create test queries with correct IDs

## New Scripts Created

### 1. `scripts/parse_pdfs_to_json.py`
Extracts all content from PDFs into structured JSON.

**Output:** `data/parsed_documents.json`

### 2. `scripts/create_chunks_from_json.py`
Creates chunks from JSON and builds vector database.

**Outputs:**
- `data/vector_db/` - FAISS database
- `data/vector_db/metadata.json` - DB info
- `data/vector_db/chunk_mapping.json` - **Key file for creating test queries!**

### 3. `scripts/run_full_pipeline.py`
Runs both steps automatically.

## How to Use

### Run Complete Pipeline
```bash
python scripts/run_full_pipeline.py
```

### With Custom Settings
```bash
python scripts/run_full_pipeline.py \
  --chunk_size 800 \
  --chunk_overlap 150
```

### Re-chunk Without Re-parsing
```bash
python scripts/run_full_pipeline.py --skip_parsing
```

## Key File: chunk_mapping.json

This file maps chunk indices to source documents:

```json
[
  {
    "chunk_index": 0,  ← Use this for test queries!
    "chunk_id": "document.pdf_chunk_0",
    "source_file": "document.pdf",
    "page": 0,
    "page_label": "1",
    "content_preview": "First 100 characters..."
  }
]
```

## Creating Accurate Test Queries

1. **Review chunk_mapping.json** to find content
2. **Note the chunk_index** of relevant chunks
3. **Update test_queries.json** with correct indices:

```json
{
  "query_id": 1,
  "query": "What is the EDD formula?",
  "relevant_docs": ["21", "22"],  ← Actual chunk_index values
  "query_type": "factual"
}
```

## Benefits

✅ **Debuggable** - See raw extracted text before chunking
✅ **Reproducible** - Re-chunk with different parameters
✅ **Accurate** - Know exact chunk indices for evaluation
✅ **Fast iteration** - Skip parsing when re-chunking

## Current Status

🔄 **PDF Parsing** - Running (45 PDFs)
⏳ **Chunking** - Pending
⏳ **Evaluation** - Pending

## Next Steps After Pipeline Completes

1. Check `data/parsed_documents.json` (raw data)
2. Review `data/vector_db/chunk_mapping.json`
3. Update test queries with correct chunk indices
4. Re-run evaluation
5. Test chatbot

## Files Structure

```
data/
├── documents/              # Input PDFs
├── parsed_documents.json   # ← Raw extracted data
└── vector_db/
    ├── index.faiss        # Vector database
    ├── index.pkl
    ├── metadata.json
    └── chunk_mapping.json # ← Use this for test queries!

scripts/
├── parse_pdfs_to_json.py
├── create_chunks_from_json.py
├── run_full_pipeline.py
└── README.md

evaluation/
├── evaluate_retrieval.py
├── README.md
└── EVALUATION_SUMMARY.md
```

## Configuration Options

You switched to **qwen2.5:0.5b** for the LLM due to GPU memory constraints.
Fallback is set to **gemini-1.5-flash** if qwen fails.

### Embedding Model
Currently using: `sentence-transformers/all-mpnet-base-v2`

### Chunking Parameters
- Chunk size: 1000 tokens
- Chunk overlap: 200 tokens
- Strategy: Recursive text splitting

## Improvements Made

1. ✅ Removed Streamlit configuration UI → Hardcoded in `CONFIG`
2. ✅ Added LLM fallback system (llama → gemini)
3. ✅ Enhanced prompt for general conversation + RAG
4. ✅ Two-stage data pipeline for better control
5. ✅ Comprehensive evaluation framework
6. ✅ Chunk mapping for accurate test queries

## Documentation Created

- `scripts/README.md` - Pipeline usage
- `evaluation/README.md` - Metrics explanation
- `evaluation/EVALUATION_SUMMARY.md` - Current eval status
- `PIPELINE_SUMMARY.md` - This file

---

**Status:** Pipeline running. Once complete, you'll have full visibility into your data and can create accurate evaluations!
