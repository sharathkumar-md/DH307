# Data Processing Scripts

## Two-Stage Pipeline

### Overview

```
PDFs → JSON → Chunks → Vector DB
```

This pipeline separates data extraction from chunking, making it easier to:
- Debug and inspect raw data
- Control chunking strategy
- Create accurate test queries
- Track document/chunk relationships

## Scripts

### 1. `parse_pdfs_to_json.py`
**Purpose:** Extract all content from PDFs into structured JSON

**Usage:**
```bash
python scripts/parse_pdfs_to_json.py \
  --pdf_dir data/documents \
  --output data/parsed_documents.json
```

**Output:** JSON file with structure:
```json
{
  "created_at": "2025-11-26T...",
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
          "content": "...",
          "metadata": {...}
        }
      ]
    }
  ]
}
```

### 2. `create_chunks_from_json.py`
**Purpose:** Create chunks from JSON and build vector database

**Usage:**
```bash
python scripts/create_chunks_from_json.py \
  --json data/parsed_documents.json \
  --output data/vector_db \
  --chunk_size 1000 \
  --chunk_overlap 200 \
  --embedding_model sentence-transformers/all-mpnet-base-v2
```

**Outputs:**
- `data/vector_db/` - FAISS vector database
- `data/vector_db/metadata.json` - Database metadata
- `data/vector_db/chunk_mapping.json` - **Important:** Maps chunk_index to source documents

### 3. `run_full_pipeline.py`
**Purpose:** Run both steps automatically

**Usage:**
```bash
# Run full pipeline
python scripts/run_full_pipeline.py

# With custom settings
python scripts/run_full_pipeline.py \
  --pdf_dir data/documents \
  --json_output data/parsed.json \
  --db_output data/my_vector_db \
  --chunk_size 800 \
  --chunk_overlap 150

# Skip parsing if JSON already exists
python scripts/run_full_pipeline.py --skip_parsing
```

## Chunk Mapping

After running the pipeline, check `data/vector_db/chunk_mapping.json`:

```json
[
  {
    "chunk_index": 0,
    "chunk_id": "document.pdf_chunk_0",
    "source_file": "document.pdf",
    "page": 0,
    "page_label": "1",
    "content_preview": "First 100 characters of chunk..."
  },
  ...
]
```

**Use this file to:**
- Find which chunk_index contains specific content
- Create test queries with correct chunk indices
- Understand document-to-chunk mapping

## Creating Test Queries

1. **Run the pipeline:**
   ```bash
   python scripts/run_full_pipeline.py
   ```

2. **Review chunk_mapping.json:**
   - Search for content you want to test
   - Note the `chunk_index` values

3. **Update test_queries.json:**
   ```json
   {
     "query_id": 1,
     "query": "What is the EDD formula?",
     "relevant_docs": ["21", "22"],  // Use actual chunk_index values!
     "query_type": "factual"
   }
   ```

4. **Run evaluation:**
   ```bash
   python evaluation/evaluate_retrieval.py
   ```

## Workflow

### Initial Setup
```bash
# 1. Place PDFs in data/documents/
# 2. Run full pipeline
python scripts/run_full_pipeline.py

# 3. Check chunk mapping
cat data/vector_db/chunk_mapping.json | head -20

# 4. Test chatbot
streamlit run rag_chat.py
```

### Re-chunking (change chunk size/overlap)
```bash
# Skip parsing, only re-chunk
python scripts/run_full_pipeline.py --skip_parsing --chunk_size 800
```

### Adding New Documents
```bash
# 1. Add new PDFs to data/documents/
# 2. Re-run full pipeline
python scripts/run_full_pipeline.py
```

## Benefits of This Approach

✅ **Separation of Concerns**
- Data extraction (PDF → JSON) separate from processing (JSON → Chunks)

✅ **Inspectable**
- Review raw extracted text in JSON before chunking
- See exact chunk boundaries and indices

✅ **Reproducible**
- JSON file serves as ground truth
- Re-chunk with different parameters without re-parsing

✅ **Accurate Evaluation**
- chunk_mapping.json shows exact indices
- Easy to create correct test queries

✅ **Debugging**
- If retrieval fails, check if problem is in extraction or chunking
- Inspect individual chunks easily

## File Structure After Pipeline

```
data/
├── documents/              # Input PDFs
│   ├── doc1.pdf
│   └── doc2.pdf
├── parsed_documents.json   # Intermediate JSON
└── vector_db/             # Output database
    ├── index.faiss
    ├── index.pkl
    ├── metadata.json      # DB metadata
    └── chunk_mapping.json # Chunk index reference (IMPORTANT!)
```

## Troubleshooting

### "No PDF files found"
- Check `--pdf_dir` path is correct
- Ensure PDFs are directly in the directory (not subdirectories)

### "No documents were created"
- Check the JSON file exists and is valid
- Verify JSON has non-empty page content

### Low recall in evaluation
- Check if test_queries.json uses correct chunk_index values
- Review chunk_mapping.json to find actual indices
- Update test queries accordingly

## Next Steps

After running the pipeline:
1. ✓ Review `chunk_mapping.json`
2. ✓ Create/update test queries with correct indices
3. ✓ Run evaluation
4. ✓ Use chatbot with `streamlit run rag_chat.py`
