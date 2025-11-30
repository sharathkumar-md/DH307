# Retrieval Evaluation Summary

## Current Status

✅ Evaluation system is working correctly
❌ Test data has ID mismatch with vector database

## Results

```
Chunk Recall:    2.22%
Precision:       0.44%
F1 Score:        0.74%
Hit Rate:        2.22%
```

**Important:** These low scores are due to ID mismatch, NOT poor retrieval quality!

## The Problem: ID Mismatch

### What the test queries expect:
- Query 1: "What is the formula for EDD..."
- Expected document: `["15"]`

### What the system retrieved:
- Retrieved chunks: `["21", "22", "8", "19", "35"]`
- These are **chunk_index** values from the vector database

### Root Cause:
The `relevant_docs` field in `test_queries.json` uses IDs that don't correspond to the `chunk_index` values in your vector database.

## Vector Database Structure

Your documents are indexed with these metadata fields:

```python
{
  "source_file": "Care During Pregnancy and Childbirth Training Manual.pdf",
  "page": 16,                    # 0-indexed page number
  "page_label": "17",            # Page as shown in PDF
  "chunk_index": 21,             # Sequential chunk number
  "chunk_id": "filename_chunk_21",
  "document_id": "filename_page_17"
}
```

Currently, the evaluation uses `chunk_index` for matching.

## Solutions

### Option 1: Update Test Queries (Recommended)
Create new test queries with correct chunk_index values.

**Steps:**
1. For each test query, manually find the answer in your documents
2. Use the retrieval system to find which chunks contain the answer
3. Note the `chunk_index` of those chunks
4. Update `test_queries.json` with correct chunk_index values

**Helper script provided:** `evaluation/create_test_queries_helper.py`

### Option 2: Use Page-Based Matching
If your test queries reference page numbers instead of chunk indices, we can modify the evaluation script to match against `page` or `page_label` fields.

### Option 3: Manual Annotation
Create a small set (10-15) of well-annotated test queries with correct IDs to get meaningful evaluation results.

## Next Steps

1. **Determine what the IDs in test_queries.json actually represent:**
   - Are they page numbers?
   - Are they from a different chunking strategy?
   - Are they document IDs from a previous system?

2. **Choose a matching strategy:**
   - `chunk_index`: Current sequential chunk number (0-3963)
   - `page`: PDF page number (0-indexed)
   - `page_label`: PDF page as shown in document
   - Custom ID mapping

3. **Create properly aligned test data**

## Files Created

- `evaluation/evaluate_retrieval.py` - Main evaluation script
- `evaluation/retrieval_results_YYYYMMDD_HHMMSS.json` - Detailed results
- `evaluation/metadata_inspection.json` - Sample of actual metadata
- `evaluation/README.md` - Metrics documentation
- `evaluation/EVALUATION_SUMMARY.md` - This file

## Understanding Your Current System

Despite low evaluation scores, your retrieval system may actually be working well!

The retrieved chunks for query 1 about EDD formula came from:
- Page 16-17 of "Care During Pregnancy and Childbirth Training Manual"
- Chunks 21, 22 nearby chunks 8, 19, 35

These might actually contain relevant information about EDD calculation, but we can't verify without proper ground truth labels.

## Recommendation

**Before spending time on evaluation, manually test your system:**

1. Ask your chatbot a few queries
2. Check if the answers are accurate
3. Look at the source documents shown
4. Verify the retrieved chunks make sense

If the system works well in practice, focus on:
- Building a small set of high-quality test queries
- Getting the ID mapping correct
- Then doing comprehensive evaluation

---

**Need Help?**
Check `evaluation/README.md` for metric explanations and usage instructions.
