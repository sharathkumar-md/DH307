# RAG Retrieval Evaluation

This folder contains scripts and results for evaluating the RAG (Retrieval-Augmented Generation) system's retrieval performance.

## Metrics Explained

### 1. Chunk Recall
**What it measures:** The proportion of relevant documents that were successfully retrieved.

**Formula:** `Recall = (Retrieved Relevant Docs) / (Total Relevant Docs)`

**Example:** If 3 documents are relevant and we retrieved 2 of them, recall = 2/3 = 0.67

**Why it matters:** High recall means the system finds most of the relevant information. Low recall means the system is missing important documents.

### 2. Precision
**What it measures:** The proportion of retrieved documents that are actually relevant.

**Formula:** `Precision = (Retrieved Relevant Docs) / (Total Retrieved Docs)`

**Example:** If we retrieved 5 documents and 2 are relevant, precision = 2/5 = 0.40

**Why it matters:** High precision means the system returns mostly relevant results. Low precision means many irrelevant documents are retrieved.

### 3. F1 Score
**What it measures:** Harmonic mean of precision and recall (balanced metric).

**Formula:** `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

**Why it matters:** Provides a single metric that balances both precision and recall.

### 4. Mean Reciprocal Rank (MRR)
**What it measures:** How high the first relevant document appears in the results.

**Formula:** `MRR = 1 / (Rank of First Relevant Doc)`

**Example:** If the first relevant doc is at position 3, MRR = 1/3 = 0.33

**Why it matters:** Measures ranking quality. Higher is better - means relevant docs appear earlier.

### 5. Hit Rate
**What it measures:** Whether at least one relevant document was found.

**Formula:** `Hit Rate = 1 if any relevant doc retrieved, else 0`

**Why it matters:** Basic measure of whether the system finds anything useful.

## Running the Evaluation

### Basic Usage

```bash
python evaluation/evaluate_retrieval.py
```

### What it does:
1. Loads the vector database
2. Runs all test queries from `data/test_queries.json`
3. Retrieves top-K documents for each query
4. Compares with ground truth
5. Calculates all metrics
6. Saves detailed results to JSON
7. Prints summary report

## Output Files

Results are saved with timestamps:
- `retrieval_results_YYYYMMDD_HHMMSS.json` - Full evaluation results

## Customizing the Evaluation

Edit these variables in `evaluate_retrieval.py`:

```python
VECTOR_DB_PATH = "data/vector_db"          # Path to your vector database
TEST_QUERIES_PATH = "data/test_queries.json"  # Path to test queries
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Must match your DB
TOP_K = 5  # Number of documents to retrieve
```

## Interpreting Results

### Good Performance:
- **Recall > 0.8:** System finds most relevant documents
- **Precision > 0.7:** Most retrieved docs are relevant
- **F1 > 0.75:** Good balance
- **MRR > 0.7:** Relevant docs appear early
- **Hit Rate > 0.95:** Almost always finds something relevant

### Areas for Improvement:
- **Low Recall:** Increase top_k, improve embeddings, or adjust chunking strategy
- **Low Precision:** Improve query processing, adjust retrieval parameters
- **Low MRR:** Improve ranking algorithm or reranking

## Query Types

The evaluation breaks down metrics by query type:
- **factual:** Straightforward fact-based questions
- **clinical:** Clinical/diagnostic questions
- **procedural:** How-to/process questions
- **safety:** Safety and contraindication questions

This helps identify which types of queries your system handles better.
