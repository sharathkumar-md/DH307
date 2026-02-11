# Complete Evaluation Process Explanation

**Healthcare RAG System Evaluation Framework**
**Author:** Sharath Kumar MD
**Project:** Maharashtra Government CHO Training System (KCDH, IIT Bombay)

---

## Table of Contents

1. [Overview](#overview)
2. [Evaluation Architecture](#evaluation-architecture)
3. [Retrieval Methods](#retrieval-methods)
4. [Metrics Explained](#metrics-explained)
5. [Configuration Parameters](#configuration-parameters)
6. [Matching Logic](#matching-logic)
7. [Complete Workflow](#complete-workflow)
8. [Test Configurations](#test-configurations)
9. [Results Analysis](#results-analysis)

---

## 1. Overview

The evaluation system tests the **Retrieval-Augmented Generation (RAG)** system's ability to retrieve relevant healthcare documents based on user queries. It measures performance across multiple dimensions using 6 different metrics and tests various retrieval strategies.

### Key Components
- **Test Queries:** 133 healthcare-related questions with known relevant pages
- **Retrieval Methods:** Dense (FAISS), Sparse (BM25), and Hybrid approaches
- **Metrics:** 6 comprehensive evaluation metrics
- **Configurations:** Multiple alpha values and top-K settings

---

## 2. Evaluation Architecture

### 2.1 System Components

```
┌─────────────────────┐
│   Test Queries      │
│  (test_queries.json)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│        Vector Database (FAISS)      │
│  - Semantic/Dense Embeddings        │
│  - 512-dimensional vectors          │
│  - Similarity search capability     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│        BM25 Index (Sparse)          │
│  - Keyword-based search             │
│  - TF-IDF scoring                   │
│  - Token matching                   │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│      Hybrid Search Engine           │
│  - Weighted combination (alpha)     │
│  - Score normalization              │
│  - Re-ranking                       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│     Retrieved Documents (K docs)    │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│       Metrics Calculation           │
│  - Precision@K, Recall@K, F1@K      │
│  - MRR, Hit Rate@K, NDCG@K          │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│         Results Storage             │
│  - CSV files with all results       │
│  - Summary statistics               │
│  - Visualizations                   │
└─────────────────────────────────────┘
```

### 2.2 Input Structure

**Test Query Format (JSON):**
```json
{
  "query": "What are the symptoms of anaemia in pregnancy?",
  "relevant_docs": ["34", "35"],
  "query_type": "factual"
}
```

**Fields:**
- `query`: The question text
- `relevant_docs`: List of page numbers containing the answer
- `query_type`: Category (factual, procedural, diagnostic, etc.)

---

## 3. Retrieval Methods

### 3.1 Dense Retrieval (FAISS)

**What it is:** Semantic search using vector embeddings

**How it works:**
1. Query is converted to a 512-dimensional embedding vector
2. FAISS index computes cosine similarity with all document vectors
3. Returns top-K documents with highest similarity scores

**Formula:**
```
similarity(query, doc) = cosine_similarity(embed(query), embed(doc))
                       = (Q · D) / (||Q|| × ||D||)
```

**Strengths:**
- Understands semantic meaning
- Handles paraphrasing and synonyms
- Good for conceptual queries

**Weaknesses:**
- May miss exact keyword matches
- Computationally expensive
- Requires good embeddings

**Configuration:**
- Alpha = 1.0 (pure dense)
- Model: `sentence-transformers/all-MiniLM-L6-v2`

### 3.2 Sparse Retrieval (BM25)

**What it is:** Keyword-based search using term frequency statistics

**How it works:**
1. Query and documents are tokenized into words
2. BM25 algorithm scores each document based on:
   - Term frequency (TF): How often query terms appear
   - Inverse document frequency (IDF): Rarity of terms
   - Document length normalization

**Formula:**
```
BM25(D, Q) = Σ IDF(qi) × [f(qi, D) × (k1 + 1)] / [f(qi, D) + k1 × (1 - b + b × |D|/avgdl)]

Where:
- qi = query term i
- f(qi, D) = frequency of qi in document D
- |D| = length of document D
- avgdl = average document length
- k1 = term frequency saturation parameter (default: 1.5)
- b = length normalization parameter (default: 0.75)
- IDF(qi) = log[(N - df(qi) + 0.5) / (df(qi) + 0.5)]
  - N = total number of documents
  - df(qi) = document frequency of term qi
```

**Strengths:**
- Exact keyword matching
- Fast computation
- No embedding required
- Good for specific term queries

**Weaknesses:**
- Doesn't understand semantics
- Struggles with synonyms
- Sensitive to exact wording

**Configuration:**
- Alpha = 0.0 (pure sparse)
- Implementation: `rank_bm25` library

### 3.3 Hybrid Retrieval

**What it is:** Weighted combination of dense and sparse methods

**How it works:**
1. Retrieve candidates from both FAISS (dense) and BM25 (sparse)
2. Normalize scores from each method to [0, 1] range
3. Combine scores using weighted sum with alpha parameter
4. Re-rank and return top-K documents

**Formula:**
```
hybrid_score(doc) = α × dense_score_norm(doc) + (1 - α) × sparse_score_norm(doc)

Where:
- α (alpha) = weight for dense retrieval [0, 1]
- (1 - α) = weight for sparse retrieval
- dense_score_norm = normalized dense similarity score
- sparse_score_norm = normalized BM25 score
```

**Score Normalization:**
```
normalized_score = (score - min_score) / (max_score - min_score)
```

**Alpha Configurations:**
- α = 1.0: Pure Dense (100% semantic, 0% keyword)
- α = 0.7: Dense-Heavy Hybrid (70% semantic, 30% keyword)
- α = 0.5: Balanced Hybrid (50% semantic, 50% keyword)
- α = 0.3: Sparse-Heavy Hybrid (30% semantic, 70% keyword)
- α = 0.0: Pure Sparse (0% semantic, 100% keyword)

**Strengths:**
- Best of both worlds
- Adapts to different query types
- Robust to various phrasings

**Weaknesses:**
- More complex computation
- Requires tuning alpha parameter
- May introduce noise if methods conflict

---

## 4. Metrics Explained

### 4.1 Precision@K

**What it measures:** Of the K documents retrieved, what fraction are relevant?

**Formula:**
```
Precision@K = (Number of Relevant Documents in Top-K) / K
```

**Calculation Example:**
```
Query: "What are symptoms of anaemia?"
Expected relevant pages: [34, 35]
Retrieved pages (K=5): [34, 78, 35, 102, 45]

Relevant retrieved: 2 (pages 34 and 35)
Precision@5 = 2/5 = 0.40 (40%)
```

**Interpretation:**
- Range: [0, 1]
- 1.0 = Perfect (all retrieved docs are relevant)
- 0.0 = Worst (no retrieved docs are relevant)
- Higher is better

**Use case:** Measures retrieval accuracy - important when you want to avoid showing irrelevant information

**Code Implementation:**
```python
def calculate_precision_at_k(retrieved_docs, relevant_docs, k):
    if not retrieved_docs or k == 0:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = 0

    for doc in retrieved_k:
        doc_page = str(doc.metadata.get('page_number', ''))
        for rel_page in relevant_docs:
            if doc_page == rel_page:  # Exact match
                relevant_retrieved += 1
                break

    return relevant_retrieved / k
```

### 4.2 Recall@K

**What it measures:** Of all relevant documents, what fraction did we retrieve in top-K?

**Formula:**
```
Recall@K = (Number of Relevant Documents in Top-K) / (Total Number of Relevant Documents)
```

**Calculation Example:**
```
Query: "What are symptoms of anaemia?"
Expected relevant pages: [34, 35, 89]  # 3 total relevant
Retrieved pages (K=5): [34, 78, 35, 102, 45]

Relevant retrieved: 2 (pages 34 and 35)
Recall@5 = 2/3 = 0.667 (66.7%)
```

**Interpretation:**
- Range: [0, 1]
- 1.0 = Perfect (found all relevant docs)
- 0.0 = Worst (found no relevant docs)
- Higher is better

**Use case:** Measures completeness - important when you need to find all relevant information

**Code Implementation:**
```python
def calculate_recall_at_k(retrieved_docs, relevant_docs, k):
    if not relevant_docs:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = 0

    # For each relevant doc, check if we found it
    for rel_page in relevant_docs:
        for doc in retrieved_k:
            doc_page = str(doc.metadata.get('page_number', ''))
            if doc_page == rel_page:
                relevant_retrieved += 1
                break

    return relevant_retrieved / len(relevant_docs)
```

### 4.3 F1 Score@K

**What it measures:** Harmonic mean of Precision and Recall

**Formula:**
```
F1@K = 2 × (Precision@K × Recall@K) / (Precision@K + Recall@K)
```

**Calculation Example:**
```
Using previous example:
Precision@5 = 0.40
Recall@5 = 0.667

F1@5 = 2 × (0.40 × 0.667) / (0.40 + 0.667)
     = 2 × 0.267 / 1.067
     = 0.500 (50%)
```

**Why harmonic mean?**
- Punishes extreme imbalances
- Requires both precision and recall to be good
- Better than arithmetic mean for this purpose

**Comparison:**
```
Scenario A: Precision=0.9, Recall=0.1
  Arithmetic mean = 0.5
  Harmonic mean (F1) = 0.18  ← Better reflects poor performance

Scenario B: Precision=0.5, Recall=0.5
  Arithmetic mean = 0.5
  Harmonic mean (F1) = 0.5   ← Balanced performance
```

**Interpretation:**
- Range: [0, 1]
- 1.0 = Perfect precision and recall
- Balances accuracy vs completeness
- Higher is better

**Use case:** Overall performance metric - best single number to compare systems

**Code Implementation:**
```python
def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
```

### 4.4 Mean Reciprocal Rank (MRR)

**What it measures:** How quickly do we find the first relevant document?

**Formula:**
```
MRR = 1 / (Rank of First Relevant Document)

If no relevant document found: MRR = 0
```

**Calculation Examples:**
```
Example 1:
Retrieved pages: [34✓, 78, 35✓, 102, 45]
First relevant at position 1
MRR = 1/1 = 1.0

Example 2:
Retrieved pages: [78, 102, 34✓, 35✓, 45]
First relevant at position 3
MRR = 1/3 = 0.333

Example 3:
Retrieved pages: [78, 102, 45, 88, 91]
No relevant document found
MRR = 0.0
```

**Interpretation:**
- Range: [0, 1]
- 1.0 = First result is relevant
- 0.5 = First relevant result at position 2
- 0.333 = First relevant result at position 3
- Higher is better

**Use case:** Important for user experience - users typically look at top results first

**Why it matters:**
- Search engines prioritize MRR
- Users rarely go beyond first few results
- First impression is critical

**Code Implementation:**
```python
def calculate_mrr(retrieved_docs, relevant_docs):
    for i, doc in enumerate(retrieved_docs):
        doc_page = str(doc.metadata.get('page_number', ''))
        for rel_page in relevant_docs:
            if doc_page == rel_page:
                return 1.0 / (i + 1)  # Position is 1-indexed
    return 0.0
```

### 4.5 Hit Rate@K (Success@K)

**What it measures:** Did we find at least one relevant document in top-K?

**Formula:**
```
Hit Rate@K = 1 if at least one relevant document in top-K
           = 0 otherwise
```

**Calculation Examples:**
```
Example 1:
Retrieved pages: [34✓, 78, 102, 45, 88]
Hit Rate@5 = 1 (found page 34)

Example 2:
Retrieved pages: [78, 102, 45, 88, 91]
Hit Rate@5 = 0 (no relevant page found)
```

**Interpretation:**
- Range: [0, 1] (binary)
- 1.0 = Success (found at least one)
- 0.0 = Failure (found none)
- Higher is better

**Average Hit Rate across queries:**
```
If 80 out of 100 queries find at least one relevant doc:
Average Hit Rate = 80/100 = 0.80 (80%)
```

**Use case:**
- Minimum viable performance metric
- Important for question-answering systems
- Users need at least one good result

**Code Implementation:**
```python
def calculate_hit_rate_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]
    for doc in retrieved_k:
        doc_page = str(doc.metadata.get('page_number', ''))
        for rel_page in relevant_docs:
            if doc_page == rel_page:
                return 1.0
    return 0.0
```

### 4.6 Normalized Discounted Cumulative Gain@K (NDCG@K)

**What it measures:** Ranking quality with position-based discounting

**Concept:**
- Relevant documents at top positions are more valuable
- Relevance is "discounted" by position using logarithm
- Normalized against ideal ranking

**Formula:**

**Step 1 - Calculate DCG (Discounted Cumulative Gain):**
```
DCG@K = Σ (relevance_i) / log₂(i + 1)  for i = 1 to K

Where:
- relevance_i = 1 if document at position i is relevant, 0 otherwise
- log₂(i + 1) = discount factor (logarithmic)
```

**Step 2 - Calculate IDCG (Ideal DCG):**
```
IDCG@K = DCG with all relevant docs at top positions

If there are R relevant docs:
IDCG@K = Σ 1 / log₂(i + 1)  for i = 1 to min(R, K)
```

**Step 3 - Calculate NDCG:**
```
NDCG@K = DCG@K / IDCG@K

If IDCG@K = 0 (no relevant docs exist): NDCG@K = 0
```

**Detailed Calculation Example:**

```
Query: "What are symptoms of anaemia?"
Expected relevant pages: [34, 35]
Retrieved pages (K=5): [34✓, 78, 35✓, 102, 45]

Step 1 - Calculate DCG:
Position 1: page 34 (relevant=1) → 1 / log₂(2) = 1 / 1.0 = 1.0
Position 2: page 78 (relevant=0) → 0 / log₂(3) = 0 / 1.585 = 0.0
Position 3: page 35 (relevant=1) → 1 / log₂(4) = 1 / 2.0 = 0.5
Position 4: page 102 (relevant=0) → 0 / log₂(5) = 0 / 2.322 = 0.0
Position 5: page 45 (relevant=0) → 0 / log₂(6) = 0 / 2.585 = 0.0

DCG@5 = 1.0 + 0.0 + 0.5 + 0.0 + 0.0 = 1.5

Step 2 - Calculate IDCG (ideal ranking: [34✓, 35✓, 78, 102, 45]):
Position 1: relevant → 1 / log₂(2) = 1.0
Position 2: relevant → 1 / log₂(3) = 0.631
Position 3: not relevant → 0.0
Position 4: not relevant → 0.0
Position 5: not relevant → 0.0

IDCG@5 = 1.0 + 0.631 = 1.631

Step 3 - Calculate NDCG:
NDCG@5 = DCG@5 / IDCG@5 = 1.5 / 1.631 = 0.920 (92%)
```

**Why Logarithmic Discount?**
```
Position 1: discount = log₂(2) = 1.0    (100% value)
Position 2: discount = log₂(3) = 1.585  (63% value)
Position 3: discount = log₂(4) = 2.0    (50% value)
Position 4: discount = log₂(5) = 2.322  (43% value)
Position 5: discount = log₂(6) = 2.585  (39% value)

This reflects diminishing attention from users
```

**Interpretation:**
- Range: [0, 1]
- 1.0 = Perfect ranking (all relevant docs at top)
- 0.0 = Worst (no relevant docs retrieved)
- Higher is better
- More sophisticated than Hit Rate or Precision

**Comparison with other metrics:**
```
Scenario A: [relevant, not, not, not, not]
Precision@5 = 1/5 = 0.20
NDCG@5 = high (relevant doc at top)

Scenario B: [not, not, not, not, relevant]
Precision@5 = 1/5 = 0.20 (same!)
NDCG@5 = low (relevant doc at bottom)

→ NDCG captures ranking quality, Precision doesn't
```

**Use case:**
- Search engine evaluation
- When result ordering matters
- Industry standard metric

**Code Implementation:**
```python
def calculate_ndcg_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]

    # Calculate DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved_k):
        doc_page = str(doc.metadata.get('page_number', ''))
        relevance = 0.0
        for rel_page in relevant_docs:
            if doc_page == rel_page:
                relevance = 1.0
                break
        dcg += relevance / np.log2(i + 2)  # i+2 because i is 0-indexed

    # Calculate IDCG (ideal: all relevant at top)
    num_relevant = len(relevant_docs)
    ideal_relevances = [1.0] * min(num_relevant, k) + [0.0] * max(0, k - num_relevant)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    # Calculate NDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg
```

### 4.7 Metrics Summary Table

| Metric | Range | What It Measures | Best For | Formula |
|--------|-------|------------------|----------|---------|
| **Precision@K** | [0, 1] | Accuracy of retrieved results | Avoiding false positives | relevant_retrieved / K |
| **Recall@K** | [0, 1] | Completeness of retrieval | Finding all relevant docs | relevant_retrieved / total_relevant |
| **F1@K** | [0, 1] | Balance of Precision & Recall | Overall performance | 2·P·R/(P+R) |
| **MRR** | [0, 1] | Speed of finding first relevant | User experience | 1 / rank_first |
| **Hit Rate@K** | [0, 1] | Basic success rate | Minimum viability | 1 if any hit else 0 |
| **NDCG@K** | [0, 1] | Ranking quality | Search engine evaluation | DCG / IDCG |

### 4.8 Which Metric to Use When?

**For Question Answering Systems:**
- Primary: **MRR** (find answer quickly)
- Secondary: **Hit Rate@K** (at least one good answer)

**For Document Retrieval:**
- Primary: **Recall@K** (find all relevant docs)
- Secondary: **Precision@K** (avoid noise)

**For Search Engines:**
- Primary: **NDCG@K** (ranking quality)
- Secondary: **F1@K** (overall performance)

**For Comparison/Benchmarking:**
- Primary: **F1@K** (balanced metric)
- Secondary: **All metrics** (comprehensive view)

**Our System (Healthcare RAG):**
- We use **ALL 6 metrics** to get comprehensive evaluation
- **F1@K** as the main comparison metric
- **MRR** for user experience
- **Recall@K** to ensure we don't miss important information

---

## 5. Configuration Parameters

### 5.1 Alpha (α) - Hybrid Weight

**What it controls:** Balance between dense (semantic) and sparse (keyword) retrieval

**Values:**
```
α = 0.0 → Pure Sparse (100% BM25)
α = 0.3 → Sparse-Heavy (30% FAISS + 70% BM25)
α = 0.5 → Balanced (50% FAISS + 50% BM25)
α = 0.7 → Dense-Heavy (70% FAISS + 30% BM25)
α = 1.0 → Pure Dense (100% FAISS)
```

**When to use what:**

| Alpha | Best For | Example Query |
|-------|----------|---------------|
| 0.0-0.3 | Exact terminology, acronyms | "What is the ICD-10 code for diabetes?" |
| 0.4-0.6 | Mixed queries | "How to treat and manage anaemia?" |
| 0.7-1.0 | Conceptual questions | "Why does vitamin deficiency cause problems?" |

**Tuning Strategy:**
1. Start with α = 0.7 (generally best)
2. Test α = 0.5 if keyword matching important
3. Evaluate on your specific query distribution
4. Use grid search: [0.0, 0.1, 0.2, ..., 1.0]

### 5.2 K - Number of Retrieved Documents

**What it controls:** How many documents to retrieve and evaluate

**Common values:** K ∈ {3, 5, 7, 10, 15, 20}

**Trade-offs:**

```
Small K (3-5):
✓ Fast retrieval
✓ High precision
✗ May miss relevant docs (low recall)
✗ Sensitive to ranking errors

Large K (15-20):
✓ High recall
✓ More robust
✗ Slower retrieval
✗ Lower precision (more noise)
```

**Our default:** K = 5
- Balances precision and recall
- Reasonable computational cost
- Typical search engine result count

### 5.3 Test Configuration Matrix

We test **all combinations** of:
- 5 Alpha values: [0.0, 0.3, 0.5, 0.7, 1.0]
- 5 K values: [3, 5, 7, 10, 15]
- 133 test queries

**Total test cases:** 5 × 5 × 133 = **3,325 evaluations**

---

## 6. Matching Logic

### 6.1 Page Number Matching

**Critical Fix:** Exact string matching (not substring!)

**The Problem (Before Fix):**
```python
# OLD CODE (WRONG):
if str(rel_page) in str(doc_page):
    # This matched "34" with "340", "134", "3456"!
```

**The Solution (After Fix):**
```python
# NEW CODE (CORRECT):
def is_page_match(doc_page_num, relevant_page_num):
    doc_page = str(doc_page_num).strip()
    rel_page = str(relevant_page_num).strip()
    return doc_page == rel_page  # Exact match only
```

**Examples:**
```
Document Page: "34"  | Expected: "34"  → ✓ MATCH
Document Page: "340" | Expected: "34"  → ✗ NO MATCH
Document Page: "134" | Expected: "34"  → ✗ NO MATCH
Document Page: " 34" | Expected: "34"  → ✓ MATCH (after strip)
```

### 6.2 Metadata Structure

**Document Metadata:**
```python
{
    'source': 'documents/Care_During_Pregnancy.pdf',
    'page_number': '34',
    'page': 33,  # 0-indexed
    'chunk_id': 'doc_34_chunk_0'
}
```

**Key field:** `page_number` (string, 1-indexed)

### 6.3 Matching Process Flow

```
For each query:
  1. Retrieve K documents from vector DB

  2. For each retrieved document:
     a. Extract page_number from metadata
     b. Convert to string and strip whitespace
     c. Compare with each expected page number
     d. Mark as "relevant" if exact match found

  3. Count relevant matches

  4. Calculate all 6 metrics based on matches
```

---

## 7. Complete Workflow

### 7.1 Initialization Phase

```python
# Load test queries
test_queries = load_json("test_queries.json")
# 133 queries with known relevant pages

# Load vector database
vector_db = load_vector_db("vector_db")
# FAISS index with embeddings

# Build BM25 index
bm25_index = build_bm25_index(vector_db)
# For sparse keyword retrieval
```

### 7.2 Evaluation Loop

```
FOR each configuration in [Pure Dense, Dense-Heavy, Balanced, Sparse-Heavy, Pure Sparse]:

    SET alpha = configuration.alpha
    SET k = configuration.k

    FOR each query in test_queries:

        # Step 1: Retrieve
        retrieved_docs = hybrid_search(
            query=query.text,
            alpha=alpha,
            k=k
        )

        # Step 2: Extract pages
        retrieved_pages = [doc.metadata['page_number'] for doc in retrieved_docs]
        expected_pages = query.relevant_docs

        # Step 3: Match
        matches = []
        for ret_page in retrieved_pages:
            if ret_page in expected_pages:
                matches.append(ret_page)

        # Step 4: Calculate metrics
        precision = len(matches) / k
        recall = len(matches) / len(expected_pages)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mrr = calculate_mrr(retrieved_docs, expected_pages)
        hit_rate = 1 if len(matches) > 0 else 0
        ndcg = calculate_ndcg(retrieved_docs, expected_pages, k)

        # Step 5: Store results
        results.append({
            'config': configuration.name,
            'query': query.text,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mrr': mrr,
            'hit_rate': hit_rate,
            'ndcg': ndcg,
            'retrieved_pages': retrieved_pages,
            'expected_pages': expected_pages
        })

    # Step 6: Aggregate per configuration
    avg_metrics = calculate_averages(results)
    print_summary(avg_metrics)

# Step 7: Find best configuration
best_config = max(results, key=lambda x: x['f1'])
```

### 7.3 Example Execution Trace

```
================================================================================
Test Configuration: Dense_Heavy_Hybrid
Description: 70% Semantic + 30% Keyword
Alpha: 0.7, K: 5
================================================================================

Query 1/133: "What are the symptoms of anaemia during pregnancy?"
Expected pages: [34, 35]
Intent: factual (confidence: 0.92)

Retrieving documents...
  Retrieved in 23.4ms

Retrieved documents:
  1. Page 34 - "Anaemia in Pregnancy... symptoms include fatigue, pale skin..."
  2. Page 78 - "Maternal health indicators..."
  3. Page 35 - "Diagnosis and treatment of anaemia..."
  4. Page 102 - "Nutritional guidelines..."
  5. Page 45 - "General pregnancy care..."

Matching:
  Page 34: ✓ MATCH
  Page 78: ✗ no match
  Page 35: ✓ MATCH
  Page 102: ✗ no match
  Page 45: ✗ no match

Metrics:
  Precision@5: 2/5 = 0.4000
  Recall@5: 2/2 = 1.0000
  F1@5: 0.5714
  MRR: 1/1 = 1.0000 (first result relevant)
  Hit Rate@5: 1.0000
  NDCG@5: 0.9519

Progress: 1/133 queries (1/665 total tests)

[... continues for all queries ...]

================================================================================
Configuration Summary: Dense_Heavy_Hybrid
================================================================================
Average Metrics (133 queries):
  Precision@5: 0.2487 ± 0.1823
  Recall@5: 0.3156 ± 0.2401
  F1@5: 0.2615 ± 0.1956
  MRR: 0.3421 ± 0.3012
  Hit Rate@5: 0.5789 ± 0.3845
  NDCG@5: 0.2891 ± 0.2234

Retrieval Time: 24.3ms ± 8.7ms

[... repeat for other configurations ...]

================================================================================
BEST CONFIGURATION
================================================================================
Config: Dense_Heavy_Hybrid (α=0.7, K=5)
Average F1 Score: 0.2615

Saved results to: evaluation_results_fixed/
```

---

## 8. Test Configurations

### 8.1 Standard Configurations

**Configuration 1: Pure Dense FAISS**
- Name: `Pure_Dense_FAISS`
- Alpha: 1.0
- K: 5
- Description: 100% Semantic Search
- Use case: Conceptual queries, paraphrasing

**Configuration 2: Dense-Heavy Hybrid**
- Name: `Dense_Heavy_Hybrid`
- Alpha: 0.7
- K: 5
- Description: 70% Semantic + 30% Keyword
- Use case: General purpose (usually best)

**Configuration 3: Balanced Hybrid**
- Name: `Balanced_Hybrid`
- Alpha: 0.5
- K: 5
- Description: 50% Semantic + 50% Keyword
- Use case: Mixed query types

**Configuration 4: Sparse-Heavy Hybrid**
- Name: `Sparse_Heavy_Hybrid`
- Alpha: 0.3
- K: 5
- Description: 30% Semantic + 70% Keyword
- Use case: Keyword-focused queries

**Configuration 5: Pure Sparse BM25**
- Name: `Pure_Sparse_BM25`
- Alpha: 0.0
- K: 5
- Description: 100% Keyword Search
- Use case: Exact term matching

### 8.2 Comprehensive Testing

For instructor presentation, we test:

**Test 1: Retrieval Methods**
- 4 methods: [Pure Dense, Dense-Heavy, Balanced, Sparse-Heavy]
- Fixed: K=5
- Queries: All 133

**Test 2: Alpha Sensitivity**
- Alpha values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- Fixed: K=5
- Queries: All 133
- Purpose: Find optimal alpha

**Test 3: Top-K Analysis**
- K values: [3, 5, 7, 10, 15]
- Fixed: α=0.7 (best from Test 2)
- Queries: All 133
- Purpose: Find optimal K

### 8.3 Output Files

**Raw Results:**
- `evaluation_results_YYYYMMDD_HHMMSS.csv` - All individual query results

**Summary Statistics:**
- `summary_statistics.csv` - Aggregated metrics per configuration

**Visualizations:**
- `1_retrieval_methods_comparison.png` - Bar charts of metrics
- `2_alpha_sensitivity.png` - Line plots showing alpha vs metrics
- `3_topk_analysis.png` - Line plots showing K vs metrics
- `4_comprehensive_comparison.png` - Side-by-side comparison
- `5_radar_chart.png` - Multi-dimensional comparison

---

## 9. Results Analysis

### 9.1 Sample Results (Typical Performance)

```
Configuration: Dense_Heavy_Hybrid (α=0.7, K=5)
================================================
Precision@5:  0.25 (25% of retrieved docs are relevant)
Recall@5:     0.32 (32% of all relevant docs found)
F1@5:         0.26 (balanced performance)
MRR:          0.34 (first relevant doc at avg position 3)
Hit Rate@5:   0.58 (58% queries find at least one)
NDCG@5:       0.29 (ranking quality moderate)

Performance by Query Type:
- Factual queries:     F1 = 0.31 ✓ (better)
- Procedural queries:  F1 = 0.24
- Diagnostic queries:  F1 = 0.28
- Conceptual queries:  F1 = 0.22 (harder)
```

### 9.2 Why Performance May Be Low

**Reason 1: Document Coverage**
```
Expected pages in test queries: 245 unique pages
Pages in vector database: 180 unique pages
Overlap: 123 pages (50% coverage)

→ Missing 122 pages that queries expect!
```

**Reason 2: Page Number Mismatch**
```
Test queries use page numbers from original PDFs
Vector DB may use different page numbering (0-indexed, etc.)
Need to align indexing systems
```

**Reason 3: Chunking Issues**
```
Documents split into chunks
Relevant info might span multiple chunks
Page number may not be preserved correctly
```

**Reason 4: Query-Document Mismatch**
```
Test queries may be too specific
Documents may use different terminology
Embeddings may not capture medical nuances
```

### 9.3 Interpreting Results

**Good Performance Indicators:**
- F1@5 > 0.5 (50%)
- Hit Rate@5 > 0.8 (80%)
- MRR > 0.5 (first relevant in top 2)

**Our Current Performance:**
- F1@5 ≈ 0.26 → **Need improvement**
- Hit Rate@5 ≈ 0.58 → **Moderate**
- MRR ≈ 0.34 → **Need improvement**

**Action Items:**
1. Verify document ingestion
2. Check page number alignment
3. Review chunking strategy
4. Possibly regenerate test queries
5. Consider medical-specific embeddings
6. Add query preprocessing
7. Implement re-ranking

### 9.4 Comparison Between Configurations

**Typical Rankings (Best to Worst):**

1. **Dense-Heavy Hybrid (α=0.7)** - F1: 0.26
   - Best overall balance
   - Good for diverse queries
   - Recommended default

2. **Balanced Hybrid (α=0.5)** - F1: 0.25
   - Close second
   - More keyword emphasis
   - Good for specific terms

3. **Pure Dense (α=1.0)** - F1: 0.23
   - Good for conceptual queries
   - Misses keyword matches
   - Struggles with acronyms

4. **Sparse-Heavy (α=0.3)** - F1: 0.22
   - Better for exact terms
   - Loses semantic understanding
   - Limited generalization

5. **Pure Sparse (α=0.0)** - F1: 0.19
   - Fast but limited
   - Only exact matches
   - Poor for paraphrasing

### 9.5 Statistical Significance

We compute:
- **Mean** - Average performance
- **Std Dev** - Variation across queries
- **Min/Max** - Range of performance
- **Percentiles** - Distribution analysis

**Example:**
```
Dense-Heavy Hybrid F1@5:
Mean: 0.2615
Std:  0.1956
Min:  0.0000
25%:  0.1000
50%:  0.2500
75%:  0.4000
Max:  1.0000

Interpretation:
- High variance (std = 0.196)
- Some queries work well (max = 1.0)
- Some queries fail completely (min = 0.0)
- Median better than mean (positive skew)
```

---

## 10. Implementation Details

### 10.1 Key Files

**Main evaluation script:**
```
automated_evaluation_FIXED.py
- Standalone metrics calculation
- Fixed page matching bug
- Comprehensive evaluation
```

**Comprehensive testing:**
```
comprehensive_evaluation.py
- Alpha sensitivity analysis
- Top-K analysis
- Visualization generation
```

**Explanation scripts:**
```
explain_evaluation_process.py
- Step-by-step walkthrough
- Debugging helper

verify_retrieval_accuracy.py
- Verification against PDFs
- Page number validation
```

### 10.2 Dependencies

```python
# Core
import numpy as np
import pandas as pd

# Retrieval
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

# Evaluation
from optimized_rag_chat import (
    load_vector_db,
    classify_query_intent,
    get_llm
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

### 10.3 Running Evaluation

**Basic evaluation:**
```bash
python automated_evaluation_FIXED.py
```

**Comprehensive evaluation with plots:**
```bash
python comprehensive_evaluation.py
```

**Debug specific query:**
```bash
python explain_evaluation_process.py
```

**Verify against PDFs:**
```bash
python verify_retrieval_accuracy.py
```

---

## 11. Summary

### Key Takeaways

1. **Multiple Metrics Required**: No single metric captures everything
   - Precision: accuracy
   - Recall: completeness
   - F1: balance
   - MRR: user experience
   - Hit Rate: minimum success
   - NDCG: ranking quality

2. **Hybrid > Pure Methods**: Combining semantic + keyword usually best
   - Recommended: α = 0.7 (70% semantic, 30% keyword)
   - Adapt based on query distribution

3. **Exact Matching Critical**: Page number matching must be exact
   - Fixed substring matching bug
   - Ensures fair evaluation

4. **Comprehensive Testing**: Test multiple configurations
   - Different alpha values
   - Different K values
   - Different query types

5. **Statistical Analysis**: Look beyond averages
   - Standard deviation
   - Min/max ranges
   - Per-query-type breakdown

### Future Improvements

1. **Query Preprocessing**: Spell check, expansion, translation
2. **Re-ranking**: Use LLM to rerank retrieved documents
3. **Medical Embeddings**: Domain-specific models
4. **Dynamic K**: Adapt K based on query type
5. **Query Classification**: Route queries to best retrieval method
6. **Document Verification**: Ensure all expected pages exist
7. **Evaluation Set Expansion**: More diverse test queries

---

## References

**Evaluation Metrics:**
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques

**BM25:**
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond

**Hybrid Retrieval:**
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering
- RAGatouille: State-of-the-art Neural Search Library

---

**Document Version:** 1.0
**Last Updated:** 2025
**Author:** Sharath Kumar MD
**Project:** KCDH, IIT Bombay - Maharashtra CHO Training System
