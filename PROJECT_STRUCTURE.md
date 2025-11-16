# DH307 Project Structure

**Healthcare RAG System for Maharashtra Government CHO Training**
**KCDH, IIT Bombay**

---

## 📁 Directory Structure

```
DH307/
│
├── src/                                    # Source Code
│   ├── __init__.py
│   │
│   ├── rag/                                # Core RAG System
│   │   ├── __init__.py
│   │   ├── optimized_rag_chat.py           # Main RAG chat application
│   │   ├── optimized_document_processor.py # Document processing & chunking
│   │   └── config.py                       # Configuration settings
│   │
│   ├── ingestion/                          # Document Ingestion
│   │   ├── __init__.py
│   │   └── batch_ingestion.py              # Batch document processing
│   │
│   ├── enhancements/                       # RAGFlow Enhancements
│   │   ├── __init__.py
│   │   ├── ragflow_citations.py            # Citation grounding (Apache 2.0)
│   │   ├── ragflow_fusion.py               # Advanced hybrid search fusion
│   │   ├── ragflow_reranker.py             # Cross-encoder reranking
│   │   └── query_preprocessing.py          # Query enhancement
│   │
│   └── evaluation/                         # Evaluation Framework
│       ├── __init__.py
│       └── automated_evaluation_FIXED.py   # Comprehensive evaluation
│
├── data/                                   # Data Files
│   ├── documents/                          # Source PDF documents
│   ├── vector_db/                          # FAISS vector database
│   ├── test_queries.json                   # Evaluation test queries (133 queries)
│   └── backups/                            # Backups
│       └── vector_db_backup/               # Vector DB backup
│
├── results/                                # Output Files
│   ├── evaluation/                         # Evaluation results
│   │   └── evaluation_results_fixed/       # CSV results, metrics
│   └── visualizations/                     # Plots & charts
│       └── presentation_plots/             # Presentation visualizations
│
├── docs/                                   # Documentation
│   └── COMPLETE_EVALUATION_EXPLANATION.md  # Comprehensive evaluation guide
│
├── .env                                    # Environment variables (not in git)
├── .env.example                            # Environment template
├── .gitignore                              # Git ignore rules
├── requirements.txt                        # Python dependencies
├── README.md                               # Project README
├── LICENSE                                 # Project license
└── PROJECT_STRUCTURE.md                    # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/your-repo/DH307.git
cd DH307

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Ingest Documents

```bash
# Place PDF files in data/documents/
# Run batch ingestion
python -m src.ingestion.batch_ingestion
```

### 3. Run RAG Chat

```bash
# Start the RAG chat system
python -m src.rag.optimized_rag_chat
```

### 4. Run Evaluation

```bash
# Evaluate system performance
python -m src.evaluation.automated_evaluation_FIXED
```

---

## 📦 Module Descriptions

### `src/rag/` - Core RAG System

**optimized_rag_chat.py**
- Main RAG application with Streamlit UI
- Query intent classification
- Hybrid retrieval (FAISS + BM25)
- LLM response generation
- Conversation memory

**optimized_document_processor.py**
- PDF document loading
- Text extraction and chunking
- Metadata management
- Vector embedding creation

**config.py**
- Configuration parameters
- Model settings
- Retrieval parameters

### `src/ingestion/` - Document Ingestion

**batch_ingestion.py**
- Batch processing of multiple PDFs
- Progress tracking
- Error handling
- Vector DB persistence

### `src/enhancements/` - RAGFlow Enhancements

**ragflow_citations.py**
- Inline citation markers [1], [2]
- Grounding evidence extraction
- Medical claim validation
- Adapted from RAGFlow (Apache 2.0)

**ragflow_fusion.py**
- Weighted sum fusion
- Reciprocal Rank Fusion (RRF)
- Score normalization
- Adaptive thresholds

**ragflow_reranker.py**
- Cross-encoder reranking
- Medical-specific boosting
- Top-K selection
- Local inference (no API)

**query_preprocessing.py**
- Spell correction
- Query expansion
- Intent detection

### `src/evaluation/` - Evaluation Framework

**automated_evaluation_FIXED.py**
- 6 metrics: Precision@K, Recall@K, F1@K, MRR, Hit Rate@K, NDCG@K
- 5 retrieval configurations tested
- Statistical analysis
- CSV output with detailed results

---

## 📊 Data Files

### `data/documents/`
Place your PDF files here for ingestion.

### `data/vector_db/`
FAISS vector database storage.
- Generated during ingestion
- Contains document embeddings
- Used for semantic search

### `data/test_queries.json`
Evaluation test queries with ground truth.
```json
{
  "query": "What are symptoms of anaemia?",
  "relevant_docs": ["34", "35"],
  "query_type": "factual"
}
```

---

## 📈 Results

### `results/evaluation/`
Contains evaluation results:
- CSV files with metrics per query
- Summary statistics
- Performance comparisons

### `results/visualizations/`
Generated plots:
- Retrieval methods comparison
- Alpha sensitivity analysis
- Top-K analysis
- Radar charts

---

## 📚 Documentation

### `docs/COMPLETE_EVALUATION_EXPLANATION.md`
Comprehensive guide covering:
- Evaluation architecture
- All 6 metrics (formulas + examples)
- Retrieval methods (Dense, Sparse, Hybrid)
- Configuration parameters
- Results interpretation

---

## 🔧 Configuration

### Environment Variables (.env)
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

### Key Parameters (src/rag/config.py)
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
ALPHA = 0.7  # 70% semantic, 30% keyword
TOP_K = 5
```

---

## 🧪 Testing & Evaluation

### Run Full Evaluation
```bash
python -m src.evaluation.automated_evaluation_FIXED
```

### Metrics Calculated
1. **Precision@K** - Accuracy of retrieved docs
2. **Recall@K** - Completeness of retrieval
3. **F1@K** - Harmonic mean of precision/recall
4. **MRR** - Mean Reciprocal Rank (first relevant position)
5. **Hit Rate@K** - Success rate (found at least one)
6. **NDCG@K** - Ranking quality with position discount

### Configurations Tested
- Pure Dense (α=1.0) - 100% semantic
- Dense-Heavy (α=0.7) - 70% semantic, 30% keyword
- Balanced (α=0.5) - 50% semantic, 50% keyword
- Sparse-Heavy (α=0.3) - 30% semantic, 70% keyword
- Pure Sparse (α=0.0) - 100% keyword

---

## 🛠️ Development

### Adding New Features

1. **New RAG Component**
   - Add to `src/rag/`
   - Update `__init__.py`

2. **New Enhancement**
   - Add to `src/enhancements/`
   - Follow RAGFlow attribution format

3. **New Evaluation Metric**
   - Update `src/evaluation/automated_evaluation_FIXED.py`
   - Add to metrics calculation

### Code Style
- Follow PEP 8
- Add docstrings to all functions
- Include type hints
- Attribution for adapted code (RAGFlow)

---

## 📄 License

This project is licensed under the terms specified in LICENSE file.

### Third-Party Attributions

**RAGFlow Components** (Apache License 2.0):
- `src/enhancements/ragflow_citations.py`
- `src/enhancements/ragflow_fusion.py`
- `src/enhancements/ragflow_reranker.py`

Copyright 2024 The InfiniFlow Authors
Source: https://github.com/infiniflow/ragflow

---

## 👥 Contributors

**Sharath Kumar MD**
Knowledge Center on Data & Health (KCDH)
IIT Bombay

---

## 📞 Support

For issues or questions:
1. Check `docs/COMPLETE_EVALUATION_EXPLANATION.md`
2. Review code comments
3. Contact project maintainers

---

## 🗺️ Roadmap

### Current Version
- ✅ Core RAG system with hybrid retrieval
- ✅ Comprehensive evaluation framework
- ✅ RAGFlow enhancements integrated
- ✅ Batch document ingestion

### Future Enhancements
- [ ] Medical-specific embeddings
- [ ] Query classification routing
- [ ] Dynamic K selection
- [ ] Multi-language support
- [ ] API deployment
- [ ] Web interface improvements

---

**Last Updated:** 2025-11-07
**Project Status:** Active Development
