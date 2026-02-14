"""
Medical Embeddings Comparison Test
Compares general embeddings (all-MiniLM-L6-v2) vs medical embeddings (PubMedBERT)

Author: Sharath Kumar MD
Date: February 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag'))

import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Import metrics from evaluation
from automated_evaluation_FIXED import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_hit_rate_at_k,
    calculate_ndcg_at_k,
    build_bm25_index_standalone,
    hybrid_search_standalone,
    BM25_INDEX,
    BM25_DOCS,
    BM25_METADATA
)

####################################################################
#                    CONFIGURATION
####################################################################

# Embedding models to test
EMBEDDING_MODELS = {
    "baseline_minilm": {
        "name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "General purpose (current baseline)"
    },
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "dimensions": 768,
        "description": "Better general purpose embeddings"
    },
    "pubmedbert": {
        "name": "pritamdeka/S-PubMedBert-MS-MARCO",
        "dimensions": 768,
        "description": "Medical domain specific"
    },
    "biobert": {
        "name": "dmis-lab/biobert-base-cased-v1.2",
        "dimensions": 768,
        "description": "Biomedical domain specific"
    }
}

# Test with optimal retrieval config from your evaluation
RETRIEVAL_CONFIG = {
    "alpha": 0.7,  # Your optimal: 70% dense, 30% sparse
    "k": 7         # Your optimal K
}

OUTPUT_DIR = Path("../../results/embedding_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

####################################################################
#                    EMBEDDING TESTING
####################################################################

def load_existing_chunks():
    """Load existing document chunks from vector DB metadata"""
    print("Loading existing document chunks...")

    # Load the baseline vector DB to get documents
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    db_path = "../../data/vector_db"
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        print(f"Vector database not found at {db_path}")
        return None

    vector_db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Extract all documents from FAISS docstore
    all_docs = []
    docstore = vector_db.docstore

    # FAISS stores document IDs in index_to_docstore_id mapping
    for doc_id in vector_db.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        if doc:
            all_docs.append(doc)

    print(f"Loaded {len(all_docs)} document chunks")

    return all_docs


def create_vector_db_with_embedding(documents, embedding_model_name, save_path):
    """Create a FAISS vector database with specified embedding model"""
    print(f"\nCreating vector DB with: {embedding_model_name}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )

        start_time = time.time()
        vector_db = FAISS.from_documents(documents, embeddings)
        embedding_time = time.time() - start_time

        # Save the vector DB
        vector_db.save_local(save_path)
        print(f"  Created in {embedding_time:.2f}s, saved to {save_path}")

        return vector_db, embedding_time

    except Exception as e:
        print(f"  Error: {e}")
        return None, 0


def evaluate_embedding_model(model_key, model_info, documents, test_queries):
    """Evaluate a single embedding model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_key}")
    print(f"Model: {model_info['name']}")
    print(f"Description: {model_info['description']}")
    print(f"{'='*60}")

    # Create vector DB
    save_path = OUTPUT_DIR / f"vector_db_{model_key}"
    vector_db, embedding_time = create_vector_db_with_embedding(
        documents,
        model_info['name'],
        str(save_path)
    )

    if vector_db is None:
        return None

    # Build BM25 index (same for all - based on text, not embeddings)
    global BM25_INDEX, BM25_DOCS, BM25_METADATA
    build_bm25_index_standalone(vector_db)

    # Run evaluation
    results = []
    alpha = RETRIEVAL_CONFIG['alpha']
    k = RETRIEVAL_CONFIG['k']

    print(f"\nEvaluating on {len(test_queries)} queries (alpha={alpha}, k={k})...")

    for idx, query_data in enumerate(test_queries):
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx+1}/{len(test_queries)}")

        query = query_data['query']
        relevant_docs = query_data['relevant_docs']
        query_type = query_data.get('query_type', 'unknown')

        try:
            # Retrieval
            start_time = time.time()
            docs_with_scores = hybrid_search_standalone(
                vector_db, query, k=k, alpha=alpha
            )
            retrieval_time = (time.time() - start_time) * 1000

            docs = [doc for doc, _ in docs_with_scores]

            # Calculate metrics
            precision = calculate_precision_at_k(docs, relevant_docs, k)
            recall = calculate_recall_at_k(docs, relevant_docs, k)
            mrr = calculate_mrr(docs, relevant_docs)
            hit_rate = calculate_hit_rate_at_k(docs, relevant_docs, k)
            ndcg = calculate_ndcg_at_k(docs, relevant_docs, k)

            f1 = 0.0
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)

            results.append({
                'query_id': idx,
                'query_type': query_type,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mrr': mrr,
                'hit_rate': hit_rate,
                'ndcg': ndcg,
                'retrieval_time_ms': retrieval_time
            })

        except Exception as e:
            print(f"  Error on query {idx}: {e}")
            results.append({
                'query_id': idx,
                'query_type': query_type,
                'precision': 0, 'recall': 0, 'f1': 0,
                'mrr': 0, 'hit_rate': 0, 'ndcg': 0,
                'retrieval_time_ms': 0
            })

    df = pd.DataFrame(results)

    # Summary
    summary = {
        'model_key': model_key,
        'model_name': model_info['name'],
        'description': model_info['description'],
        'embedding_time_s': embedding_time,
        'avg_precision': df['precision'].mean(),
        'avg_recall': df['recall'].mean(),
        'avg_f1': df['f1'].mean(),
        'avg_mrr': df['mrr'].mean(),
        'avg_hit_rate': df['hit_rate'].mean(),
        'avg_ndcg': df['ndcg'].mean(),
        'avg_latency_ms': df['retrieval_time_ms'].mean()
    }

    print(f"\nResults for {model_key}:")
    print(f"  F1: {summary['avg_f1']:.4f}")
    print(f"  Precision: {summary['avg_precision']:.4f}")
    print(f"  Recall: {summary['avg_recall']:.4f}")
    print(f"  Hit Rate: {summary['avg_hit_rate']:.4f}")
    print(f"  MRR: {summary['avg_mrr']:.4f}")
    print(f"  Latency: {summary['avg_latency_ms']:.1f}ms")

    return summary, df


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("MEDICAL EMBEDDINGS COMPARISON TEST")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load documents
    documents = load_existing_chunks()
    if documents is None:
        print("Failed to load documents!")
        return

    # Load test queries
    queries_file = "../../data/test_queries.json"
    print(f"\nLoading test queries from: {queries_file}")
    with open(queries_file, 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
    print(f"Loaded {len(test_queries)} test queries")

    # Test each embedding model
    all_summaries = []
    all_details = {}

    for model_key, model_info in EMBEDDING_MODELS.items():
        try:
            summary, details = evaluate_embedding_model(
                model_key, model_info, documents, test_queries
            )
            if summary:
                all_summaries.append(summary)
                all_details[model_key] = details
        except Exception as e:
            print(f"Failed to test {model_key}: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison report
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    if all_summaries:
        comparison_df = pd.DataFrame(all_summaries)
        comparison_df = comparison_df.sort_values('avg_f1', ascending=False)

        print("\nRanking by F1 Score:")
        print("-" * 50)
        for i, row in comparison_df.iterrows():
            improvement = ""
            baseline_f1 = comparison_df[comparison_df['model_key'] == 'baseline_minilm']['avg_f1'].values
            if len(baseline_f1) > 0 and row['model_key'] != 'baseline_minilm':
                diff = (row['avg_f1'] - baseline_f1[0]) / baseline_f1[0] * 100
                improvement = f" ({'+' if diff > 0 else ''}{diff:.1f}% vs baseline)"

            print(f"{row['model_key']:20s} F1={row['avg_f1']:.4f}{improvement}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        comparison_path = OUTPUT_DIR / f"embedding_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved: {comparison_path}")

        # Save detailed results
        for model_key, details in all_details.items():
            details_path = OUTPUT_DIR / f"details_{model_key}_{timestamp}.csv"
            details.to_csv(details_path, index=False)

        # Best model recommendation
        best = comparison_df.iloc[0]
        print(f"\n{'='*70}")
        print(f"RECOMMENDATION")
        print(f"{'='*70}")
        print(f"Best Model: {best['model_key']}")
        print(f"  Name: {best['model_name']}")
        print(f"  F1 Score: {best['avg_f1']:.4f}")
        print(f"  Hit Rate: {best['avg_hit_rate']:.4f}")

        if best['model_key'] != 'baseline_minilm':
            baseline = comparison_df[comparison_df['model_key'] == 'baseline_minilm'].iloc[0]
            f1_improvement = (best['avg_f1'] - baseline['avg_f1']) / baseline['avg_f1'] * 100
            print(f"\n  Improvement over baseline: {f1_improvement:+.1f}% F1")
        else:
            print("\n  Current baseline is already optimal!")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
