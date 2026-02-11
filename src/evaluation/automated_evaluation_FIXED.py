"""
FIXED Automated RAG System Evaluation
Corrects the page number matching bug and other issues

Author: Sharath Kumar MD
Date: October 18, 2025
"""

import sys
import io
# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import dotenv
from dotenv import load_dotenv
load_dotenv()

# Import from your RAG system
from optimized_rag_chat import (
    load_vector_db,
    classify_query_intent,
    get_llm
)
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import pickle

####################################################################
#                    CONFIGURATION
####################################################################

RETRIEVAL_CONFIGS = [
    {"name": "Pure_Dense_FAISS", "alpha": 1.0, "k": 5, "description": "100% Semantic Search"},
    {"name": "Dense_Heavy_Hybrid", "alpha": 0.7, "k": 5, "description": "70% Semantic + 30% Keyword"},
    {"name": "Balanced_Hybrid", "alpha": 0.5, "k": 5, "description": "50% Semantic + 50% Keyword"},
    {"name": "Sparse_Heavy_Hybrid", "alpha": 0.3, "k": 5, "description": "30% Semantic + 70% Keyword"},
    {"name": "Pure_Sparse_BM25", "alpha": 0.0, "k": 5, "description": "100% Keyword Search"},
]

OUTPUT_DIR = Path("evaluation_results_fixed")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global BM25 index and documents (initialized in run_evaluation)
BM25_INDEX = None
BM25_DOCS = None
BM25_METADATA = None

####################################################################
#                    STANDALONE RETRIEVAL FUNCTIONS
####################################################################

def build_bm25_index_standalone(vector_db):
    """Build BM25 index from vector database documents - FIXED for FAISS"""
    global BM25_INDEX, BM25_DOCS, BM25_METADATA

    try:
        print("📚 Building BM25 index for keyword search...")

        # FIXED: FAISS doesn't have get() method like Chroma
        # Use similarity_search with empty string to get documents
        all_docs = vector_db.similarity_search("", k=10000)

        if not all_docs:
            print("❌ No documents found in vector database!")
            return False, []

        print(f"   Found {len(all_docs)} documents")

        # Extract content and metadata
        BM25_DOCS = [doc.page_content for doc in all_docs]
        BM25_METADATA = [doc.metadata for doc in all_docs]

        # Tokenize documents
        tokenized_corpus = [doc.lower().split() for doc in BM25_DOCS]

        # Create BM25 index
        BM25_INDEX = BM25Okapi(tokenized_corpus)

        print("✅ BM25 index built successfully!")
        return True, BM25_METADATA

    except Exception as e:
        print(f"❌ Error building BM25 index: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def bm25_search_standalone(query, k=5):
    """Perform BM25 search"""
    global BM25_INDEX, BM25_DOCS, BM25_METADATA
    
    if BM25_INDEX is None or BM25_DOCS is None:
        return []
    
    # Tokenize query
    query_tokens = query.lower().split()
    
    # Get scores
    scores = BM25_INDEX.get_scores(query_tokens)
    
    # Get top-k indices
    top_k_idx = np.argsort(scores)[::-1][:k]
    
    # Create Document objects with scores
    results = []
    for idx in top_k_idx:
        if idx < len(BM25_DOCS):
            doc = Document(
                page_content=BM25_DOCS[idx],
                metadata=BM25_METADATA[idx] if idx < len(BM25_METADATA) else {}
            )
            results.append((doc, float(scores[idx])))
    
    return results

def hybrid_search_standalone(vector_db, query, k=5, alpha=0.5):
    """
    Standalone hybrid search combining FAISS and BM25
    
    Args:
        vector_db: FAISS vector database
        query: Search query
        k: Number of results to return
        alpha: Weight for dense search (0.0 = pure BM25, 1.0 = pure FAISS)
    
    Returns:
        List of (Document, score) tuples
    """
    # Handle edge cases
    if alpha == 1.0:
        # Pure FAISS
        docs_and_scores = vector_db.similarity_search_with_score(query, k=k)
        return docs_and_scores
    
    elif alpha == 0.0:
        # Pure BM25
        return bm25_search_standalone(query, k=k)
    
    else:
        # Hybrid approach
        # Get more results from each method, then combine
        dense_k = min(k * 2, 20)
        sparse_k = min(k * 2, 20)
        
        # Dense retrieval
        dense_results = vector_db.similarity_search_with_score(query, k=dense_k)
        
        # Sparse retrieval
        sparse_results = bm25_search_standalone(query, k=sparse_k)
        
        # Normalize scores
        def normalize_scores(results):
            if not results:
                return []
            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [(doc, 0.5) for doc, _ in results]
            return [(doc, (score - min_score) / (max_score - min_score)) 
                    for doc, score in results]
        
        dense_normalized = normalize_scores(dense_results)
        sparse_normalized = normalize_scores(sparse_results)
        
        # Combine scores
        combined = {}
        for doc, score in dense_normalized:
            doc_id = doc.page_content[:100]  # Use first 100 chars as ID
            combined[doc_id] = {
                'doc': doc,
                'score': alpha * score,
                'sources': ['dense']
            }
        
        for doc, score in sparse_normalized:
            doc_id = doc.page_content[:100]
            if doc_id in combined:
                combined[doc_id]['score'] += (1 - alpha) * score
                combined[doc_id]['sources'].append('sparse')
            else:
                combined[doc_id] = {
                    'doc': doc,
                    'score': (1 - alpha) * score,
                    'sources': ['sparse']
                }
        
        # Sort by combined score and return top-k
        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return [(item['doc'], item['score']) for item in sorted_results[:k]]

####################################################################
#                    FIXED METRICS CALCULATION
####################################################################

def is_page_match(doc_page_num, relevant_page_num):
    """
    Check if document page matches relevant page - EXACT MATCH ONLY
    
    FIXED: Now does exact string matching instead of substring
    """
    doc_page = str(doc_page_num).strip()
    rel_page = str(relevant_page_num).strip()
    return doc_page == rel_page

def calculate_precision_at_k(retrieved_docs, relevant_docs, k):
    """Calculate Precision@K - FIXED"""
    if not retrieved_docs or k == 0:
        return 0.0
    
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = 0
    for doc in retrieved_k:
        doc_page = str(doc.metadata.get('page_number', ''))
        for rel_page in relevant_docs:
            if is_page_match(doc_page, rel_page):
                relevant_retrieved += 1
                break  # Count each doc only once
    
    return relevant_retrieved / k

def calculate_recall_at_k(retrieved_docs, relevant_docs, k):
    """Calculate Recall@K - FIXED"""
    if not relevant_docs:
        return 0.0
    
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = 0
    
    # For each relevant doc, check if we found it
    for rel_page in relevant_docs:
        for doc in retrieved_k:
            doc_page = str(doc.metadata.get('page_number', ''))
            if is_page_match(doc_page, rel_page):
                relevant_retrieved += 1
                break  # Found this relevant doc, move to next
    
    return relevant_retrieved / len(relevant_docs)

def calculate_mrr(retrieved_docs, relevant_docs):
    """Calculate Mean Reciprocal Rank - FIXED"""
    for i, doc in enumerate(retrieved_docs):
        doc_page = str(doc.metadata.get('page_number', ''))
        for rel_page in relevant_docs:
            if is_page_match(doc_page, rel_page):
                return 1.0 / (i + 1)
    return 0.0

def calculate_hit_rate_at_k(retrieved_docs, relevant_docs, k):
    """Calculate Hit Rate@K - FIXED"""
    retrieved_k = retrieved_docs[:k]
    for doc in retrieved_k:
        doc_page = str(doc.metadata.get('page_number', ''))
        for rel_page in relevant_docs:
            if is_page_match(doc_page, rel_page):
                return 1.0
    return 0.0

def calculate_ndcg_at_k(retrieved_docs, relevant_docs, k):
    """Calculate Normalized Discounted Cumulative Gain@K - FIXED"""
    retrieved_k = retrieved_docs[:k]

    # DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved_k):
        doc_page = str(doc.metadata.get('page_number', ''))
        relevance = 0.0
        for rel_page in relevant_docs:
            if is_page_match(doc_page, rel_page):
                relevance = 1.0
                break
        dcg += relevance / np.log2(i + 2)

    # IDCG (ideal DCG - all relevant docs at top)
    ideal_relevances = [1.0] * min(len(relevant_docs), k) + [0.0] * max(0, k - len(relevant_docs))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    return dcg / idcg if idcg > 0 else 0.0

def calculate_metrics(retrieved_docs, relevant_docs, k):
    """
    Wrapper function to calculate all metrics at once

    Args:
        retrieved_docs: List of Document objects or list of (Document, score) tuples
        relevant_docs: List of relevant page numbers
        k: Number of documents to consider

    Returns:
        Dict with all metric values
    """
    # Handle both list of docs and list of (doc, score) tuples
    if retrieved_docs and isinstance(retrieved_docs[0], tuple):
        docs = [doc for doc, _ in retrieved_docs]
    else:
        docs = retrieved_docs

    # Calculate all metrics
    precision = calculate_precision_at_k(docs, relevant_docs, k)
    recall = calculate_recall_at_k(docs, relevant_docs, k)
    mrr = calculate_mrr(docs, relevant_docs)
    hit_rate = calculate_hit_rate_at_k(docs, relevant_docs, k)
    ndcg = calculate_ndcg_at_k(docs, relevant_docs, k)

    # Calculate F1 score
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1,
        'mrr': mrr,
        'hit_rate@k': hit_rate,
        'ndcg@k': ndcg
    }

####################################################################
#                    RUN EVALUATION
####################################################################

def run_evaluation(queries_file):
    """Run comprehensive evaluation"""
    
    # Load test queries
    print(f"\n📋 Loading test queries from: {queries_file}")
    with open(queries_file, 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
    print(f"✅ Loaded {len(test_queries)} test queries")
    
    # Load vector database
    print("\n🔍 Loading vector database...")
    db = load_vector_db()
    if db is None:
        print("❌ Failed to load vector database!")
        return None
    print(f"✅ Vector database loaded: {db.index.ntotal} vectors")
    
    # Build BM25 index
    print("\n🔍 Building BM25 index for sparse retrieval...")
    bm25_success, metadata_list = build_bm25_index_standalone(db)
    if not bm25_success:
        print("⚠️ BM25 index build failed, sparse retrieval may not work")
    else:
        print("✅ BM25 index ready")
    
    # Initialize results storage
    all_results = []
    
    # Run evaluation for each configuration
    total_tests = len(RETRIEVAL_CONFIGS) * len(test_queries)
    current_test = 0
    
    for config in RETRIEVAL_CONFIGS:
        print(f"\n{'='*80}")
        print(f"🧪 Testing Configuration: {config['name']}")
        print(f"   {config['description']}")
        print(f"   Alpha: {config['alpha']}, K: {config['k']}")
        print(f"{'='*80}")
        
        for idx, query_data in enumerate(test_queries):
            current_test += 1
            query = query_data['query']
            relevant_docs = query_data['relevant_docs']
            query_type = query_data['query_type']
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"   Progress: {idx+1}/{len(test_queries)} queries ({current_test}/{total_tests} total tests)")
            
            try:
                # Intent classification
                is_non_medical, confidence, intent = classify_query_intent(query)
                
                # Retrieval
                start_time = time.time()
                docs_with_scores = hybrid_search_standalone(
                    db, 
                    query, 
                    k=config['k'], 
                    alpha=config['alpha']
                )
                retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Extract documents and scores
                docs = [doc for doc, _ in docs_with_scores]
                scores = [score for _, score in docs_with_scores]
                
                # Calculate metrics
                precision_k = calculate_precision_at_k(docs, relevant_docs, config['k'])
                recall_k = calculate_recall_at_k(docs, relevant_docs, config['k'])
                mrr = calculate_mrr(docs, relevant_docs)
                hit_rate = calculate_hit_rate_at_k(docs, relevant_docs, config['k'])
                ndcg = calculate_ndcg_at_k(docs, relevant_docs, config['k'])
                
                # F1 Score
                f1_score = 0.0
                if precision_k + recall_k > 0:
                    f1_score = 2 * (precision_k * recall_k) / (precision_k + recall_k)
                
                # Store results
                all_results.append({
                    'config_name': config['name'],
                    'alpha': config['alpha'],
                    'k': config['k'],
                    'query_id': idx,
                    'query': query,
                    'query_type': query_type,
                    'intent_detected': intent,
                    'intent_confidence': confidence,
                    'num_relevant_docs': len(relevant_docs),
                    'num_retrieved_docs': len(docs),
                    'retrieval_time_ms': retrieval_time,
                    'precision_at_k': precision_k,
                    'recall_at_k': recall_k,
                    'f1_score': f1_score,
                    'mrr': mrr,
                    'hit_rate_at_k': hit_rate,
                    'ndcg_at_k': ndcg,
                    'avg_score': np.mean(scores) if scores else 0.0,
                    'max_score': np.max(scores) if scores else 0.0,
                    'min_score': np.min(scores) if scores else 0.0,
                    'retrieved_pages': [str(doc.metadata.get('page_number', 'N/A')) for doc in docs],
                    'relevant_pages': relevant_docs
                })
                
            except Exception as e:
                print(f"   ⚠️ Error on query {idx}: {str(e)[:100]}")
                all_results.append({
                    'config_name': config['name'],
                    'alpha': config['alpha'],
                    'k': config['k'],
                    'query_id': idx,
                    'query': query,
                    'query_type': query_type,
                    'error': str(e),
                    'retrieval_time_ms': 0,
                    'precision_at_k': 0,
                    'recall_at_k': 0,
                    'f1_score': 0,
                    'mrr': 0,
                    'hit_rate_at_k': 0,
                    'ndcg_at_k': 0,
                    'retrieved_pages': [],
                    'relevant_pages': relevant_docs
                })
    
    print(f"\n✅ Evaluation complete! Processed {len(all_results)} test cases")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"evaluation_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Raw results saved: {csv_path}")
    
    # Create summary statistics
    summary = df.groupby('config_name').agg({
        'precision_at_k': ['mean', 'std'],
        'recall_at_k': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'mrr': ['mean', 'std'],
        'hit_rate_at_k': ['mean', 'std'],
        'ndcg_at_k': ['mean', 'std'],
        'retrieval_time_ms': ['mean', 'std', 'min', 'max']
    })
    
    summary_path = OUTPUT_DIR / "summary_statistics.csv"
    summary.to_csv(summary_path)
    print(f"📊 Summary statistics saved: {summary_path}")
    
    return df

####################################################################
#                    MAIN EXECUTION
####################################################################

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("🏥 HEALTHCARE RAG SYSTEM - FIXED EVALUATION")
    print("="*80)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output Directory: {OUTPUT_DIR.absolute()}")
    print("="*80)
    
    # Run evaluation
    df = run_evaluation("data/test_queries.json")
    
    if df is None:
        print("\n❌ Evaluation failed!")
        return
    
    # Sanity checks
    print("\n" + "="*80)
    print("🔍 SANITY CHECKS")
    print("="*80)
    
    # Check for impossible values
    bad_recall = df[df['recall_at_k'] > 1.0]
    if len(bad_recall) > 0:
        print(f"❌ WARNING: {len(bad_recall)} queries have recall > 1.0!")
        print(bad_recall[['config_name', 'query_id', 'recall_at_k']].head())
    else:
        print("✅ All recall values are valid (0.0 to 1.0)")
    
    bad_precision = df[df['precision_at_k'] > 1.0]
    if len(bad_precision) > 0:
        print(f"❌ WARNING: {len(bad_precision)} queries have precision > 1.0!")
    else:
        print("✅ All precision values are valid (0.0 to 1.0)")
    
    # Check row counts
    expected_rows = len(RETRIEVAL_CONFIGS) * len(df['query_id'].unique())
    actual_rows = len(df)
    if expected_rows == actual_rows:
        print(f"✅ Row count correct: {actual_rows} ({len(RETRIEVAL_CONFIGS)} configs × {len(df['query_id'].unique())} queries)")
    else:
        print(f"❌ Row count mismatch: expected {expected_rows}, got {actual_rows}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"📊 Total Queries Tested: {len(df['query_id'].unique())}")
    print(f"🔧 Configurations Tested: {len(RETRIEVAL_CONFIGS)}")
    print(f"📈 Total Test Cases: {len(df)}")
    print(f"💾 Results saved to: {OUTPUT_DIR.absolute()}")
    print(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Best configuration
    best_config = df.groupby('config_name')['f1_score'].mean().idxmax()
    best_f1 = df.groupby('config_name')['f1_score'].mean().max()
    print(f"\n🏆 BEST CONFIGURATION: {best_config}")
    print(f"   Average F1 Score: {best_f1:.4f}")
    
    print("\n🎉 Fixed evaluation complete!")

if __name__ == "__main__":
    main()
