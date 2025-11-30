"""
Evaluation Script for RAG Retrieval System
Evaluates chunk recall, precision, F1, and other retrieval metrics.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch


def get_device():
    """Detect and return the best available device (GPU/CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class RetrievalEvaluator:
    """Evaluates retrieval performance for RAG system."""

    def __init__(
        self,
        vector_db_path: str,
        test_queries_path: str,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        top_k: int = 5
    ):
        """
        Initialize the evaluator.

        Args:
            vector_db_path: Path to the FAISS vector database
            test_queries_path: Path to test queries JSON file
            embedding_model: Embedding model name
            top_k: Number of documents to retrieve
        """
        self.vector_db_path = vector_db_path
        self.test_queries_path = test_queries_path
        self.top_k = top_k

        print(f"Loading embeddings model: {embedding_model}")
        device = get_device()
        print(f"Using device: {device}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device}
        )

        print(f"Loading vector database from: {vector_db_path}")
        self.vectorstore = FAISS.load_local(
            vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector database loaded successfully")

        print(f"Loading test queries from: {test_queries_path}")
        with open(test_queries_path, 'r', encoding='utf-8') as f:
            self.test_queries = json.load(f)
        print(f"Loaded {len(self.test_queries)} test queries")

    def retrieve_documents(self, query: str) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve documents for a query.

        Args:
            query: Query string

        Returns:
            List of (doc_id, score, metadata) tuples
        """
        # Retrieve documents with scores
        results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)

        # Extract document IDs from metadata
        retrieved = []
        for doc, score in results:
            # Use chunk_index as the primary identifier
            doc_id = doc.metadata.get('chunk_index', 'unknown')

            retrieved.append((str(doc_id), float(score), doc.metadata))

        return retrieved

    def calculate_recall(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate recall: proportion of relevant docs that were retrieved.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs (ground truth)

        Returns:
            Recall score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0

        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        hits = len(retrieved_set.intersection(relevant_set))
        return hits / len(relevant_set)

    def calculate_precision(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate precision: proportion of retrieved docs that are relevant.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs (ground truth)

        Returns:
            Precision score (0.0 to 1.0)
        """
        if not retrieved_ids:
            return 0.0

        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        hits = len(retrieved_set.intersection(relevant_set))
        return hits / len(retrieved_ids)

    def calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: List of relevant document IDs

        Returns:
            MRR score
        """
        relevant_set = set(relevant_ids)

        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    def calculate_hit_rate(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate hit rate: whether at least one relevant doc was retrieved.

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        return 1.0 if len(retrieved_set.intersection(relevant_set)) > 0 else 0.0

    def evaluate_single_query(self, query_data: Dict) -> Dict:
        """
        Evaluate a single query.

        Args:
            query_data: Dictionary containing query information

        Returns:
            Dictionary with evaluation metrics
        """
        query = query_data['query']
        relevant_docs = [str(doc_id) for doc_id in query_data['relevant_docs']]

        # Retrieve documents
        retrieved = self.retrieve_documents(query)
        retrieved_ids = [doc_id for doc_id, _, _ in retrieved]
        retrieved_scores = [score for _, score, _ in retrieved]
        retrieved_metadata = [metadata for _, _, metadata in retrieved]

        # Calculate metrics
        recall = self.calculate_recall(retrieved_ids, relevant_docs)
        precision = self.calculate_precision(retrieved_ids, relevant_docs)
        f1 = self.calculate_f1(precision, recall)
        mrr = self.calculate_mrr(retrieved_ids, relevant_docs)
        hit_rate = self.calculate_hit_rate(retrieved_ids, relevant_docs)

        return {
            'query_id': query_data['query_id'],
            'query': query,
            'query_type': query_data.get('query_type', 'unknown'),
            'relevant_docs': relevant_docs,
            'retrieved_docs': retrieved_ids,
            'retrieved_scores': retrieved_scores,
            'retrieved_metadata': retrieved_metadata,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'mrr': mrr,
            'hit_rate': hit_rate
        }

    def evaluate_all(self) -> Dict:
        """
        Evaluate all test queries.

        Returns:
            Dictionary with overall and per-query results
        """
        print("\n" + "="*80)
        print("STARTING EVALUATION")
        print("="*80)

        results = []

        for i, query_data in enumerate(self.test_queries, 1):
            print(f"\nEvaluating query {i}/{len(self.test_queries)}: {query_data['query_id']}")
            result = self.evaluate_single_query(query_data)
            results.append(result)

            # Print progress
            print(f"  Recall: {result['recall']:.3f} | Precision: {result['precision']:.3f} | F1: {result['f1']:.3f}")

        # Calculate overall metrics
        overall = {
            'num_queries': len(results),
            'avg_recall': np.mean([r['recall'] for r in results]),
            'avg_precision': np.mean([r['precision'] for r in results]),
            'avg_f1': np.mean([r['f1'] for r in results]),
            'avg_mrr': np.mean([r['mrr'] for r in results]),
            'hit_rate': np.mean([r['hit_rate'] for r in results]),
            'top_k': self.top_k
        }

        # Calculate metrics by query type
        query_types = set(r['query_type'] for r in results)
        by_type = {}

        for qtype in query_types:
            type_results = [r for r in results if r['query_type'] == qtype]
            by_type[qtype] = {
                'num_queries': len(type_results),
                'avg_recall': np.mean([r['recall'] for r in type_results]),
                'avg_precision': np.mean([r['precision'] for r in type_results]),
                'avg_f1': np.mean([r['f1'] for r in type_results]),
                'avg_mrr': np.mean([r['mrr'] for r in type_results]),
                'hit_rate': np.mean([r['hit_rate'] for r in type_results])
            }

        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'vector_db_path': self.vector_db_path,
                'embedding_model': self.embeddings.model_name,
                'top_k': self.top_k
            },
            'overall': overall,
            'by_query_type': by_type,
            'per_query_results': results
        }

    def print_report(self, evaluation_results: Dict):
        """Print a formatted evaluation report."""
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)

        overall = evaluation_results['overall']

        print("\nOVERALL METRICS")
        print("-" * 80)
        print(f"Number of queries: {overall['num_queries']}")
        print(f"Top-K retrieved: {overall['top_k']}")
        print()
        print(f"Chunk Recall (avg):    {overall['avg_recall']:.4f} ({overall['avg_recall']*100:.2f}%)")
        print(f"Precision (avg):       {overall['avg_precision']:.4f} ({overall['avg_precision']*100:.2f}%)")
        print(f"F1 Score (avg):        {overall['avg_f1']:.4f} ({overall['avg_f1']*100:.2f}%)")
        print(f"MRR (avg):             {overall['avg_mrr']:.4f}")
        print(f"Hit Rate:              {overall['hit_rate']:.4f} ({overall['hit_rate']*100:.2f}%)")

        print("\nMETRICS BY QUERY TYPE")
        print("-" * 80)

        for qtype, metrics in evaluation_results['by_query_type'].items():
            print(f"\n{qtype.upper()} ({metrics['num_queries']} queries)")
            print(f"  Recall:    {metrics['avg_recall']:.4f} ({metrics['avg_recall']*100:.2f}%)")
            print(f"  Precision: {metrics['avg_precision']:.4f} ({metrics['avg_precision']*100:.2f}%)")
            print(f"  F1 Score:  {metrics['avg_f1']:.4f} ({metrics['avg_f1']*100:.2f}%)")
            print(f"  MRR:       {metrics['avg_mrr']:.4f}")
            print(f"  Hit Rate:  {metrics['hit_rate']:.4f} ({metrics['hit_rate']*100:.2f}%)")

        print("\n" + "="*80)

    def save_results(self, evaluation_results: Dict, output_path: str):
        """Save evaluation results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Results saved to: {output_path}")


def main():
    """Main evaluation function."""
    # Configuration
    VECTOR_DB_PATH = "data/vector_db"
    TEST_QUERIES_PATH = "data/test_queries.json"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    TOP_K = 5

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation/retrieval_results_{timestamp}.json"

    # Create evaluator
    evaluator = RetrievalEvaluator(
        vector_db_path=VECTOR_DB_PATH,
        test_queries_path=TEST_QUERIES_PATH,
        embedding_model=EMBEDDING_MODEL,
        top_k=TOP_K
    )

    # Run evaluation
    results = evaluator.evaluate_all()

    # Print report
    evaluator.print_report(results)

    # Save results
    evaluator.save_results(results, output_path)

    print("\n[OK] Evaluation complete!")


if __name__ == "__main__":
    main()
