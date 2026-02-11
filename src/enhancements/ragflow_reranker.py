"""
Reranking Module - Adapted from RAGFlow
=====================================

SOURCE: https://github.com/infiniflow/ragflow
FILE: ragflow/rag/llm/rerank_model.py
LICENSE: Apache License 2.0

Copyright 2024 The InfiniFlow Authors

This code is adapted from RAGFlow for the Maharashtra Government CHO Training
RAG System (KCDH, IIT Bombay).

Modifications:
- Simplified for local use (removed cloud API dependencies)
- Added medical-specific reranking logic
- Integration with existing FAISS + BM25 system

WHAT THIS DOES:
- Takes retrieved documents from your hybrid search
- Re-scores them using cross-encoder models for better relevance
- Returns top-K documents with improved ranking

IMPROVEMENT EXPECTED: 10-30% better Precision@5
"""

import numpy as np
from typing import List, Tuple
from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalReranker:
    """
    ⭐ TAKEN FROM RAGFLOW: ragflow/rag/llm/rerank_model.py (Base class structure)

    Local reranking using sentence-transformers CrossEncoder
    No API calls needed - runs completely on your machine
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker

        Args:
            model_name: CrossEncoder model name
                - ms-marco-MiniLM-L-6-v2 (fast, good)
                - ms-marco-MiniLM-L-12-v2 (slower, better)
                - bge-reranker-base (best quality)
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> Tuple[List[int], np.ndarray]:
        """
        ⭐ TAKEN FROM RAGFLOW: ragflow/rag/llm/rerank_model.py (similarity method pattern)

        Rerank documents based on query relevance

        Args:
            query: User query
            documents: List of document texts
            top_k: Return top K documents (if None, return all sorted)

        Returns:
            Tuple of (indices, scores)
            - indices: Original indices sorted by relevance
            - scores: Relevance scores
        """
        if not documents:
            return [], np.array([])

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]

        # Apply top-k if specified
        if top_k is not None and top_k < len(sorted_indices):
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]

        logger.info(f"Reranked {len(documents)} docs, top score: {sorted_scores[0]:.4f}")

        return sorted_indices.tolist(), sorted_scores


class MedicalReranker(LocalReranker):
    """
    ⭐ NEW: Medical-specific reranking (YOUR CUSTOMIZATION)

    Extends LocalReranker with medical domain knowledge
    Boosts documents containing medical terms
    """

    # Medical terms that should boost document relevance
    MEDICAL_BOOST_TERMS = {
        'hypertension': 1.1,
        'diabetes': 1.1,
        'treatment': 1.05,
        'symptoms': 1.05,
        'diagnosis': 1.05,
        'blood pressure': 1.1,
        'medication': 1.05,
        'prevention': 1.05,
        'cho': 1.15,  # CHO-specific content gets highest boost
        'community health': 1.15,
    }

    def rerank_with_medical_boost(self, query: str, documents: List[str],
                                   metadata: List[dict] = None,
                                   top_k: int = None) -> Tuple[List[int], np.ndarray]:
        """
        ⭐ NEW: Medical-aware reranking

        Rerank with medical term boosting

        Args:
            query: User query
            documents: List of document texts
            metadata: Optional metadata for each document
            top_k: Return top K documents

        Returns:
            Tuple of (indices, scores) with medical boosting applied
        """
        # Get base reranking scores
        indices, scores = self.rerank(query, documents, top_k=None)

        # Apply medical boosting
        boosted_scores = scores.copy()
        for i, (idx, score) in enumerate(zip(indices, scores)):
            doc = documents[idx].lower()
            boost_factor = 1.0

            # Check for medical terms
            for term, boost in self.MEDICAL_BOOST_TERMS.items():
                if term in doc:
                    boost_factor *= boost

            # Apply boost
            boosted_scores[i] = score * boost_factor

        # Re-sort with boosted scores
        boosted_indices_order = np.argsort(boosted_scores)[::-1]
        final_indices = [indices[i] for i in boosted_indices_order]
        final_scores = boosted_scores[boosted_indices_order]

        # Apply top-k
        if top_k is not None and top_k < len(final_indices):
            final_indices = final_indices[:top_k]
            final_scores = final_scores[:top_k]

        logger.info(f"Medical reranking: top score after boost: {final_scores[0]:.4f}")

        return final_indices, final_scores


def rerank_hybrid_results(query: str, retrieved_docs: List,
                          reranker: LocalReranker = None,
                          top_k: int = 5) -> List:
    """
    ⭐ INTEGRATION FUNCTION: Use this in your optimized_rag_chat.py

    Rerank documents retrieved from hybrid search

    Args:
        query: User query
        retrieved_docs: Documents from your hybrid_search function
        reranker: Reranker instance (will create if None)
        top_k: Return top K documents

    Returns:
        List of reranked documents

    Example:
        >>> # In your optimized_rag_chat.py:
        >>> hybrid_results = hybrid_search(query, vector_db, alpha=0.7, k=10)
        >>> reranked_results = rerank_hybrid_results(query, hybrid_results, top_k=5)
        >>> # Use reranked_results for LLM generation
    """
    if not retrieved_docs:
        return []

    # Create reranker if not provided
    if reranker is None:
        reranker = MedicalReranker()

    # Extract document texts
    doc_texts = [doc.page_content for doc in retrieved_docs]

    # Rerank
    if isinstance(reranker, MedicalReranker):
        # Use medical-aware reranking
        metadata = [doc.metadata for doc in retrieved_docs]
        indices, scores = reranker.rerank_with_medical_boost(
            query, doc_texts, metadata=metadata, top_k=top_k
        )
    else:
        # Use standard reranking
        indices, scores = reranker.rerank(query, doc_texts, top_k=top_k)

    # Reorder documents
    reranked_docs = [retrieved_docs[idx] for idx in indices]

    # Add reranking scores to metadata
    for doc, score in zip(reranked_docs, scores):
        doc.metadata['rerank_score'] = float(score)

    logger.info(f"Reranked from {len(retrieved_docs)} to {len(reranked_docs)} docs")

    return reranked_docs


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_reranker():
    """Test the reranker with sample medical queries"""
    print("\n" + "="*70)
    print("TESTING RERANKER - Adapted from RAGFlow")
    print("="*70)

    # Sample query
    query = "What is the treatment for hypertension?"

    # Sample documents (simulating retrieved docs)
    documents = [
        "Hypertension, or high blood pressure, is a common condition. Treatment includes lifestyle changes and medication.",
        "Diabetes mellitus is managed through diet, exercise, and insulin therapy.",
        "The CHO (Community Health Officer) provides primary healthcare services in rural areas.",
        "Treatment for hypertension includes ACE inhibitors, beta-blockers, and diuretics. Lifestyle modifications are also important.",
        "Regular exercise can help prevent various diseases including obesity and heart disease.",
    ]

    print(f"\nQuery: {query}")
    print(f"\nDocuments to rerank: {len(documents)}")

    # Test 1: Standard reranker
    print("\n" + "-"*70)
    print("Test 1: Standard Reranker")
    print("-"*70)

    standard_reranker = LocalReranker()
    indices, scores = standard_reranker.rerank(query, documents, top_k=3)

    print("\nReranked Results:")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   Doc: {documents[idx][:100]}...")

    # Test 2: Medical reranker
    print("\n" + "-"*70)
    print("Test 2: Medical-Aware Reranker")
    print("-"*70)

    medical_reranker = MedicalReranker()
    indices, scores = medical_reranker.rerank_with_medical_boost(query, documents, top_k=3)

    print("\nMedical Reranked Results:")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   Doc: {documents[idx][:100]}...")

    print("\n" + "="*70)
    print("✅ Reranker test complete!")
    print("="*70)


if __name__ == "__main__":
    # Run tests
    test_reranker()
