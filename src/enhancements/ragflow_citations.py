"""
Citation Grounding - Adapted from RAGFlow
=========================================

SOURCE: https://github.com/infiniflow/ragflow
FILE: ragflow/rag/nlp/search.py (lines 170-200+ insert_citations method)
LICENSE: Apache License 2.0

Copyright 2024 The InfiniFlow Authors

This code is adapted from RAGFlow for the Maharashtra Government CHO Training
RAG System (KCDH, IIT Bombay).

WHAT THIS DOES:
- Extracts exact passages from retrieved documents that support the LLM answer
- Adds inline citations [1], [2] to the response
- Provides grounding evidence (what exact text was used)
- CRITICAL for medical/government applications (accountability!)

IMPROVEMENT: Trustworthiness, explainability, compliance
"""

import re
from typing import List, Dict, Tuple, Set
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CitationGrounder:
    """
    ⭐ TAKEN FROM RAGFLOW: ragflow/rag/nlp/search.py (insert_citations method pattern)

    Adds citations and grounding evidence to LLM responses
    """

    @staticmethod
    def insert_citations(answer: str,
                        chunks: List[Document],
                        similarity_threshold: float = 0.5) -> Tuple[str, List[Dict]]:
        """
        ⭐ TAKEN FROM RAGFLOW: ragflow/rag/nlp/search.py (lines 170-230)

        Insert citation markers into answer and extract supporting evidence

        Args:
            answer: LLM generated answer
            chunks: Retrieved document chunks
            similarity_threshold: Minimum similarity to consider a chunk relevant

        Returns:
            Tuple of (answer_with_citations, citation_evidence)
        """
        if not chunks or not answer:
            return answer, []

        # Split answer into sentences
        sentences = CitationGrounder._split_into_sentences(answer)

        # Track citations
        citations = []
        cited_chunks = set()
        answer_with_citations = []

        for sent_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                answer_with_citations.append(sentence)
                continue

            # Find supporting chunks for this sentence
            supporting_chunks = CitationGrounder._find_supporting_chunks(
                sentence, chunks, similarity_threshold
            )

            # Add citation markers
            if supporting_chunks:
                citation_nums = []
                for chunk_idx in supporting_chunks:
                    if chunk_idx not in cited_chunks:
                        cited_chunks.add(chunk_idx)
                        citation_nums.append(len(cited_chunks))

                        # Add to citations list
                        chunk = chunks[chunk_idx]
                        citations.append({
                            'citation_num': len(cited_chunks),
                            'chunk_index': chunk_idx,
                            'source': chunk.metadata.get('source', 'Unknown'),
                            'page': chunk.metadata.get('page_number', 'N/A'),
                            'content': chunk.page_content[:200] + "...",  # First 200 chars
                            'full_content': chunk.page_content,
                            'metadata': chunk.metadata
                        })
                    else:
                        # Already cited, find its citation number
                        for cit in citations:
                            if cit['chunk_index'] == chunk_idx:
                                citation_nums.append(cit['citation_num'])
                                break

                # Add citations to sentence
                if citation_nums:
                    citation_str = ''.join([f'[{num}]' for num in sorted(set(citation_nums))])
                    sentence_with_citation = sentence.strip() + ' ' + citation_str
                    answer_with_citations.append(sentence_with_citation)
                else:
                    answer_with_citations.append(sentence)
            else:
                answer_with_citations.append(sentence)

        # Reconstruct answer
        cited_answer = ' '.join(answer_with_citations)

        logger.info(f"Added {len(citations)} citations to answer")

        return cited_answer, citations

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        ⭐ TAKEN FROM RAGFLOW: ragflow/rag/nlp/search.py (lines 175-196)

        Split text into sentences (handles code blocks)
        """
        # Handle code blocks (don't split inside ```)
        pieces = re.split(r"(```)", text)

        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    # Split by sentence terminators
                    pieces_.extend(
                        re.split(
                            r"([^\|][।।।!。\n]|[a-z][.?;!][ \n])",  # Indian & English punctuation
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            # Simple sentence splitting
            pieces = re.split(r"([^\|][।।।!。\n]|[a-z][.?;!][ \n])", text)

        # Reconstruct sentences
        sentences = []
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][।।।!।\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]

        sentences = [p for p in pieces if p.strip()]
        return sentences

    @staticmethod
    def _find_supporting_chunks(sentence: str,
                               chunks: List[Document],
                               threshold: float = 0.5) -> List[int]:
        """
        ⭐ NEW: Find chunks that support a given sentence

        Uses simple keyword overlap (fast)
        For production, could use sentence embeddings (slower but better)

        Args:
            sentence: Sentence to find support for
            chunks: List of document chunks
            threshold: Minimum overlap ratio (0-1)

        Returns:
            List of chunk indices that support the sentence
        """
        sentence_words = set(sentence.lower().split())
        supporting = []

        for idx, chunk in enumerate(chunks):
            chunk_words = set(chunk.page_content.lower().split())

            # Calculate overlap
            overlap = len(sentence_words & chunk_words)
            total = len(sentence_words)

            if total > 0 and (overlap / total) >= threshold:
                supporting.append(idx)

        return supporting


def format_citations_for_display(citations: List[Dict]) -> str:
    """
    ⭐ NEW: Format citations for user display

    Creates a clean citation list to show after the answer

    Args:
        citations: List of citation dictionaries

    Returns:
        Formatted citation string

    Example output:
        Sources:
        [1] CHO_Training_Manual.pdf (Page 45)
            "Hypertension is defined as blood pressure above 140/90 mmHg..."

        [2] Medical_Guidelines.pdf (Page 12)
            "Treatment includes lifestyle modifications and medication..."
    """
    if not citations:
        return ""

    citation_text = "\n\n📚 **Sources:**\n"

    for cit in citations:
        citation_text += f"\n[{cit['citation_num']}] {cit['source']}"

        if cit['page'] != 'N/A':
            citation_text += f" (Page {cit['page']})"

        citation_text += f"\n    \"{cit['content']}\"\n"

    return citation_text


def create_grounded_response(query: str,
                             answer: str,
                             retrieved_chunks: List[Document],
                             include_full_content: bool = False) -> Dict:
    """
    ⭐ INTEGRATION FUNCTION: Use this in your optimized_rag_chat.py

    Create a fully grounded response with citations

    Args:
        query: User query
        answer: LLM generated answer
        retrieved_chunks: Chunks used for generation
        include_full_content: Include full chunk content in response

    Returns:
        Dictionary with grounded response

    Example:
        >>> # In your optimized_rag_chat.py:
        >>> answer = llm.generate(query, context)
        >>> grounded = create_grounded_response(query, answer, retrieved_chunks)
        >>> st.write(grounded['answer_with_citations'])
        >>> st.markdown(grounded['citation_display'])
    """
    # Insert citations
    cited_answer, citations = CitationGrounder.insert_citations(
        answer, retrieved_chunks
    )

    # Format for display
    citation_display = format_citations_for_display(citations)

    # Create response object
    response = {
        'query': query,
        'answer': answer,
        'answer_with_citations': cited_answer,
        'citation_display': citation_display,
        'citations': citations,
        'num_citations': len(citations),
        'is_grounded': len(citations) > 0
    }

    # Optionally include full content
    if include_full_content:
        response['full_chunks'] = [
            {
                'content': chunk.page_content,
                'metadata': chunk.metadata
            }
            for chunk in retrieved_chunks
        ]

    logger.info(f"Created grounded response with {len(citations)} citations")

    return response


# ============================================================================
# MEDICAL-SPECIFIC GROUNDING
# ============================================================================

class MedicalCitationValidator:
    """
    ⭐ NEW: Medical-specific citation validation

    For government healthcare systems, ensure ALL medical claims are cited
    """

    # Medical claim keywords that MUST be cited
    MEDICAL_CLAIM_KEYWORDS = [
        'treatment', 'medication', 'drug', 'therapy',
        'diagnosis', 'symptoms', 'causes',
        'risk', 'side effect', 'complication',
        'dosage', 'contraindication',
        'prevention', 'screening'
    ]

    @staticmethod
    def validate_medical_claims(answer: str, citations: List[Dict]) -> Dict:
        """
        ⭐ NEW: Validate that medical claims are properly cited

        Args:
            answer: Answer with citations
            citations: Citation list

        Returns:
            Validation results
        """
        sentences = CitationGrounder._split_into_sentences(answer)

        uncited_claims = []
        for sentence in sentences:
            # Check if sentence contains medical claim
            has_claim = any(
                keyword in sentence.lower()
                for keyword in MedicalCitationValidator.MEDICAL_CLAIM_KEYWORDS
            )

            # Check if cited
            has_citation = bool(re.search(r'\[\d+\]', sentence))

            if has_claim and not has_citation:
                uncited_claims.append(sentence)

        is_valid = len(uncited_claims) == 0

        return {
            'is_valid': is_valid,
            'num_uncited_claims': len(uncited_claims),
            'uncited_claims': uncited_claims,
            'warning': "⚠️ Medical claims found without citations!" if not is_valid else None
        }


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_citations():
    """Test citation grounding"""
    print("\n" + "="*70)
    print("TESTING CITATION GROUNDING - Adapted from RAGFlow")
    print("="*70)

    # Sample answer
    answer = """
    Hypertension is a common condition affecting many people.
    Treatment includes lifestyle changes such as diet and exercise.
    Medication like ACE inhibitors may also be prescribed by doctors.
    Regular monitoring of blood pressure is essential for management.
    """

    # Sample chunks
    from langchain_core.documents import Document

    chunks = [
        Document(
            page_content="Hypertension, also known as high blood pressure, is a common medical condition.",
            metadata={'source': 'CHO_Manual.pdf', 'page_number': 45}
        ),
        Document(
            page_content="Treatment for hypertension includes lifestyle modifications like diet, exercise, and stress reduction.",
            metadata={'source': 'CHO_Manual.pdf', 'page_number': 46}
        ),
        Document(
            page_content="ACE inhibitors are a class of medication used to treat high blood pressure.",
            metadata={'source': 'Medical_Guide.pdf', 'page_number': 12}
        ),
        Document(
            page_content="Regular blood pressure monitoring helps track treatment effectiveness.",
            metadata={'source': 'CHO_Manual.pdf', 'page_number': 47}
        ),
    ]

    print("\n📝 Original Answer:")
    print(answer)

    print("\n" + "-"*70)

    # Add citations
    cited_answer, citations = CitationGrounder.insert_citations(answer, chunks, similarity_threshold=0.3)

    print("\n✅ Answer with Citations:")
    print(cited_answer)

    print("\n" + "-"*70)
    print("\n📚 Citation Details:")
    citation_display = format_citations_for_display(citations)
    print(citation_display)

    print("\n" + "-"*70)

    # Validate medical claims
    validation = MedicalCitationValidator.validate_medical_claims(cited_answer, citations)
    print("\n🏥 Medical Claim Validation:")
    print(f"Valid: {validation['is_valid']}")
    if not validation['is_valid']:
        print(f"Uncited claims: {validation['num_uncited_claims']}")
        for claim in validation['uncited_claims']:
            print(f"  - {claim}")

    print("\n" + "="*70)
    print("✅ Citation test complete!")
    print("="*70)


if __name__ == "__main__":
    test_citations()
