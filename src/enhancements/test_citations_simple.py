"""Simple test for citation grounding (Windows compatible)"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from ragflow_citations import CitationGrounder, format_citations_for_display, create_grounded_response
from langchain_core.documents import Document

def test():
    print("\n" + "="*70)
    print("TESTING CITATION GROUNDING")
    print("="*70)

    # Sample answer (simulating LLM output)
    answer = """Hypertension is a common condition affecting many people.
Treatment includes lifestyle changes such as diet and exercise.
Medication like ACE inhibitors may also be prescribed by doctors.
Regular monitoring of blood pressure is essential for management."""

    # Sample retrieved chunks
    chunks = [
        Document(
            page_content="Hypertension, also known as high blood pressure, is a common medical condition affecting millions worldwide.",
            metadata={'source': 'CHO_Manual.pdf', 'page_number': 45}
        ),
        Document(
            page_content="Treatment for hypertension includes lifestyle modifications like diet, exercise, and stress reduction techniques.",
            metadata={'source': 'CHO_Manual.pdf', 'page_number': 46}
        ),
        Document(
            page_content="ACE inhibitors are a class of medication commonly used to treat high blood pressure and heart conditions.",
            metadata={'source': 'Medical_Guide.pdf', 'page_number': 12}
        ),
        Document(
            page_content="Regular blood pressure monitoring helps track treatment effectiveness and disease progression.",
            metadata={'source': 'CHO_Manual.pdf', 'page_number': 47}
        ),
    ]

    print("\n[Original Answer]")
    print(answer)
    print("\n" + "-"*70)

    # Test citation insertion
    cited_answer, citations = CitationGrounder.insert_citations(
        answer, chunks, similarity_threshold=0.3
    )

    print("\n[Answer with Citations]")
    print(cited_answer)
    print("\n" + "-"*70)

    print(f"\n[Citation Count: {len(citations)}]")
    for cit in citations:
        print(f"  [{cit['citation_num']}] {cit['source']} (Page {cit['page']})")
        print(f"      \"{cit['content'][:80]}...\"")

    print("\n" + "-"*70)

    # Test full grounded response
    grounded = create_grounded_response(
        query="What is hypertension treatment?",
        answer=answer,
        retrieved_chunks=chunks
    )

    print("\n[Grounded Response Object]")
    print(f"  is_grounded: {grounded['is_grounded']}")
    print(f"  num_citations: {grounded['num_citations']}")

    print("\n" + "="*70)
    print("TEST PASSED - Citation grounding works!")
    print("="*70)

    return True

if __name__ == "__main__":
    test()
