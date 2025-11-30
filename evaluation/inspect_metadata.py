"""Quick script to inspect vector database metadata."""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Load embeddings
device = get_device()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device}
)

# Load vector database
vectorstore = FAISS.load_local(
    "data/vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# Get a sample query
results = vectorstore.similarity_search("What is the formula for EDD?", k=10)

# Collect metadata info
metadata_info = []
for i, doc in enumerate(results, 1):
    info = {
        "doc_num": i,
        "content_preview": doc.page_content[:200],
        "metadata_keys": list(doc.metadata.keys()),
        "metadata": doc.metadata
    }
    metadata_info.append(info)

# Save to file
output_path = "evaluation/metadata_inspection.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(metadata_info, f, indent=2, ensure_ascii=False)

print(f"Metadata inspection saved to: {output_path}")
print(f"\nFound {len(results)} documents")
print(f"Metadata keys in first document: {list(results[0].metadata.keys())}")
print(f"Sample metadata: {results[0].metadata}")
