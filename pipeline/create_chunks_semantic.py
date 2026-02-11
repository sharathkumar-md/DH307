"""
Semantic Chunking - Create chunks using semantic similarity
Uses LlamaIndex for intelligent semantic-based text splitting.
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import torch
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Try to import LlamaIndex for semantic chunking
try:
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Document as LlamaDocument
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logger.info("LlamaIndex not available. Install with: pip install llama-index")
    LLAMAINDEX_AVAILABLE = False


def get_device():
    """Detect GPU."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("[GPU] CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("[GPU] Apple Silicon (MPS)")
    else:
        device = "cpu"
        logger.info("[CPU] Using CPU")
    return device


class SemanticChunker:
    """Semantic chunking using LlamaIndex."""

    def __init__(
        self,
        json_path: str,
        output_db_path: str,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        breakpoint_percentile_threshold: int = 95
    ):
        self.json_path = Path(json_path)
        self.output_db_path = Path(output_db_path)
        self.device = get_device()

        # Load JSON
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info("Loaded {self.data['num_documents']} documents")

        # Initialize embeddings for vector DB
        logger.info("Loading embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={
                "batch_size": 32 if self.device == "cuda" else 8,
                "show_progress_bar": True,
                "normalize_embeddings": True
            }
        )

        # Initialize LlamaIndex embeddings for semantic splitting
        if LLAMAINDEX_AVAILABLE:
            self.llama_embeddings = HuggingFaceEmbedding(
                model_name=embedding_model,
                device=self.device
            )

            self.semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                embed_model=self.llama_embeddings
            )
            logger.info("Semantic chunking enabled (threshold: {breakpoint_percentile_threshold})")

    def create_semantic_chunks(self) -> List[Document]:
        """Create chunks using semantic similarity."""
        if not LLAMAINDEX_AVAILABLE:
            logger.error(" LlamaIndex not available!")
            return []

        logger.info("\n" + "="*80)
        logger.info("CREATING SEMANTIC CHUNKS")
        logger.info("="*80)

        all_documents = []
        chunk_counter = 0

        for doc_idx, doc in enumerate(self.data['documents']):
            filename = doc['filename']
            logger.info("\nProcessing: {filename}")

            # Combine all non-empty pages into document text
            full_text = ""
            page_boundaries = []

            for page in doc['pages']:
                if page['content'] and page['content'].strip():
                    start_pos = len(full_text)
                    full_text += page['content'] + "\n\n"
                    page_boundaries.append({
                        'start': start_pos,
                        'end': len(full_text),
                        'page_num': page['page_number'],
                        'page_label': page['page_label']
                    })

            if not full_text.strip():
                continue

            # Create LlamaIndex document
            llama_doc = LlamaDocument(text=full_text)

            # Apply semantic splitting
            nodes = self.semantic_splitter.get_nodes_from_documents([llama_doc])

            # Convert nodes to LangChain documents
            for node_idx, node in enumerate(nodes):
                # Find which page this chunk belongs to
                chunk_start = full_text.find(node.text)
                page_info = None
                for page_boundary in page_boundaries:
                    if (chunk_start >= page_boundary['start'] and
                        chunk_start < page_boundary['end']):
                        page_info = page_boundary
                        break

                metadata = {
                    "source_file": filename,
                    "file_path": doc['filepath'],
                    "chunk_id": f"{filename}_chunk_{chunk_counter}",
                    "chunk_index": chunk_counter,
                    "chunk_size": len(node.text),
                    "chunking_strategy": "semantic",
                    "document_index": doc_idx,
                    "created_at": datetime.now().isoformat()
                }

                if page_info:
                    metadata.update({
                        "page": page_info['page_num'],
                        "page_label": page_info['page_label']
                    })

                lang_doc = Document(
                    page_content=node.text,
                    metadata=metadata
                )
                all_documents.append(lang_doc)
                chunk_counter += 1

            logger.info("  Created {chunk_counter} semantic chunks")

        logger.info("\n{'='*80}")
        logger.info("Total semantic chunks: {len(all_documents)}")
        return all_documents

    def create_vector_database(self, documents: List[Document]):
        """Create FAISS vector database."""
        logger.info("\n" + "="*80)
        logger.info("CREATING VECTOR DATABASE")
        logger.info("="*80)

        self.output_db_path.mkdir(parents=True, exist_ok=True)

        logger.info("Embedding {len(documents)} chunks with GPU acceleration...")

        vectorstore = FAISS.from_documents(documents, self.embeddings)
        vectorstore.save_local(str(self.output_db_path))

        logger.info("\n[OK] Vector database saved to: {self.output_db_path}")

        # Save metadata
        self.save_metadata(documents)

    def save_metadata(self, documents: List[Document]):
        """Save metadata."""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_chunks": len(documents),
            "chunking_strategy": "semantic",
            "embedding_model": self.embeddings.model_name
        }

        metadata_path = self.output_db_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info("[OK] Metadata saved")

        # Chunk mapping
        chunk_mapping = []
        for doc in documents:
            chunk_mapping.append({
                "chunk_index": doc.metadata['chunk_index'],
                "chunk_id": doc.metadata['chunk_id'],
                "source_file": doc.metadata['source_file'],
                "page": doc.metadata.get('page', 'N/A'),
                "content_preview": doc.page_content[:100]
            })

        mapping_path = self.output_db_path / "chunk_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_mapping, f, indent=2)
        logger.info("[OK] Chunk mapping saved")

    def run(self):
        """Run semantic chunking pipeline."""
        logger.info("\n" + "="*80)
        logger.info("SEMANTIC CHUNKING PIPELINE")
        logger.info("="*80)

        documents = self.create_semantic_chunks()
        if documents:
            self.create_vector_database(documents)
            logger.info("\n[OK] Pipeline complete!")
            return True
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic chunking with LlamaIndex")
    parser.add_argument("--json", default="data/parsed_documents.json")
    parser.add_argument("--output", default="data/vector_db_semantic")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--threshold", type=int, default=95,
                       help="Semantic breakpoint threshold (90-99)")

    args = parser.parse_args()

    if not LLAMAINDEX_AVAILABLE:
        logger.error(" Please install LlamaIndex:")
        logger.info("  pip install llama-index llama-index-embeddings-huggingface")
        return False

    chunker = SemanticChunker(
        json_path=args.json,
        output_db_path=args.output,
        embedding_model=args.embedding_model,
        breakpoint_percentile_threshold=args.threshold
    )

    return chunker.run()


if __name__ == "__main__":
    import sys
    if not main():
        sys.exit(1)
