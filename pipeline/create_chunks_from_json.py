"""
JSON to Vector DB - Create chunks from parsed JSON and build vector database
Reads structured JSON data and creates a vector database with proper chunk indexing.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import torch


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device():
    """Detect and return the best available device (GPU/CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"[GPU] CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("[CPU] No GPU detected, using CPU")
    return device


class JSONChunker:
    """Create chunks from parsed JSON and build vector database."""

    def __init__(
        self,
        json_path: str,
        output_db_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize chunker.

        Args:
            json_path: Path to parsed JSON file
            output_db_path: Path to save vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: HuggingFace embedding model name
        """
        self.json_path = Path(json_path)
        self.output_db_path = Path(output_db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load JSON data
        logger.info("Loading data from: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info("  Documents: {self.data['num_documents']}")
        logger.info("  Total pages: {self.data['total_pages']}")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialize embeddings with GPU acceleration
        device = get_device()
        logger.info("Loading embeddings: {embedding_model}")

        model_kwargs = {
            "device": device,
            "trust_remote_code": True
        }

        # Enable GPU optimizations if available
        if device == "cuda":
            model_kwargs.update({
                "device": device,
            })
            logger.info("  GPU acceleration ENABLED")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs={
                "batch_size": 32 if device == "cuda" else 8,  # Larger batch on GPU
                "show_progress_bar": True,
                "normalize_embeddings": True
            }
        )

    def create_documents_from_json(self) -> List[Document]:
        """
        Create LangChain documents from JSON data.

        Returns:
            List of Document objects
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING DOCUMENTS FROM JSON")
        logger.info("="*80)

        documents = []
        chunk_counter = 0

        for doc_idx, doc in enumerate(self.data['documents']):
            filename = doc['filename']
            logger.info("\nProcessing: {filename}")

            # Process each page
            for page in doc['pages']:
                page_num = page['page_number']
                page_label = page['page_label']
                content = page['content']

                # Skip empty pages
                if not content or not content.strip():
                    continue

                # Create chunks for this page
                chunks = self.text_splitter.split_text(content)

                # Create Document objects with metadata
                for chunk_idx, chunk_text in enumerate(chunks):
                    metadata = {
                        # Source information
                        "source_file": filename,
                        "file_path": doc['filepath'],

                        # Page information
                        "page": page_num,
                        "page_label": page_label,

                        # Chunk information
                        "chunk_id": f"{filename}_chunk_{chunk_counter}",
                        "chunk_index": chunk_counter,
                        "chunk_within_page": chunk_idx,
                        "chunk_size": len(chunk_text),

                        # Document structure
                        "document_index": doc_idx,
                        "total_chunks_in_page": len(chunks),

                        # Processing metadata
                        "chunking_strategy": "recursive",
                        "chunk_config": f"size={self.chunk_size},overlap={self.chunk_overlap}",
                        "created_at": datetime.now().isoformat()
                    }

                    # Add original page metadata
                    if 'metadata' in page:
                        for k, v in page['metadata'].items():
                            if k not in metadata:  # Don't override
                                metadata[k] = v

                    doc_obj = Document(
                        page_content=chunk_text,
                        metadata=metadata
                    )
                    documents.append(doc_obj)
                    chunk_counter += 1

            logger.info("  Created {chunk_counter} chunks so far")

        logger.info("\n{'='*80}")
        logger.info("Total chunks created: {len(documents)}")
        return documents

    def create_vector_database(self, documents: List[Document]):
        """
        Create FAISS vector database from documents.

        Args:
            documents: List of Document objects
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING VECTOR DATABASE")
        logger.info("="*80)

        # Create output directory
        self.output_db_path.mkdir(parents=True, exist_ok=True)

        logger.info("Embedding {len(documents)} chunks...")
        logger.info("This may take a while...")

        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(documents, self.embeddings)

        # Save to disk
        db_path = str(self.output_db_path)
        vectorstore.save_local(db_path)

        logger.info("\n[OK] Vector database saved to: {db_path}")

        # Save metadata
        self.save_metadata(documents)

    def save_metadata(self, documents: List[Document]):
        """
        Save metadata about the vector database.

        Args:
            documents: List of Document objects
        """
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_chunks": len(documents),
            "total_documents": self.data['num_documents'],
            "total_pages": self.data['total_pages'],
            "embedding_model": self.embeddings.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunks_per_document": {
                doc['filename']: sum(
                    1 for d in documents
                    if d.metadata['source_file'] == doc['filename']
                )
                for doc in self.data['documents']
            }
        }

        metadata_path = self.output_db_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("[OK] Metadata saved to: {metadata_path}")

        # Also save chunk index mapping for reference
        chunk_mapping = []
        for doc in documents:
            chunk_mapping.append({
                "chunk_index": doc.metadata['chunk_index'],
                "chunk_id": doc.metadata['chunk_id'],
                "source_file": doc.metadata['source_file'],
                "page": doc.metadata['page'],
                "page_label": doc.metadata['page_label'],
                "content_preview": doc.page_content[:100]
            })

        mapping_path = self.output_db_path / "chunk_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_mapping, f, indent=2, ensure_ascii=False)

        logger.info("[OK] Chunk mapping saved to: {mapping_path}")

    def run(self):
        """Run the complete chunking and embedding pipeline."""
        logger.info("\n" + "="*80)
        logger.info("JSON TO VECTOR DB PIPELINE")
        logger.info("="*80)

        # Create documents
        documents = self.create_documents_from_json()

        if not documents:
            logger.error(" No documents were created")
            return False

        # Create vector database
        self.create_vector_database(documents)

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*80)
        return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create chunks from JSON and build vector database"
    )
    parser.add_argument(
        "--json",
        type=str,
        default="data/parsed_documents.json",
        help="Input JSON file from PDF parser"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/vector_db",
        help="Output directory for vector database"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of text chunks"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="HuggingFace embedding model"
    )

    args = parser.parse_args()

    # Create chunker and run
    chunker = JSONChunker(
        json_path=args.json,
        output_db_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model
    )

    success = chunker.run()

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
