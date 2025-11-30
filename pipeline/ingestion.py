"""

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

Document Ingestion Script
Loads documents, splits text, creates embeddings, and stores in vector database.
"""

import argparse
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from dotenv import load_dotenv
import torch

import logging

load_dotenv()


def get_device():
    """Detect and return the best available device (GPU/CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("[GPU] CUDA detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
        logger.info("[GPU] Apple Silicon detected (MPS)")
    else:
        device = "cpu"
        logger.info("[CPU] No GPU detected, using CPU")
    return device


class DocumentIngestion:
    """Handles document loading, splitting, and vector database creation."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_provider: str = "llama",
        embedding_model: str = None
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_provider: "llama", "openai", "google", or "huggingface"
            embedding_model: Specific model name (optional)
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Initialize embeddings based on provider
        if embedding_provider == "llama":
            # Use Ollama for Llama embeddings (default: llama3.2:1b)
            model = embedding_model or "llama3.2:1b"
            self.embeddings = OllamaEmbeddings(
                model=model,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            logger.info("Using Llama embeddings via Ollama: {model}")
        elif embedding_provider == "huggingface":
            # Use HuggingFace embeddings (default: all-MiniLM-L6-v2)
            model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
            device = get_device()
            model_kwargs = {"device": device}
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs=model_kwargs
            )
            logger.info("Using HuggingFace embeddings: {model} on {device}")
        elif embedding_provider == "google":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        else:  # openai
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

    def load_document(self, input_path: str) -> List:
        """
        Load document from file or URL.

        Args:
            input_path: Path to file or URL

        Returns:
            List of loaded documents
        """
        if input_path.startswith("http"):
            # Load from URL
            logger.info("Loading URL: {input_path}")
            loader = UnstructuredURLLoader(urls=[input_path])
        elif input_path.endswith(".pdf"):
            # Load PDF
            logger.info("Loading PDF: {input_path}")
            loader = PyPDFLoader(input_path)
        elif input_path.endswith(".txt"):
            # Load text file
            logger.info("Loading TXT: {input_path}")
            loader = TextLoader(input_path)
        else:
            raise ValueError(f"Unsupported file type: {input_path}")

        return loader.load()

    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks.

        Args:
            documents: List of documents

        Returns:
            List of text chunks
        """
        logger.info("Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info("Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(
        self,
        chunks: List,
        db_path: str,
        use_faiss: bool = True
    ):
        """
        Create and save vector database.

        Args:
            chunks: List of text chunks
            db_path: Path to save the vector database
            use_faiss: If True, use FAISS; otherwise use Chroma
        """
        logger.info("Creating embeddings and vector database...")

        if use_faiss:
            # Create FAISS vector store
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            # Save to disk
            vectorstore.save_local(db_path)
            logger.info("FAISS vector database saved to: {db_path}")
        else:
            # Create Chroma vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=db_path
            )
            logger.info("Chroma vector database saved to: {db_path}")

        return vectorstore

    def ingest(
        self,
        input_path: str,
        db_path: str,
        use_faiss: bool = True
    ):
        """
        Complete ingestion pipeline.

        Args:
            input_path: Path to document or URL
            db_path: Path to save vector database
            use_faiss: If True, use FAISS; otherwise use Chroma
        """
        # Load document
        documents = self.load_document(input_path)

        # Split into chunks
        chunks = self.split_documents(documents)

        # Create and save vector database
        self.create_vectorstore(chunks, db_path, use_faiss)

        logger.info("Ingestion complete!")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector database"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to document (PDF/TXT) or URL"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/vector_db",
        help="Path to save vector database"
    )
    parser.add_argument(
        "--faiss",
        action="store_true",
        help="Use FAISS (default is Chroma if not specified)"
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
        "--embedding",
        type=str,
        default="llama",
        choices=["llama", "huggingface", "openai", "google"],
        help="Embedding provider (default: llama)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Specific embedding model (e.g., llama2, llama3, all-MiniLM-L6-v2)"
    )

    args = parser.parse_args()

    # Create ingestion instance
    ingestion = DocumentIngestion(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_provider=args.embedding,
        embedding_model=args.embedding_model
    )

    # Run ingestion
    ingestion.ingest(
        input_path=args.input,
        db_path=args.db_path,
        use_faiss=args.faiss
    )


if __name__ == "__main__":
    main()
