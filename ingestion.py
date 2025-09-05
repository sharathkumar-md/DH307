"""
ingestion.py
Module for loading documents, splitting text, creating embeddings, and ingesting into a vector database (FAISS/Chroma).
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# Load environment variables (API keys, etc.)
load_dotenv()

# Initialize embedding model only when needed (lazy loading)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Supported loaders for different file types
def load_document(file_path_or_url):
    """
    Load a document from a PDF, text file, or URL.
    """
    if file_path_or_url.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path_or_url)
    elif file_path_or_url.lower().endswith('.txt'):
        loader = TextLoader(file_path_or_url)
    elif file_path_or_url.startswith('http'):
        loader = WebBaseLoader(file_path_or_url)
    else:
        raise ValueError("Unsupported file type or URL.")
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_vector_db(chunks, db_path=None, use_faiss=True):
    """
    Create or update a vector database (FAISS or Chroma) with embedded chunks.
    """
    embedding_model = get_embedding_model()
    if use_faiss:
        db = FAISS.from_documents(chunks, embedding_model)
        if db_path:
            db.save_local(db_path)
    else:
        db = Chroma.from_documents(chunks, embedding_model, persist_directory=db_path)
        db.persist()
    return db

def ingest(file_path_or_url, db_path="vector_db", use_faiss=True):
    """
    Full ingestion pipeline: load, split, embed, and store documents.
    """
    print(f"Loading document: {file_path_or_url}")
    docs = load_document(file_path_or_url)
    print(f"Splitting document into chunks...")
    chunks = split_documents(docs)
    print(f"Creating vector database and storing embeddings...")
    db = create_vector_db(chunks, db_path=db_path, use_faiss=use_faiss)
    print(f"Ingestion complete. Vector DB stored at: {db_path}")
    return db

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest documents into vector DB.")
    parser.add_argument("--input", required=True, help="Path to PDF/TXT file or URL.")
    parser.add_argument("--db_path", default="vector_db", help="Path to store the vector DB.")
    parser.add_argument("--faiss", action="store_true", help="Use FAISS (default). If not set, uses Chroma.")
    args = parser.parse_args()
    ingest(args.input, db_path=args.db_path, use_faiss=args.faiss)
