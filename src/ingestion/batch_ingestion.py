"""
batch_ingestion.py
Enhanced ingestion module that supports multiple files and batch processing.
"""
import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# Load environment variables
load_dotenv()

def get_embedding_model():
    """Get embedding model - reusable across ingestions"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_supported_files(directory):
    """Get all supported files from a directory"""
    supported_extensions = ['.pdf', '.txt']
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
    return files

def load_document(file_path_or_url):
    """Load a single document"""
    if file_path_or_url.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path_or_url)
    elif file_path_or_url.lower().endswith('.txt'):
        loader = TextLoader(file_path_or_url, encoding='utf-8')
    elif file_path_or_url.startswith('http'):
        loader = WebBaseLoader(file_path_or_url)
    else:
        raise ValueError(f"Unsupported file type: {file_path_or_url}")
    return loader.load()

def load_multiple_documents(file_paths):
    """Load multiple documents and combine them"""
    all_documents = []
    failed_files = []
    
    for file_path in file_paths:
        try:
            print(f"📄 Loading: {os.path.basename(file_path)}")
            docs = load_document(file_path)
            # Add enhanced metadata for source tracking
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    'source_file': os.path.basename(file_path),
                    'full_path': file_path,
                    'page_number': doc.metadata.get('page', i + 1),
                    'document_id': f"{os.path.basename(file_path)}_page_{i + 1}",
                    'content_length': len(doc.page_content),
                    'document_type': 'healthcare_guideline'
                })
            all_documents.extend(docs)
            print(f"✅ Loaded {len(docs)} pages from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {str(e)}")
            failed_files.append(file_path)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully loaded: {len(file_paths) - len(failed_files)} files")
    print(f"   ❌ Failed to load: {len(failed_files)} files")
    print(f"   📑 Total pages/documents: {len(all_documents)}")
    
    return all_documents, failed_files

def split_documents(documents, chunk_size=350, chunk_overlap=50):
    """Split documents into chunks with enhanced metadata"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        original_source = chunk.metadata.get('source_file', 'Unknown')
        chunk.metadata.update({
            'chunk_id': f"{original_source}_chunk_{i}",
            'chunk_index': i,
            'chunk_size': len(chunk.page_content),
            'total_chunks': len(chunks)
        })
    
    print(f"📝 Created {len(chunks)} text chunks (avg size: {chunk_size} chars)")
    return chunks

def create_or_update_vector_db(chunks, db_path="data/vector_db", use_faiss=True):
    """Create new or update existing vector database"""
    embedding_model = get_embedding_model()
    
    # Check if there are chunks to process
    if not chunks:
        print("⚠️ No chunks to process, skipping database update")
        if use_faiss and os.path.exists(os.path.join(db_path, "index.faiss")):
            return FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        elif not use_faiss and os.path.exists(db_path):
            return Chroma(persist_directory=db_path, embedding_function=embedding_model)
        return None
    
    # Check if database already exists
    db_exists = False
    if use_faiss:
        db_exists = os.path.exists(os.path.join(db_path, "index.faiss"))
    else:
        db_exists = os.path.exists(db_path) and os.listdir(db_path)
    
    if db_exists:
        print(f"🔄 Updating existing vector database at {db_path}")
        if use_faiss:
            # Load existing FAISS database
            existing_db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
            print(f"📊 Current database has {existing_db.index.ntotal} vectors")
            
            # Create new database from new chunks
            new_db = FAISS.from_documents(chunks, embedding_model)
            print(f"📈 Adding {new_db.index.ntotal} new vectors")
            
            # Merge databases
            existing_db.merge_from(new_db)
            print(f"✅ Updated database now has {existing_db.index.ntotal} vectors")
            
            # Save updated database
            existing_db.save_local(db_path)
            return existing_db
        else:
            # Update Chroma database
            db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
            db.add_documents(chunks)
            db.persist()
            return db
    else:
        print(f"🆕 Creating new vector database at {db_path}")
        if use_faiss:
            db = FAISS.from_documents(chunks, embedding_model)
            db.save_local(db_path)
        else:
            db = Chroma.from_documents(chunks, embedding_model, persist_directory=db_path)
            db.persist()
        print(f"✅ Created database with {len(chunks)} vectors")
        return db

def batch_ingest(input_path, db_path="data/vector_db", use_faiss=True, chunk_size=350, chunk_overlap=50, batch_size=10):
    """
    Batch ingestion pipeline - supports single files, directories, or file lists
    Processes files in batches to handle large datasets efficiently
    """
    print(f"🚀 Starting batch ingestion...")
    print(f"📁 Input: {input_path}")
    print(f"💾 Database: {db_path}")
    print(f"🔧 Using: {'FAISS' if use_faiss else 'Chroma'}")
    print(f"📦 Processing in batches of: {batch_size} files")
    
    # Determine input type
    file_paths = []
    
    if os.path.isfile(input_path):
        # Single file
        file_paths = [input_path]
    elif os.path.isdir(input_path):
        # Directory - find all supported files
        file_paths = get_supported_files(input_path)
        if not file_paths:
            print(f"❌ No supported files found in {input_path}")
            return None
    else:
        print(f"❌ Input path does not exist: {input_path}")
        return None
    
    print(f"📋 Found {len(file_paths)} files to process")
    
    # Process files in batches for memory efficiency
    total_processed = 0
    total_chunks = 0
    failed_files = []
    
    # Split files into batches
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(file_paths) + batch_size - 1) // batch_size
        
        print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        
        # Load documents in this batch
        batch_documents, batch_failed = load_multiple_documents(batch_files)
        failed_files.extend(batch_failed)
        
        if not batch_documents:
            print(f"⚠️ No documents loaded in batch {batch_num}")
            continue
        
        # Split into chunks
        batch_chunks = split_documents(batch_documents, chunk_size, chunk_overlap)
        total_chunks += len(batch_chunks)
        
        # Create or update vector database
        db = create_or_update_vector_db(batch_chunks, db_path, use_faiss)
        
        total_processed += len(batch_files) - len(batch_failed)
        
        print(f"✅ Batch {batch_num} complete: {len(batch_chunks)} chunks added")
        
        # Clear memory
        del batch_documents, batch_chunks
    
    print(f"\n🎉 Batch ingestion complete!")
    print(f"   📚 Total files processed: {total_processed}/{len(file_paths)}")
    print(f"   ❌ Failed files: {len(failed_files)}")
    print(f"   📝 Total chunks added: {total_chunks}")
    print(f"   💾 Database saved to: {db_path}")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for file in failed_files:
            print(f"   - {os.path.basename(file)}")
    
    return db

def get_database_info(db_path="data/vector_db", use_faiss=True):
    """Get information about the current database"""
    try:
        embedding_model = get_embedding_model()
        if use_faiss:
            db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
            return f"FAISS database with {db.index.ntotal} vectors"
        else:
            db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
            return f"Chroma database in {db_path}"
    except Exception as e:
        return f"Database not found or error: {e}"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch ingest documents into vector DB.")
    parser.add_argument("--input", required=True, help="Path to file, directory, or URL")
    parser.add_argument("--db_path", default="data/vector_db", help="Path to store the vector DB")
    parser.add_argument("--faiss", action="store_true", default=True, help="Use FAISS (default)")
    parser.add_argument("--chroma", action="store_true", help="Use Chroma instead of FAISS")
    parser.add_argument("--chunk_size", type=int, default=350, help="Text chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Text chunk overlap")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of files to process per batch")
    parser.add_argument("--info", action="store_true", help="Show database information")
    
    args = parser.parse_args()
    
    use_faiss = not args.chroma  # Default to FAISS unless --chroma is specified
    
    if args.info:
        print(get_database_info(args.db_path, use_faiss))
    else:
        batch_ingest(
            input_path=args.input,
            db_path=args.db_path,
            use_faiss=use_faiss,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size
        )
