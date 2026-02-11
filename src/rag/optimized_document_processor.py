"""
Optimized Document Processing Module
Handles batch processing, chunking, and metadata enrichment for healthcare RAG

Features:
- Multi-format support (PDF, TXT, CSV, DOCX)
- Intelligent chunking strategies
- Metadata enrichment
- Progress tracking
- Error handling and validation
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Utilities
from tqdm import tqdm
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

####################################################################
#                    CONFIGURATION
####################################################################

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.csv', '.docx']

MEDICAL_SEPARATORS = [
    "\n\n### ",  # Headers
    "\n\n## ",
    "\n\n# ",
    "\n\n",      # Paragraphs
    "\n",        # Lines
    ". ",        # Sentences
    " ",         # Words
    ""
]

####################################################################
#                    DOCUMENT LOADERS
####################################################################

class DocumentLoader:
    """Unified document loader for multiple formats"""
    
    def __init__(self):
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_pages': 0,
            'total_chunks': 0
        }
    
    def load_single_file(self, file_path: str) -> Tuple[List, Optional[str]]:
        """
        Load a single document file
        Returns: (documents, error_message)
        """
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext not in SUPPORTED_EXTENSIONS:
                return [], f"Unsupported file type: {ext}"
            
            # Select appropriate loader
            loaders = {
                '.pdf': PyPDFLoader,
                '.txt': lambda p: TextLoader(p, encoding='utf-8'),
                '.csv': CSVLoader,
                '.docx': Docx2txtLoader
            }
            
            loader = loaders[ext](file_path)
            documents = loader.load()
            
            # Enhance metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source_file': Path(file_path).name,
                    'file_path': file_path,
                    'page_number': doc.metadata.get('page', i + 1),
                    'file_type': ext[1:],  # Remove dot
                    'loaded_at': datetime.now().isoformat(),
                    'document_id': f"{Path(file_path).stem}_page_{i + 1}"
                })
            
            self.stats['files_processed'] += 1
            self.stats['total_pages'] += len(documents)
            
            logger.info(f"✅ Loaded {len(documents)} pages from {Path(file_path).name}")
            return documents, None
            
        except Exception as e:
            self.stats['files_failed'] += 1
            error_msg = f"Failed to load {Path(file_path).name}: {str(e)}"
            logger.error(error_msg)
            return [], error_msg
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List:
        """
        Load all supported documents from a directory
        """
        all_documents = []
        errors = []
        
        # Find all supported files
        directory = Path(directory_path)
        
        if recursive:
            files = [f for f in directory.rglob('*') if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        else:
            files = [f for f in directory.glob('*') if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        
        if not files:
            logger.warning(f"No supported files found in {directory_path}")
            return all_documents
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process each file with progress bar
        for file_path in tqdm(files, desc="Loading documents"):
            docs, error = self.load_single_file(str(file_path))
            
            if docs:
                all_documents.extend(docs)
            if error:
                errors.append(error)
        
        logger.info(f"""
        ✅ Document Loading Complete:
        - Files Processed: {self.stats['files_processed']}
        - Files Failed: {self.stats['files_failed']}
        - Total Pages: {self.stats['total_pages']}
        """)
        
        if errors:
            logger.warning(f"Errors encountered:\n" + "\n".join(errors))
        
        return all_documents
    
    def get_stats(self) -> Dict:
        """Return loading statistics"""
        return self.stats.copy()

####################################################################
#                    TEXT CHUNKING
####################################################################

class MedicalTextSplitter:
    """Specialized text splitter for medical documents"""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=MEDICAL_SEPARATORS,
            length_function=len
        )
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into optimized chunks
        """
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        
        chunks = self.splitter.split_documents(documents)
        
        # Enhance chunk metadata
        chunk_counter = {}
        for chunk in chunks:
            source_file = chunk.metadata.get('source_file', 'unknown')
            
            if source_file not in chunk_counter:
                chunk_counter[source_file] = 0
            chunk_counter[source_file] += 1
            
            chunk.metadata.update({
                'chunk_id': f"{source_file}_chunk_{chunk_counter[source_file]}",
                'chunk_index': chunk_counter[source_file],
                'chunk_size': len(chunk.page_content),
                'chunking_strategy': 'medical_recursive'
            })
        
        logger.info(f"✅ Created {len(chunks)} text chunks")
        
        return chunks
    
    def analyze_chunks(self, chunks: List) -> Dict:
        """Analyze chunk statistics"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'unique_sources': len(set(c.metadata.get('source_file', '') for c in chunks))
        }

####################################################################
#                    VECTOR DATABASE CREATION
####################################################################

class VectorDatabaseBuilder:
    """Build and manage FAISS vector databases"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.stats = {}
    
    def get_embeddings(self):
        """Initialize embedding model"""
        if self.embeddings is None:
            logger.info(f"Loading embedding model: {self.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
        return self.embeddings
    
    def create_vector_db(self, chunks: List, save_path: str = "data/vector_db") -> Optional[FAISS]:
        """
        Create FAISS vector database from document chunks
        """
        try:
            if not chunks:
                logger.error("No chunks provided for vector database creation")
                return None
            
            logger.info(f"Creating FAISS index for {len(chunks)} chunks...")
            
            embeddings = self.get_embeddings()
            
            # Create FAISS index
            vector_db = FAISS.from_documents(chunks, embeddings)
            
            # Save database
            Path(save_path).mkdir(parents=True, exist_ok=True)
            vector_db.save_local(save_path)
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'total_vectors': vector_db.index.ntotal,
                'total_chunks': len(chunks),
                'embedding_model': self.embedding_model,
                'chunk_size': chunks[0].metadata.get('chunk_size', 'unknown'),
                'unique_sources': len(set(c.metadata.get('source_file', '') for c in chunks))
            }
            
            with open(Path(save_path) / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.stats = metadata
            
            logger.info(f"""
            ✅ Vector Database Created Successfully:
            - Location: {save_path}
            - Vectors: {metadata['total_vectors']}
            - Sources: {metadata['unique_sources']}
            """)
            
            return vector_db
            
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            return None
    
    def load_vector_db(self, db_path: str = "data/vector_db") -> Optional[FAISS]:
        """Load existing FAISS database"""
        try:
            embeddings = self.get_embeddings()
            
            if not Path(db_path).exists():
                logger.error(f"Database path does not exist: {db_path}")
                return None
            
            logger.info(f"Loading vector database from {db_path}...")
            
            vector_db = FAISS.load_local(
                db_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata if exists
            metadata_path = Path(db_path) / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.stats = json.load(f)
            
            logger.info(f"✅ Loaded database with {vector_db.index.ntotal} vectors")
            
            return vector_db
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Return database statistics"""
        return self.stats.copy()

####################################################################
#                    BATCH PROCESSING PIPELINE
####################################################################

class DocumentProcessor:
    """Complete document processing pipeline"""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.loader = DocumentLoader()
        self.splitter = MedicalTextSplitter(chunk_size, chunk_overlap)
        self.db_builder = VectorDatabaseBuilder(embedding_model)
        
        self.pipeline_stats = {}
    
    def process_directory(self, input_dir: str, output_dir: str = "data/vector_db",
                         recursive: bool = True) -> Optional[FAISS]:
        """
        Complete pipeline: Load → Split → Embed → Save
        """
        start_time = datetime.now()
        
        logger.info(f"Starting document processing pipeline...")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")
        
        # Step 1: Load documents
        logger.info("\n=== STEP 1: Loading Documents ===")
        documents = self.loader.load_directory(input_dir, recursive)
        
        if not documents:
            logger.error("No documents loaded. Aborting pipeline.")
            return None
        
        # Step 2: Split into chunks
        logger.info("\n=== STEP 2: Creating Text Chunks ===")
        chunks = self.splitter.split_documents(documents)
        
        if not chunks:
            logger.error("No chunks created. Aborting pipeline.")
            return None
        
        # Analyze chunks
        chunk_stats = self.splitter.analyze_chunks(chunks)
        logger.info(f"Chunk Statistics: {chunk_stats}")
        
        # Step 3: Create vector database
        logger.info("\n=== STEP 3: Building Vector Database ===")
        vector_db = self.db_builder.create_vector_db(chunks, output_dir)
        
        if not vector_db:
            logger.error("Failed to create vector database.")
            return None
        
        # Calculate pipeline statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.pipeline_stats = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'loading_stats': self.loader.get_stats(),
            'chunking_stats': chunk_stats,
            'database_stats': self.db_builder.get_stats()
        }
        
        # Save pipeline report
        report_path = Path(output_dir) / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.pipeline_stats, f, indent=2)
        
        logger.info(f"""
        ╔════════════════════════════════════════╗
        ║   PIPELINE COMPLETED SUCCESSFULLY      ║
        ╚════════════════════════════════════════╝
        
        Duration: {duration:.2f} seconds
        Files Processed: {self.pipeline_stats['loading_stats']['files_processed']}
        Total Chunks: {chunk_stats['total_chunks']}
        Vector Database: {output_dir}
        
        Report saved to: {report_path}
        """)
        
        return vector_db
    
    def get_pipeline_stats(self) -> Dict:
        """Return complete pipeline statistics"""
        return self.pipeline_stats.copy()

####################################################################
#                    CLI INTERFACE
####################################################################

def main():
    """Command-line interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process healthcare documents and create vector database"
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help="Directory containing documents to process"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/vector_db',
        help="Output directory for vector database (default: data/vector_db)"
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})"
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help="HuggingFace embedding model (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help="Don't process subdirectories"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input_dir).exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Create processor
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model
    )
    
    # Run pipeline
    vector_db = processor.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output,
        recursive=not args.no_recursive
    )
    
    if vector_db:
        logger.info("✅ Processing complete!")
    else:
        logger.error("❌ Processing failed!")

if __name__ == "__main__":
    main()
