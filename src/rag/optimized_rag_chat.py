"""
Optimized Healthcare RAG Chatbot
Combines best features from multiple implementations for CHO training assistance

Features:
- Hybrid Retrieval (Dense FAISS + Sparse BM25)
- Smart Intent Classification
- Conversational Memory
- Healthcare Safety Guardrails
- File Upload Support
- Multiple Document Formats
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

# LangChain imports
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
# Lazy import for ConversationalRetrievalChain - not needed for evaluation
ConversationalRetrievalChain = None
ConversationBufferMemory = None

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)

# Text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Advanced retrieval - lazy loading for optional features
EmbeddingsFilter = None
ContextualCompressionRetriever = None

# Utilities
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict
import pickle

# Load environment
from dotenv import load_dotenv
load_dotenv()

####################################################################
#                    CONFIGURATION
####################################################################

PAGE_CONFIG = {
    "page_title": "Healthcare RAG Chatbot - CHO Assistant",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

MEDICAL_KEYWORDS = [
    'diabetes', 'hypertension', 'malaria', 'tuberculosis', 'covid', 'fever',
    'patient', 'treatment', 'medicine', 'symptom', 'diagnosis', 'therapy',
    'CHO', 'doctor', 'nurse', 'healthcare', 'medical', 'clinical',
    'vaccination', 'immunization', 'pregnancy', 'birth', 'emergency'
]

NON_MEDICAL_TEMPLATES = [
    "hello how are you", "hi there", "good morning", "thank you",
    "thanks", "bye", "goodbye", "what can you help", "how does this work"
]

####################################################################
#                    GLOBAL STATE MANAGEMENT
####################################################################

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'chat_history': [],
        'conversation_memory': None,
        'db_loaded': False,
        'db': None,
        'conversational_chain': None,
        'hybrid_alpha': 0.7,
        'show_hybrid_debug': False,
        'intent_classifier': None,
        'bm25_index': None,
        'document_corpus': None,
        'uploaded_files_processed': False,
        'vector_store_path': 'data/vector_db',
        'use_compression': False,
        'retriever_k': 5
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

####################################################################
#                    INTENT CLASSIFICATION
####################################################################

def get_intent_classifier():
    """Initialize and cache intent classifier"""
    if st.session_state.intent_classifier is None:
        try:
            st.session_state.intent_classifier = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading intent classifier: {e}")
    return st.session_state.intent_classifier

def classify_query_intent(query: str, threshold: float = 0.65) -> Tuple[bool, float, str]:
    """
    Classify query as medical or non-medical using embeddings
    Returns: (is_non_medical, confidence, intent_type)
    """
    try:
        classifier = get_intent_classifier()
        if classifier is None:
            return False, 0.0, "unknown"
        
        query_embedding = classifier.encode([query.lower().strip()])
        template_embeddings = classifier.encode(NON_MEDICAL_TEMPLATES)
        
        similarities = np.dot(query_embedding, template_embeddings.T)[0]
        max_similarity = np.max(similarities)
        best_idx = np.argmax(similarities)
        
        best_template = NON_MEDICAL_TEMPLATES[best_idx]
        if any(w in best_template for w in ['hello', 'hi', 'morning']):
            intent_type = "greeting"
        elif any(w in best_template for w in ['thank']):
            intent_type = "gratitude"
        elif any(w in best_template for w in ['bye', 'goodbye']):
            intent_type = "farewell"
        else:
            intent_type = "meta_query"
        
        return max_similarity > threshold, max_similarity, intent_type
    
    except Exception as e:
        return False, 0.0, "error"

def has_medical_keywords(query: str) -> bool:
    """Check if query contains medical keywords"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in MEDICAL_KEYWORDS)

def generate_non_medical_response(intent_type: str) -> str:
    """Generate appropriate response for non-medical queries"""
    responses = {
        "greeting": """Hello! I'm your CHO Healthcare Assistant specializing in Community Health Officer training materials.

**I can help you with:**
• CHO roles and responsibilities
• Disease management protocols (diabetes, hypertension, malaria, etc.)
• Emergency response procedures
• Maternal and child health guidelines
• Mental health care protocols
• Preventive care and vaccination schedules

Please ask specific questions about CHO training materials!""",

        "gratitude": """You're welcome! I'm here to provide evidence-based healthcare information from CHO training materials.

Feel free to ask about specific diseases, procedures, or guidelines.""",

        "farewell": """Goodbye! Remember to always follow official CHO protocols and guidelines in your healthcare practice.

Stay safe and keep serving your community's health needs!""",

        "meta_query": """I'm a specialized AI assistant trained on Community Health Officer (CHO) training materials and healthcare guidelines.

**What I provide:**
• Evidence-based information from verified CHO documents
• Disease management protocols and procedures
• Emergency response guidelines
• Patient care standards

**Important:** All responses are based strictly on official training materials to ensure healthcare safety.

**Try asking:**
• "What are the diabetes management protocols for CHOs?"
• "How to handle emergency childbirth situations?"
• "Vaccination schedules for children"
• "Mental health assessment procedures" """
    }
    
    return responses.get(intent_type, responses["meta_query"])

####################################################################
#                    HYBRID RETRIEVAL SYSTEM
####################################################################

def preprocess_for_bm25(text: str) -> List[str]:
    """Preprocess text for BM25 indexing"""
    text = text.lower()
    text = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    
    medical_abbrevs = {'bp', 'hr', 'iv', 'im', 'cho', 'ncd', 'dm', 'tb'}
    tokens = [t for t in tokens if len(t) >= 2 or t in medical_abbrevs]
    return tokens

def build_bm25_index(vector_db):
    """Build BM25 index from vector database"""
    try:
        all_docs = vector_db.similarity_search("", k=10000)
        
        if not all_docs:
            return False
        
        corpus = [preprocess_for_bm25(doc.page_content) for doc in all_docs]
        
        st.session_state.bm25_index = BM25Okapi(corpus)
        st.session_state.document_corpus = all_docs
        
        return True
    except Exception as e:
        st.error(f"BM25 indexing error: {e}")
        return False

def bm25_search(query: str, k: int = 10) -> List[Dict]:
    """Perform BM25 sparse retrieval"""
    if st.session_state.bm25_index is None:
        return []
    
    try:
        query_tokens = preprocess_for_bm25(query)
        scores = st.session_state.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [
            {
                'document': st.session_state.document_corpus[idx],
                'score': scores[idx],
                'index': idx
            }
            for idx in top_indices if scores[idx] > 0
        ]
    except Exception as e:
        return []

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range"""
    if not scores:
        return scores
    
    scores_array = np.array(scores)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return ((scores_array - min_score) / (max_score - min_score)).tolist()

def hybrid_search(vector_db, query: str, k: int = 5, alpha: float = 0.7):
    """
    Hybrid search combining FAISS (dense) and BM25 (sparse)
    alpha: weight for dense retrieval (0.7 = 70% dense, 30% sparse)
    """
    try:
        # Dense retrieval
        dense_results = vector_db.similarity_search_with_score(query, k=k*2)
        
        # Sparse retrieval
        sparse_results = bm25_search(query, k=k*2)
        
        # Normalize scores
        dense_scores = [1 / (1 + score) for _, score in dense_results]
        dense_scores_norm = normalize_scores(dense_scores)
        
        sparse_scores = [r['score'] for r in sparse_results]
        sparse_scores_norm = normalize_scores(sparse_scores)
        
        # Combine results
        candidates = {}
        
        for i, (doc, original_score) in enumerate(dense_results):
            doc_id = id(doc)
            candidates[doc_id] = {
                'document': doc,
                'dense_score': dense_scores_norm[i] if i < len(dense_scores_norm) else 0,
                'sparse_score': 0,
                'original_score': original_score
            }
        
        for i, result in enumerate(sparse_results):
            doc = result['document']
            doc_id = id(doc)
            
            if doc_id in candidates:
                candidates[doc_id]['sparse_score'] = sparse_scores_norm[i] if i < len(sparse_scores_norm) else 0
            else:
                candidates[doc_id] = {
                    'document': doc,
                    'dense_score': 0,
                    'sparse_score': sparse_scores_norm[i] if i < len(sparse_scores_norm) else 0,
                    'original_score': 1.0
                }
        
        # Calculate hybrid scores
        for doc_id in candidates:
            dense = candidates[doc_id]['dense_score']
            sparse = candidates[doc_id]['sparse_score']
            candidates[doc_id]['hybrid_score'] = alpha * dense + (1 - alpha) * sparse
        
        # Sort and return top-k
        sorted_candidates = sorted(candidates.values(), 
                                   key=lambda x: x['hybrid_score'], 
                                   reverse=True)
        
        return [(c['document'], c['original_score']) for c in sorted_candidates[:k]]
    
    except Exception as e:
        st.error(f"Hybrid search error: {e}")
        return vector_db.similarity_search_with_score(query, k=k)

####################################################################
#                    DOCUMENT PROCESSING
####################################################################

def load_document(file_path: str):
    """Load document based on file extension"""
    ext = Path(file_path).suffix.lower()
    
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': lambda p: TextLoader(p, encoding='utf-8'),
        '.csv': CSVLoader,
        '.docx': Docx2txtLoader
    }
    
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    
    loader = loaders[ext](file_path)
    return loader.load()

def process_uploaded_files(uploaded_files) -> List:
    """Process uploaded files and return documents"""
    all_docs = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                temp_path = Path(temp_dir) / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load document
                docs = load_document(str(temp_path))
                
                # Enhance metadata
                for i, doc in enumerate(docs):
                    doc.metadata.update({
                        'source_file': uploaded_file.name,
                        'page_number': doc.metadata.get('page', i + 1),
                        'document_id': f"{uploaded_file.name}_page_{i + 1}"
                    })
                
                all_docs.extend(docs)
                st.success(f"✅ Loaded {len(docs)} pages from {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Failed to load {uploaded_file.name}: {str(e)}")
    
    return all_docs

def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 100):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"{chunk.metadata.get('source_file', 'unknown')}_chunk_{i}"
        chunk.metadata['chunk_index'] = i
    
    return chunks

####################################################################
#                    CONVERSATIONAL CHAIN
####################################################################

def get_llm():
    """Initialize LLM"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key not found")
        return None
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )

def create_conversational_chain(vector_db):
    """Create conversational retrieval chain with memory"""
    try:
        llm = get_llm()
        if llm is None:
            return None
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create base retriever
        alpha = st.session_state.get('hybrid_alpha', 0.7)
        k = st.session_state.get('retriever_k', 5)
        
        # Custom retriever function for hybrid search
        class HybridRetriever(BaseRetriever):
            vector_db: object
            alpha: float
            k: int
            
            class Config:
                arbitrary_types_allowed = True
            
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
            ) -> list[Document]:
                results = hybrid_search(self.vector_db, query, k=self.k, alpha=self.alpha)
                return [doc for doc, _ in results]
        
        retriever = HybridRetriever(vector_db=vector_db, alpha=alpha, k=k)
        
        # Optional: Add contextual compression
        if st.session_state.get('use_compression', False):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            embeddings_filter = EmbeddingsFilter(
                embeddings=embeddings,
                similarity_threshold=0.75
            )
            retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=retriever
            )
        
        # Healthcare-specific prompt
        prompt_template = """You are a specialized healthcare assistant for Community Health Officers (CHOs). 
You provide information STRICTLY based on official CHO training materials.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the context doesn't contain relevant information, state: "I couldn't find this information in the CHO training materials."
3. Always maintain healthcare safety and accuracy
4. Cite sources when possible

Context from CHO training materials:
{context}

Chat History:
{chat_history}

Current Question: {question}

Provide a detailed, accurate answer based ONLY on the context above:"""

        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
        )
        
        # Create conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_template(prompt_template)}
        )
        
        return chain
    
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None

####################################################################
#                    VECTOR DATABASE
####################################################################

def get_embeddings():
    """Initialize embedding model"""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def create_vector_db(documents, save_path: str = "data/vector_db"):
    """Create FAISS vector database from documents"""
    try:
        with st.spinner("Creating embeddings and building vector database..."):
            embeddings = get_embeddings()
            
            # Split documents
            chunks = split_documents(documents)
            st.info(f"Created {len(chunks)} text chunks")
            
            # Create FAISS index
            vector_db = FAISS.from_documents(chunks, embeddings)
            
            # Save locally
            vector_db.save_local(save_path)
            st.success(f"Vector database saved to {save_path}")
            
            # Build BM25 index
            if build_bm25_index(vector_db):
                st.success("Hybrid retrieval system ready!")
            
            return vector_db
    
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None

def load_vector_db(db_path: str = "data/vector_db"):
    """Load existing vector database"""
    try:
        embeddings = get_embeddings()
        
        if not os.path.exists(os.path.join(db_path, "index.faiss")):
            st.error(f"Vector database not found at {db_path}")
            return None
        
        vector_db = FAISS.load_local(
            db_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        st.success(f"Loaded FAISS database with {vector_db.index.ntotal} vectors")
        
        # Build BM25 index
        with st.spinner("Building BM25 index for hybrid retrieval..."):
            if build_bm25_index(vector_db):
                st.success("Hybrid retrieval system ready!")
        
        return vector_db
    
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return None

####################################################################
#                    STREAMLIT UI
####################################################################

def display_chat_history():
    """Display chat history in a clean format"""
    if not st.session_state.chat_history:
        st.info("👋 Welcome! Ask your first question about CHO training materials.")
        return
    
    st.subheader("Chat History")
    
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.markdown(f"**{chat['timestamp']}**")
            st.markdown(chat['question'])
        
        # Assistant message
        with st.chat_message("assistant"):
            st.markdown(chat['answer'])
            
            # Show sources if available
            if chat.get('sources'):
                with st.expander(f"📚 Sources ({len(chat['sources'])} found)"):
                    for j, source in enumerate(chat['sources'], 1):
                        st.text(f"📄 {source['document']} (Page {source['page']}) - Score: {source['score']:.3f}")
                        st.caption(source['preview'])
                        if j < len(chat['sources']):
                            st.divider()

def sidebar_ui():
    """Create sidebar with controls"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Instructions
        with st.expander("📖 How to Use", expanded=True):
            st.markdown("""
            **Quick Start:**
            1. Upload CHO training documents OR use existing database
            2. Adjust retrieval settings (optional)
            3. Ask questions about healthcare topics
            
            **Features:**
            • 🔄 Hybrid Retrieval (Semantic + Keyword)
            • 💬 Conversational Memory
            • 🛡️ Healthcare Safety Guardrails
            • 📊 Source Attribution
            """)
        
        st.divider()
        
        # Database Management
        st.subheader("📚 Database Management")
        
        tab1, tab2 = st.tabs(["Upload New", "Load Existing"])
        
        with tab1:
            uploaded_files = st.file_uploader(
                "Upload CHO Training Materials",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'csv', 'docx'],
                help="Upload healthcare training documents"
            )
            
            if uploaded_files and st.button("Process & Create Database"):
                docs = process_uploaded_files(uploaded_files)
                if docs:
                    db = create_vector_db(docs, st.session_state.vector_store_path)
                    if db:
                        st.session_state.db = db
                        st.session_state.db_loaded = True
                        st.session_state.uploaded_files_processed = True
                        st.rerun()
        
        with tab2:
            db_path = st.text_input("Database Path", value="data/vector_db")
            if st.button("Load Database"):
                db = load_vector_db(db_path)
                if db:
                    st.session_state.db = db
                    st.session_state.db_loaded = True
                    st.rerun()
        
        # Show database status
        if st.session_state.db_loaded:
            st.success("✅ Database Ready")
            
            # Hybrid Retrieval Settings
            st.divider()
            st.subheader("🔧 Retrieval Settings")
            
            with st.expander("Hybrid Retrieval", expanded=False):
                alpha = st.slider(
                    "Dense vs Sparse Weight",
                    0.0, 1.0, 0.7, 0.1,
                    help="0.7 = 70% semantic + 30% keyword"
                )
                st.session_state.hybrid_alpha = alpha
                
                if alpha >= 0.8:
                    st.info("🎯 Primarily Semantic Search")
                elif alpha <= 0.3:
                    st.info("🔍 Primarily Keyword Search")
                else:
                    st.info("⚖️ Balanced Hybrid Search")
                
                st.session_state.retriever_k = st.slider(
                    "Number of Results",
                    3, 10, 5,
                    help="Documents to retrieve"
                )
                
                st.session_state.use_compression = st.checkbox(
                    "Use Contextual Compression",
                    help="Filter out irrelevant content"
                )
        else:
            st.warning("⚠️ No Database Loaded")
        
        st.divider()
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.conversation_memory = None
            st.session_state.conversational_chain = None
            st.rerun()

def main():
    """Main application"""
    st.set_page_config(**PAGE_CONFIG)
    
    st.title("🏥 Healthcare RAG Chatbot")
    st.caption("Specialized AI Assistant for Community Health Officer Training")
    
    # Initialize
    init_session_state()
    
    # Sidebar
    sidebar_ui()
    
    # Main chat interface
    if st.session_state.db_loaded and st.session_state.db:
        
        # Create conversational chain if not exists
        if st.session_state.conversational_chain is None:
            with st.spinner("Initializing conversational system..."):
                chain = create_conversational_chain(st.session_state.db)
                if chain:
                    st.session_state.conversational_chain = chain
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        user_query = st.chat_input(
            "Ask about CHO training materials...",
            key="chat_input"
        )
        
        if user_query:
            # Process query
            with st.spinner("Processing your query..."):
                # Intent classification
                is_non_medical, confidence, intent_type = classify_query_intent(user_query)
                
                if is_non_medical and confidence > 0.65:
                    # Handle non-medical query
                    answer = generate_non_medical_response(intent_type)
                    sources = []
                else:
                    # Handle medical query with conversational chain
                    if st.session_state.conversational_chain:
                        try:
                            result = st.session_state.conversational_chain({
                                "question": user_query
                            })
                            
                            answer = result.get('answer', 'Error generating response')
                            
                            # Extract sources
                            sources = []
                            if 'source_documents' in result:
                                for doc in result['source_documents'][:3]:
                                    sources.append({
                                        'document': doc.metadata.get('source_file', 'Unknown'),
                                        'page': doc.metadata.get('page_number', 'Unknown'),
                                        'score': 0.0,  # ConversationalRetrievalChain doesn't return scores
                                        'preview': doc.page_content[:200] + "..."
                                    })
                        except Exception as e:
                            answer = f"Error: {str(e)}"
                            sources = []
                    else:
                        answer = "Conversational system not initialized. Please try again."
                        sources = []
                
                # Add to history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'question': user_query,
                    'answer': answer,
                    'sources': sources
                })
                
                st.rerun()
    
    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h2>Welcome to Healthcare RAG Chatbot</h2>
                <p style='font-size: 18px;'>Your AI assistant for CHO training materials</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🚀 Getting Started")
            st.markdown("""
            1. **Upload Documents** or **Load Existing Database** from sidebar
            2. **Adjust Settings** for optimal retrieval (optional)
            3. **Start Chatting** with your healthcare knowledge base
            """)
            
            st.markdown("### ✨ Key Features")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("""
                - 🔄 **Hybrid Retrieval**
                - 💬 **Conversational Memory**
                - 🛡️ **Healthcare Safety**
                """)
            with col_b:
                st.markdown("""
                - 📚 **Multi-Format Support**
                - 🎯 **Smart Intent Detection**
                - 📊 **Source Attribution**
                """)

if __name__ == "__main__":
    main()
