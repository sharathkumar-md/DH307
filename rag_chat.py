import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables (API keys, etc.)
load_dotenv()

def check_query_relevance(db, query, threshold=0.6):
    """
    Check if the query has relevant content in the database before processing.
    """
    try:
        # Perform similarity search with scores
        docs_and_scores = db.similarity_search_with_score(query, k=3)
        
        if not docs_and_scores:
            return False, "No documents found"
        
        # Check if the best match meets our threshold
        best_score = docs_and_scores[0][1] if docs_and_scores else 1.0
        
        # For FAISS, lower scores are better (distance), so we check if it's below threshold
        is_relevant = best_score < threshold
        
        return is_relevant, f"Best relevance score: {best_score:.3f}"
    except Exception as e:
        return True, f"Error checking relevance: {e}"  # Default to allowing the query

# Initialize models with Streamlit-compatible approach
@st.cache_resource
def get_embedding_model():
    """Get embedding model using HuggingFace (no async issues)"""
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error initializing embedding model: {e}")
        return None

@st.cache_resource
def get_llm():
    """Get LLM - using Google Gemini"""
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

# Load vector database
def load_vector_db(db_path="vector_db", use_faiss=True):
    """
    Load the vector database (FAISS or Chroma) from disk.
    """
    try:
        embedding_model = get_embedding_model()
        if embedding_model is None:
            st.error("Failed to initialize embedding model")
            return None
            
        if use_faiss:
            db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        else:
            db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
        return db
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return None

def get_qa_chain(db):
    """
    Create a RetrievalQA chain using the vector DB and LLM with strict document adherence.
    """
    try:
        llm = get_llm()
        if llm is None or db is None:
            st.error("Failed to initialize LLM or database is None")
            return None
            
        # Configure retriever to return more relevant documents
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.7,  # Higher threshold for better relevance
                "k": 5  # Retrieve top 5 most relevant chunks
            }
        )
        
        # Create a custom prompt that enforces document adherence
        from langchain.prompts import PromptTemplate
        
        strict_prompt = PromptTemplate(
            template="""You are a helpful assistant that answers questions STRICTLY based on the provided document content.

IMPORTANT RULES:
1. ONLY use information from the provided context below
2. If the context doesn't contain relevant information to answer the question, respond with: "I couldn't find relevant information about this topic in the document."
3. DO NOT use your general knowledge or make assumptions
4. DO NOT provide information not present in the context
5. If you're unsure, say "The document doesn't provide clear information about this."

Context from document:
{context}

Question: {question}

Answer based ONLY on the context above:""",
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": strict_prompt},
            return_source_documents=True  # This helps us see what was retrieved
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

# Streamlit interface
def main():
    st.title("RAG Chatbot")
    st.markdown("### Chat with Your PDF Documents!")
    
    # Instructions
    with st.expander("How to Use", expanded=True):
        st.markdown("""
        **Step 1**: Load your vector database (your ingested PDF)
        **Step 2**: Ask questions about your document content
        **Step 3**: Get AI-powered answers based on your PDFs
        
        **Tip**: Ask specific questions like "What is this document about?" or "Explain the main concepts"
        """)
    
    st.markdown("---")
    st.subheader("Setup")
    
    db_path = st.text_input("Vector Database Path:", value="vector_db", 
                           help="Path to your vector database folder")
    use_faiss = st.checkbox("Use FAISS", value=True, help="Keep this checked for better performance")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        load_button = st.button("Load Vector DB", type="primary")
    
    if load_button:
        with st.spinner("Loading vector database..."):
            db = load_vector_db(db_path, use_faiss)
            if db is not None:
                st.session_state["db"] = db
                st.success("Vector DB loaded successfully! You can now ask questions.")
            else:
                st.error("Failed to load vector database. Check the path and try again.")
    
    st.markdown("---")
    
    # Chat interface
    if "db" in st.session_state and st.session_state["db"] is not None:
        st.subheader("Chat with Your Documents")
        
        # Show database info
        st.info("Database loaded and ready! Ask your questions below.")
        
        user_query = st.text_input("Your question:", 
                                  placeholder="e.g., What is this document about? Explain the main concepts...",
                                  help="Type your question about the document content")
        
        if user_query:
            # First, check if the query is relevant to the document
            is_relevant, relevance_info = check_query_relevance(st.session_state["db"], user_query)
            
            if not is_relevant:
                st.markdown("### Answer:")
                st.warning("I couldn't find relevant information about this topic in the document.")
                st.info("Try asking about topics that are more likely to be covered in your uploaded document.")
                with st.expander("Relevance Details", expanded=False):
                    st.text(relevance_info)
                st.markdown("---")
            else:
                with st.spinner("AI is searching the document..."):
                    qa_chain = get_qa_chain(st.session_state["db"])
                    if qa_chain is not None:
                        try:
                            # Get answer with source documents
                            result = qa_chain({"query": user_query})
                            answer = result["result"]
                            source_docs = result.get("source_documents", [])
                            
                            # Check if the answer indicates no relevant information
                            no_info_phrases = [
                                "couldn't find relevant information",
                                "document doesn't provide",
                                "not present in the context",
                                "no information about",
                                "doesn't contain information"
                            ]
                            
                            if any(phrase in answer.lower() for phrase in no_info_phrases):
                                st.markdown("### Answer:")
                                st.warning("I couldn't find relevant information about this topic in the document.")
                                st.info("The document may not cover this specific topic. Try asking about different aspects or main themes of the document.")
                            else:
                                st.markdown("### Answer:")
                                st.markdown(f"**{answer}**")
                            
                            # Show relevance information
                            if source_docs:
                                with st.expander("Source Information", expanded=False):
                                    st.markdown(f"**Found {len(source_docs)} relevant sections in the document**")
                                    for i, doc in enumerate(source_docs[:3]):  # Show first 3 sources
                                        st.markdown(f"**Source {i+1}:**")
                                        st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                            
                            # Add a separator for multiple questions
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"Error generating answer: {e}")
                            st.info("Try rephrasing your question or ask about topics more likely to be in the document.")
                    else:
                        st.error("Failed to create QA chain")
    else:
        st.warning("Please load the vector database first to start chatting!")

if __name__ == "__main__":
    main()
