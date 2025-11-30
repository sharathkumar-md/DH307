"""
RAG Chatbot with Streamlit Interface
Retrieval-Augmented Generation chatbot using vector database and LLM.
"""

import os
from typing import List, Dict
import logging

import streamlit as st
import torch
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# ========================================
# CONFIGURATION SETTINGS
# ========================================
# Modify these values to configure the RAG chatbot
CONFIG = {
    # Vector Database Settings
    "db_path": "data/vector_db",
    "use_faiss": True,

    # LLM Settings (Primary)
    "llm_provider": "llama",  # Primary: "llama", "openai", "google"
    "model_name": "qwen2.5:0.5b",  # Smaller model to avoid GPU memory issues

    # LLM Fallback Settings (if primary fails)
    "fallback_llm_provider": "google",
    "fallback_model_name": "gemini-1.5-flash",

    # Embedding Settings
    # IMPORTANT: Must match the model used to create the vector database
    "embedding_provider": "huggingface",
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",  # Must match vector DB

    # Generation Parameters
    "temperature": 0.3,  # Lower = more focused and factual (better for medical Q&A)
    "top_k": 5,  # Number of documents to retrieve (increased for better context)
}
# ========================================


def get_device():
    """Detect and return the best available device (GPU/CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


class RAGChatbot:
    """RAG-based chatbot with conversation memory."""

    def __init__(
        self,
        db_path: str,
        use_faiss: bool = True,
        llm_provider: str = "llama",
        model_name: str = "llama3.2:1b",
        embedding_provider: str = "llama",
        embedding_model: str = None,
        temperature: float = 0.7,
        top_k: int = 4,
        fallback_llm_provider: str = None,
        fallback_model_name: str = None
    ):
        """
        Initialize the RAG chatbot.

        Args:
            db_path: Path to vector database
            use_faiss: If True, load FAISS; otherwise load Chroma
            llm_provider: "llama", "openai", or "google"
            model_name: Name of the LLM model
            embedding_provider: "llama", "huggingface", "openai", or "google"
            embedding_model: Specific embedding model name
            temperature: LLM temperature
            top_k: Number of documents to retrieve
            fallback_llm_provider: Fallback LLM provider if primary fails
            fallback_model_name: Fallback model name
        """
        self.db_path = db_path
        self.top_k = top_k
        self.temperature = temperature
        self.fallback_llm_provider = fallback_llm_provider
        self.fallback_model_name = fallback_model_name
        self.active_llm_provider = None
        self.active_model_name = None

        # Initialize embeddings
        if embedding_provider == "llama":
            model = embedding_model or "llama3.2:1b"
            self.embeddings = OllamaEmbeddings(
                model=model,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
        elif embedding_provider == "huggingface":
            model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
            device = get_device()
            model_kwargs = {"device": device}
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs=model_kwargs
            )
        elif embedding_provider == "google":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        else:  # openai
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

        # Load vector database
        self.vectorstore = self._load_vectorstore(use_faiss)

        # Initialize LLM with fallback
        self.llm = self._initialize_llm(llm_provider, model_name)

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Create RAG chain
        self.chain = self._create_chain()

    def _initialize_llm(self, provider: str, model: str):
        """
        Initialize LLM with fallback support.

        Args:
            provider: Primary LLM provider
            model: Primary model name

        Returns:
            Initialized LLM instance
        """
        try:
            logger.info("Attempting to initialize {provider} with model {model}...")
            llm = self._create_llm(provider, model)

            # Test the LLM with a simple query to verify it works
            try:
                test_response = llm.invoke("test")
                self.active_llm_provider = provider
                self.active_model_name = model
                logger.info("✓ Successfully initialized {provider} ({model})")
                return llm
            except Exception as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "cudaMalloc" in error_msg:
                    raise MemoryError(f"GPU memory error with {provider}")
                raise

        except (MemoryError, Exception) as e:
            logger.info("✗ Failed to initialize {provider}: {e}")

            if self.fallback_llm_provider:
                logger.info("Attempting fallback to {self.fallback_llm_provider} ({self.fallback_model_name})...")
                try:
                    llm = self._create_llm(self.fallback_llm_provider, self.fallback_model_name)
                    self.active_llm_provider = self.fallback_llm_provider
                    self.active_model_name = self.fallback_model_name
                    logger.info("✓ Successfully initialized fallback {self.fallback_llm_provider} ({self.fallback_model_name})")
                    return llm
                except Exception as fallback_error:
                    logger.info("✗ Fallback also failed: {fallback_error}")
                    raise RuntimeError(f"Both primary and fallback LLM initialization failed")
            else:
                raise

    def _create_llm(self, provider: str, model: str):
        """Create an LLM instance based on provider and model."""
        if provider == "llama":
            return ChatOllama(
                model=model or "qwen2.5:0.5b",
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=self.temperature,
                num_predict=2048,  # Enough for detailed medical responses
                top_p=0.9,  # Nucleus sampling for better quality
                top_k=40,  # Top-k sampling
                repeat_penalty=1.1  # Reduce repetition
            )
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=model or "gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=self.temperature,
                convert_system_message_to_human=True
            )
        else:  # openai
            return ChatOpenAI(
                model_name=model or "gpt-3.5-turbo",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=self.temperature
            )

    def _load_vectorstore(self, use_faiss: bool):
        """Load the vector database."""
        logger.info("Loading vector database from: {self.db_path}")

        if use_faiss:
            vectorstore = FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )

        logger.info("Vector database loaded successfully")
        return vectorstore

    def _create_chain(self):
        """Create the conversational retrieval chain."""
        # Custom prompt template optimized for Qwen and other small models
        prompt_template = """You are a knowledgeable medical AI assistant specializing in Community Health Officer (CHO) training and healthcare guidelines.

TASK: Answer the question using ONLY the provided context. Be accurate, clear, and concise.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- If the context contains the answer, provide a clear, direct response with specific details
- If the context doesn't contain relevant information, respond: "I don't have information about this in the provided documents."
- For greetings (hi/hello), respond briefly then ask how you can help with health-related questions
- Use medical terminology from the context when appropriate
- Be precise with numbers, dosages, and procedures

ANSWER:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        return chain

    def chat(self, question: str) -> Dict:
        """
        Process a chat query.

        Args:
            question: User question

        Returns:
            Dictionary with answer and source documents
        """
        response = self.chain({"question": question})

        return {
            "answer": response["answer"],
            "sources": response.get("source_documents", [])
        }

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()


def main():
    """Streamlit UI for the chatbot."""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG Chatbot")
    st.markdown("Chat with me! I can answer questions about your documents or just have a friendly conversation.")

    # Sidebar with system info and controls
    with st.sidebar:
        st.header("⚙️ System Info")

        # Display GPU info
        device = get_device()
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"🚀 GPU: {gpu_name}")
        elif device == "mps":
            st.success("🚀 GPU: Apple Silicon (MPS)")
        else:
            st.info("💻 Using CPU")

        st.markdown("---")

        # Display current configuration
        st.header("📋 Active Configuration")

        # Show active LLM if chatbot is initialized
        if "chatbot" in st.session_state:
            active_provider = st.session_state.chatbot.active_llm_provider
            active_model = st.session_state.chatbot.active_model_name

            if active_provider == CONFIG["llm_provider"]:
                st.success(f"**LLM:** {active_provider} ({active_model})")
            else:
                st.warning(f"**LLM (Fallback):** {active_provider} ({active_model})")
        else:
            st.info(f"**LLM (Configured):** {CONFIG['llm_provider']} ({CONFIG['model_name']})")

        st.info(f"**Embeddings:** {CONFIG['embedding_provider']}")
        st.info(f"**Temperature:** {CONFIG['temperature']}")
        st.info(f"**Top K:** {CONFIG['top_k']}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                if "chatbot" in st.session_state:
                    st.session_state.chatbot.clear_memory()
                st.success("Chat history cleared!")

        with col2:
            if st.button("Reinitialize Chatbot"):
                if "chatbot" in st.session_state:
                    del st.session_state.chatbot
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()

    # Initialize chatbot
    if "chatbot" not in st.session_state:
        try:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = RAGChatbot(
                    db_path=CONFIG["db_path"],
                    use_faiss=CONFIG["use_faiss"],
                    llm_provider=CONFIG["llm_provider"],
                    model_name=CONFIG["model_name"],
                    embedding_provider=CONFIG["embedding_provider"],
                    embedding_model=CONFIG["embedding_model"],
                    temperature=CONFIG["temperature"],
                    top_k=CONFIG["top_k"],
                    fallback_llm_provider=CONFIG.get("fallback_llm_provider"),
                    fallback_model_name=CONFIG.get("fallback_model_name")
                )

            # Show which LLM is being used
            active_provider = st.session_state.chatbot.active_llm_provider
            active_model = st.session_state.chatbot.active_model_name

            if active_provider == CONFIG["llm_provider"]:
                st.success(f"✅ Chatbot initialized with {active_provider} ({active_model})")
            else:
                st.warning(f"⚠️ Primary LLM failed. Using fallback: {active_provider} ({active_model})")

        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {e}")
            st.stop()

    # Initialize chat messages with welcome message
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! 👋 I'm your AI assistant. I can help you with questions about the documents in my knowledge base, or we can just chat. How can I help you today?"
            }
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:500] + "...")
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)

                    st.markdown(response["answer"])

                    # Display sources
                    if response["sources"]:
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(doc.page_content[:500] + "...")
                                st.markdown("---")

                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
