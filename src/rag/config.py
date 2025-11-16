"""
Configuration file for Optimized Healthcare RAG System
Centralized settings for easy customization
"""

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

####################################################################
#                    APPLICATION SETTINGS
####################################################################

@dataclass
class AppConfig:
    """Main application configuration"""
    
    # Application Info
    APP_NAME: str = "Healthcare RAG Chatbot"
    APP_VERSION: str = "2.0.0-optimized"
    APP_DESCRIPTION: str = "AI Assistant for Community Health Officer Training"
    
    # Page Configuration
    PAGE_TITLE: str = "Healthcare RAG Chatbot - CHO Assistant"
    PAGE_ICON: str = "🏥"
    LAYOUT: str = "wide"
    SIDEBAR_STATE: str = "expanded"
    
    # Paths
    VECTOR_DB_PATH: str = "data/vector_db"
    DOCUMENTS_PATH: str = "data/documents"
    LOGS_PATH: str = "logs"

####################################################################
#                    LLM SETTINGS
####################################################################

@dataclass
class LLMConfig:
    """Large Language Model configuration"""
    
    # Model Selection
    PROVIDER: str = "google"  # google, openai, anthropic
    MODEL_NAME: str = "gemini-1.5-pro"
    
    # Alternative Models
    MODELS: Dict[str, str] = None
    
    def __post_init__(self):
        if self.MODELS is None:
            self.MODELS = {
                "google_pro": "gemini-1.5-pro",
                "google_flash": "gemini-1.5-flash",
                "openai": "gpt-4-turbo-preview",
                "anthropic": "claude-3-opus-20240229"
            }
    
    # Generation Parameters
    TEMPERATURE: float = 0.3  # Lower = more deterministic
    MAX_TOKENS: int = 2048
    TOP_P: float = 0.95
    TOP_K: int = 40
    
    # Safety Settings (for healthcare)
    STRICT_MODE: bool = True
    REQUIRE_SOURCES: bool = True

####################################################################
#                    EMBEDDING SETTINGS
####################################################################

@dataclass
class EmbeddingConfig:
    """Text embedding configuration"""
    
    # Model Selection
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # Alternative Models
    MODELS: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.MODELS is None:
            self.MODELS = {
                "mini": {
                    "name": "all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "speed": "fast",
                    "quality": "good"
                },
                "mpnet": {
                    "name": "all-mpnet-base-v2",
                    "dimensions": 768,
                    "speed": "medium",
                    "quality": "excellent"
                },
                "medical": {
                    "name": "pubmedbert-base-uncased",
                    "dimensions": 768,
                    "speed": "medium",
                    "quality": "medical-specific"
                }
            }
    
    # Device Settings
    DEVICE: str = "cpu"  # cpu, cuda, mps
    NORMALIZE_EMBEDDINGS: bool = True

####################################################################
#                    RETRIEVAL SETTINGS
####################################################################

@dataclass
class RetrievalConfig:
    """Retrieval system configuration"""
    
    # Hybrid Retrieval
    DEFAULT_ALPHA: float = 0.7  # 70% dense, 30% sparse
    MIN_ALPHA: float = 0.0
    MAX_ALPHA: float = 1.0
    ALPHA_STEP: float = 0.1
    
    # Dense Retrieval (FAISS)
    DENSE_K: int = 5
    DENSE_SEARCH_TYPE: str = "similarity"  # similarity, mmr
    
    # Sparse Retrieval (BM25)
    SPARSE_K: int = 5
    BM25_K1: float = 1.5  # Term frequency saturation
    BM25_B: float = 0.75  # Length normalization
    
    # Contextual Compression
    ENABLE_COMPRESSION: bool = False
    COMPRESSION_THRESHOLD: float = 0.75
    
    # Retrieval Strategy
    RETRIEVER_TYPE: str = "hybrid"  # dense, sparse, hybrid

####################################################################
#                    CHUNKING SETTINGS
####################################################################

@dataclass
class ChunkingConfig:
    """Document chunking configuration"""
    
    # Chunk Parameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MIN_CHUNK_SIZE: int = 50
    MAX_CHUNK_SIZE: int = 1000
    
    # Separators (in priority order)
    SEPARATORS: List[str] = None
    
    def __post_init__(self):
        if self.SEPARATORS is None:
            self.SEPARATORS = [
                "\n\n### ",  # H3 headers
                "\n\n## ",   # H2 headers
                "\n\n# ",    # H1 headers
                "\n\n",      # Paragraphs
                "\n",        # Lines
                ". ",        # Sentences
                " ",         # Words
                ""
            ]
    
    # Medical-Specific
    PRESERVE_ABBREVIATIONS: bool = True
    MEDICAL_ABBREVS: List[str] = None
    
    def __post_init__(self):
        if self.MEDICAL_ABBREVS is None:
            self.MEDICAL_ABBREVS = [
                'bp', 'hr', 'iv', 'im', 'po', 'pr', 'sc',
                'cho', 'ncd', 'dm', 'tb', 'hiv', 'aids',
                'copd', 'ckd', 'cvd', 'mi', 'cva', 'pe',
                'dvt', 'uti', 'rti', 'gi', 'cns', 'pns'
            ]

####################################################################
#                    INTENT CLASSIFICATION
####################################################################

@dataclass
class IntentConfig:
    """Intent classification configuration"""
    
    # Classification Settings
    ENABLE_INTENT_CLASSIFICATION: bool = True
    CONFIDENCE_THRESHOLD: float = 0.65
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # Non-Medical Templates
    NON_MEDICAL_TEMPLATES: List[str] = None
    
    def __post_init__(self):
        if self.NON_MEDICAL_TEMPLATES is None:
            self.NON_MEDICAL_TEMPLATES = [
                "hello how are you",
                "hi there",
                "good morning",
                "good afternoon",
                "good evening",
                "thank you",
                "thanks",
                "bye",
                "goodbye",
                "see you",
                "what can you help",
                "how does this work",
                "tell me about yourself",
                "who are you",
                "what are you",
                "help",
                "how to use",
                "instructions",
                "features"
            ]
    
    # Medical Keywords
    MEDICAL_KEYWORDS: List[str] = None
    
    def __post_init__(self):
        if self.MEDICAL_KEYWORDS is None:
            self.MEDICAL_KEYWORDS = [
                # Diseases
                'diabetes', 'hypertension', 'malaria', 'tuberculosis', 'tb',
                'covid', 'fever', 'cancer', 'asthma', 'copd', 'ckd', 'hiv',
                'pneumonia', 'dengue', 'typhoid', 'cholera', 'diarrhea',
                
                # Medical Terms
                'patient', 'treatment', 'medicine', 'drug', 'medication',
                'symptom', 'diagnosis', 'therapy', 'surgery', 'procedure',
                'examination', 'screening', 'test', 'laboratory', 'imaging',
                
                # Healthcare Roles
                'cho', 'doctor', 'nurse', 'physician', 'surgeon', 'midwife',
                'pharmacist', 'paramedic', 'healthcare', 'medical', 'clinical',
                
                # Procedures
                'vaccination', 'immunization', 'injection', 'infusion',
                'delivery', 'pregnancy', 'birth', 'emergency', 'trauma',
                'resuscitation', 'first aid', 'triage', 'ambulance',
                
                # Body Systems
                'cardiac', 'respiratory', 'gastrointestinal', 'neurological',
                'renal', 'hepatic', 'endocrine', 'musculoskeletal'
            ]

####################################################################
#                    CONVERSATION SETTINGS
####################################################################

@dataclass
class ConversationConfig:
    """Conversational AI configuration"""
    
    # Memory Settings
    ENABLE_MEMORY: bool = True
    MEMORY_TYPE: str = "buffer"  # buffer, summary, window
    MAX_MEMORY_LENGTH: int = 10
    
    # Conversation Parameters
    CONDENSE_QUESTIONS: bool = True
    RETURN_SOURCE_DOCUMENTS: bool = True
    VERBOSE_MODE: bool = False

####################################################################
#                    DOCUMENT PROCESSING
####################################################################

@dataclass
class DocumentConfig:
    """Document processing configuration"""
    
    # Supported Formats
    SUPPORTED_EXTENSIONS: List[str] = None
    
    def __post_init__(self):
        if self.SUPPORTED_EXTENSIONS is None:
            self.SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.csv', '.docx']
    
    # Processing Settings
    RECURSIVE_LOAD: bool = True
    BATCH_SIZE: int = 100
    MAX_FILE_SIZE_MB: int = 50
    
    # Metadata Enhancement
    ADD_TIMESTAMPS: bool = True
    GENERATE_IDS: bool = True

####################################################################
#                    UI SETTINGS
####################################################################

@dataclass
class UIConfig:
    """User interface configuration"""
    
    # Chat Settings
    SHOW_CHAT_HISTORY: bool = True
    MAX_DISPLAYED_CHATS: int = 50
    TIMESTAMP_FORMAT: str = "%H:%M:%S"
    
    # Source Display
    MAX_SOURCES_DISPLAYED: int = 3
    SOURCE_PREVIEW_LENGTH: int = 200
    SHOW_SOURCE_SCORES: bool = True
    
    # Sidebar Settings
    SHOW_INSTRUCTIONS: bool = True
    SHOW_STATS: bool = True
    ENABLE_DEBUG_MODE: bool = True
    
    # Theme
    PRIMARY_COLOR: str = "#FF4B4B"
    BACKGROUND_COLOR: str = "#FFFFFF"
    SECONDARY_BG_COLOR: str = "#F0F2F6"

####################################################################
#                    LOGGING & MONITORING
####################################################################

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_TO_FILE: bool = True
    LOG_FILE: str = "logs/app.log"
    
    # Monitoring
    TRACK_METRICS: bool = True
    TRACK_LATENCY: bool = True
    TRACK_TOKEN_USAGE: bool = True

####################################################################
#                    HEALTHCARE SAFETY
####################################################################

@dataclass
class SafetyConfig:
    """Healthcare-specific safety configuration"""
    
    # Safety Rules
    REQUIRE_SOURCE_ATTRIBUTION: bool = True
    DISABLE_GUESSING: bool = True
    MEDICAL_DISCLAIMER: bool = True
    
    # Response Validation
    CHECK_HARMFUL_CONTENT: bool = True
    VERIFY_MEDICAL_ACCURACY: bool = True
    
    # Disclaimer Text
    DISCLAIMER: str = (
        "⚠️ **Important**: This system provides information from CHO training materials "
        "for educational purposes only. Always consult official medical guidelines and "
        "healthcare professionals for clinical decisions."
    )

####################################################################
#                    PROMPTS
####################################################################

class PromptTemplates:
    """Prompt templates for different scenarios"""
    
    MEDICAL_QA_PROMPT = """You are a specialized healthcare assistant for Community Health Officers (CHOs).
You provide information STRICTLY based on official CHO training materials.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the context doesn't contain relevant information, state: "I couldn't find this information in the CHO training materials."
3. Always maintain healthcare safety and accuracy
4. Cite sources when possible
5. Never guess or infer beyond the provided context

Context from CHO training materials:
{context}

Chat History:
{chat_history}

Current Question: {question}

Provide a detailed, accurate answer based ONLY on the context above:"""

    NON_MEDICAL_GREETING = """Hello! I'm your CHO Healthcare Assistant specializing in Community Health Officer training materials.

**I can help you with:**
• CHO roles and responsibilities
• Disease management protocols (diabetes, hypertension, malaria, etc.)
• Emergency response procedures
• Maternal and child health guidelines
• Mental health care protocols
• Preventive care and vaccination schedules

Please ask specific questions about CHO training materials!"""

    NON_MEDICAL_GRATITUDE = """You're welcome! I'm here to provide evidence-based healthcare information from CHO training materials.

Feel free to ask about specific diseases, procedures, or guidelines."""

    NON_MEDICAL_FAREWELL = """Goodbye! Remember to always follow official CHO protocols and guidelines in your healthcare practice.

Stay safe and keep serving your community's health needs!"""

    CONDENSE_QUESTION_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:"""

####################################################################
#                    LOAD CONFIGURATION
####################################################################

def load_config():
    """Load all configurations"""
    return {
        'app': AppConfig(),
        'llm': LLMConfig(),
        'embedding': EmbeddingConfig(),
        'retrieval': RetrievalConfig(),
        'chunking': ChunkingConfig(),
        'intent': IntentConfig(),
        'conversation': ConversationConfig(),
        'document': DocumentConfig(),
        'ui': UIConfig(),
        'logging': LoggingConfig(),
        'safety': SafetyConfig(),
        'prompts': PromptTemplates()
    }

# Global config instance
CONFIG = load_config()

if __name__ == "__main__":
    # Print configuration summary
    config = load_config()
    
    print("=" * 60)
    print("Healthcare RAG System Configuration")
    print("=" * 60)
    
    print(f"\n[Application]")
    print(f"  - Name: {config['app'].APP_NAME}")
    print(f"  - Version: {config['app'].APP_VERSION}")

    print(f"\n[LLM]")
    print(f"  - Provider: {config['llm'].PROVIDER}")
    print(f"  - Model: {config['llm'].MODEL_NAME}")
    print(f"  - Temperature: {config['llm'].TEMPERATURE}")

    print(f"\n[Embeddings]")
    print(f"  - Model: {config['embedding'].MODEL_NAME}")
    print(f"  - Device: {config['embedding'].DEVICE}")

    print(f"\n[Retrieval]")
    print(f"  - Type: {config['retrieval'].RETRIEVER_TYPE}")
    print(f"  - Alpha: {config['retrieval'].DEFAULT_ALPHA}")
    print(f"  - K: {config['retrieval'].DENSE_K}")

    print(f"\n[Chunking]")
    print(f"  - Size: {config['chunking'].CHUNK_SIZE}")
    print(f"  - Overlap: {config['chunking'].CHUNK_OVERLAP}")

    print(f"\n[Intent Classification]")
    print(f"  - Enabled: {config['intent'].ENABLE_INTENT_CLASSIFICATION}")
    print(f"  - Threshold: {config['intent'].CONFIDENCE_THRESHOLD}")

    print(f"\n[Safety]")
    print(f"  - Source Attribution: {config['safety'].REQUIRE_SOURCE_ATTRIBUTION}")
    print(f"  - Disable Guessing: {config['safety'].DISABLE_GUESSING}")
    
    print("\n" + "=" * 60)
