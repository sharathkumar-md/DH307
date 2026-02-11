"""
Answer Generation Evaluation Across Different Retrieval Methods

Evaluates the quality of generated answers using different retrieval strategies:
- BM25 (Sparse)
- FAISS Dense
- Hybrid with different alpha values (0.2, 0.4, 0.5, 0.6, 0.8)

Metrics evaluated:
- Faithfulness: Answer consistency with retrieved context
- Answer Relevancy: How relevant the answer is to the question
- Context Precision: Quality of retrieved contexts
- Context Recall: Coverage of relevant information
- Answer Correctness: Semantic similarity with ground truth
- Generation Time: Speed of answer generation
- Retrieval Time: Speed of context retrieval

Author: Sharath Kumar MD
Date: November 2025
"""

import sys
import os
import io
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# LangChain imports - Google imports are conditional (loaded only when USE_OLLAMA=False)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# Load environment
from dotenv import load_dotenv
load_dotenv()

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

# Logging setup
import logging

LOG_LEVEL = os.getenv('EVAL_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

####################################################################
#                    OLLAMA HELPER
####################################################################

import requests

OLLAMA_URL = "http://127.0.0.1:11434"

def ollama_generate(model: str, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Send prompt to local Ollama model and return generated text.
    
    Args:
        model: Ollama model name (e.g., 'llama3.2:1b', 'llama3.1:8b')
        prompt: Prompt string
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature (0.0 = deterministic)
    
    Returns:
        Generated text string
    """
    endpoint = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Get full response at once
        "options": {
            "num_predict": max_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    }
    
    try:
        resp = requests.post(endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        # Ollama returns {"response": "generated text", ...}
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API error: {e}")
        return f"Error: {str(e)}"

def test_ollama_connection(model: str = "llama3.2:1b"):
    """Test Ollama connection and model availability"""
    try:
        prompt = "Answer briefly: What is anaemia?"
        response = ollama_generate(model, prompt, max_tokens=100)
        logger.info(f"Ollama test successful with model '{model}'")
        logger.info(f"  Response: {response[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Ollama test failed: {e}")
        logger.error("Make sure Ollama is running: 'ollama serve'")
        logger.error(f"And model is pulled: 'ollama pull {model}'")
        return False

import asyncio


class OllamaLLM:
    """Wrapper to make Ollama compatible with LangChain-style LLM interface and RAGAS"""

    def __init__(self, model: str = "llama3.2:1b", max_tokens: int = 512, temperature: float = 0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.metadata = {"model_name": model}  # RAGAS compatibility

    def invoke(self, prompt: str):
        """LangChain-compatible invoke method"""
        response_text = ollama_generate(
            self.model,
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        # Return object with .content attribute like LangChain LLMs
        class Response:
            def __init__(self, text):
                self.content = text
        return Response(response_text)

    def _invoke_with_overrides(self, prompt: str, max_tokens: int = None, temperature: float = None):
        """Internal helper to call Ollama with optional overrides without mutating self."""
        mt = self.max_tokens if max_tokens is None else max_tokens
        temp = self.temperature if temperature is None else temperature
        response_text = ollama_generate(self.model, prompt, max_tokens=mt, temperature=temp)

        class Response:
            def __init__(self, text):
                self.content = text

        return Response(response_text)

    async def agenerate_prompt(self, prompts, **kwargs):
        """Async method for RAGAS compatibility - handles list of prompts"""
        loop = asyncio.get_event_loop()
        tasks = []
        for p in prompts:
            # Extract text from prompt object if needed
            if hasattr(p, 'to_string'):
                prompt_text = p.to_string()
            elif hasattr(p, 'text'):
                prompt_text = p.text
            else:
                prompt_text = str(p)

            tasks.append(
                loop.run_in_executor(
                    None,
                    self._invoke_with_overrides,
                    prompt_text,
                    kwargs.get('max_tokens', None),
                    kwargs.get('temperature', None)
                )
            )

        responses = await asyncio.gather(*tasks)

        generations = []
        for resp in responses:
            class Generation:
                def __init__(self, text):
                    self.text = text
                    self.message = type('Message', (), {'content': text})()

            generations.append([Generation(resp.content)])

        class LLMResult:
            def __init__(self, generations):
                self.generations = generations
                self.llm_output = {}

        return LLMResult(generations)

    async def generate(self, prompts: List[str], **kwargs):
        """Async LangChain-style `generate` method compatible with RAGAS.

        Runs synchronous `invoke` calls in a thread pool executor so callers
        can `await llm.generate(...)` safely.
        """
        loop = asyncio.get_event_loop()
        tasks = []
        for p in prompts:
            tasks.append(
                loop.run_in_executor(
                    None,
                    self._invoke_with_overrides,
                    p,
                    kwargs.get('max_tokens', None),
                    kwargs.get('temperature', None)
                )
            )

        responses = await asyncio.gather(*tasks)

        generations = []
        for resp in responses:
            class Generation:
                def __init__(self, text):
                    self.text = text

            generations.append([Generation(resp.content)])

        class LLMResult:
            def __init__(self, generations):
                self.generations = generations

        return LLMResult(generations)

####################################################################
#                    CONFIGURATION
####################################################################

VECTOR_DB_PATH = "data/vector_db"
TEST_QUERIES_PATH = "data/test_queries.json"
EMBEDDING_MODEL = "all-mpnet-base-v2"

# LLM Configuration
USE_OLLAMA = True  # Using local Ollama with custom evaluation metrics
OLLAMA_MODEL = "qwen2.5:0.5b"  # Local model for both answer generation and evaluation

RESULTS_DIR = "evaluation_results_answer_generation"
NUM_TEST_QUERIES = 50  # Limit to first 50 queries for faster evaluation

# Retrieval configurations to test
RETRIEVAL_CONFIGS = [
    {"name": "BM25 (Sparse)", "alpha": 0.0, "method": "hybrid"},
    {"name": "Hybrid (alpha=0.2)", "alpha": 0.2, "method": "hybrid"},
    {"name": "Hybrid (alpha=0.4)", "alpha": 0.4, "method": "hybrid"},
    {"name": "Hybrid (alpha=0.5)", "alpha": 0.5, "method": "hybrid"},
    {"name": "Hybrid (alpha=0.6)", "alpha": 0.6, "method": "hybrid"},
    {"name": "Hybrid (alpha=0.8)", "alpha": 0.8, "method": "hybrid"},
    {"name": "FAISS (Dense)", "alpha": 1.0, "method": "hybrid"},
]

K_VALUE = 5  # Number of contexts to retrieve

# Create output directory
Path(RESULTS_DIR).mkdir(exist_ok=True)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

####################################################################
#                    RETRIEVAL METHODS
####################################################################

def get_hybrid_retriever(vector_db, documents: List, alpha: float, k: int = 5):
    """
    Get hybrid retriever with specified alpha
    alpha=0: BM25 only
    alpha=1: Dense only
    alpha=0.5: Equal weight
    """
    # Dense retriever
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": k})

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    # Hybrid ensemble
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[1-alpha, alpha]
    )

    return ensemble

def retrieve_contexts(retriever, query: str) -> List[str]:
    """Retrieve contexts using given retriever"""
    try:
        docs = retriever.get_relevant_documents(query)
        contexts = [doc.page_content for doc in docs]
        return contexts
    except Exception as e:
        logger.error(f"Error retrieving contexts: {e}")
        return []

####################################################################
#                    ANSWER GENERATION
####################################################################

HEALTHCARE_PROMPT_TEMPLATE = """You are a specialized healthcare assistant for Community Health Officers (CHOs).
You provide information STRICTLY based on official CHO training materials.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the context doesn't contain relevant information, state: "I couldn't find this information in the CHO training materials."
3. Always maintain healthcare safety and accuracy
4. Never guess or infer beyond the provided context
5. Be concise but comprehensive

Context from CHO training materials:
{context}

Question: {question}

Provide a detailed, accurate answer based ONLY on the context above:"""

def generate_answer(llm, query: str, contexts: List[str]) -> str:
    """Generate answer using LLM and contexts"""
    try:
        context_str = "\n\n".join(contexts)
        prompt = HEALTHCARE_PROMPT_TEMPLATE.format(
            context=context_str,
            question=query
        )

        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error: {str(e)}"

def extract_ground_truth(vector_db, relevant_pages: List, k: int = 3) -> str:
    """Extract ground truth from relevant pages"""
    try:
        # Retrieve all documents
        all_docs = []
        docstore = vector_db.docstore
        for doc_id in vector_db.index_to_docstore_id.values():
            doc = docstore.search(doc_id)
            if doc:
                all_docs.append(doc)

        # Filter by relevant pages
        relevant_docs = []
        for doc in all_docs:
            doc_page = str(doc.metadata.get('page_number', ''))
            if doc_page in [str(p) for p in relevant_pages]:
                relevant_docs.append(doc.page_content)
                if len(relevant_docs) >= k:
                    break

        if relevant_docs:
            return "\n\n".join(relevant_docs[:k])
        else:
            return "Information about this topic from CHO training materials."
    except Exception as e:
        logger.error(f"Error extracting ground truth: {e}")
        return "Ground truth not available"

####################################################################
#                    CUSTOM EVALUATION METRICS
####################################################################

def evaluate_faithfulness(llm, answer: str, contexts: List[str]) -> float:
    """
    Evaluate if the answer is faithful to the retrieved contexts.
    Returns a score between 0 and 1.
    """
    try:
        context_str = "\n\n".join(contexts[:3])  # Use top 3 contexts
        prompt = f"""You are evaluating the faithfulness of an answer to given contexts.

Context:
{context_str}

Answer:
{answer}

Question: Is the answer faithful to the context? Does it only contain information from the context without adding external knowledge or hallucinations?

Respond with ONLY a score from 0 to 10, where:
- 0 = completely unfaithful, contains hallucinations
- 10 = perfectly faithful, all information comes from context

Score:"""

        response = llm.invoke(prompt)
        score_text = response.content.strip()

        # Extract first number from response
        import re
        numbers = re.findall(r'\d+', score_text)
        if numbers:
            score = float(numbers[0]) / 10.0  # Normalize to 0-1
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        return 0.5  # Default if parsing fails

    except Exception as e:
        logger.warning(f"Faithfulness evaluation error: {e}")
        return 0.5

def evaluate_answer_relevancy(llm, question: str, answer: str) -> float:
    """
    Evaluate how relevant the answer is to the question.
    Returns a score between 0 and 1.
    """
    try:
        prompt = f"""You are evaluating the relevancy of an answer to a question.

Question:
{question}

Answer:
{answer}

Question: How relevant is the answer to the question? Does it address what was asked?

Respond with ONLY a score from 0 to 10, where:
- 0 = completely irrelevant
- 10 = perfectly relevant, directly answers the question

Score:"""

        response = llm.invoke(prompt)
        score_text = response.content.strip()

        # Extract first number from response
        import re
        numbers = re.findall(r'\d+', score_text)
        if numbers:
            score = float(numbers[0]) / 10.0
            return min(max(score, 0.0), 1.0)
        return 0.5

    except Exception as e:
        logger.warning(f"Answer relevancy evaluation error: {e}")
        return 0.5

def evaluate_context_precision(llm, question: str, contexts: List[str]) -> float:
    """
    Evaluate the precision of retrieved contexts - are they relevant to the question?
    Returns a score between 0 and 1.
    """
    try:
        context_str = "\n\n".join([f"Context {i+1}: {c[:200]}..." for i, c in enumerate(contexts[:3])])
        prompt = f"""You are evaluating the precision of retrieved contexts for a question.

Question:
{question}

Retrieved Contexts:
{context_str}

Question: How many of these contexts are relevant and useful for answering the question?

Respond with ONLY a score from 0 to 10, where:
- 0 = no contexts are relevant
- 10 = all contexts are highly relevant

Score:"""

        response = llm.invoke(prompt)
        score_text = response.content.strip()

        # Extract first number from response
        import re
        numbers = re.findall(r'\d+', score_text)
        if numbers:
            score = float(numbers[0]) / 10.0
            return min(max(score, 0.0), 1.0)
        return 0.5

    except Exception as e:
        logger.warning(f"Context precision evaluation error: {e}")
        return 0.5

def evaluate_context_recall(llm, question: str, contexts: List[str], ground_truth: str) -> float:
    """
    Evaluate if the retrieved contexts contain all necessary information from ground truth.
    Returns a score between 0 and 1.
    """
    try:
        context_str = "\n\n".join(contexts[:3])
        prompt = f"""You are evaluating context recall - whether retrieved contexts contain necessary information.

Question:
{question}

Ground Truth Information:
{ground_truth[:500]}

Retrieved Contexts:
{context_str[:500]}

Question: Do the retrieved contexts contain the key information needed to answer the question (as shown in ground truth)?

Respond with ONLY a score from 0 to 10, where:
- 0 = contexts missing all key information
- 10 = contexts contain all necessary information

Score:"""

        response = llm.invoke(prompt)
        score_text = response.content.strip()

        # Extract first number from response
        import re
        numbers = re.findall(r'\d+', score_text)
        if numbers:
            score = float(numbers[0]) / 10.0
            return min(max(score, 0.0), 1.0)
        return 0.5

    except Exception as e:
        logger.warning(f"Context recall evaluation error: {e}")
        return 0.5

def evaluate_answer_correctness(llm, question: str, answer: str, ground_truth: str) -> float:
    """
    Evaluate the correctness of the answer compared to ground truth.
    Returns a score between 0 and 1.
    """
    try:
        prompt = f"""You are evaluating the correctness of an answer against ground truth.

Question:
{question}

Ground Truth:
{ground_truth[:500]}

Generated Answer:
{answer[:500]}

Question: How correct is the generated answer compared to the ground truth?

Respond with ONLY a score from 0 to 10, where:
- 0 = completely incorrect
- 10 = perfectly correct, semantically equivalent to ground truth

Score:"""

        response = llm.invoke(prompt)
        score_text = response.content.strip()

        # Extract first number from response
        import re
        numbers = re.findall(r'\d+', score_text)
        if numbers:
            score = float(numbers[0]) / 10.0
            return min(max(score, 0.0), 1.0)
        return 0.5

    except Exception as e:
        logger.warning(f"Answer correctness evaluation error: {e}")
        return 0.5

####################################################################
#                    EVALUATION
####################################################################

def evaluate_retrieval_method(
    retriever,
    llm,
    vector_db,
    test_queries: List[Dict],
    method_name: str
) -> List[Dict]:
    """Evaluate a single retrieval method"""

    results = []
    logger.info(f"Evaluating {method_name}...")

    for query_data in tqdm(test_queries, desc=f"  {method_name}", leave=False):
        query = query_data['query']
        relevant_docs = query_data.get('relevant_docs', [])

        try:
            # Retrieve contexts
            start_time = time.time()
            contexts = retrieve_contexts(retriever, query)
            retrieval_time = (time.time() - start_time) * 1000  # ms

            # Generate answer
            start_time = time.time()
            answer = generate_answer(llm, query, contexts)
            generation_time = (time.time() - start_time) * 1000  # ms

            # Extract ground truth
            ground_truth = extract_ground_truth(vector_db, relevant_docs, k=3)

            results.append({
                'question': query,
                'answer': answer,
                'contexts': contexts,
                'ground_truth': ground_truth,
                'retrieval_time_ms': retrieval_time,
                'generation_time_ms': generation_time,
                'num_contexts': len(contexts),
                'method': method_name
            })

        except Exception as e:
            logger.error(f"Error on query: {str(e)[:100]}")
            results.append({
                'question': query,
                'answer': f"Error: {str(e)}",
                'contexts': [],
                'ground_truth': "",
                'retrieval_time_ms': 0,
                'generation_time_ms': 0,
                'num_contexts': 0,
                'method': method_name,
                'error': str(e)
            })

    return results

def run_comprehensive_evaluation():
    """Run evaluation for all retrieval methods"""
    logger.info("%s", "="*80)
    logger.info("ANSWER GENERATION EVALUATION - COMPARING RETRIEVAL METHODS")
    logger.info("%s", "="*80)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("%s", "="*80)

    # Load resources
    logger.info("[1/5] Loading vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    vector_db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info(f"Loaded {vector_db.index.ntotal} vectors")

    logger.info("[2/5] Loading test queries...")
    with open(TEST_QUERIES_PATH, 'r', encoding='utf-8') as f:
        all_queries = json.load(f)
    # Filter out comments
    all_valid_queries = [q for q in all_queries if not any(k.startswith('_') for k in q.keys())]
    # Limit to first N queries for faster evaluation
    test_queries = all_valid_queries[:NUM_TEST_QUERIES]
    logger.info(f"Loaded {len(test_queries)} queries (limited from {len(all_valid_queries)} total)")

    logger.info("[3/5] Preparing documents for BM25...")
    all_docs = []
    docstore = vector_db.docstore
    for doc_id in vector_db.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        if doc:
            all_docs.append(doc)
    logger.info(f"Loaded {len(all_docs)} documents")

    logger.info("[4/5] Initializing LLM...")

    # Initialize Ollama LLM for both answer generation and evaluation
    logger.info(f"Using Ollama model: {OLLAMA_MODEL}")
    if not test_ollama_connection(OLLAMA_MODEL):
        logger.error("OLLAMA connection failed")
        logger.error("Tip: Run 'ollama serve' in another terminal")
        return None

    llm = OllamaLLM(model=OLLAMA_MODEL, max_tokens=512, temperature=0.3)
    eval_llm = OllamaLLM(model=OLLAMA_MODEL, max_tokens=256, temperature=0.0)  # For evaluation metrics
    logger.info(f"Initialized Ollama: {OLLAMA_MODEL}")

    logger.info("[5/5] Running evaluations...")

    all_results = []

    for config in RETRIEVAL_CONFIGS:
        method_name = config['name']
        alpha = config['alpha']

        logger.info(f"Testing {method_name} (alpha={alpha})")

        # Create retriever
        retriever = get_hybrid_retriever(vector_db, all_docs, alpha, k=K_VALUE)

        # Evaluate
        results = evaluate_retrieval_method(
            retriever,
            llm,
            vector_db,
            test_queries,
            method_name
        )

        all_results.extend(results)

    logger.info(f"Total evaluations: {len(all_results)}")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save intermediate results
    csv_path = f"{RESULTS_DIR}/answer_generation_raw_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw results saved: {csv_path}")

    return df, eval_llm

def compute_custom_scores(df: pd.DataFrame, eval_llm):
    """Compute custom evaluation scores for each method using our own metrics"""

    logger.info("="*80)
    logger.info("COMPUTING CUSTOM EVALUATION SCORES")
    logger.info("="*80)

    methods = df['method'].unique()
    eval_results = []

    for method in methods:
        logger.info(f"Computing scores for {method}...")

        method_df = df[df['method'] == method].copy()

        # Evaluate using custom metrics
        try:
            logger.info(f"  Starting custom evaluation for {method}...")
            logger.info(f"  Dataset size: {len(method_df)} samples")

            # Collect scores for all samples
            faithfulness_scores = []
            relevancy_scores = []
            precision_scores = []
            recall_scores = []
            correctness_scores = []

            for idx, row in method_df.iterrows():
                # Evaluate each metric
                faith = evaluate_faithfulness(eval_llm, row['answer'], row['contexts'])
                rel = evaluate_answer_relevancy(eval_llm, row['question'], row['answer'])
                prec = evaluate_context_precision(eval_llm, row['question'], row['contexts'])
                rec = evaluate_context_recall(eval_llm, row['question'], row['contexts'], row['ground_truth'])
                corr = evaluate_answer_correctness(eval_llm, row['question'], row['answer'], row['ground_truth'])

                faithfulness_scores.append(faith)
                relevancy_scores.append(rel)
                precision_scores.append(prec)
                recall_scores.append(rec)
                correctness_scores.append(corr)

                logger.debug(f"    Sample {idx+1}/{len(method_df)}: F={faith:.2f}, R={rel:.2f}, P={prec:.2f}, RC={rec:.2f}, C={corr:.2f}")

            # Calculate averages
            faithfulness_score = np.mean(faithfulness_scores)
            relevancy_score = np.mean(relevancy_scores)
            precision_score = np.mean(precision_scores)
            recall_score = np.mean(recall_scores)
            correctness_score = np.mean(correctness_scores)

            eval_results.append({
                'method': method,
                'faithfulness': faithfulness_score,
                'answer_relevancy': relevancy_score,
                'context_precision': precision_score,
                'context_recall': recall_score,
                'answer_correctness': correctness_score,
                'avg_retrieval_time_ms': method_df['retrieval_time_ms'].mean(),
                'avg_generation_time_ms': method_df['generation_time_ms'].mean(),
                'total_time_ms': method_df['retrieval_time_ms'].mean() + method_df['generation_time_ms'].mean()
            })

            logger.info(f"  ✓ Custom evaluation completed for {method}")
            logger.info(f"    Faithfulness: {faithfulness_score:.3f}")
            logger.info(f"    Answer Relevancy: {relevancy_score:.3f}")
            logger.info(f"    Context Precision: {precision_score:.3f}")
            logger.info(f"    Context Recall: {recall_score:.3f}")
            logger.info(f"    Answer Correctness: {correctness_score:.3f}")

        except Exception as e:
            logger.error(f"  ERROR computing scores for {method}: {e}")
            import traceback
            logger.error(f"  Traceback:\n{traceback.format_exc()}")

            eval_results.append({
                'method': method,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'answer_correctness': 0.0,
                'avg_retrieval_time_ms': method_df['retrieval_time_ms'].mean(),
                'avg_generation_time_ms': method_df['generation_time_ms'].mean(),
                'total_time_ms': method_df['retrieval_time_ms'].mean() + method_df['generation_time_ms'].mean(),
                'error': str(e)
            })

            logger.warning(f"  Continuing with zeros for {method}...")

    scores_df = pd.DataFrame(eval_results)

    # Save evaluation results
    scores_csv = f"{RESULTS_DIR}/ragas_scores_by_method.csv"
    scores_df.to_csv(scores_csv, index=False)
    logger.info(f"Custom evaluation scores saved: {scores_csv}")

    return scores_df

####################################################################
#                    VISUALIZATION
####################################################################

def create_visualizations(ragas_df: pd.DataFrame):
    """Create comprehensive visualizations"""
    logger.info("%s", "="*80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("%s", "="*80)

    methods = ragas_df['method'].tolist()

    # 1. RAGAS Metrics Comparison
    logger.info("[1/6] Creating RAGAS metrics comparison...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Answer Generation Quality: RAGAS Metrics Comparison', fontsize=16, fontweight='bold')

    metrics = ['faithfulness', 'answer_relevancy', 'context_precision',
               'context_recall', 'answer_correctness']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        values = ragas_df[metric].tolist()
        bars = ax.bar(range(len(methods)), values,
                      color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
        ax.set_xlabel('Retrieval Method', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    # Remove empty subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/1_ragas_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Overall Performance Radar Chart
    logger.info("[2/6] Creating radar chart...")
    from math import pi

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

    categories = ['Faithfulness', 'Answer\nRelevancy', 'Context\nPrecision',
                  'Context\nRecall', 'Answer\nCorrectness']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for idx, method in enumerate(methods):
        row = ragas_df[ragas_df['method'] == method][metrics]
        if row.empty:
            values = [0.0] * len(metrics)
        else:
            vals = row.values[0].tolist()
            values = [0.0 if (isinstance(v, float) and np.isnan(v)) else v for v in vals]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Answer Generation Quality: Radar Comparison', size=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/2_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Performance vs Speed Trade-off
    logger.info("[3/6] Creating performance vs speed plot...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use average of key metrics as "performance score"
    ragas_df['performance_score'] = ragas_df[['faithfulness', 'answer_relevancy', 'answer_correctness']].mean(axis=1)

    scatter = ax.scatter(
        ragas_df['total_time_ms'],
        ragas_df['performance_score'],
        s=200,
        c=range(len(ragas_df)),
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )

    # Add labels
    for idx, row in ragas_df.iterrows():
        ax.annotate(
            row['method'],
            (row['total_time_ms'], row['performance_score']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5)
        )

    ax.set_xlabel('Total Time (Retrieval + Generation) [ms]', fontsize=12)
    ax.set_ylabel('Performance Score\n(Avg of Faithfulness, Relevancy, Correctness)', fontsize=12)
    ax.set_title('Answer Generation: Performance vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/3_performance_vs_speed.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Time Breakdown
    logger.info("[4/6] Creating time breakdown chart...")
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(methods))
    width = 0.35

    retrieval_times = ragas_df['avg_retrieval_time_ms'].tolist()
    generation_times = ragas_df['avg_generation_time_ms'].tolist()

    bars1 = ax.bar(x - width/2, retrieval_times, width, label='Retrieval Time', color='skyblue')
    bars2 = ax.bar(x + width/2, generation_times, width, label='Generation Time', color='lightcoral')

    ax.set_xlabel('Retrieval Method', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Time Breakdown: Retrieval vs Generation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/4_time_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Heatmap of all metrics
    logger.info("[5/6] Creating metrics heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap_data = ragas_df[metrics].T
    heatmap_data.columns = methods

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                linewidths=0.5, ax=ax)

    ax.set_xlabel('Retrieval Method', fontsize=12)
    ax.set_ylabel('RAGAS Metric', fontsize=12)
    ax.set_title('Answer Generation Quality Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/5_metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Best Method Summary
    logger.info("[6/6] Creating best method summary...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Find best method for each metric (handle all-NaN case)
    best_methods = {}
    for metric in metrics:
        if ragas_df[metric].dropna().empty:
            best_methods[metric] = {'method': 'N/A', 'score': 0.0}
            continue

        best_idx = ragas_df[metric].idxmax()
        try:
            best_methods[metric] = {
                'method': ragas_df.loc[best_idx, 'method'],
                'score': float(ragas_df.loc[best_idx, metric])
            }
        except Exception:
            best_methods[metric] = {'method': 'N/A', 'score': 0.0}

    metric_names = [m.replace('_', ' ').title() for m in metrics]
    best_scores = [best_methods[m]['score'] for m in metrics]
    best_labels = [best_methods[m]['method'] for m in metrics]

    bars = ax.barh(metric_names, best_scores, color=plt.cm.Greens(np.linspace(0.4, 0.9, len(metrics))))

    ax.set_xlabel('Best Score', fontsize=12)
    ax.set_title('Best Performing Method for Each Metric', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='x')

    # Add method names and scores
    for idx, (bar, label) in enumerate(zip(bars, best_labels)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f' {label} ({width:.3f})',
               ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/6_best_methods.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("All plots saved!")

def print_summary(ragas_df: pd.DataFrame):
    """Print summary of results"""
    logger.info("%s", "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("%s", "="*80)

    logger.info("RAGAS Scores by Method:\n%s", ragas_df[['method', 'faithfulness', 'answer_relevancy', 'answer_correctness']].to_string(index=False))

    logger.info("Performance Metrics:\n%s", ragas_df[['method', 'avg_retrieval_time_ms', 'avg_generation_time_ms', 'total_time_ms']].to_string(index=False))

    # Best method
    ragas_df['overall_score'] = ragas_df[['faithfulness', 'answer_relevancy', 'answer_correctness']].mean(axis=1)
    if ragas_df['overall_score'].dropna().empty:
        best_method = {
            'method': 'N/A',
            'overall_score': 0.0,
            'faithfulness': 0.0,
            'answer_relevancy': 0.0,
            'answer_correctness': 0.0,
            'total_time_ms': 0.0
        }
    else:
        best_idx = ragas_df['overall_score'].idxmax()
        best_method = ragas_df.loc[best_idx]

    logger.info("%s", "="*80)
    logger.info("RECOMMENDED METHOD (Best Overall Score):")
    logger.info("%s", "="*80)
    logger.info(f"Method: {best_method['method']}")
    logger.info(f"Overall Score: {best_method['overall_score']:.4f}")
    logger.info(f"Faithfulness: {best_method['faithfulness']:.4f}")
    logger.info(f"Answer Relevancy: {best_method['answer_relevancy']:.4f}")
    logger.info(f"Answer Correctness: {best_method['answer_correctness']:.4f}")
    logger.info(f"Avg Total Time: {best_method['total_time_ms']:.2f} ms")

####################################################################
#                    MAIN
####################################################################

def main():
    logger.info("%s", "="*80)
    logger.info("ANSWER GENERATION EVALUATION ACROSS RETRIEVAL METHODS")
    logger.info("%s", "="*80)

    # Run evaluation
    result = run_comprehensive_evaluation()
    if result is None:
        logger.error("Evaluation failed!")
        return

    df, eval_llm = result

    # Compute custom evaluation scores
    scores_df = compute_custom_scores(df, eval_llm)

    # Create visualizations
    create_visualizations(scores_df)

    # Print summary
    print_summary(scores_df)

    logger.info("%s", "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("%s", "="*80)
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Results saved to: %s/", RESULTS_DIR)
    logger.info(" - answer_generation_raw_results.csv")
    logger.info(" - ragas_scores_by_method.csv")
    logger.info(" - 1_ragas_metrics_comparison.png")
    logger.info(" - 2_radar_chart.png")
    logger.info(" - 3_performance_vs_speed.png")
    logger.info(" - 4_time_breakdown.png")
    logger.info(" - 5_metrics_heatmap.png")
    logger.info(" - 6_best_methods.png")

if __name__ == "__main__":
    main()
