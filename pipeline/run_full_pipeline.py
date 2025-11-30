"""

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

Full Pipeline - Parse PDFs to JSON, then create vector database
Runs the complete two-stage pipeline.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and handle errors.

    Args:
        cmd: Command to run as list
        description: Description of what the command does

    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "="*80)
    logger.info("{description}")
    logger.info("="*80)
    logger.info("Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.info("\n✗ Error: {e}")
        return False


def main():
    """Run the full pipeline."""
    import argparse

import logging

    parser = argparse.ArgumentParser(
        description="Run full pipeline: PDFs → JSON → Vector DB"
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data/documents",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default="data/parsed_documents.json",
        help="Intermediate JSON file"
    )
    parser.add_argument(
        "--db_output",
        type=str,
        default="data/vector_db",
        help="Output vector database directory"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Chunk overlap"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model"
    )
    parser.add_argument(
        "--skip_parsing",
        action="store_true",
        help="Skip PDF parsing (use existing JSON)"
    )

    args = parser.parse_args()

    # Get script directory
    scripts_dir = Path(__file__).parent

    logger.info("\n" + "="*80)
    logger.info("FULL PIPELINE: PDFs → JSON → VECTOR DB")
    logger.info("="*80)
    logger.info("PDF directory: {args.pdf_dir}")
    logger.info("JSON output: {args.json_output}")
    logger.info("Vector DB output: {args.db_output}")
    logger.info("Chunk size: {args.chunk_size}")
    logger.info("Chunk overlap: {args.chunk_overlap}")
    logger.info("Embedding model: {args.embedding_model}")

    # Step 1: Parse PDFs to JSON
    if not args.skip_parsing:
        cmd1 = [
            sys.executable,
            str(scripts_dir / "parse_pdfs_to_json.py"),
            "--pdf_dir", args.pdf_dir,
            "--output", args.json_output
        ]

        if not run_command(cmd1, "STEP 1: Parsing PDFs to JSON"):
            logger.info("\n✗ Pipeline failed at Step 1")
            return False
    else:
        logger.info("\n[SKIP] Skipping PDF parsing, using existing JSON")

    # Step 2: Create chunks and vector database
    cmd2 = [
        sys.executable,
        str(scripts_dir / "create_chunks_from_json.py"),
        "--json", args.json_output,
        "--output", args.db_output,
        "--chunk_size", str(args.chunk_size),
        "--chunk_overlap", str(args.chunk_overlap),
        "--embedding_model", args.embedding_model
    ]

    if not run_command(cmd2, "STEP 2: Creating chunks and vector database"):
        logger.info("\n✗ Pipeline failed at Step 2")
        return False

    # Success!
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info("\n[OK] Parsed JSON: {args.json_output}")
    logger.info("[OK] Vector DB: {args.db_output}")
    logger.info("[OK] Chunk mapping: {args.db_output}/chunk_mapping.json")
    logger.info("\nNext steps:")
    logger.info("1. Review chunk_mapping.json to see all chunk indices")
    logger.info("2. Create test queries with correct chunk_index values")
    logger.info("3. Run evaluation: python evaluation/evaluate_retrieval.py")
    logger.info("4. Start chatbot: streamlit run rag_chat.py")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
