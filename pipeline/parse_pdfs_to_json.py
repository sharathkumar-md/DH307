"""

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

PDF Parser - Extract content from PDFs to JSON
Parses all PDFs and saves structured data to JSON for later chunking.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader


class PDFParser:
    """Parse PDFs and save to structured JSON format."""

    def __init__(self, pdf_dir: str, output_json: str):
        """
        Initialize PDF parser.

        Args:
            pdf_dir: Directory containing PDF files
            output_json: Output JSON file path
        """
        self.pdf_dir = Path(pdf_dir)
        self.output_json = Path(output_json)

        # Create output directory if needed
        self.output_json.parent.mkdir(parents=True, exist_ok=True)

    def parse_single_pdf(self, pdf_path: Path) -> Dict:
        """
        Parse a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing PDF content and metadata
        """
        logger.info("\nParsing: {pdf_path.name}")

        try:
            # Load PDF
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Extract content per page
            # Fields to exclude from metadata
            exclude_fields = [
                'source', 'producer', 'creator', 'creationdate',
                'moddate', 'trapped', 'author', 'title'
            ]

            pages_data = []
            for page_num, page in enumerate(pages):
                # Only keep useful metadata
                filtered_metadata = {
                    k: v for k, v in page.metadata.items()
                    if k not in exclude_fields
                }

                page_data = {
                    "page_number": page_num,
                    "page_label": str(page_num + 1),  # 1-indexed for display
                    "content": page.page_content,
                    "metadata": filtered_metadata
                }
                pages_data.append(page_data)

            # Build document structure
            doc_data = {
                "filename": pdf_path.name,
                "filepath": str(pdf_path),
                "total_pages": len(pages),
                "parsed_at": datetime.now().isoformat(),
                "pages": pages_data
            }

            logger.info("  [OK] Parsed {len(pages)} pages")
            return doc_data

        except Exception as e:
            logger.info("  [ERROR] Error parsing {pdf_path.name}: {e}")
            return None

    def parse_all_pdfs(self) -> List[Dict]:
        """
        Parse all PDFs in the directory.

        Returns:
            List of document dictionaries
        """
        # Find all PDF files
        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            logger.info("No PDF files found in {self.pdf_dir}")
            return []

        logger.info("Found {len(pdf_files)} PDF files")
        logger.info("="*80)

        # Parse each PDF
        documents = []
        successful = 0
        failed = 0

        for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
            doc_data = self.parse_single_pdf(pdf_path)
            if doc_data:
                documents.append(doc_data)
                successful += 1
            else:
                failed += 1

        logger.info("\n" + "="*80)
        logger.info("Parsing complete!")
        logger.info("  Successful: {successful}")
        logger.info("  Failed: {failed}")
        logger.info("  Total pages: {sum(doc['total_pages'] for doc in documents)}")

        return documents

    def save_to_json(self, documents: List[Dict]):
        """
        Save parsed documents to JSON file.

        Args:
            documents: List of document dictionaries
        """
        # Create output structure
        output = {
            "created_at": datetime.now().isoformat(),
            "num_documents": len(documents),
            "total_pages": sum(doc['total_pages'] for doc in documents),
            "source_directory": str(self.pdf_dir),
            "documents": documents
        }

        # Save to JSON
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info("\n[OK] Saved to: {self.output_json}")
        logger.info("  File size: {self.output_json.stat().st_size / 1024 / 1024:.2f} MB")

    def run(self):
        """Run the complete parsing pipeline."""
        logger.info("\n" + "="*80)
        logger.info("PDF TO JSON PARSER")
        logger.info("="*80)

        # Parse all PDFs
        documents = self.parse_all_pdfs()

        if documents:
            # Save to JSON
            self.save_to_json(documents)
            return True
        else:
            logger.error(" No documents were successfully parsed")
            return False


def main():
    """Main entry point."""
    import argparse

import logging

    parser = argparse.ArgumentParser(
        description="Parse PDFs and save to structured JSON"
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data/documents",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/parsed_documents.json",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # Create parser and run
    pdf_parser = PDFParser(
        pdf_dir=args.pdf_dir,
        output_json=args.output
    )

    success = pdf_parser.run()

    if success:
        logger.info("\n[OK] Pipeline complete!")
    else:
        logger.error(" Pipeline failed")
        exit(1)


if __name__ == "__main__":
    main()
