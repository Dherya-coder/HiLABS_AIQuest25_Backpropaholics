#!/usr/bin/env python3
"""
Complete PDF Processing Pipeline

Python implementation using imports instead of shell commands.
Processes 12 PDFs across 4 categories with error handling.
"""

import sys
import logging
from pathlib import Path
from typing import Dict
import json

from util.pdf_parsing import process_pdf
from util.chunking import chunk_markdown_files
from util.embedding import process_contract_chunks_to_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Pipeline configuration
MAX_FILE_SIZE_MB = 50  # Skip files larger than this size
PIPELINE_CONFIG: Dict[str, Dict] = {
    "TNredacted": {
        "input_dir": str(PROJECT_ROOT / "data/Contracts_data/TN"),
        "output_dir": str(PROJECT_ROOT / "outputs/pdf_parsed/TN"),
        "collection_name": "TNredacted",
        "description": "TN Contract PDFs (5 files)",
    },
    "WAredacted": {
        "input_dir": str(PROJECT_ROOT / "data/Contracts_data/WA"),
        "output_dir": str(PROJECT_ROOT / "outputs/pdf_parsed/WA"),
        "collection_name": "WAredacted",
        "description": "WA Contract PDFs (5 files)",
    },
    "TNstandard": {
        "input_dir": str(PROJECT_ROOT / "data/Contracts_data/Standard Templates_data"),
        "input_file": "TN_Standard_Template_Redacted.pdf",
        "output_dir": str(PROJECT_ROOT / "outputs/pdf_parsed/TN_standard"),
        "collection_name": "TNstandard",
        "description": "TN Standard Template (1 file)",
    },
    "WAstandard": {
        "input_dir": str(PROJECT_ROOT / "data/Contracts_data/Standard Templates_data"),
        "input_file": "WA_Standard_Redacted.pdf",
        "output_dir": str(PROJECT_ROOT / "outputs/pdf_parsed/WA_standard"),
        "collection_name": "WAstandard",
        "description": "WA Standard Template (1 file)",
    },
}


class PipelineProcessor:
    """Runs parsing, chunking, and embedding stages across configured categories."""

    def __init__(self, db_path: str = None) -> None:
        self.db_path = db_path or str(PROJECT_ROOT / "chroma_db_qwen")
        self.results: Dict[str, bool] = {}

    @staticmethod
    def get_file_size_mb(file_path: Path) -> float:
        """Return file size in megabytes."""
        return file_path.stat().st_size / (1024 * 1024)

    @staticmethod
    def is_already_processed(pdf_file: Path, output_dir: Path) -> bool:
        """Return True if the PDF already has corresponding .md and _chunks.json outputs."""
        md_file = output_dir / f"{pdf_file.stem}.md"
        chunks_file = output_dir / f"{pdf_file.stem}_chunks.json"
        return md_file.exists() and chunks_file.exists()

    def check_dependencies(self) -> bool:
        """Verify required third-party modules are importable."""
        required_modules = [
            "pdfplumber",
            "pytesseract",
            "pdf2image",
            "pypdf",
            "tiktoken",
            "langchain_text_splitters",
            "chromadb",
            "requests",
        ]
        missing = []

        logger.info("Checking dependencies...")
        for module in required_modules:
            try:
                __import__(module)
                logger.debug("OK: %s", module)
            except ImportError:
                missing.append(module)
                logger.error("Missing: %s", module)

        if missing:
            logger.error("Missing dependencies: %s", missing)
            logger.info("Install with: pip install %s", " ".join(missing))
            return False

        logger.info("All dependencies available")
        return True

    def process_parsing_stage(self, category: str, config: Dict) -> bool:
        """Parse PDFs to markdown for a category."""
        logger.info("PARSING STAGE: %s", category)

        input_dir = Path(config["input_dir"])
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        total_files = 0

        try:
            if "input_file" in config:
                # Single file processing (standard templates)
                input_file = input_dir / config["input_file"]
                if not input_file.exists():
                    logger.error("Input file not found: %s", input_file)
                    return False

                total_files = 1
                logger.info("Processing: %s", config["input_file"])

                try:
                    process_pdf(input_file, output_dir)
                    success_count = 1
                    logger.info("Successfully processed: %s", config["input_file"])
                except Exception as e:
                    logger.error("Failed to process %s: %s", config["input_file"], e)
            else:
                # Multiple files
                pdf_files = list(input_dir.glob("*.pdf"))
                if not pdf_files:
                    logger.error("No PDF files found in %s", input_dir)
                    return False

                total_files = len(pdf_files)
                logger.info("Found %d PDF files to process", total_files)

                for pdf_file in pdf_files:
                    if self.is_already_processed(pdf_file, output_dir):
                        logger.info("Skipping %s (already processed)", pdf_file.name)
                        success_count += 1
                        continue

                    file_size_mb = self.get_file_size_mb(pdf_file)
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        logger.warning(
                            "Skipping %s (%.1f MB exceeds %d MB limit)",
                            pdf_file.name,
                            file_size_mb,
                            MAX_FILE_SIZE_MB,
                        )
                        continue

                    logger.info("Processing: %s (%.1f MB)", pdf_file.name, file_size_mb)
                    try:
                        process_pdf(pdf_file, output_dir)
                        success_count += 1
                        logger.info("Successfully processed: %s", pdf_file.name)
                    except Exception as e:
                        logger.error("Failed to process %s: %s", pdf_file.name, e)

        except Exception as e:
            logger.error("Error in parsing stage for %s: %s", category, e)
            return False

        logger.info("Parsing completed: %d/%d files processed", success_count, total_files)
        return success_count > 0

    def process_chunking_stage(self, category: str, config: Dict) -> bool:
        """Split markdown into chunks for a category."""
        logger.info("CHUNKING STAGE: %s", category)

        output_dir = Path(config["output_dir"])
        if not output_dir.exists():
            logger.error("Output directory not found: %s", output_dir)
            return False

        md_files = list(output_dir.glob("*.md"))
        if not md_files:
            logger.error("No markdown files found in %s", output_dir)
            return False

        logger.info("Found %d markdown files to chunk", len(md_files))

        try:
            chunk_markdown_files(output_dir, output_dir)
            logger.info("Successfully chunked files for %s", category)
            return True
        except Exception as e:
            logger.error("Failed to chunk files for %s: %s", category, e)
            return False

    def process_embedding_stage(self, category: str, config: Dict) -> bool:
        """Generate embeddings for chunk JSON files in a category."""
        logger.info("EMBEDDING STAGE: %s", category)

        output_dir = Path(config["output_dir"])
        collection_name = config["collection_name"]

        if not output_dir.exists():
            logger.error("Output directory not found: %s", output_dir)
            return False

        json_files = list(output_dir.glob("*_chunks.json"))
        if not json_files:
            logger.error("No chunk JSON files found in %s", output_dir)
            return False

        logger.info("Found %d chunk files for embedding", len(json_files))

        try:
            process_contract_chunks_to_embeddings(
                chunks_dir=output_dir,
                collection_name=collection_name,
                db_path=self.db_path,
                model_name="qwen3-embedding:0.6b",
                batch_size=5,
            )
            logger.info("Successfully generated embeddings for %s", category)
            return True
        except Exception as e:
            logger.error("Failed to generate embeddings for %s: %s", category, e)
            return False

    def process_category(self, category: str, config: Dict) -> bool:
        """Run parsing, chunking, and embeddings for a single category."""
        logger.info("\n%s", "=" * 20 + f" PROCESSING {category.upper()} " + "=" * 20)

        if not self.process_parsing_stage(category, config):
            logger.error("Parsing failed for %s", category)
            return False

        if not self.process_chunking_stage(category, config):
            logger.error("Chunking failed for %s", category)
            return False

        if not self.process_embedding_stage(category, config):
            logger.error("Embedding generation failed for %s", catebeddgory)
            return False

        logger.info("%s pipeline completed successfully", category)
        return True

    def create_summary(self) -> None:
        """Persist a summary of pipeline execution."""
        logger.info("\n%s", "=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("%s", "=" * 60)

        total_categories = len(PIPELINE_CONFIG)
        successful_categories = sum(1 for success in self.results.values() if success)

        for category, config in PIPELINE_CONFIG.items():
            status = "SUCCESS" if self.results.get(category, False) else "FAILED"
            logger.info("%-12s | %-25s | %s", category, config["description"], status)

        logger.info(
            "\nOverall: %d/%d categories completed successfully",
            successful_categories,
            total_categories,
        )

        summary_file = Path("../outputs/pipeline_summary.json")
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        summary_data = {
            "pipeline_config": PIPELINE_CONFIG,
            "execution_results": self.results,
            "success_rate": f"{successful_categories}/{total_categories}",
            "successful_categories": successful_categories,
            "total_categories": total_categories,
        }

        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                # Keep ensure_ascii=False if downstream expects UTF-8 JSON;
                # change to True only if you require strictly ASCII JSON.
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info("Pipeline summary saved to: %s", summary_file)
        except Exception as e:
            logger.error("Failed to save summary: %s", e)

    def run_pipeline(self) -> bool:
        """Run the complete pipeline for all categories."""
        logger.info("Starting Complete PDF Processing Pipeline")
        logger.info("%s", "=" * 60)

        if not self.check_dependencies():
            logger.error("Dependency check failed")
            return False

        logger.info("Pipeline Configuration:")
        for category, config in PIPELINE_CONFIG.items():
            logger.info("  %s: %s", category, config["description"])

        for category, config in PIPELINE_CONFIG.items():
            try:
                self.results[category] = self.process_category(category, config)
            except Exception as e:
                logger.error("Unexpected error processing %s: %s", category, e)
                self.results[category] = False

        self.create_summary()

        all_successful = all(self.results.values()) if self.results else False
        if all_successful:
            logger.info("\nALL PIPELINES COMPLETED SUCCESSFULLY")
        else:
            logger.warning("\nSOME PIPELINES FAILED - CHECK LOGS ABOVE")

        return all_successful


def main() -> int:
    """CLI entrypoint."""
    try:
        processor = PipelineProcessor()
        success = processor.run_pipeline()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
