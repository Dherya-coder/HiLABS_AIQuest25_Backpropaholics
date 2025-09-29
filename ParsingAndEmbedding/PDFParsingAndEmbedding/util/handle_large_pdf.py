#!/usr/bin/env python3
"""
Handle Large PDF Processing

Split large PDFs into smaller chunks and process them safely.
Specifically designed for WA_5_Redacted.pdf (103MB).
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import gc
import psutil
import os

# Local modules
from parsing import process_pdf
from chunking import chunk_markdown_files
from embedding import process_contract_chunks_to_embeddings

# Configure logging (ASCII-only)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_memory_usage() -> float:
    """Return current process RSS memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def split_large_pdf_processing(pdf_path: Path, output_dir: Path, max_pages_per_batch: int = 10) -> tuple[List[Path], Optional[Path]]:
    """
    Split a large PDF into smaller batch PDFs to avoid memory issues.

    Parameters
    ----------
    pdf_path : Path
        Path to the input PDF.
    output_dir : Path
        Directory where batch PDFs and outputs will be written.
    max_pages_per_batch : int
        Number of pages per batch PDF.

    Returns
    -------
    (batch_files, temp_dir) : (List[Path], Optional[Path])
        List of created batch PDF paths and the temporary directory used.
    """
    logger.info(f"Processing large PDF: {pdf_path.name}")
    logger.info(f"Initial memory usage: {get_memory_usage():.1f} MB")

    try:
        import pypdf

        # Determine total pages
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)

        logger.info(f"Total pages: {total_pages}")
        logger.info(f"Processing in batches of {max_pages_per_batch} pages")

        # Temporary directory for batches
        temp_dir = output_dir / "temp_batches"
        temp_dir.mkdir(exist_ok=True)

        batch_files: List[Path] = []

        # Create each batch PDF
        for batch_start in range(0, total_pages, max_pages_per_batch):
            batch_end = min(batch_start + max_pages_per_batch, total_pages)
            batch_num = (batch_start // max_pages_per_batch) + 1

            logger.info(f"Creating batch {batch_num}: pages {batch_start + 1}-{batch_end}")

            batch_filename = f"{pdf_path.stem}_batch_{batch_num:03d}.pdf"
            batch_path = temp_dir / batch_filename

            try:
                with open(pdf_path, "rb") as input_file:
                    pdf_reader = pypdf.PdfReader(input_file)
                    pdf_writer = pypdf.PdfWriter()

                    for page_num in range(batch_start, batch_end):
                        pdf_writer.add_page(pdf_reader.pages[page_num])

                    with open(batch_path, "wb") as output_file:
                        pdf_writer.write(output_file)

                batch_files.append(batch_path)
                logger.info(f"Created batch: {batch_filename}")

                gc.collect()  # Proactive memory cleanup

            except Exception as e:
                logger.error(f"Failed to create batch {batch_num}: {e}")
                continue

        logger.info(f"Created {len(batch_files)} batch files")
        return batch_files, temp_dir

    except Exception as e:
        logger.error(f"Failed to split PDF: {e}")
        return [], None


def process_pdf_batches(batch_files: List[Path], output_dir: Path, original_name: str) -> Optional[Path]:
    """
    Process each batch PDF via process_pdf, then combine resulting markdowns.

    Parameters
    ----------
    batch_files : List[Path]
        Paths to batch PDFs.
    output_dir : Path
        Directory where outputs are written.
    original_name : str
        Base name used for the combined markdown file.

    Returns
    -------
    Path or None
        Path to the combined markdown if successful, else None.
    """
    all_markdown_content: List[str] = []
    batch_md_files: List[Path] = []

    for i, batch_file in enumerate(batch_files, 1):
        logger.info(f"Processing batch {i}/{len(batch_files)}: {batch_file.name}")
        logger.info(f"Memory usage: {get_memory_usage():.1f} MB")

        try:
            process_pdf(batch_file, output_dir)

            batch_md = output_dir / f"{batch_file.stem}.md"
            if batch_md.exists():
                with open(batch_md, "r", encoding="utf-8") as f:
                    content = f.read()
                all_markdown_content.append(content)
                batch_md_files.append(batch_md)
                logger.info(f"Processed batch {i}")
            else:
                logger.warning(f"No markdown generated for batch {i}")

            gc.collect()

        except Exception as e:
            logger.error(f"Failed to process batch {i}: {e}")
            continue

    if all_markdown_content:
        combined_md_path = output_dir / f"{original_name}.md"
        logger.info(f"Combining {len(all_markdown_content)} batch results into {combined_md_path.name}")

        with open(combined_md_path, "w", encoding="utf-8") as f:
            f.write(f"# {original_name}\n\n")
            for i, content in enumerate(all_markdown_content, 1):
                f.write(f"\n\n--- Batch {i} ---\n\n")
                f.write(content)

        logger.info(f"Combined markdown saved: {combined_md_path.name}")

        # Clean up per-batch markdown files
        for batch_md in batch_md_files:
            try:
                batch_md.unlink()
                logger.debug(f"Cleaned up: {batch_md.name}")
            except Exception:
                pass

        return combined_md_path

    return None


def process_wa5_safely() -> bool:
    """
    Orchestrate safe processing for WA_5_Redacted.pdf:
    - Split into batches
    - Process batches and combine markdown
    - Chunk markdowns
    - Generate embeddings for WAredacted collection
    """
    logger.info("PROCESSING WA_5_REDACTED.PDF SAFELY")
    logger.info("=" * 50)

    wa5_pdf = Path("../data/Contracts_data/WA/WA_5_Redacted.pdf")
    output_dir = Path("../outputs/pdf_parsed/WA")

    if not wa5_pdf.exists():
        logger.error(f"WA_5 PDF not found: {wa5_pdf}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Split and process PDF
        logger.info("Step 1: Split and process PDF")
        batch_files, temp_dir = split_large_pdf_processing(wa5_pdf, output_dir, max_pages_per_batch=5)

        if not batch_files:
            logger.error("Failed to create PDF batches")
            return False

        # Step 2: Process batches and combine
        combined_md = process_pdf_batches(batch_files, output_dir, "WA_5_Redacted")
        if not combined_md:
            logger.error("Failed to create combined markdown")
            return False

        # Step 3: Clean up temporary files
        if temp_dir and temp_dir.exists():
            logger.info("Cleaning up temporary files")
            for batch_file in batch_files:
                try:
                    batch_file.unlink()
                except Exception:
                    pass
            try:
                temp_dir.rmdir()
            except Exception:
                pass

        logger.info("WA_5 parsing completed successfully")

        # Step 4: Chunk markdowns
        logger.info("Step 2: Chunking WA_5 markdown")
        chunk_markdown_files(output_dir, output_dir)
        logger.info("WA_5 chunking completed")

        # Step 5: Generate embeddings
        logger.info("Step 3: Generating embeddings for WAredacted collection")

        available_model = check_and_get_embedding_model()
        if not available_model:
            logger.error("No embedding model available")
            return False

        process_contract_chunks_to_embeddings(
            chunks_dir=output_dir,
            collection_name="WAredacted",
            db_path="../chroma_db_qwen",
            model_name=available_model,
            batch_size=3,  # Small batch size for safety
        )

        logger.info("WA embeddings completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error processing WA_5: {e}")
        return False


def check_and_get_embedding_model() -> Optional[str]:
    """
    Inspect available Ollama models and select a preferred embedding model.
    Preference order: nomic-embed-text, mxbai-embed-large, all-minilm.
    Returns the chosen model name or None if none available.
    """
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models if isinstance(model, dict) and "name" in model]
            logger.info(f"Available models: {model_names}")

            preferred = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
            for model in preferred:
                if model in model_names:
                    logger.info(f"Using embedding model: {model}")
                    return model

            if model_names:
                fallback = model_names[0]
                logger.warning(f"Using fallback model: {fallback}")
                return fallback

        logger.error("No models available in Ollama")
        return None

    except Exception as e:
        logger.error(f"Error checking models: {e}")
        return None


def complete_wa_collection() -> bool:
    """
    Complete processing for the entire WA collection:
    - Ensure WA_5 is processed
    - Ensure markdowns are chunked
    - Generate embeddings for the full collection
    """
    logger.info("COMPLETING WA COLLECTION")
    logger.info("=" * 50)

    wa_output_dir = Path("../outputs/pdf_parsed/WA")

    # Current status
    md_files = list(wa_output_dir.glob("*.md"))
    json_files = list(wa_output_dir.glob("*_chunks.json"))
    logger.info("Current status:")
    logger.info(f"   Markdown files: {len(md_files)}")
    logger.info(f"   Chunk files: {len(json_files)}")

    # Process WA_5 if missing
    wa5_md = wa_output_dir / "WA_5_Redacted.md"
    if not wa5_md.exists():
        logger.info("WA_5 markdown missing - processing now")
        if not process_wa5_safely():
            logger.error("Failed to process WA_5")
            return False
    else:
        logger.info("WA_5 markdown already exists")

    # Ensure all files are chunked
    md_files = list(wa_output_dir.glob("*.md"))
    json_files = list(wa_output_dir.glob("*_chunks.json"))
    if len(md_files) > len(json_files):
        logger.info("Chunking remaining markdown files")
        chunk_markdown_files(wa_output_dir, wa_output_dir)

    # Generate embeddings
    logger.info("Generating embeddings for complete WA collection")

    available_model = check_and_get_embedding_model()
    if not available_model:
        logger.error("No embedding model available")
        return False

    try:
        process_contract_chunks_to_embeddings(
            chunks_dir=wa_output_dir,
            collection_name="WAredacted",
            db_path="../chroma_db_qwen",
            model_name=available_model,
            batch_size=3,
        )
        logger.info("WA collection completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Failed to generate WA embeddings: {e}")
        return False


def main() -> int:
    """Program entry point for WA collection processing."""
    logger.info("HANDLING LARGE WA_5 PDF")
    logger.info("=" * 60)

    success = complete_wa_collection()

    if success:
        logger.info("\nWA COLLECTION COMPLETED SUCCESSFULLY")
        logger.info("All WA files (including WA_5) processed and embedded in WAredacted collection")
    else:
        logger.error("\nWA COLLECTION PROCESSING FAILED")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
