#!/usr/bin/env python3
"""
PDF Embedding Generation for Multiple Collections

Generates embeddings for PDF chunks and stores them in ChromaDB collections:
- TNredacted: TN contract PDFs
- WAredacted: WA contract PDFs
- TNstandard: TN standard template
- WAstandard: WA standard template

Usage:
    python embedding.py --chunks-dir "../outputs/pdf_parsed/TN" --collection-name "TNredacted"
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import chromadb
from chromadb.config import Settings

# Suppress noisy warnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure structured logging (ASCII-only)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OllamaEmbeddingGenerator:
    """
    Generate embeddings via a local Ollama server.

    This class calls the Ollama embeddings API:
        POST {ollama_url}/api/embeddings
        payload: {"model": <model_name>, "prompt": <text>}
    """

    def __init__(self, model_name: str = "qwen3-embedding:0.6b", ollama_url: str = "http://localhost:11434") -> None:
        self.model_name: str = model_name
        self.ollama_url: str = ollama_url
        self.embed_url: str = f"{ollama_url}/api/embeddings"

        logger.info(f"Using Ollama model: {model_name}")
        logger.info(f"Ollama URL: {ollama_url}")

        # Validate connectivity and model availability
        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to the Ollama server and check if the model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models if isinstance(model, dict)]
                logger.info(f"Available Ollama models: {model_names}")

                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in available models")
                    logger.info("You may need to pull the model first: ollama pull qwen3-embedding:0.6b")
                else:
                    logger.info(f"Model {self.model_name} is available")
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding vector for the given text.

        Returns
        -------
        List[float]
            Normalized embedding vector. Returns an empty list on failure.
        """
        try:
            payload = {"model": self.model_name, "prompt": text}
            response = requests.post(self.embed_url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                if embedding:
                    vec = np.asarray(embedding, dtype=float)
                    norm = np.linalg.norm(vec)
                    if norm == 0.0 or not np.isfinite(norm):
                        logger.error("Received zero or non-finite norm embedding from Ollama")
                        return []
                    vec = vec / norm
                    return vec.tolist()
                logger.error("No embedding returned from Ollama")
                return []
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return []
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings sequentially for a batch of texts.

        Notes
        -----
        Uses a simple loop to avoid overwhelming a local Ollama server.
        Falls back to a zero vector of length 768 if generation fails.
        """
        embeddings: List[List[float]] = []
        for i, text in enumerate(texts):
            logger.info(f"Generating embedding {i + 1}/{len(texts)}")
            embedding = self.generate_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                logger.error(f"Failed to generate embedding for text {i + 1}")
                embeddings.append([0.0] * 768)  # Keep the original fallback dimension
        return embeddings


def prepare_contract_metadata_for_chroma(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare flattened metadata for ChromaDB (string, int, float, bool only).
    """
    content = chunk_data.get("content", "")
    header_path = chunk_data.get("header_path", "")

    metadata: Dict[str, Any] = {
        # Core identifiers
        "chunk_id": chunk_data.get("chunk_id", "unknown"),
        "page_number": chunk_data.get("page_number", 0),
        "chunk_index": chunk_data.get("chunk_index", 0),
        "source_file": chunk_data.get("source_file", "unknown"),

        # Content metadata
        "header_path": header_path,
        "token_count": chunk_data.get("token_count", 0),
        "document_type": chunk_data.get("document_type", "legal_contract"),

        # Embedding metadata
        "embedding_model": "qwen3-embedding:0.6b",
        "semantic_type": "contract_chunk",
        "domain": "legal_healthcare",

        # Content analysis flags
        "has_header": bool(header_path.strip() and header_path != "document_content"),
        "is_definition": "definition" in header_path.lower(),
        "is_clause": "clause:" in header_path.lower(),
        "is_section": "section:" in header_path.lower(),

        # Content type detection
        "contains_compensation": any(
            term in content.lower() for term in ["compensation", "payment", "reimbursement", "rate"]
        ),
        "contains_provider": "provider" in content.lower(),
        "contains_member": "member" in content.lower(),
        "contains_medicaid": "medicaid" in content.lower(),
        "contains_medicare": "medicare" in content.lower(),
        "contains_claims": "claim" in content.lower(),
        "contains_network": "network" in content.lower(),

        # Length indicators
        "is_short_chunk": chunk_data.get("token_count", 0) < 50,
        "is_long_chunk": chunk_data.get("token_count", 0) > 200,

        # Page context
        "is_first_page": chunk_data.get("page_number", 0) == 1,
        "page_range": f"page_{chunk_data.get('page_number', 0) // 10 * 10 + 1}-"
                      f"{(chunk_data.get('page_number', 0) // 10 + 1) * 10}",
    }

    return metadata


def load_contract_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """
    Load chunk JSON files from a directory.

    Expects files ending with *_chunks.json. Accepts either a list of chunks or a single chunk object.
    """
    chunks: List[Dict[str, Any]] = []
    json_files = list(chunks_dir.glob("*_chunks.json"))

    logger.info(f"Found {len(json_files)} JSON chunk files in {chunks_dir}")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                chunks.extend(data)
                logger.info(f"Loaded {len(data)} chunks from {json_file.name}")
            else:
                chunks.append(data)
                logger.info(f"Loaded 1 chunk from {json_file.name}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue

    logger.info(f"Total chunks loaded: {len(chunks)}")
    return chunks


def process_contract_chunks_to_embeddings(
    chunks_dir: Path,
    collection_name: str = "redacted",
    db_path: str = "../chroma_db_qwen",
    model_name: str = "qwen3-embedding:0.6b",
    batch_size: int = 5,
) -> None:
    """
    Main pipeline to embed contract chunks with Ollama and store in ChromaDB.
    """
    logger.info("Initializing Ollama embedding generator and ChromaDB client...")
    embedding_generator = OllamaEmbeddingGenerator(model_name=model_name)

    logger.info(f"Loading contract chunks from {chunks_dir}")
    chunks = load_contract_chunks(chunks_dir)
    if not chunks:
        logger.error("No chunks found to process")
        return

    # Prepare ChromaDB persistent client
    db_path_obj = Path(db_path)
    db_path_obj.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(db_path_obj),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

    # Replace any existing collection (preserves behavior of original script)
    replace_existing = True
    if replace_existing:
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            # Collection may not exist; ignore
            pass

    # Create the target collection
    try:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Legal contract document chunks with Qwen embeddings"},
        )
        logger.info(f"Created collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return

    # Process in batches to avoid overloading the local embedding server
    total = len(chunks)
    logger.info(f"Processing {total} chunks in batches of {batch_size}")

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts: List[str] = []
        batch_documents: List[str] = []
        batch_metadatas: List[Dict[str, Any]] = []
        batch_ids: List[str] = []

        logger.info(f"Processing batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

        for chunk_idx, chunk_data in enumerate(batch_chunks):
            content = chunk_data.get("content", "")
            header_path = chunk_data.get("header_path", "")

            # Include header context when available to improve embedding quality
            if header_path and header_path != "document_content":
                text_for_embedding = f"Section: {header_path} | Content: {content}"
            else:
                text_for_embedding = content

            batch_texts.append(text_for_embedding)
            batch_documents.append(content)
            batch_metadatas.append(prepare_contract_metadata_for_chroma(chunk_data))

            # Preserve original id selection behavior
            chunk_id = chunk_data.get("chunk_id", f"chunk_{i}_{len(batch_ids)}")
            batch_ids.append(chunk_id)

        # Generate embeddings
        logger.info(f"Generating embeddings for batch of {len(batch_texts)} texts using Ollama...")
        batch_embeddings = embedding_generator.generate_batch_embeddings(batch_texts)
        if not batch_embeddings or len(batch_embeddings) != len(batch_texts):
            logger.error(f"Failed to generate embeddings for batch {i // batch_size + 1}")
            continue

        # Persist to ChromaDB
        logger.info("Adding batch to ChromaDB...")
        try:
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
            logger.info(f"Successfully processed batch {i // batch_size + 1}")
        except Exception as e:
            logger.error(f"Failed to add batch {i // batch_size + 1} to ChromaDB: {e}")
            continue

    # Optional quick sanity check with a single query
    logger.info("Testing similarity search with contract embeddings...")
    if chunks:
        test_text = "provider compensation and payment terms"
        test_embedding = embedding_generator.generate_embedding(test_text)
        if test_embedding:
            results = collection.query(query_embeddings=[test_embedding], n_results=3)
            logger.info(f"Test query returned {len(results.get('documents', [[]])[0])} results")

            if results.get("documents") and results["documents"][0]:
                logger.info("Top result previews:")
                for j, (doc, metadata) in enumerate(
                    zip(results["documents"][0][:3], results["metadatas"][0][:3])
                ):
                    header = metadata.get("header_path", "No header")
                    page = metadata.get("page_number", "Unknown")
                    preview = (doc[:100] + "...") if len(doc) > 100 else doc
                    logger.info(f"  {j + 1}. Page {page}, {header}: {preview}")

    logger.info("Contract embedding generation completed successfully.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for contract JSON chunks using Ollama model"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="./outputs/chunked123",
        help="Directory containing JSON chunk files",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="redacted",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="../chroma_db_qwen",
        help="ChromaDB database path",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3-embedding:0.6b",
        help="Ollama model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for processing (smaller for Ollama)",
    )

    args = parser.parse_args()
    chunks_dir = Path(args.chunks_dir)

    if not chunks_dir.exists():
        raise SystemExit(f"Chunks directory not found: {chunks_dir}")

    process_contract_chunks_to_embeddings(
        chunks_dir=chunks_dir,
        collection_name=args.collection_name,
        db_path=args.db_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
