#!/usr/bin/env python3
"""
Generate paraphrase and Qwen embeddings for processed datasets and store in ChromaDB.

Inputs (default): outputs/precise_similarity/processed_datasets/
- TNstandard_dataset.json
- WAstandard_dataset.json
- TNredacted_*.json (≈5 files)
- WAredacted_*.json (≈5 files)

Collections:
- similarityTemplate   -> for TNstandard, WAstandard (Qwen vector; paraphrase vector saved in metadata)
- SimilarityRedacted   -> for TNredacted_*, WAredacted_* (Qwen vector; paraphrase vector saved in metadata)

Notes:
- We store ONE vector per record in Chroma (Qwen). Paraphrase embeddings are stored as metadata to avoid
  mixed vector dimensions in a single collection.
- You can still create a separate collection for paraphrase vectors if needed later.

CLI:
  python similarity_reprocessing_embedding/generate_processed_embeddings.py \
    --input-dir outputs/precise_similarity/processed_datasets \
    --db-path chroma_db_qwen \
    --ollama-url http://localhost:11434 \
    --qwen-model qwen3-embedding:0.6b \
    --paraphrase-model paraphrase-MiniLM-L6-v2 \
    --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import uuid
import numpy as np

import chromadb
from chromadb.config import Settings

# Sentence-Transformers for paraphrase embeddings
from sentence_transformers import SentenceTransformer

# HTTP for Ollama
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Ollama Qwen embedding helper --------------------
class OllamaEmbeddingGenerator:
    def __init__(self, model_name: str = "qwen3-embedding:0.6b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.embed_url = f"{self.ollama_url}/api/embeddings"
        self._test_connection()

    def _test_connection(self) -> None:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if r.status_code == 200:
                names = [m.get("name") for m in r.json().get("models", [])]
                if self.model_name not in names:
                    logger.warning(f"Model {self.model_name} not found in Ollama tags. Try: ollama pull {self.model_name}")
            else:
                logger.warning(f"Ollama tags status: {r.status_code}")
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {e}")

    def embed_one(self, text: str) -> List[float]:
        try:
            payload = {"model": self.model_name, "prompt": text}
            r = requests.post(self.embed_url, json=payload, timeout=30)
            if r.status_code != 200:
                logger.error(f"Ollama error {r.status_code}: {r.text[:200]}")
                return []
            data = r.json()
            vec = data.get("embedding") or []
            if not vec:
                return []
            v = np.array(vec, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            return v.tolist()
        except Exception as e:
            logger.error(f"Ollama embed error: {e}")
            return []

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        embs: List[List[float]] = []
        for i, t in enumerate(texts):
            logger.info(f"Qwen embedding {i+1}/{len(texts)}")
            e = self.embed_one(t)
            if not e:
                embs.append([])
            else:
                embs.append(e)
        return embs

# -------------------- Chroma helpers --------------------
class ChromaManager:
    def __init__(self, db_path: Path, collection_name: str, description: str):
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Using existing Chroma collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": description}
            )
            logger.info(f"Created Chroma collection: {collection_name}")

    def add(self, embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        # Filter out any empty embeddings (keep indices aligned)
        good_idx = [i for i, e in enumerate(embeddings) if e]
        if not good_idx:
            return
        embs = [embeddings[i] for i in good_idx]
        docs = [documents[i] for i in good_idx]
        metas = [metadatas[i] for i in good_idx]
        idz = [ids[i] for i in good_idx]
        self.collection.add(embeddings=embs, documents=docs, metadatas=metas, ids=idz)

# -------------------- Core logic --------------------
STANDARD_KEYS = {"TNstandard", "WAstandard"}


def is_standard_row(row: Dict[str, Any]) -> bool:
    return row.get("collection_key") in STANDARD_KEYS


def build_metadata(row: Dict[str, Any], embedding_model: str, embedding_type: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "collection_key": row.get("collection_key"),
        "collection_name": row.get("collection_name"),
        "attribute_number": row.get("attribute_number"),
        "attribute_name": row.get("attribute_name"),
        "source_file": row.get("source_file"),
        "page": row.get("page"),
        "section": row.get("section"),
        "chunk_id": row.get("chunk_id"),
        "rrf_score": row.get("rrf_score"),
        "bm25_score": row.get("bm25_score"),
        "preprocessed_token_count": row.get("preprocessed_token_count"),
        "has_redaction": row.get("has_redaction"),
        "preprocessed_sha256": row.get("preprocessed_sha256"),
        "preprocessing_version": row.get("preprocessing_version"),
        "embedding_model": embedding_model,
        "embedding_type": embedding_type,  # "qwen" or "paraphrase_in_metadata"
        "embedding_source": "processed_dataset",
    }
    return meta


def load_rows(file_path: Path) -> List[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        return []


def dataset_files(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    files = sorted(input_dir.glob("*.json"))
    std_files: List[Path] = []
    red_files: List[Path] = []
    for fp in files:
        try:
            rows = load_rows(fp)
            if not rows:
                continue
            ck = rows[0].get("collection_key")
            if ck in STANDARD_KEYS:
                std_files.append(fp)
            else:
                red_files.append(fp)
        except Exception:
            logger.warning(f"Skipping unreadable file: {fp}")
    return std_files, red_files


def make_id(row: Dict[str, Any]) -> str:
    # Prefer stable id using sha, fallback to uuid
    sha = (row.get("preprocessed_sha256") or "")[:16]
    chunk = row.get("chunk_id") or "chunk"
    attr = str(row.get("attribute_number") or "attr")
    col = row.get("collection_key") or "col"
    src = Path(str(row.get("source_file") or "src")).stem
    page = str(row.get("page") or 0)
    base = f"proc::{col}::{attr}::{src}::{chunk}::{page}::{sha}"
    return base


def process_and_store(
    input_dir: Path,
    db_path: Path,
    ollama_url: str,
    qwen_model: str,
    paraphrase_model_name: str,
    batch_size: int,
):
    # Collections (separate per embedding dimension)
    tmplt_qwen = ChromaManager(db_path, "similarityTemplate_qwen", "Standard template datasets with Qwen vectors")
    tmplt_para = ChromaManager(db_path, "similarityTemplate_para", "Standard template datasets with paraphrase vectors")
    redct_qwen = ChromaManager(db_path, "SimilarityRedacted_qwen", "Redacted datasets with Qwen vectors")
    redct_para = ChromaManager(db_path, "SimilarityRedacted_para", "Redacted datasets with paraphrase vectors")

    # Models
    logger.info("Loading paraphrase model...")
    para_model = SentenceTransformer(paraphrase_model_name)

    qwen = OllamaEmbeddingGenerator(model_name=qwen_model, ollama_url=ollama_url)

    std_files, red_files = dataset_files(input_dir)
    logger.info(f"Found {len(std_files)} standard datasets, {len(red_files)} redacted datasets")

    def handle_file(fp: Path, q_target: ChromaManager, p_target: ChromaManager):
        rows = load_rows(fp)
        if not rows:
            logger.warning(f"No rows in {fp.name}")
            return
        # Prepare texts
        texts = [r.get("preprocessed_final_content") or "" for r in rows]
        ids = [make_id(r) for r in rows]

        # Paraphrase embeddings (batched)
        logger.info(f"Encoding paraphrase embeddings for {fp.name} ({len(texts)} rows)...")
        para_embs = para_model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        if isinstance(para_embs, list):
            para_embs = np.array(para_embs)

        # Qwen embeddings (sequential)
        logger.info(f"Encoding Qwen embeddings for {fp.name} ...")
        qwen_embs = qwen.embed_many(texts)

        # Prepare and add Qwen records
        q_docs: List[str] = []
        q_metas: List[Dict[str, Any]] = []
        q_ids: List[str] = []
        q_vecs: List[List[float]] = []

        # Prepare and add Paraphrase records
        p_docs: List[str] = []
        p_metas: List[Dict[str, Any]] = []
        p_ids: List[str] = []
        p_vecs: List[List[float]] = []

        for i, row in enumerate(rows):
            doc = texts[i]
            if not doc:
                continue

            # Qwen
            qvec = qwen_embs[i] if i < len(qwen_embs) else []
            q_meta = build_metadata(row, embedding_model=qwen_model, embedding_type="qwen")
            q_id = ids[i] + "::qwen"
            q_docs.append(doc)
            q_metas.append(q_meta)
            q_ids.append(q_id)
            q_vecs.append(qvec)

            # Paraphrase
            pvec_np = para_embs[i]
            pvec = pvec_np.tolist() if hasattr(pvec_np, "tolist") else list(pvec_np)
            p_meta = build_metadata(row, embedding_model="paraphrase-MiniLM-L6-v2", embedding_type="paraphrase")
            p_id = ids[i] + "::para"
            p_docs.append(doc)
            p_metas.append(p_meta)
            p_ids.append(p_id)
            p_vecs.append(pvec)

        # Add to Chroma
        logger.info(f"Adding {len(q_ids)} Qwen records from {fp.name} to Chroma ({q_target.collection.name})...")
        q_target.add(embeddings=q_vecs, documents=q_docs, metadatas=q_metas, ids=q_ids)
        logger.info(f"Adding {len(p_ids)} Paraphrase records from {fp.name} to Chroma ({p_target.collection.name})...")
        p_target.add(embeddings=p_vecs, documents=p_docs, metadatas=p_metas, ids=p_ids)

    # Standard datasets -> similarityTemplate
    for fp in std_files:
        handle_file(fp, tmplt_qwen, tmplt_para)

    # Redacted datasets -> SimilarityRedacted
    for fp in red_files:
        handle_file(fp, redct_qwen, redct_para)

    logger.info("✅ Finished embedding generation for processed datasets")


def main():
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Generate embeddings (paraphrase + Qwen) for processed datasets into ChromaDB")
    parser.add_argument("--input-dir", default=str(repo_root / "outputs/precise_similarity/processed_datasets"), help="Directory of processed datasets")
    parser.add_argument("--db-path", default=str(repo_root / "chroma_db_qwen"), help="ChromaDB path")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--qwen-model", default="qwen3-embedding:0.6b", help="Qwen embedding model name")
    parser.add_argument("--paraphrase-model", default="paraphrase-MiniLM-L6-v2", help="Sentence-Transformers model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for paraphrase model")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset collections before ingesting")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Allow fallback to 'preprocess_data' if requested directory doesn't exist
    if not input_dir.exists():
        alt = repo_root / "outputs/precise_similarity/preprocess_data"
        if alt.exists():
            logger.warning(f"Input dir not found: {input_dir}, falling back to {alt}")
            input_dir = alt
        else:
            raise SystemExit(f"Input directory not found: {input_dir}")

    db_path = Path(args.db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    # Reset collections unless skipped
    if not args.no_reset:
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        to_delete = [
            # legacy names (runs before split)
            "similarityTemplate",
            "SimilarityRedacted",
            # new split collections
            "similarityTemplate_qwen",
            "similarityTemplate_para",
            "SimilarityRedacted_qwen",
            "SimilarityRedacted_para",
        ]
        for name in to_delete:
            try:
                client.delete_collection(name)
                logger.info(f"Deleted existing collection: {name}")
            except Exception:
                pass

    process_and_store(
        input_dir=input_dir,
        db_path=db_path,
        ollama_url=args.ollama_url,
        qwen_model=args.qwen_model,
        paraphrase_model_name=args.paraphrase_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
