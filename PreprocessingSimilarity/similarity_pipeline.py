#!/usr/bin/env python3
"""
Similarity Processing Pipeline

Single script that combines preprocessing and embedding in one go:
1. Preprocesses precise similarity results from Ranker output
2. Generates paraphrase and Qwen embeddings 
3. Stores in ChromaDB collections

Usage (from anywhere):
    python PreprocessingSimilarity/similarity_pipeline.py

Features:
- Processes TNredacted, WAredacted, TNstandard, WAstandard datasets
- Text preprocessing with stopword removal, normalization
- Dual embeddings: Qwen (via Ollama) + Paraphrase (SentenceTransformers)
- ChromaDB storage in separate collections by type
"""

import json
import logging
import re
import unicodedata
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import argparse
import numpy as np
import requests

# ChromaDB
import chromadb
from chromadb.config import Settings

# Sentence-Transformers for paraphrase embeddings
from sentence_transformers import SentenceTransformer

# Optional imports
try:
    import ftfy
    _FTFY_AVAILABLE = True
except Exception:
    ftfy = None
    _FTFY_AVAILABLE = False

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get project root directory (2 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Configuration
CONFIG = {
    "input_dir": PROJECT_ROOT / "outputs/precise_similarity",
    "output_dir": PROJECT_ROOT / "outputs/precise_similarity/processed_datasets",
    "db_path": PROJECT_ROOT / "chroma_db_qwen",
    "ollama_url": "http://localhost:11434",
    "qwen_model": "qwen3-embedding:0.6b",
    "paraphrase_model": "paraphrase-MiniLM-L6-v2",
    "batch_size": 64
}

# Preprocessing constants
PREPROCESSING_VERSION = f"mtime:{datetime.fromtimestamp(Path(__file__).stat().st_mtime).isoformat()}"

_STOPWORDS = set([
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "can","can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each","few","for","from","further",
    "had","hadn't","has","hasn't","have","haven't","having","he","her","here","hers","herself","him","himself","his","how",
    "i","if","in","into","is","isn't","it","its","itself",
    "let's","me","more","most","mustn't","my","myself",
    "no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","she","should","shouldn't","so","some","such",
    "than","that","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too",
    "under","until","up","very",
    "was","wasn't","we","were","weren't","what","when","where","which","while","who","whom","why","with","won't","would","wouldn't",
    "you","your","yours","yourself","yourselves",
    "shall","may","including","without","limitation","pursuant","herein","thereof","therein","thereby",
    "such","any","all","within","upon","prior","thereafter","thereon","therefrom","hereby","herewith"
])

_NUMBER_WORDS = set([
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen",
    "twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety",
    "hundred","thousand","million","billion"
])

_MD_FENCE_RE = re.compile(r"`{3,}")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

_RED_PATTERNS = [
    re.compile(r"\[REDACTED\]", re.IGNORECASE),
    re.compile(r"x{3,}", re.IGNORECASE),
    re.compile(r"\u2588+"),
    re.compile(r"\[\[.+?\]\]")
]

class SimilarityPipeline:
    """Complete similarity processing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_dir = Path(config["input_dir"])
        self.output_dir = Path(config["output_dir"])
        self.db_path = Path(config["db_path"])
        
        # Initialize models
        self.ollama_generator = None
        self.paraphrase_model = None
        
    def preprocess_text(self, text: Optional[str]) -> str:
        """Preprocess text with normalization, stopword removal, etc."""
        if not text:
            return ""
        
        t = text
        if _FTFY_AVAILABLE:
            try:
                t = ftfy.fix_text(t)
            except Exception:
                pass
        
        t = unicodedata.normalize("NFKC", t)
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = t.replace("\u201c", '"').replace("\u201d", '"')
        t = t.lower()
        
        # Remove markdown/code fences and table pipes
        t = _MD_FENCE_RE.sub(" ", t)
        t = t.replace("|", " ")
        
        # Remove non-alphanumeric chars, keep spaces
        t = _NON_ALNUM_SPACE_RE.sub(" ", t)
        t = _MULTI_SPACE_RE.sub(" ", t).strip()
        
        if not t:
            return ""
        
        # Tokenize and process
        tokens = t.split()
        tokens = [tok for tok in tokens if tok not in _STOPWORDS]
        tokens = ["num" if any(ch.isdigit() for ch in tok) else tok for tok in tokens]
        tokens = ["num" if tok in _NUMBER_WORDS else tok for tok in tokens]
        
        return " ".join(tokens)
    
    def get_token_count(self, text: Optional[str]) -> Tuple[int, str]:
        """Get token count using tiktoken if available."""
        s = text or ""
        if _TIKTOKEN_AVAILABLE:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(s)), "tiktoken"
            except Exception:
                pass
        return len(s.split()), "split"
    
    def detect_redaction(self, text: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Detect redaction patterns in text."""
        if not text:
            return False, None
        for pat in _RED_PATTERNS:
            m = pat.search(text)
            if m:
                snippet = text[max(0, m.start()-20): m.end()+20]
                return True, snippet[:200]
        return False, None
    
    def compute_sha256(self, text: str) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    
    def enrich_row(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Add preprocessing fields to row."""
        row = dict(base)
        raw = base.get("full_content_raw") or ""
        preprocessed = self.preprocess_text(raw)
        
        tok_count, tok_name = self.get_token_count(preprocessed)
        has_redact, red_snip = self.detect_redaction(raw)
        sha = self.compute_sha256(preprocessed)
        
        row["preprocessed_final_content"] = preprocessed
        row["preprocessed_token_count"] = tok_count
        row["preprocessed_sha256"] = sha
        row["preprocessing_version"] = PREPROCESSING_VERSION
        row["has_redaction"] = has_redact
        row["redaction_snippet"] = red_snip
        
        return row
    
    def step1_preprocess_datasets(self) -> bool:
        """Step 1: Preprocess similarity results to datasets."""
        logger.info("🔄 Step 1: Preprocessing similarity results")
        
        # Input files
        input_files = [
            "TNredacted_precise_attribute_similarities.json",
            "WAredacted_precise_attribute_similarities.json", 
            "TNstandard_precise_attribute_similarities.json",
            "WAstandard_precise_attribute_similarities.json"
        ]
        
        # Validate inputs
        missing = []
        for fname in input_files:
            fpath = self.input_dir / fname
            if not fpath.exists():
                missing.append(fpath)
        
        if missing:
            for p in missing:
                logger.error(f"Missing input: {p}")
            return False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for fname in input_files:
            fpath = self.input_dir / fname
            logger.info(f"Processing {fname}")
            
            if "redacted" in fname:
                self.process_redacted_file(fpath)
            else:
                self.process_standard_file(fpath)
        
        logger.info("✅ Step 1 Complete: Preprocessing finished")
        return True
    
    def process_redacted_file(self, sim_path: Path):
        """Process redacted similarity file into per-source datasets."""
        with open(sim_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        collection_info = data.get("collection_info", {})
        collection_key = collection_info.get("collection_key", sim_path.stem)
        collection_name = collection_info.get("collection_name", collection_key)
        
        per_source_rows = {}
        matches = data.get("matches", {}) or {}
        
        # Per-attribute limits
        per_attr_limit = {1: 7, 2: 7, 3: 3, 4: 5, 5: 5}
        
        for attr_name, attr_data in matches.items():
            attr_info = attr_data.get("attribute_info", {})
            attr_num = attr_info.get("number", 0)
            
            # Group by source file
            by_source = {}
            for match in attr_data.get("matches", []):
                source_file = match.get("source_file", "unknown")
                by_source.setdefault(source_file, []).append(match)
            
            # Process each source
            for source_file, match_list in by_source.items():
                limit = per_attr_limit.get(attr_num, len(match_list))
                for m in match_list[:limit]:
                    base = {
                        "collection_key": collection_key,
                        "collection_name": collection_name,
                        "source_file": source_file,
                        "attribute_number": attr_num,
                        "attribute_name": attr_name,
                        "rank_in_global": m.get("rank_in_global"),
                        "page": m.get("page"),
                        "section": m.get("section"),
                        "chunk_id": m.get("chunk_id"),
                        "rrf_score": m.get("rrf_score"),
                        "content_preview": m.get("content_preview"),
                        "full_content_raw": m.get("full_content")
                    }
                    
                    # Add score breakdown
                    sb = m.get("score_breakdown", {}) or {}
                    base["dense_similarity"] = sb.get("dense_similarity")
                    base["bm25_score"] = sb.get("bm25_score")
                    
                    enriched = self.enrich_row(base)
                    per_source_rows.setdefault(source_file, []).append(enriched)
        
        # Save per-source datasets
        for source_file, rows in per_source_rows.items():
            src_slug = Path(source_file).stem
            src_slug = re.sub(r"[^A-Za-z0-9_-]", "_", src_slug)
            out_path = self.output_dir / f"{collection_key}_{src_slug}_dataset.json"
            self.save_json(out_path, rows)
    
    def process_standard_file(self, sim_path: Path):
        """Process standard similarity file into single dataset."""
        with open(sim_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        collection_info = data.get("collection_info", {})
        collection_key = collection_info.get("collection_key", sim_path.stem)
        collection_name = collection_info.get("collection_name", collection_key)
        
        rows = []
        matches = data.get("matches", {}) or {}
        
        per_attr_limit = {1: 7, 2: 7, 3: 3, 4: 5, 5: 5}
        
        for attr_name, attr_data in matches.items():
            attr_info = attr_data.get("attribute_info", {})
            attr_num = attr_info.get("number", 0)
            
            top_matches = attr_data.get("top_matches") or attr_data.get("matches") or []
            limit = per_attr_limit.get(attr_num, len(top_matches))
            
            for m in top_matches[:limit]:
                base = {
                    "collection_key": collection_key,
                    "collection_name": collection_name,
                    "attribute_number": attr_num,
                    "attribute_name": attr_name,
                    "rank": m.get("rank"),
                    "page": m.get("page"),
                    "section": m.get("section"),
                    "chunk_id": m.get("chunk_id"),
                    "source_file": m.get("source_file"),
                    "rrf_score": m.get("rrf_score"),
                    "content_preview": m.get("content_preview"),
                    "full_content_raw": m.get("full_content")
                }
                
                # Add score breakdown
                sb = m.get("score_breakdown", {}) or {}
                base["dense_similarity"] = sb.get("dense_similarity")
                base["bm25_score"] = sb.get("bm25_score")
                
                enriched = self.enrich_row(base)
                rows.append(enriched)
        
        # Save dataset
        out_path = self.output_dir / f"{collection_key}_dataset.json"
        self.save_json(out_path, rows)
    
    def save_json(self, path: Path, obj: Any):
        """Save JSON with logging."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote: {path}")
    
    def step2_generate_embeddings(self) -> bool:
        """Step 2: Generate embeddings and store in ChromaDB."""
        logger.info("🔄 Step 2: Generating embeddings and storing in ChromaDB")
        
        # Initialize models
        self.init_models()
        
        # Setup ChromaDB
        client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        # Create collections
        collections = {
            "similarityTemplate_qwen": client.get_or_create_collection("similarityTemplate_qwen"),
            "similarityTemplate_para": client.get_or_create_collection("similarityTemplate_para"),
            "SimilarityRedacted_qwen": client.get_or_create_collection("SimilarityRedacted_qwen"),
            "SimilarityRedacted_para": client.get_or_create_collection("SimilarityRedacted_para")
        }
        
        # Process dataset files
        dataset_files = list(self.output_dir.glob("*.json"))
        standard_files = []
        redacted_files = []
        
        for fp in dataset_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                if rows and rows[0].get("collection_key") in {"TNstandard", "WAstandard"}:
                    standard_files.append(fp)
                else:
                    redacted_files.append(fp)
            except Exception:
                logger.warning(f"Skipping unreadable file: {fp}")
        
        # Process standard files
        for fp in standard_files:
            self.process_file_embeddings(
                fp, 
                collections["similarityTemplate_qwen"],
                collections["similarityTemplate_para"]
            )
        
        # Process redacted files  
        for fp in redacted_files:
            self.process_file_embeddings(
                fp,
                collections["SimilarityRedacted_qwen"], 
                collections["SimilarityRedacted_para"]
            )
        
        logger.info("✅ Step 2 Complete: Embeddings generated and stored")
        return True
    
    def init_models(self):
        """Initialize embedding models."""
        logger.info("Initializing embedding models...")
        
        # Ollama Qwen embeddings
        self.ollama_generator = OllamaEmbeddingGenerator(
            model_name=self.config["qwen_model"],
            ollama_url=self.config["ollama_url"]
        )
        
        # Paraphrase embeddings
        self.paraphrase_model = SentenceTransformer(self.config["paraphrase_model"])
        logger.info("✅ Models initialized")
    
    def process_file_embeddings(self, fp: Path, qwen_collection, para_collection):
        """Process embeddings for a single file."""
        logger.info(f"Processing embeddings for {fp.name}")
        
        with open(fp, "r", encoding="utf-8") as f:
            rows = json.load(f)
        
        if not rows:
            return
        
        # Prepare texts and IDs
        texts = [r.get("preprocessed_final_content") or "" for r in rows]
        base_ids = [self.make_id(r) for r in rows]
        
        # Generate paraphrase embeddings (batched)
        logger.info(f"Generating paraphrase embeddings for {len(texts)} texts...")
        para_embs = self.paraphrase_model.encode(
            texts, 
            normalize_embeddings=True, 
            batch_size=self.config["batch_size"],
            show_progress_bar=False
        )
        
        # Generate Qwen embeddings (sequential)
        logger.info(f"Generating Qwen embeddings for {len(texts)} texts...")
        qwen_embs = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                logger.info(f"Qwen embedding {i+1}/{len(texts)}")
            emb = self.ollama_generator.embed_one(text)
            qwen_embs.append(emb)
        
        # Prepare data for ChromaDB
        qwen_data = {"embeddings": [], "documents": [], "metadatas": [], "ids": []}
        para_data = {"embeddings": [], "documents": [], "metadatas": [], "ids": []}
        
        for i, row in enumerate(rows):
            doc = row.get("preprocessed_final_content") or ""
            
            # Qwen data
            if qwen_embs[i]:
                qwen_data["embeddings"].append(qwen_embs[i])
                qwen_data["documents"].append(doc)
                qwen_data["metadatas"].append(self.build_metadata(row, "qwen"))
                qwen_data["ids"].append(base_ids[i] + "::qwen")
            
            # Paraphrase data
            if len(para_embs[i]) > 0:
                para_data["embeddings"].append(para_embs[i].tolist())
                para_data["documents"].append(doc)
                para_data["metadatas"].append(self.build_metadata(row, "paraphrase"))
                para_data["ids"].append(base_ids[i] + "::para")
        
        # Add to ChromaDB
        if qwen_data["embeddings"]:
            qwen_collection.add(**qwen_data)
            logger.info(f"Added {len(qwen_data['embeddings'])} Qwen embeddings")
        
        if para_data["embeddings"]:
            para_collection.add(**para_data)
            logger.info(f"Added {len(para_data['embeddings'])} paraphrase embeddings")
    
    def make_id(self, row: Dict[str, Any]) -> str:
        """Generate stable ID for row."""
        sha = (row.get("preprocessed_sha256") or "")[:16]
        chunk = row.get("chunk_id") or "chunk"
        attr = str(row.get("attribute_number") or "attr")
        col = row.get("collection_key") or "col"
        src = Path(str(row.get("source_file") or "src")).stem
        page = str(row.get("page") or 0)
        return f"proc::{col}::{attr}::{src}::{chunk}::{page}::{sha}"
    
    def build_metadata(self, row: Dict[str, Any], embedding_type: str) -> Dict[str, Any]:
        """Build metadata for ChromaDB."""
        return {
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
            "embedding_type": embedding_type,
            "embedding_source": "processed_dataset"
        }
    
    def run_pipeline(self) -> bool:
        """Run the complete similarity processing pipeline."""
        logger.info("🚀 Starting Similarity Processing Pipeline")
        logger.info("=" * 60)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"ChromaDB path: {self.db_path}")
        
        # Step 1: Preprocess datasets
        if not self.step1_preprocess_datasets():
            logger.error("❌ Pipeline failed at Step 1: Preprocessing")
            return False
        
        # Step 2: Generate embeddings
        if not self.step2_generate_embeddings():
            logger.error("❌ Pipeline failed at Step 2: Embeddings")
            return False
        
        logger.info("=" * 60)
        logger.info("🎉 Similarity Processing Pipeline Completed!")
        logger.info(f"📁 Processed datasets: {self.output_dir}")
        logger.info(f"🗄️ ChromaDB collections updated")
        
        return True

class OllamaEmbeddingGenerator:
    """Ollama embedding generator for Qwen."""
    
    def __init__(self, model_name: str, ollama_url: str):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.embed_url = f"{self.ollama_url}/api/embeddings"
        self._test_connection()
    
    def _test_connection(self):
        """Test Ollama connection."""
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if r.status_code == 200:
                names = [m.get("name") for m in r.json().get("models", [])]
                if self.model_name not in names:
                    logger.warning(f"Model {self.model_name} not found. Try: ollama pull {self.model_name}")
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {e}")
    
    def embed_one(self, text: str) -> List[float]:
        """Generate single embedding."""
        try:
            payload = {"model": self.model_name, "prompt": text}
            r = requests.post(self.embed_url, json=payload, timeout=30)
            if r.status_code != 200:
                logger.error(f"Ollama error {r.status_code}")
                return []
            
            data = r.json()
            vec = data.get("embedding") or []
            if not vec:
                return []
            
            # Normalize
            v = np.array(vec, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            return v.tolist()
        except Exception as e:
            logger.error(f"Ollama embed error: {e}")
            return []

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Similarity Processing Pipeline")
    parser.add_argument("--input-dir", default=str(CONFIG["input_dir"]), help="Input directory")
    parser.add_argument("--output-dir", default=str(CONFIG["output_dir"]), help="Output directory") 
    parser.add_argument("--db-path", default=str(CONFIG["db_path"]), help="ChromaDB path")
    parser.add_argument("--ollama-url", default=CONFIG["ollama_url"], help="Ollama URL")
    parser.add_argument("--qwen-model", default=CONFIG["qwen_model"], help="Qwen model name")
    parser.add_argument("--paraphrase-model", default=CONFIG["paraphrase_model"], help="Paraphrase model name")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="Batch size")
    
    args = parser.parse_args()
    
    # Update config with args
    config = CONFIG.copy()
    config.update({
        "input_dir": Path(args.input_dir),
        "output_dir": Path(args.output_dir),
        "db_path": Path(args.db_path),
        "ollama_url": args.ollama_url,
        "qwen_model": args.qwen_model,
        "paraphrase_model": args.paraphrase_model,
        "batch_size": args.batch_size
    })
    
    logger.info("Similarity Processing Pipeline")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # Run pipeline
    pipeline = SimilarityPipeline(config)
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("✅ Pipeline completed successfully!")
        return 0
    else:
        logger.error("❌ Pipeline failed!")
        return 1

if __name__ == "__main__":
    exit(main())
