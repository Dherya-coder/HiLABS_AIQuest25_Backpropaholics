#!/usr/bin/env python3
"""
Attribute Precise Similarity Pipeline

Requirements from user:
- Use embeddings from attribute collection (attributes_simple).
- For each attribute:
  - TNstandard: top 20 clauses
  - WAstandard: top 20 clauses
  - TNredacted: top 15 clauses PER SOURCE PDF (5 PDFs)
  - WAredacted: top 15 clauses PER SOURCE PDF (5 PDFs)
- Store results in outputs/precise_similarity/
- Use Ranker/rrf_attribute_matcher.py logic (RRFAttributeMatcher) for scoring.

This script directly leverages the public methods of RRFAttributeMatcher to compute
Dense, BM25, and fused RRF rankings, then post-processes results by source file where needed.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple
import traceback

# Add utils directory to path
sys.path.append(str(Path(__file__).parent / "utils"))

from rrf_attribute_matcher import RRFAttributeMatcher

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = str(PROJECT_ROOT / "chroma_db_qwen")
OUTPUT_DIR = PROJECT_ROOT / "outputs/precise_similarity"
ATTRIBUTE_COLLECTION = "attributes_simple"

# Collections configuration
STANDARD_COLLECTIONS = {
    "TNstandard": {"collection_name": "TNstandard", "top_k": 20, "group_per_source": False},
    "WAstandard": {"collection_name": "WAstandard", "top_k": 20, "group_per_source": False},
}

REDACTED_COLLECTIONS = {
    "TNredacted": {"collection_name": "TNredacted", "per_pdf_top_k": 15, "group_per_source": True},
    "WAredacted": {"collection_name": "WAredacted", "per_pdf_top_k": 15, "group_per_source": True},
}


def _format_match(doc_idx: int, rrf_score: float, details: Dict[str, float], matcher: RRFAttributeMatcher) -> Dict[str, Any]:
    """Build a match record from indices and details."""
    doc = matcher.contract_documents[doc_idx]
    meta = matcher.contract_metadatas[doc_idx] if doc_idx < len(matcher.contract_metadatas) else {}

    # Section fallback
    section = meta.get("header_path", "") or "No header"

    return {
        "rrf_score": float(rrf_score),
        "page": int(meta.get("page_number", 0) or 0),
        "section": section,
        "chunk_id": meta.get("chunk_id", ""),
        "source_file": meta.get("source_file", "unknown"),
        "content_preview": (doc[:150] + "...") if isinstance(doc, str) and len(doc) > 150 else doc,
        "full_content": doc,
        "score_breakdown": {
            "dense_similarity": float(details.get("dense_score", 0.0)),
            "bm25_score": float(details.get("bm25_score", 0.0)),
            "dense_rank": int(details.get("dense_rank", 0) or 0),
            "bm25_rank": int(details.get("bm25_rank", 0) or 0),
            "rrf_contribution_dense": float(details.get("dense_rrf", 0.0)),
            "rrf_contribution_bm25": float(details.get("bm25_rrf", 0.0)),
        },
    }


def _get_attributes(matcher: RRFAttributeMatcher) -> Tuple[List[str], List[Dict[str, Any]], List[List[float]]]:
    """Fetch attribute docs, metadatas, and embeddings."""
    try:
        attr_results = matcher.attr_collection.get(include=["documents", "metadatas", "embeddings"])
        docs = attr_results.get("documents", [])
        metas = attr_results.get("metadatas", [])
        embeds = attr_results.get("embeddings", [])
        return docs, metas, embeds
    except Exception as e:
        logger.error(f"Failed to fetch attributes: {e}")
        raise


def _rank_for_attribute(matcher: RRFAttributeMatcher, attr_doc: str, attr_embedding: List[float], full_top_k: int) -> List[Tuple[int, float, Dict[str, float]]]:
    """Compute fused RRF ranking for a single attribute; return up to full_top_k results."""
    dense_scores = matcher.get_dense_similarity_scores(attr_embedding)  # [(doc_idx, sim)]
    bm25_scores = matcher.get_bm25_scores(attr_doc)                     # [(doc_idx, score)]

    # Ask for as many as needed (cap by corpus size)
    corpus_size = len(matcher.contract_documents)
    k = min(full_top_k, corpus_size)
    rrf_scores = matcher.apply_rrf(dense_scores, bm25_scores, top_k=k)
    return rrf_scores


def process_standard_collection(collection_key: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Top-K across entire collection for each attribute (no per-PDF grouping)."""
    collection_name = config["collection_name"]
    top_k = int(config["top_k"])

    matcher = RRFAttributeMatcher(db_path=DB_PATH)
    matcher.load_collections(ATTRIBUTE_COLLECTION, collection_name)

    attr_docs, attr_metas, attr_embeds = _get_attributes(matcher)

    results: Dict[str, Any] = {
        "collection_info": {
            "collection_key": collection_key,
            "collection_name": collection_name,
            "mode": "global_top_k",
            "top_k": top_k,
            "processed_at": datetime.now().isoformat(),
        },
        "summary": {
            "method": "RRF (Dense + BM25)",
            "total_attributes": len(attr_docs),
            "total_contracts": len(matcher.contract_documents),
            "rrf_k": matcher.rrf_k,
        },
        "matches": {},
    }

    for i, (doc, meta, emb) in enumerate(zip(attr_docs, attr_metas, attr_embeds), 1):
        attr_name = (meta or {}).get("attribute_name", f"Attribute_{i}")
        attr_num = (meta or {}).get("attribute_number", i)

        logger.info(f"[{collection_key}] Attribute {i}/{len(attr_docs)}: {attr_name}")
        try:
            ranked = _rank_for_attribute(matcher, doc, emb, top_k)

            matches = []
            for rank, (doc_idx, rrf_score, details) in enumerate(ranked, 1):
                m = _format_match(doc_idx, rrf_score, details, matcher)
                m["rank"] = rank
                matches.append(m)

            # Statistics
            rrf_scores = [m["rrf_score"] for m in matches]
            dense_scores = [m["score_breakdown"]["dense_similarity"] for m in matches]
            bm25_scores = [m["score_breakdown"]["bm25_score"] for m in matches]

            results["matches"][attr_name] = {
                "attribute_info": {
                    "number": attr_num,
                    "name": attr_name,
                    "content_preview": (doc[:100] + "...") if len(doc) > 100 else doc,
                },
                "statistics": {
                    "avg_rrf_score": float(sum(rrf_scores) / len(rrf_scores)) if rrf_scores else 0.0,
                    "max_rrf_score": float(max(rrf_scores)) if rrf_scores else 0.0,
                    "avg_dense_score": float(sum(dense_scores) / len(dense_scores)) if dense_scores else 0.0,
                    "avg_bm25_score": float(sum(bm25_scores) / len(bm25_scores)) if bm25_scores else 0.0,
                    "high_rrf_matches": int(len([s for s in rrf_scores if s > 0.01])),
                },
                "top_matches": matches,
            }
        except Exception as e:
            logger.error(f"Error processing attribute '{attr_name}' for {collection_key}: {e}")
            logger.error(traceback.format_exc())
            continue

    return results


def process_redacted_collection(collection_key: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Top-N per source_file (PDF) for each attribute."""
    collection_name = config["collection_name"]
    per_pdf_top_k = int(config["per_pdf_top_k"])  # 15

    matcher = RRFAttributeMatcher(db_path=DB_PATH)
    matcher.load_collections(ATTRIBUTE_COLLECTION, collection_name)

    attr_docs, attr_metas, attr_embeds = _get_attributes(matcher)

    # Build index mapping from document index to its source_file
    source_for_idx: Dict[int, str] = {}
    for idx, meta in enumerate(matcher.contract_metadatas):
        source_for_idx[idx] = (meta or {}).get("source_file", "unknown")

    results: Dict[str, Any] = {
        "collection_info": {
            "collection_key": collection_key,
            "collection_name": collection_name,
            "mode": "per_pdf_top_k",
            "per_pdf_top_k": per_pdf_top_k,
            "processed_at": datetime.now().isoformat(),
        },
        "summary": {
            "method": "RRF (Dense + BM25)",
            "total_attributes": len(attr_docs),
            "total_contracts": len(matcher.contract_documents),
            "rrf_k": matcher.rrf_k,
        },
        "matches": {},
    }

    for i, (doc, meta, emb) in enumerate(zip(attr_docs, attr_metas, attr_embeds), 1):
        attr_name = (meta or {}).get("attribute_name", f"Attribute_{i}")
        attr_num = (meta or {}).get("attribute_number", i)

        logger.info(f"[{collection_key}] Attribute {i}/{len(attr_docs)}: {attr_name}")
        try:
            # Rank across the full corpus to allow per-PDF selection
            full_k = len(matcher.contract_documents)
            ranked = _rank_for_attribute(matcher, doc, emb, full_k)

            # Group top matches by source_file
            per_source: Dict[str, List[Dict[str, Any]]] = {}
            # Maintain counters per source
            counts: Dict[str, int] = {}

            for (_, (doc_idx, rrf_score, details)) in enumerate(ranked, 1):
                src = source_for_idx.get(doc_idx, "unknown")
                counts.setdefault(src, 0)
                if counts[src] >= per_pdf_top_k:
                    continue

                match_obj = _format_match(doc_idx, rrf_score, details, matcher)
                match_obj["rank_in_global"] = len(per_source.get(src, [])) + 1

                per_source.setdefault(src, []).append(match_obj)
                counts[src] += 1

                # Early stop if we already have desired totals for 5 PDFs (typical case)
                # but we don't hardcode 5; we just collect per source up to per_pdf_top_k each.
                if all(c >= per_pdf_top_k for c in counts.values()) and len(counts) >= 5:
                    # Enough data gathered for common 5-PDF case; still safe even if more sources exist.
                    break

            # Statistics across all per-source matches
            all_matches_flat = [m for lst in per_source.values() for m in lst]
            rrf_scores = [m["rrf_score"] for m in all_matches_flat]
            dense_scores = [m["score_breakdown"]["dense_similarity"] for m in all_matches_flat]
            bm25_scores = [m["score_breakdown"]["bm25_score"] for m in all_matches_flat]

            results["matches"][attr_name] = {
                "attribute_info": {
                    "number": attr_num,
                    "name": attr_name,
                    "content_preview": (doc[:100] + "...") if len(doc) > 100 else doc,
                },
                "statistics": {
                    "avg_rrf_score": float(sum(rrf_scores) / len(rrf_scores)) if rrf_scores else 0.0,
                    "max_rrf_score": float(max(rrf_scores)) if rrf_scores else 0.0,
                    "avg_dense_score": float(sum(dense_scores) / len(dense_scores)) if dense_scores else 0.0,
                    "avg_bm25_score": float(sum(bm25_scores) / len(bm25_scores)) if bm25_scores else 0.0,
                    "high_rrf_matches": int(len([s for s in rrf_scores if s > 0.01])),
                },
                "per_source_top_matches": per_source,
            }
        except Exception as e:
            logger.error(f"Error processing attribute '{attr_name}' for {collection_key}: {e}")
            logger.error(traceback.format_exc())
            continue

    return results


def save_results(collection_key: str, data: Dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{collection_key}_precise_attribute_similarities.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved: {out_file}")
    return out_file


def run_pipeline() -> bool:
    logger.info("🚀 STARTING PRECISE ATTRIBUTE SIMILARITY PIPELINE")
    logger.info("=" * 60)

    all_ok = True

    # Standard collections (global top 20)
    for key, cfg in STANDARD_COLLECTIONS.items():
        logger.info(f"\n{'='*20} {key.upper()} {'='*20}")
        try:
            res = process_standard_collection(key, cfg)
            save_results(key, res)
        except Exception as e:
            logger.error(f"Failed for {key}: {e}")
            all_ok = False

    # Redacted collections (per-PDF top 15)
    for key, cfg in REDACTED_COLLECTIONS.items():
        logger.info(f"\n{'='*20} {key.upper()} {'='*20}")
        try:
            res = process_redacted_collection(key, cfg)
            save_results(key, res)
        except Exception as e:
            logger.error(f"Failed for {key}: {e}")
            all_ok = False

    logger.info("\n🎯 Precise similarity pipeline completed")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    return all_ok


def main() -> int:
    try:
        ok = run_pipeline()
        return 0 if ok else 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
