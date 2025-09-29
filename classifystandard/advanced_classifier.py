#!/usr/bin/env python3
"""
Advanced classifier (Steps 2–7) for Standard vs Non-Standard clauses.

Prereq: Step 1 (exact structure match) has already marked isStandard on
redacted datasets in outputs/precise_similarity/processed_datasets/.

This script:
- For every redacted row where isStandard != 1, applies Steps 2–7:
  Step 2. Semantic similarity (Qwen) against same-state template clauses with same attribute.
          If cosine >= 0.80 -> Standard.
  Step 3. Paraphrase similarity (SentenceTransformer) against templates (same attribute).
          If cosine >= 0.75 -> proceed to Step 4, else Non-Standard.
  Step 4. NLI (two-way) using transformers MNLI model.
          - Both entailments true -> Standard.
          - One-way -> Non-Standard.
          - Contradiction -> Non-Standard.
          - Neutral both ways & paraphrase >= 0.90 -> Standard.
  Step 5. Negation scope analysis (spaCy if available, else regex heuristic).
          - If negation polarity or scope differs -> Non-Standard override.
  Step 6. Rule flags (lexical/regex). If any triggers -> Non-Standard override.
  Step 7. Final aggregation: we mark each row. Additionally, we export Standard-only
          rows into classifystandard/standard_final/.

- Updates the original redacted processed files in-place (isStandard: 1/0) and 
  writes a Standard-only view to classifystandard/standard_final/<filename>.json

Run:
  python classifystandard/advanced_classifier.py \
    --input-dir outputs/precise_similarity/processed_datasets \
    --db-path chroma_db_qwen \
    --ollama-url http://localhost:11434 \
    --qwen-model qwen3-embedding:0.6b \
    --paraphrase-collection similarityTemplate_para \
    --qwen-collection similarityTemplate_qwen

Notes:
- We query templates by where filters: same attribute_number and same state.
- We reuse existing embeddings from Chroma. If an embedding is missing for a row,
  we fallback to on-the-fly encoding when possible.
- Dependencies (optional): transformers (MNLI), spacy (negation scope). Fallbacks available.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np

import chromadb
from chromadb.config import Settings

# Optional: transformers for NLI
try:
    from transformers import pipeline
    _HAVE_TRANSFORMERS = True
except Exception:
    pipeline = None  # type: ignore
    _HAVE_TRANSFORMERS = False

# Optional: spaCy for better negation scope
try:
    import spacy  # type: ignore
    _SPACY = spacy.load("en_core_web_sm")  # may fail if model not installed
    _HAVE_SPACY = True
except Exception:
    _SPACY = None
    _HAVE_SPACY = False

# Optional: SentenceTransformer to fallback encode paraphrase if missing
try:
    from sentence_transformers import SentenceTransformer
    _HAVE_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAVE_ST = False

# Optional: Qwen via Ollama to fallback encode if missing
try:
    import requests  # type: ignore
    _HAVE_REQUESTS = True
except Exception:
    _HAVE_REQUESTS = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thresholds
TAU_SEMANTIC = 0.80
TAU_PARAPHRASE = 0.75
TAU_PARAPHRASE_STRONG = 0.90  # Step 4 neutral override

STANDARD_KEYS = {"TNstandard": "TN", "WAstandard": "WA"}
REDACTED_KEYS = {"TNredacted": "TN", "WAredacted": "WA"}

# Rule flag regex patterns
RULE_PATTERNS = {
    "exception_carveouts": re.compile(r"\b(except|unless|provided that|subject to|notwithstanding)\b", re.I),
    "methodology_shifts": re.compile(r"\b(Medicare|UCR|RBRVS|per diem|capitation|bundled|stop[- ]?loss)\b", re.I),
    "re_anchoring": re.compile(r"\b(lesser of charges|allowed amount|most favored nation)\b", re.I),
    "forbidden_conditions": re.compile(r"\b(indemnify|hold harmless|sole discretion)\b", re.I),
    # Add domain-specific missing protection checks as needed
}

NEGATION_TERMS = re.compile(r"\b(no|not|never|without|except|shall not|may not|cannot|won't|don't|doesn't|isn't|aren't)\b", re.I)


def normalize_attr_num(n: Any) -> Any:
    try:
        return int(n)
    except Exception:
        return n


def state_from_collection_key(key: str) -> Optional[str]:
    if key in REDACTED_KEYS:
        return REDACTED_KEYS[key]
    if key in STANDARD_KEYS:
        return STANDARD_KEYS[key]
    return None


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


# ===== Chroma helpers =====
class ChromaWrap:
    def __init__(self, db_path: Path):
        self.client = chromadb.PersistentClient(path=str(db_path), settings=Settings(anonymized_telemetry=False))
        self.col_tmplt_qwen = self.client.get_or_create_collection("similarityTemplate_qwen")
        self.col_tmplt_para = self.client.get_or_create_collection("similarityTemplate_para")
        self.col_red_qwen = self.client.get_or_create_collection("SimilarityRedacted_qwen")
        self.col_red_para = self.client.get_or_create_collection("SimilarityRedacted_para")

    def query_templates(self, state: str, attr_num: Any, query_emb: List[float], which: str, top_k: int = 5):
        col = self.col_tmplt_qwen if which == "qwen" else self.col_tmplt_para
        where = {"attribute_number": attr_num, "collection_key": f"{state}standard"}
        res = col.query(query_embeddings=[query_emb], n_results=top_k, where=where, include=["distances", "metadatas", "documents", "ids"])
        return res

    def get_candidate_embedding(self, row: Dict[str, Any], which: str) -> Optional[List[float]]:
        col = self.col_red_qwen if which == "qwen" else self.col_red_para
        cid = make_id(row) + ("::qwen" if which == "qwen" else "::para")
        try:
            got = col.get(ids=[cid], include=["embeddings"])
            embs = got.get("embeddings") or []
            if embs and embs[0]:
                return embs[0]
        except Exception as e:
            logger.warning(f"Chroma get embedding failed for {cid}: {e}")
        return None


# ===== ID builder (must mirror generate_processed_embeddings.make_id) =====
def make_id(row: Dict[str, Any]) -> str:
    sha = (row.get("preprocessed_sha256") or "")[:16]
    chunk = row.get("chunk_id") or "chunk"
    attr = str(row.get("attribute_number") or "attr")
    col = row.get("collection_key") or "col"
    src = Path(str(row.get("source_file") or "src")).stem
    page = str(row.get("page") or 0)
    base = f"proc::{col}::{attr}::{src}::{chunk}::{page}::{sha}"
    return base


# ===== Fallback encoders (optional) =====
class QwenOllama:
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.embed_url = f"{self.base_url}/api/embeddings"

    def embed(self, text: str) -> Optional[List[float]]:
        if not _HAVE_REQUESTS:
            return None
        try:
            r = requests.post(self.embed_url, json={"model": self.model_name, "prompt": text}, timeout=30)
            if r.status_code != 200:
                return None
            vec = r.json().get("embedding") or []
            v = np.array(vec, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            return v.tolist()
        except Exception:
            return None


class ParaEncoder:
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name) if _HAVE_ST else None

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.model:
            return None
        try:
            v = self.model.encode([text], normalize_embeddings=True)[0]
            return v.tolist() if hasattr(v, "tolist") else list(v)
        except Exception:
            return None


# ===== NLI & NLP helpers =====
class NLIModel:
    def __init__(self):
        if _HAVE_TRANSFORMERS:
            try:
                self.clf = pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True)
            except Exception:
                self.clf = None
        else:
            self.clf = None

    def infer(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Return (label, score) with label in {ENTAILMENT, CONTRADICTION, NEUTRAL}."""
        if not self.clf:
            # Fallback: naive lexical heuristic
            p = premise.lower()
            h = hypothesis.lower()
            if h in p:
                return ("ENTAILMENT", 0.6)
            if any(w in p and w not in h for w in ["not ", "except", "without", "unless"]):
                return ("CONTRADICTION", 0.6)
            return ("NEUTRAL", 0.5)
        out = self.clf({"text": premise, "text_pair": hypothesis})
        # out is list of dicts with labels
        scores = {d["label"].upper(): d["score"] for d in out[0]}
        # Map labels if needed
        ent = scores.get("ENTAILMENT", 0.0)
        con = scores.get("CONTRADICTION", 0.0)
        neu = scores.get("NEUTRAL", 0.0)
        if ent >= max(con, neu):
            return ("ENTAILMENT", ent)
        if con >= max(ent, neu):
            return ("CONTRADICTION", con)
        return ("NEUTRAL", neu)


def detect_negations(text: str) -> Set[str]:
    if _HAVE_SPACY and _SPACY is not None:
        doc = _SPACY(text)
        scopes = set()
        for token in doc:
            if token.dep_ == "neg":
                head = token.head.lemma_.lower()
                scopes.add(head)
        return scopes
    # Fallback: collect words near negation cues
    scopes = set()
    for m in NEGATION_TERMS.finditer(text):
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 20)
        snippet = text[start:end].lower()
        # naive head extraction: last content word
        words = re.findall(r"[a-z]+", snippet)
        if words:
            scopes.add(words[-1])
    return scopes


def rule_flags(text: str) -> List[str]:
    hits = []
    for name, pat in RULE_PATTERNS.items():
        if pat.search(text):
            hits.append(name)
    return hits


# ===== Core Steps =====

def cosine_from_distance(d: float) -> float:
    # Chroma returns cosine distance by default when using cosine metric: dist = 1 - cos_sim
    # Clamp into [0,1]
    sim = 1.0 - float(d)
    return max(0.0, min(1.0, sim))


def step2_semantic_qwen(row: Dict[str, Any], chroma: ChromaWrap, qwen_fallback: Optional[QwenOllama]) -> Tuple[bool, float, Dict[str, Any]]:
    state = state_from_collection_key(row.get("collection_key", ""))
    if not state:
        return (False, 0.0, {})
    attr = normalize_attr_num(row.get("attribute_number"))
    cand_emb = chroma.get_candidate_embedding(row, which="qwen")
    if not cand_emb and qwen_fallback is not None:
        cand_emb = qwen_fallback.embed(row.get("preprocessed_final_content") or "")
    if not cand_emb:
        return (False, 0.0, {})
    res = chroma.query_templates(state, attr, cand_emb, which="qwen", top_k=3)
    best_sim = 0.0
    best_meta: Dict[str, Any] = {}
    if res and res.get("distances"):
        dists = res["distances"][0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        for i, d in enumerate(dists):
            sim = cosine_from_distance(d)
            if sim > best_sim:
                best_sim = sim
                best_meta = {"id": ids[i], "metadata": metas[i], "document": docs[i]}
    passed = best_sim >= TAU_SEMANTIC
    return (passed, best_sim, best_meta)


def step3_paraphrase(row: Dict[str, Any], chroma: ChromaWrap, para_fallback: Optional[ParaEncoder]) -> Tuple[bool, float, Dict[str, Any]]:
    state = state_from_collection_key(row.get("collection_key", ""))
    if not state:
        return (False, 0.0, {})
    attr = normalize_attr_num(row.get("attribute_number"))
    cand_emb = chroma.get_candidate_embedding(row, which="para")
    if not cand_emb and para_fallback is not None:
        cand_emb = para_fallback.embed(row.get("preprocessed_final_content") or "")
    if not cand_emb:
        return (False, 0.0, {})
    res = chroma.query_templates(state, attr, cand_emb, which="para", top_k=3)
    best_sim = 0.0
    best_meta: Dict[str, Any] = {}
    if res and res.get("distances"):
        dists = res["distances"][0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        for i, d in enumerate(dists):
            sim = cosine_from_distance(d)
            if sim > best_sim:
                best_sim = sim
                best_meta = {"id": ids[i], "metadata": metas[i], "document": docs[i]}
    passed = best_sim >= TAU_PARAPHRASE
    return (passed, best_sim, best_meta)


def step4_nli(template_text: str, candidate_text: str, nli: NLIModel, paraphrase_sim: float) -> Tuple[bool, Dict[str, Any]]:
    t2c, t2c_score = nli.infer(template_text, candidate_text)
    c2t, c2t_score = nli.infer(candidate_text, template_text)
    result = {
        "t2c": t2c,
        "t2c_score": float(t2c_score),
        "c2t": c2t,
        "c2t_score": float(c2t_score),
    }
    if t2c == "ENTAILMENT" and c2t == "ENTAILMENT":
        return (True, result)
    if t2c == "CONTRADICTION" or c2t == "CONTRADICTION":
        return (False, result)
    # Neutral both ways but paraphrase very high
    if t2c == "NEUTRAL" and c2t == "NEUTRAL" and paraphrase_sim >= TAU_PARAPHRASE_STRONG:
        return (True, result)
    return (False, result)


def step5_negation_scope(template_text: str, candidate_text: str) -> Tuple[bool, Dict[str, Any]]:
    t_negs = detect_negations(template_text)
    c_negs = detect_negations(candidate_text)
    info = {"template_negs": list(t_negs), "candidate_negs": list(c_negs)}
    if not t_negs and not c_negs:
        return (True, info)  # no negations, fine
    if (t_negs and not c_negs) or (c_negs and not t_negs):
        return (False, info)  # polarity flip
    # both have negations: require overlap in heads
    if t_negs.intersection(c_negs):
        return (True, info)
    return (False, info)


def step6_rule_flags(candidate_text: str) -> Tuple[bool, List[str]]:
    hits = rule_flags(candidate_text)
    return (len(hits) == 0, hits)


# ===== Orchestrator =====

def process_file(fp: Path, chroma: ChromaWrap, qwen_fb: Optional[QwenOllama], para_fb: Optional[ParaEncoder], out_dir: Path) -> Tuple[int, int]:
    rows = load_rows(fp)
    if not rows:
        return (0, 0)
    state = state_from_collection_key(rows[0].get("collection_key", ""))
    if state not in ("TN", "WA"):
        logger.info(f"Skip {fp.name} (not a redacted dataset)")
        return (0, 0)

    nli = NLIModel()

    matched_rows: List[Dict[str, Any]] = []
    updated_rows: List[Dict[str, Any]] = []
    matched = 0
    total = 0

    for r in rows:
        total += 1
        # Respect existing Step 1 exact match
        if r.get("isStandard") == 1:
            updated_rows.append(r)
            matched_rows.append(r)
            matched += 1
            continue

        candidate = r.get("preprocessed_final_content") or ""
        if not candidate:
            r2 = dict(r)
            r2["isStandard"] = 0
            r2["std_reason"] = "empty"
            updated_rows.append(r2)
            continue

        # Step 2: Semantic (Qwen)
        s2_pass, s2_sim, s2_meta = step2_semantic_qwen(r, chroma, qwen_fb)
        if s2_pass:
            r2 = dict(r)
            r2["isStandard"] = 1
            r2["std_reason"] = f"semantic_qwen({s2_sim:.3f})"
            r2["std_match"] = s2_meta
            updated_rows.append(r2)
            matched_rows.append(r2)
            matched += 1
            continue

        # Step 3: Paraphrase
        s3_pass, s3_sim, s3_meta = step3_paraphrase(r, chroma, para_fb)
        if not s3_pass:
            r2 = dict(r)
            r2["isStandard"] = 0
            r2["std_reason"] = f"paraphrase_below({s3_sim:.3f})"
            updated_rows.append(r2)
            continue

        # Step 4: NLI
        template_text = (s3_meta.get("document") or "") if s3_meta else ""
        s4_pass, s4_info = step4_nli(template_text=template_text, candidate_text=candidate, nli=nli, paraphrase_sim=s3_sim)
        if not s4_pass:
            r2 = dict(r)
            r2["isStandard"] = 0
            r2["std_reason"] = f"nli_fail(paraphrase={s3_sim:.3f})"
            r2["nli"] = s4_info
            updated_rows.append(r2)
            continue

        # Step 5: Negation scope analysis
        s5_pass, s5_info = step5_negation_scope(template_text, candidate)
        if not s5_pass:
            r2 = dict(r)
            r2["isStandard"] = 0
            r2["std_reason"] = "negation_scope_mismatch"
            r2["negation"] = s5_info
            updated_rows.append(r2)
            continue

        # Step 6: Rule flags
        s6_pass, s6_hits = step6_rule_flags(candidate)
        if not s6_pass:
            r2 = dict(r)
            r2["isStandard"] = 0
            r2["std_reason"] = f"rule_flags:{','.join(s6_hits)}"
            r2["rule_flags"] = s6_hits
            updated_rows.append(r2)
            continue

        # Success
        r2 = dict(r)
        r2["isStandard"] = 1
        r2["std_reason"] = f"paraphrase+NLI_ok({s3_sim:.3f})"
        r2["nli"] = s4_info
        r2["negation"] = s5_info
        r2["std_match"] = s3_meta
        updated_rows.append(r2)
        matched_rows.append(r2)
        matched += 1

    # Write updated source file
    save_rows(fp, updated_rows)

    # Write standard-only view
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fp.name
    save_rows(out_path, matched_rows)

    logger.info(f"{fp.name}: matched {matched}/{total} after Steps 2–7 → wrote {len(matched_rows)} rows to {out_path} and updated source file.")
    return (matched, total)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    ap = argparse.ArgumentParser(description="Advanced Standard classifier (Steps 2–7)")
    ap.add_argument("--input-dir", default=str(repo_root / "outputs/precise_similarity/processed_datasets"), help="Processed datasets directory")
    ap.add_argument("--db-path", default=str(repo_root / "chroma_db_qwen"), help="Chroma DB path")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL (fallback)")
    ap.add_argument("--qwen-model", default="qwen3-embedding:0.6b", help="Qwen model name (fallback)")
    ap.add_argument("--paraphrase-model", default="paraphrase-MiniLM-L6-v2", help="Paraphrase model (fallback)")
    ap.add_argument("--standard-final-dir", default=str(repo_root / "classifystandard/standard_final"), help="Output dir for final Standard rows")
    ap.add_argument("--k", type=int, default=3, help="Top-k templates to consider (query)")
    ap.add_argument("--tau-semantic", type=float, default=TAU_SEMANTIC, help="Qwen semantic threshold")
    ap.add_argument("--tau-paraphrase", type=float, default=TAU_PARAPHRASE, help="Paraphrase threshold")
    args = ap.parse_args()

    global TAU_SEMANTIC, TAU_PARAPHRASE
    TAU_SEMANTIC = args.tau_semantic
    TAU_PARAPHRASE = args.tau_paraphrase

    input_dir = Path(args.input_dir)
    out_dir = Path(args.standard_final_dir)
    db_path = Path(args.db_path)

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    # Chroma
    chroma = ChromaWrap(db_path)

    # Fallback encoders
    qwen_fb = QwenOllama(args.qwen_model, args.ollama_url) if _HAVE_REQUESTS else None
    para_fb = ParaEncoder(args.paraphrase_model) if _HAVE_ST else None

    files = sorted(input_dir.glob("*.json"))

    tot_m, tot_t = 0, 0
    for fp in files:
        if "redacted_" not in fp.name.lower():
            continue
        m, t = process_file(fp, chroma, qwen_fb, para_fb, out_dir)
        tot_m += m
        tot_t += t

    logger.info(f"\nSummary (Steps 2–7): matched {tot_m}/{tot_t} across all redacted datasets.")


if __name__ == "__main__":
    main()
