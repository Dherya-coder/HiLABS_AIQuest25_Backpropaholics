#!/usr/bin/env python3
"""
Preprocess precise similarity results to produce 12 JSON datasets with cleaned `full_content`.

Inputs (from outputs/precise_similarity/):
- TNredacted_precise_attribute_similarities.json     -> 5 datasets (per source PDF)
- WAredacted_precise_attribute_similarities.json     -> 5 datasets (per source PDF)
- TNstandard_precise_attribute_similarities.json     -> 1 dataset (global)
- WAstandard_precise_attribute_similarities.json     -> 1 dataset (global)

Output directory:
- outputs/precise_similarity/processed_datasets/

Each dataset is a JSON array of rows. For redacted collections, each dataset is built per source_file
(PDF) and contains up to top 15 matches per attribute for that source as provided by the pipeline.
For standard collections, a single dataset aggregates top matches across the entire collection per attribute.

Row schema (common superset; some fields may be absent depending on collection):
{
  "collection_key": str,
  "collection_name": str,
  "source_file": str,           # present for redacted; present for standard if available
  "attribute_number": int,
  "attribute_name": str,
  "rank": int,                  # standard
  "rank_in_global": int,        # redacted (rank within that source subset accumulation)
  "page": int,
  "section": str,
  "chunk_id": str,
  "rrf_score": float,
  "dense_similarity": float,
  "bm25_score": float,
  "content_preview": str,
  "full_content_raw": str,
  "preprocessed_final_content": str,
}

Preprocessing steps:
- Unicode normalize
- Lowercase
- Remove markdown artifacts and punctuation (retain alphanumerics and spaces)
- Tokenize by whitespace
- Remove stopwords (built-in static list)
- Lemmatize and/or stem if libraries available (NLTK PorterStemmer if installed). Fallback to lightweight rules.
- Join tokens back as space-separated string
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
from datetime import datetime

# Optional NLP imports (safe fallbacks applied if unavailable)
try:
    from nltk.stem import PorterStemmer  # type: ignore
    _PORTER = PorterStemmer()
except Exception:
    _PORTER = None

# Optional text fixing (OCR/unicode issues)
try:
    import ftfy  # type: ignore
    _FTFY_AVAILABLE = True
except Exception:
    ftfy = None  # type: ignore
    _FTFY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Preprocessing version/provenance (git-less): based on script mtime or runtime timestamp fallback
try:
    PREPROCESSING_VERSION = f"mtime:{datetime.fromtimestamp(Path(__file__).stat().st_mtime).isoformat()}"
except Exception:
    PREPROCESSING_VERSION = f"runtime:{datetime.now().isoformat()}"

# Resolve paths relative to repository root (parent of this script's directory)
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "outputs/precise_similarity"
OUTPUT_DIR = INPUT_DIR / "processed_datasets"

# Minimal, static English stopword list (avoids runtime downloads)
_STOPWORDS = set(
    [
        "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
        "be","because","been","before","being","below","between","both","but","by",
        "can","can't","cannot","could","couldn't",
        "did","didn't","do","does","doesn't","doing","don't","down","during",
        "each",
        "few","for","from","further",
        "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's",
        "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
        "let's",
        "me","more","most","mustn't","my","myself",
        "no","nor","not",
        "of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
        "same","she","she'd","she'll","she's","should","shouldn't","so","some","such",
        "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too",
        "under","until","up",
        "very",
        "was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
        "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves",
        # domain-ish filler
        "shall","may","including","without","limitation","pursuant","herein","thereof","therein","thereby",
        "such","any","all","within","upon","prior","thereafter","thereon","therefrom","hereby","herewith",
    ]
)

_MD_FENCE_RE = re.compile(r"`{3,}")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Basic list of English number words to mask
_NUMBER_WORDS = set(
    [
        "zero","one","two","three","four","five","six","seven","eight","nine",
        "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen",
        "twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety",
        "hundred","thousand","million","billion",
    ]
)

def _unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _basic_lemma_rules(token: str) -> str:
    # Extremely lightweight lemmatization rules as a fallback
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]  # classes -> class
    if token.endswith("ss"):
        return token
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token


def get_token_count(text: Optional[str]) -> Tuple[int, str]:
    """Return (token_count, tokenizer_name). Prefer tiktoken if available, else whitespace split.
    Uses gpt-4o-mini encoding when available; falls back to cl100k_base.
    """
    s = text or ""
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s)), "tiktoken"
    except Exception:
        return len(s.split()), "split"


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


_RED_PATTERNS = [
    re.compile(r"\[REDACTED\]", re.IGNORECASE),
    re.compile(r"x{3,}", re.IGNORECASE),  # xxxxx sequences
    re.compile(r"\u2588+"),               # █ blocks
    re.compile(r"\[\[.+?\]\]"),       # [[ ... ]]
]


def detect_redaction(text: Optional[str]) -> Tuple[bool, Optional[str]]:
    if not text:
        return False, None
    for pat in _RED_PATTERNS:
        m = pat.search(text)
        if m:
            snippet = text[max(0, m.start()-20): m.end()+20]
            return True, snippet[:200]
    return False, None


def preprocess_text(text: Optional[str]) -> str:
    if not text:
        return ""

    # Normalize & lowercase
    t = text
    if _FTFY_AVAILABLE:
        try:
            t = ftfy.fix_text(t)  # type: ignore
        except Exception:
            pass
    t = _unicode_normalize(t)
    t = t.replace("\u2018", "'").replace("\u2019", "'")  # fancy quotes to ascii
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.lower()

    # Remove markdown/code fences and table pipes
    t = _MD_FENCE_RE.sub(" ", t)
    t = t.replace("|", " ")

    # Remove non-alphanumeric chars, keep spaces
    t = _NON_ALNUM_SPACE_RE.sub(" ", t)

    # Collapse whitespace
    t = _MULTI_SPACE_RE.sub(" ", t).strip()

    if not t:
        return ""

    # Tokenize
    tokens = t.split()

    # Remove stopwords
    tokens = [tok for tok in tokens if tok not in _STOPWORDS]

    # Replace tokens containing any digits with 'num'
    tokens = ["num" if any(ch.isdigit() for ch in tok) else tok for tok in tokens]

    # Replace spelled-out number words with 'num'
    tokens = ["num" if tok in _NUMBER_WORDS else tok for tok in tokens]

    # Lemma/stem
    processed: List[str] = list(tokens)

    # Final join
    return " ".join(processed)


def _ensure_out_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_common_fields(match: Dict[str, Any]) -> Dict[str, Any]:
    sb = match.get("score_breakdown", {}) or {}
    return {
        "page": match.get("page"),
        "section": match.get("section"),
        "chunk_id": match.get("chunk_id"),
        "source_file": match.get("source_file"),
        "rrf_score": match.get("rrf_score"),
        "dense_similarity": sb.get("dense_similarity"),
        "bm25_score": sb.get("bm25_score"),
        "content_preview": match.get("content_preview"),
        "full_content_raw": match.get("full_content"),
    }


def _row_with_preproc(base: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(base)
    raw = base.get("full_content_raw") or ""
    pre = preprocess_text(raw)

    # Enrichments
    tok_count, tok_name = get_token_count(pre)
    has_redact, red_snip = detect_redaction(raw)
    sha = compute_sha256(pre)

    row["preprocessed_final_content"] = pre
    row["preprocessed_token_count"] = tok_count
    row["preprocessed_sha256"] = sha
    row["preprocessing_version"] = PREPROCESSING_VERSION
    row["has_redaction"] = has_redact
    row["redaction_snippet"] = red_snip
    row["preprocessing_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "used_stemmer": False,
        "used_ftfy": _FTFY_AVAILABLE,
        "tokenizer": tok_name,
        "pipeline_steps": [
            *( ["ftfy_fix"] if _FTFY_AVAILABLE else [] ),
            "unicode_nfkc",
            "lowercase",
            "md_fence_strip",
            "non_alnum_strip",
            "collapse_whitespace",
            "stopword_removal",
            "number_masking",
            "number_word_masking",
        ],
    }
    return row


def build_standard_dataset(sim_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    data = _load_json(sim_path)
    collection_info = data.get("collection_info", {})
    collection_key = collection_info.get("collection_key", sim_path.stem)
    collection_name = collection_info.get("collection_name", collection_key)

    rows: List[Dict[str, Any]] = []
    matches = data.get("matches", {}) or {}

    PER_ATTR_LIMIT = {1: 7, 2: 7, 3: 3, 4: 5, 5: 5}

    for attr_name, attr_data in matches.items():
        attr_info = attr_data.get("attribute_info", {})
        attr_num_raw = attr_info.get("number")
        try:
            attr_num = int(attr_num_raw)
        except Exception:
            attr_num = attr_num_raw

        # Standard format uses `top_matches`
        top_matches = (
            attr_data.get("top_matches")
            or attr_data.get("matches")
            or []
        )

        # Enforce per-attribute limit
        limit = PER_ATTR_LIMIT.get(attr_num, len(top_matches))
        for m in top_matches[:limit]:
            base = {
                "collection_key": collection_key,
                "collection_name": collection_name,
                "attribute_number": attr_num,
                "attribute_name": attr_name,
                "rank": m.get("rank"),
            }
            base.update(_extract_common_fields(m))
            rows.append(_row_with_preproc(base))

    return collection_key, rows


def build_redacted_datasets(sim_path: Path) -> Tuple[str, Dict[str, List[Dict[str, Any]]]]:
    data = _load_json(sim_path)
    collection_info = data.get("collection_info", {})
    collection_key = collection_info.get("collection_key", sim_path.stem)
    collection_name = collection_info.get("collection_name", collection_key)

    # Gather per source
    per_source_rows: Dict[str, List[Dict[str, Any]]] = {}
    matches = data.get("matches", {}) or {}

    # Enforce per-attribute top-N per source
    PER_ATTR_LIMIT = {1: 7, 2: 7, 3: 3, 4: 5, 5: 5}

    # Union of all sources across attributes
    for attr_name, attr_data in matches.items():
        attr_info = attr_data.get("attribute_info", {})
        attr_num_raw = attr_info.get("number")
        try:
            attr_num = int(attr_num_raw)
        except Exception:
            attr_num = attr_num_raw
        per_source = attr_data.get("per_source_top_matches", {}) or {}

        for source_file, match_list in per_source.items():
            # Enforce per-attribute limit within this source
            limit = PER_ATTR_LIMIT.get(attr_num, len(match_list))
            for m in match_list[:limit]:
                base = {
                    "collection_key": collection_key,
                    "collection_name": collection_name,
                    "source_file": source_file,
                    "attribute_number": attr_num,
                    "attribute_name": attr_name,
                    "rank_in_global": m.get("rank_in_global"),
                }
                base.update(_extract_common_fields(m))
                per_source_rows.setdefault(source_file, []).append(_row_with_preproc(base))

    return collection_key, per_source_rows


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote: {path}")


def main() -> int:
    _ensure_out_dir()

    # Input files
    tn_red_path = INPUT_DIR / "TNredacted_precise_attribute_similarities.json"
    wa_red_path = INPUT_DIR / "WAredacted_precise_attribute_similarities.json"
    tn_std_path = INPUT_DIR / "TNstandard_precise_attribute_similarities.json"
    wa_std_path = INPUT_DIR / "WAstandard_precise_attribute_similarities.json"

    # Validate existence
    missing = [p for p in [tn_red_path, wa_red_path, tn_std_path, wa_std_path] if not p.exists()]
    if missing:
        for p in missing:
            logger.error(f"Missing input: {p}")
        return 1

    # 1) Standard datasets (TNstandard, WAstandard)
    for std_path in [tn_std_path, wa_std_path]:
        collection_key, rows = build_standard_dataset(std_path)
        out_path = OUTPUT_DIR / f"{collection_key}_dataset.json"
        save_json(out_path, rows)

    # 2) Redacted datasets per source (TNredacted, WAredacted)
    for red_path in [tn_red_path, wa_red_path]:
        collection_key, per_source = build_redacted_datasets(red_path)
        # Expecting ~5 sources per collection
        for source_file, rows in per_source.items():
            # Sanitize filename segment
            src_slug = Path(source_file).stem  # drop .md
            src_slug = re.sub(r"[^A-Za-z0-9_-]", "_", src_slug)
            out_path = OUTPUT_DIR / f"{collection_key}_{src_slug}_dataset.json"
            save_json(out_path, rows)

    logger.info("\n✅ Completed preprocessing and dataset generation.")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
