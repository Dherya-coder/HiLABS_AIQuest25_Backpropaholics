#!/usr/bin/env python3
"""
Exact-structure classifier for Standard vs Non-Standard clauses.

Reads preprocessed datasets from outputs/precise_similarity/processed_datasets/ and
marks clauses from redacted datasets as Standard if their `preprocessed_final_content`
exactly matches any of the corresponding state's standard template clauses for the
same attribute number.

Outputs:
1) Update original redacted preprocessed datasets in-place by adding `isStandard` flag
   (1 if matches a template for same attribute; else 0).
2) Also write a Standard-only view to classifystandard/standard/ (rows with `isStandard: 1`).

Run:
  python classifystandard/exact_structure_classifier.py \
    --input-dir outputs/precise_similarity/processed_datasets \
    --output-dir classifystandard/standard

Notes:
- We rely on preprocessing that already normalizes structure and masks numerics.
- We compare only `preprocessed_final_content` per attribute.
- We add `isStandard: 1` only for redacted rows that match; standard rows are not re-written.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STANDARD_KEYS = {"TNstandard": "TN", "WAstandard": "WA"}
REDACTED_KEYS = {"TNredacted": "TN", "WAredacted": "WA"}


def normalize_attr_num(n: Any) -> Any:
    try:
        return int(n)
    except Exception:
        return n


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def collect_standard_maps(input_dir: Path) -> Dict[str, Dict[Any, Set[str]]]:
    """Return mapping: state -> { attribute_number -> set(preprocessed_final_content) }"""
    maps: Dict[str, Dict[Any, Set[str]]] = {"TN": {}, "WA": {}}

    for std_name in ("TNstandard_dataset.json", "WAstandard_dataset.json"):
        fp = input_dir / std_name
        if not fp.exists():
            logger.warning(f"Standard dataset not found: {fp}")
            continue
        rows = load_rows(fp)
        if not rows:
            continue
        col_key = rows[0].get("collection_key")
        state = STANDARD_KEYS.get(col_key)
        if not state:
            # fallback by filename
            state = "TN" if std_name.startswith("TN") else "WA"
        m = maps[state]
        for r in rows:
            attr = normalize_attr_num(r.get("attribute_number"))
            text = r.get("preprocessed_final_content") or ""
            if not text:
                continue
            m.setdefault(attr, set()).add(text)
        logger.info(f"Collected {sum(len(s) for s in m.values())} standard templates for {state}")

    return maps


def state_from_collection_key(key: str) -> str | None:
    if key in REDACTED_KEYS:
        return REDACTED_KEYS[key]
    if key in STANDARD_KEYS:
        return STANDARD_KEYS[key]
    return None


def classify_redacted_file(fp: Path, standards: Dict[str, Dict[Any, Set[str]]], output_dir: Path) -> Tuple[int, int]:
    rows = load_rows(fp)
    if not rows:
        return (0, 0)
    state = state_from_collection_key(rows[0].get("collection_key", ""))
    if state not in ("TN", "WA"):
        logger.warning(f"Skipping {fp.name}: unknown state")
        return (0, 0)

    std_map = standards.get(state, {})

    standard_rows: List[Dict[str, Any]] = []
    updated_rows: List[Dict[str, Any]] = []
    total = 0
    matched = 0

    for r in rows:
        total += 1
        attr = normalize_attr_num(r.get("attribute_number"))
        text = r.get("preprocessed_final_content") or ""
        if not text:
            r2 = dict(r)
            r2["isStandard"] = 0
            updated_rows.append(r2)
            continue
        cand_set = std_map.get(attr, set())
        if text in cand_set:
            r2 = dict(r)
            r2["isStandard"] = 1
            standard_rows.append(r2)
            updated_rows.append(r2)
            matched += 1
        else:
            r2 = dict(r)
            r2["isStandard"] = 0
            updated_rows.append(r2)

    # 1) Write only standard rows
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / fp.name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(standard_rows, f, indent=2, ensure_ascii=False)
    # 2) Update original redacted dataset with isStandard flags
    with fp.open("w", encoding="utf-8") as f:
        json.dump(updated_rows, f, indent=2, ensure_ascii=False)
    logger.info(f"{fp.name}: matched {matched}/{total} -> wrote {len(standard_rows)} standard rows to {out_path} and updated source file with isStandard flags")
    return (matched, total)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    p = argparse.ArgumentParser(description="Exact-structure classifier using preprocessed datasets")
    p.add_argument("--input-dir", default=str(repo_root / "outputs/precise_similarity/processed_datasets"), help="Processed datasets directory")
    p.add_argument("--output-dir", default=str(repo_root / "classifystandard/standard"), help="Output directory for standard rows")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    standards = collect_standard_maps(input_dir)

    # Classify each redacted dataset
    files = sorted(input_dir.glob("*.json"))
    tot_matched = 0
    tot_rows = 0
    for fp in files:
        # Identify redacted by filename or collection_key inside
        if "redacted_" not in fp.name.lower():
            # skip standard
            continue
        m, t = classify_redacted_file(fp, standards, output_dir)
        tot_matched += m
        tot_rows += t

    logger.info(f"\nSummary: matched {tot_matched}/{tot_rows} rows across all redacted datasets.")


if __name__ == "__main__":
    main()
