#!/usr/bin/env python3
"""
Standard Classification Pipeline Orchestrator

Runs the 3-step flow in order using absolute paths:
1) exact_structure_classifier.py
2) multi_step_classifier.py
3) analysis_summary.py

Usage (from anywhere):
  python StandardClassification/pipeline.py

Optional overrides:
  --db-path /abs/path/to/chroma_db_qwen
  --input-dir /abs/path/to/outputs/precise_similarity/processed_datasets
  --standard-dir /abs/path/to/classifystandard/standard
  --final-dir /abs/path/to/classifystandard/standard_final
  --summary-file /abs/path/to/StandardClassification/classification_summary.json
  --report-file /abs/path/to/StandardClassification/classification_report.txt
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_step(name: str, cmd: list[str]) -> None:
    logger.info("\n=== Running: %s ===", name)
    logger.debug("Command: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ %s completed", name)
    except subprocess.CalledProcessError as e:
        logger.error("❌ %s failed with exit code %s", name, e.returncode)
        raise


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    default_input_dir = str((repo_root / "outputs/precise_similarity/processed_datasets").resolve())
    default_db_path = str((repo_root / "chroma_db_qwen").resolve())
    default_standard_dir = str((repo_root / "StandardClassification/standard").resolve())
    default_final_dir = str((repo_root / "StandardClassification/standard_final").resolve())
    default_summary_file = str((script_dir / "classification_summary.json").resolve())
    default_report_file = str((script_dir / "classification_report.txt").resolve())

    parser = argparse.ArgumentParser(description="Run StandardClassification full pipeline")
    parser.add_argument("--db-path", default=default_db_path, help="ChromaDB path")
    parser.add_argument("--input-dir", default=default_input_dir, help="Processed datasets input dir")
    parser.add_argument("--standard-dir", default=default_standard_dir, help="Output dir for exact-match standard rows")
    parser.add_argument("--final-dir", default=default_final_dir, help="Output dir for final standard rows")
    parser.add_argument("--summary-file", default=default_summary_file, help="Output JSON for analysis summary")
    parser.add_argument("--report-file", default=default_report_file, help="Output TXT for readable report")
    args = parser.parse_args()

    # Ensure parents
    Path(args.standard_dir).mkdir(parents=True, exist_ok=True)
    Path(args.final_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # 1) Exact structure classifier
    exact_cmd = [
        py,
        str(script_dir / "exact_structure_classifier.py"),
        "--input-dir", args.input_dir,
        "--output-dir", args.standard_dir,
    ]
    run_step("Step 1 - exact_structure_classifier", exact_cmd)

    # 2) Multi-step classifier
    multi_cmd = [
        py,
        str(script_dir / "multi_step_classifier.py"),
        "--input-dir", args.input_dir,
        "--db-path", args.db_path,
        "--output-dir", args.final_dir,
    ]
    run_step("Step 2 - multi_step_classifier", multi_cmd)

    # 3) Analysis summary
    analysis_cmd = [
        py,
        str(script_dir / "analysis_summary.py"),
        "--input-dir", args.input_dir,
        "--output-file", args.summary_file,
        "--report-file", args.report_file,
    ]
    run_step("Step 3 - analysis_summary", analysis_cmd)

    logger.info("\n🎉 StandardClassification pipeline completed successfully")
    logger.info("Processed datasets: %s", args.input_dir)
    logger.info("Exact-match outputs: %s", args.standard_dir)
    logger.info("Final standard outputs: %s", args.final_dir)
    logger.info("Summary JSON: %s", args.summary_file)
    logger.info("Report TXT: %s", args.report_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
