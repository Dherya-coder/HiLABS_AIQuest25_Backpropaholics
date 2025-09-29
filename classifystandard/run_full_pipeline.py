#!/usr/bin/env python3
"""
Complete pipeline runner for Standard vs Non-Standard classification.

This script runs the complete 7-step classification pipeline:
1. Step 1: Exact structural match (already done by exact_structure_classifier.py)
2. Steps 2-7: Multi-step analysis (semantic, paraphrase, NLI, negation, rule flags)

Run:
  python classifystandard/run_full_pipeline.py
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    # Step 1: Run exact structure classifier (if not already done)
    step1_script = script_dir / "exact_structure_classifier.py"
    if step1_script.exists():
        success = run_command([
            sys.executable, str(step1_script),
            "--input-dir", str(repo_root / "outputs/precise_similarity/processed_datasets"),
            "--output-dir", str(repo_root / "classifystandard/standard")
        ], "Step 1: Exact Structure Classification")
        
        if not success:
            logger.error("Step 1 failed, aborting pipeline")
            return 1
    else:
        logger.warning("Step 1 script not found, assuming already completed")
    
    # Steps 2-7: Multi-step classifier
    step2_script = script_dir / "multi_step_classifier.py"
    if step2_script.exists():
        success = run_command([
            sys.executable, str(step2_script),
            "--input-dir", str(repo_root / "outputs/precise_similarity/processed_datasets"),
            "--db-path", str(repo_root / "chroma_db_qwen"),
            "--output-dir", str(repo_root / "classifystandard/standard_final")
        ], "Steps 2-7: Multi-step Classification")
        
        if not success:
            logger.error("Multi-step classification failed")
            return 1
    else:
        logger.error("Multi-step classifier script not found")
        return 1
    
    logger.info("🎉 Complete classification pipeline finished successfully!")
    logger.info("Results:")
    logger.info("  - Updated datasets with classification details: outputs/precise_similarity/processed_datasets/")
    logger.info("  - Standard-only Step 1 results: classifystandard/standard/")
    logger.info("  - Final standard classifications: classifystandard/standard_final/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
