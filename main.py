#!/usr/bin/env python3
"""
Complete HiLABS Preprocessing Pipeline

Orchestrates the entire preprocessing pipeline from raw data to final analysis:

1. PDF Parsing & Embedding (ParsingAndEmbedding)
2. Attribute Ranking (Ranker) 
3. Similarity Preprocessing & Embedding (PreprocessingSimilarity)
4. Standard Classification (StandardClassification)
5. Contract Analysis Chatbot (backend)

Usage:
    python main.py [--step STEP] [--skip-steps STEPS]

Examples:
    python main.py                    # Run complete pipeline
    python main.py --step 3           # Run only step 3 (similarity processing)
    python main.py --skip-steps 1,2   # Skip steps 1 and 2
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

class PipelineOrchestrator:
    """Orchestrates the complete HiLABS preprocessing pipeline."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.python_exe = sys.executable
        
        # Pipeline steps configuration
        self.steps = {
            1: {
                "name": "PDF Parsing & Embedding",
                "description": "Parse PDFs and create embeddings for ChromaDB",
                "scripts": [
                    {
                        "name": "WA5 PDF Processing",
                        "path": "ParsingAndEmbedding/PDFParsingAndEmbedding/process_wa5_pipeline.py",
                        "description": "Process WA_5_Redacted.pdf (99MB) with OCR and chunking"
                    }
                ],
                "dependencies": ["ollama", "tesseract"],
                "estimated_time": "15-30 minutes"
            },
            2: {
                "name": "Attribute Ranking", 
                "description": "Generate precise similarity rankings for all attributes",
                "scripts": [
                    {
                        "name": "Attribute Precise Similarity",
                        "path": "Ranker/attribute_precise_similarity_pipeline.py",
                        "description": "Generate similarity rankings for TNredacted, WAredacted, TNstandard, WAstandard"
                    }
                ],
                "dependencies": ["chromadb", "sentence-transformers"],
                "estimated_time": "10-20 minutes"
            },
            3: {
                "name": "Similarity Preprocessing & Embedding",
                "description": "Process similarity results and generate embeddings",
                "scripts": [
                    {
                        "name": "Similarity Pipeline",
                        "path": "PreprocessingSimilarity/similarity_pipeline.py", 
                        "description": "Preprocess similarity results and create Qwen + Paraphrase embeddings"
                    }
                ],
                "dependencies": ["ollama", "sentence-transformers"],
                "estimated_time": "5-15 minutes"
            },
            4: {
                "name": "Standard Classification",
                "description": "Classify clauses as standard vs non-standard",
                "scripts": [
                    {
                        "name": "Classification Pipeline",
                        "path": "StandardClassification/pipeline.py",
                        "description": "Multi-step classification: exact match → semantic → NLI → rule flags"
                    }
                ],
                "dependencies": ["ollama", "transformers", "spacy"],
                "estimated_time": "20-40 minutes"
            },
            5: {
                "name": "Contract Analysis Chatbot",
                "description": "Start the contract analysis API server",
                "scripts": [
                    {
                        "name": "FastAPI Server",
                        "path": "backend/main.py",
                        "description": "Start contract analysis chatbot API server",
                        "background": True
                    }
                ],
                "dependencies": ["fastapi", "ollama"],
                "estimated_time": "Continuous (background)"
            }
        }
    
    def check_dependencies(self, step_num: int) -> bool:
        """Check if dependencies for a step are available."""
        step = self.steps[step_num]
        dependencies = step.get("dependencies", [])
        
        logger.info(f"Checking dependencies for Step {step_num}: {dependencies}")
        
        missing = []
        for dep in dependencies:
            if dep == "ollama":
                if not self.check_ollama():
                    missing.append("ollama (not running or missing models)")
            elif dep == "tesseract":
                if not self.check_tesseract():
                    missing.append("tesseract-ocr")
            else:
                if not self.check_python_package(dep):
                    missing.append(f"python package: {dep}")
        
        if missing:
            logger.warning(f"Missing dependencies for Step {step_num}: {missing}")
            return False
        
        logger.info(f"✅ All dependencies satisfied for Step {step_num}")
        return True
    
    def check_ollama(self) -> bool:
        """Check if Ollama is running with required models."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m.get("name", "") for m in response.json().get("models", [])]
                required_models = ["qwen3-embedding:0.6b", "phi3:mini"]
                missing_models = [m for m in required_models if m not in models]
                if missing_models:
                    logger.warning(f"Missing Ollama models: {missing_models}")
                    logger.info("Run: ollama pull qwen3-embedding:0.6b && ollama pull phi3:mini")
                    return False
                return True
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")
        return False
    
    def check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            result = subprocess.run(["tesseract", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def check_python_package(self, package: str) -> bool:
        """Check if Python package is installed."""
        try:
            __import__(package.replace("-", "_"))
            return True
        except ImportError:
            return False
    
    def run_script(self, script_config: dict, step_num: int) -> bool:
        """Run a single script."""
        script_path = self.project_root / script_config["path"]
        script_name = script_config["name"]
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        logger.info(f"🚀 Running: {script_name}")
        logger.info(f"📄 Script: {script_path}")
        logger.info(f"📝 Description: {script_config['description']}")
        
        # Prepare command
        cmd = [self.python_exe, str(script_path)]
        
        # Add step-specific arguments
        if step_num == 4:  # StandardClassification
            cmd.extend([
                "--db-path", str(self.project_root / "chroma_db_qwen"),
                "--input-dir", str(self.project_root / "outputs/precise_similarity/processed_datasets"),
                "--standard-dir", str(self.project_root / "StandardClassification/standard"),
                "--final-dir", str(self.project_root / "StandardClassification/standard_final")
            ])
        
        try:
            if script_config.get("background", False):
                # Run in background (for server)
                logger.info(f"Starting {script_name} in background...")
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info(f"✅ {script_name} started (PID: {process.pid})")
                logger.info("🌐 Server should be available at: http://localhost:8000")
                return True
            else:
                # Run synchronously
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    cwd=str(self.project_root),
                    check=True,
                    text=True,
                    capture_output=False
                )
                duration = time.time() - start_time
                logger.info(f"✅ {script_name} completed in {duration:.1f} seconds")
                return True
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {script_name} failed with exit code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"❌ {script_name} failed: {e}")
            return False
    
    def run_step(self, step_num: int) -> bool:
        """Run a complete pipeline step."""
        if step_num not in self.steps:
            logger.error(f"Invalid step number: {step_num}")
            return False
        
        step = self.steps[step_num]
        logger.info("=" * 80)
        logger.info(f"🔄 STEP {step_num}: {step['name']}")
        logger.info(f"📋 Description: {step['description']}")
        logger.info(f"⏱️ Estimated time: {step['estimated_time']}")
        logger.info("=" * 80)
        
        # Check dependencies
        if not self.check_dependencies(step_num):
            logger.error(f"❌ Step {step_num} failed: Missing dependencies")
            return False
        
        # Run all scripts in the step
        for script_config in step["scripts"]:
            if not self.run_script(script_config, step_num):
                logger.error(f"❌ Step {step_num} failed at script: {script_config['name']}")
                return False
        
        logger.info(f"✅ STEP {step_num} COMPLETED: {step['name']}")
        return True
    
    def run_pipeline(self, steps_to_run: Optional[List[int]] = None, 
                    steps_to_skip: Optional[List[int]] = None) -> bool:
        """Run the complete pipeline or specified steps."""
        
        if steps_to_run is None:
            steps_to_run = list(self.steps.keys())
        
        if steps_to_skip:
            steps_to_run = [s for s in steps_to_run if s not in steps_to_skip]
        
        logger.info("🚀 STARTING HILABS PREPROCESSING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"📁 Project root: {self.project_root}")
        logger.info(f"🐍 Python executable: {self.python_exe}")
        logger.info(f"📋 Steps to run: {steps_to_run}")
        if steps_to_skip:
            logger.info(f"⏭️ Steps to skip: {steps_to_skip}")
        logger.info("=" * 80)
        
        # Show pipeline overview
        self.show_pipeline_overview(steps_to_run)
        
        # Run each step
        start_time = time.time()
        failed_steps = []
        
        for step_num in steps_to_run:
            step_start = time.time()
            
            if self.run_step(step_num):
                step_duration = time.time() - step_start
                logger.info(f"✅ Step {step_num} completed in {step_duration:.1f} seconds")
            else:
                failed_steps.append(step_num)
                logger.error(f"❌ Step {step_num} failed")
                
                # Ask user if they want to continue
                if step_num < max(steps_to_run):
                    response = input(f"\nStep {step_num} failed. Continue with remaining steps? (y/N): ")
                    if response.lower() != 'y':
                        break
        
        # Pipeline summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        
        if failed_steps:
            logger.error(f"❌ PIPELINE COMPLETED WITH FAILURES")
            logger.error(f"Failed steps: {failed_steps}")
        else:
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        
        logger.info(f"⏱️ Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        logger.info(f"📊 Steps completed: {len(steps_to_run) - len(failed_steps)}/{len(steps_to_run)}")
        logger.info("=" * 80)
        
        return len(failed_steps) == 0
    
    def show_pipeline_overview(self, steps_to_run: List[int]):
        """Show overview of pipeline steps."""
        logger.info("📋 PIPELINE OVERVIEW:")
        logger.info("-" * 80)
        
        for step_num in steps_to_run:
            step = self.steps[step_num]
            logger.info(f"Step {step_num}: {step['name']}")
            logger.info(f"  📝 {step['description']}")
            logger.info(f"  ⏱️ Estimated time: {step['estimated_time']}")
            logger.info(f"  🔧 Dependencies: {', '.join(step.get('dependencies', []))}")
            
            for script in step['scripts']:
                logger.info(f"    → {script['name']}: {script['description']}")
            logger.info("")
        
        logger.info("-" * 80)
    
    def show_status(self):
        """Show current pipeline status and outputs."""
        logger.info("📊 PIPELINE STATUS CHECK")
        logger.info("=" * 80)
        
        # Check key directories and files
        checks = [
            ("Raw Data", self.project_root / "data/Contracts_data/WA/WA_5_Redacted.pdf"),
            ("PDF Parsed", self.project_root / "outputs/pdf_parsed/WA/WA_5_Redacted.md"),
            ("ChromaDB", self.project_root / "chroma_db_qwen"),
            ("Similarity Results", self.project_root / "outputs/precise_similarity"),
            ("Processed Datasets", self.project_root / "outputs/precise_similarity/processed_datasets"),
            ("Classification Results", self.project_root / "StandardClassification/classification_summary.json"),
            ("Chatbot Backend", self.project_root / "backend/routes/chatbot_routes.py")
        ]
        
        for name, path in checks:
            if path.exists():
                if path.is_file():
                    size = path.stat().st_size
                    logger.info(f"✅ {name}: {path} ({size:,} bytes)")
                else:
                    items = len(list(path.iterdir())) if path.is_dir() else 0
                    logger.info(f"✅ {name}: {path} ({items} items)")
            else:
                logger.info(f"❌ {name}: {path} (missing)")
        
        logger.info("=" * 80)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HiLABS Complete Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --step 3           # Run only step 3
  python main.py --skip-steps 1,2   # Skip steps 1 and 2
  python main.py --status           # Show pipeline status
        """
    )
    
    parser.add_argument(
        "--step", 
        type=int, 
        choices=[1, 2, 3, 4, 5],
        help="Run only specific step (1-5)"
    )
    
    parser.add_argument(
        "--skip-steps",
        type=str,
        help="Comma-separated list of steps to skip (e.g., '1,2')"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current pipeline status and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Handle status check
    if args.status:
        orchestrator.show_status()
        return 0
    
    # Determine steps to run
    steps_to_run = None
    steps_to_skip = None
    
    if args.step:
        steps_to_run = [args.step]
    
    if args.skip_steps:
        try:
            steps_to_skip = [int(s.strip()) for s in args.skip_steps.split(",")]
        except ValueError:
            logger.error("Invalid skip-steps format. Use comma-separated integers (e.g., '1,2')")
            return 1
    
    # Run pipeline
    try:
        success = orchestrator.run_pipeline(steps_to_run, steps_to_skip)
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\n🛑 Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Pipeline failed with unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
