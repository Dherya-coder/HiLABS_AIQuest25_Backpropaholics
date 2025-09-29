#!/usr/bin/env python3
"""
Complete PDF Processing Pipeline

Proper Python implementation using imports instead of shell commands.
Processes 12 PDFs across 4 categories with proper error handling.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Import our modules
from util.parsing import process_pdf
from util.chunking import chunk_markdown_files
from util.embedding import process_contract_chunks_to_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pipeline configuration
PIPELINE_CONFIG = {
    "TNredacted": {
        "input_dir": "../data/Contracts_data/TN",
        "output_dir": "../outputs/pdf_parsed/TN",
        "collection_name": "TNredacted",
        "description": "TN Contract PDFs (5 files)"
    },
    "WAredacted": {
        "input_dir": "../data/Contracts_data/WA", 
        "output_dir": "../outputs/pdf_parsed/WA",
        "collection_name": "WAredacted",
        "description": "WA Contract PDFs (5 files)"
    },
    "TNstandard": {
        "input_dir": "../data/Contracts_data/Standard Templates_data",
        "input_file": "TN_Standard_Template_Redacted.pdf",
        "output_dir": "../outputs/pdf_parsed/TN_standard",
        "collection_name": "TNstandard", 
        "description": "TN Standard Template (1 file)"
    },
    "WAstandard": {
        "input_dir": "../data/Contracts_data/Standard Templates_data",
        "input_file": "WA_Standard_Redacted.pdf", 
        "output_dir": "../outputs/pdf_parsed/WA_standard",
        "collection_name": "WAstandard",
        "description": "WA Standard Template (1 file)"
    }
}

class PipelineProcessor:
    """Main pipeline processor class."""
    
    def __init__(self, db_path: str = "../chroma_db_qwen"):
        self.db_path = db_path
        self.results = {}
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        required_modules = [
            'pdfplumber', 'pytesseract', 'pdf2image', 'pypdf', 
            'tiktoken', 'langchain_text_splitters', 'chromadb', 'requests'
        ]
        missing = []
        
        logger.info("Checking dependencies...")
        for module in required_modules:
            try:
                __import__(module)
                logger.debug(f"✅ {module}")
            except ImportError:
                missing.append(module)
                logger.error(f"❌ {module}")
        
        if missing:
            logger.error(f"Missing dependencies: {missing}")
            logger.info("Install with: pip install " + " ".join(missing))
            return False
        
        logger.info("✅ All dependencies available")
        return True
    
    def process_parsing_stage(self, category: str, config: Dict) -> bool:
        """Process parsing stage for a category."""
        logger.info(f"📄 PARSING STAGE: {category}")
        
        input_dir = Path(config["input_dir"])
        output_dir = Path(config["output_dir"])
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        total_files = 0
        
        try:
            if "input_file" in config:
                # Single file processing (standards)
                input_file = input_dir / config["input_file"]
                if not input_file.exists():
                    logger.error(f"Input file not found: {input_file}")
                    return False
                
                total_files = 1
                logger.info(f"Processing: {config['input_file']}")
                
                try:
                    process_pdf(input_file, output_dir)
                    success_count = 1
                    logger.info(f"✅ Successfully processed: {config['input_file']}")
                except Exception as e:
                    logger.error(f"❌ Failed to process {config['input_file']}: {e}")
                    
            else:
                # Multiple file processing
                pdf_files = list(input_dir.glob("*.pdf"))
                if not pdf_files:
                    logger.error(f"No PDF files found in {input_dir}")
                    return False
                
                total_files = len(pdf_files)
                logger.info(f"Found {total_files} PDF files to process")
                
                for pdf_file in pdf_files:
                    logger.info(f"Processing: {pdf_file.name}")
                    try:
                        process_pdf(pdf_file, output_dir)
                        success_count += 1
                        logger.info(f"✅ Successfully processed: {pdf_file.name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error in parsing stage for {category}: {e}")
            return False
        
        logger.info(f"📊 Parsing completed: {success_count}/{total_files} files processed")
        return success_count > 0
    
    def process_chunking_stage(self, category: str, config: Dict) -> bool:
        """Process chunking stage for a category."""
        logger.info(f"🔪 CHUNKING STAGE: {category}")
        
        output_dir = Path(config["output_dir"])
        
        if not output_dir.exists():
            logger.error(f"Output directory not found: {output_dir}")
            return False
            
        md_files = list(output_dir.glob("*.md"))
        if not md_files:
            logger.error(f"No markdown files found in {output_dir}")
            return False
        
        logger.info(f"Found {len(md_files)} markdown files to chunk")
        
        try:
            chunk_markdown_files(output_dir, output_dir)
            logger.info(f"✅ Successfully chunked files for {category}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to chunk files for {category}: {e}")
            return False
    
    def process_embedding_stage(self, category: str, config: Dict) -> bool:
        """Process embedding stage for a category."""
        logger.info(f"🧠 EMBEDDING STAGE: {category}")
        
        output_dir = Path(config["output_dir"])
        collection_name = config["collection_name"]
        
        if not output_dir.exists():
            logger.error(f"Output directory not found: {output_dir}")
            return False
            
        json_files = list(output_dir.glob("*_chunks.json"))
        if not json_files:
            logger.error(f"No chunk JSON files found in {output_dir}")
            return False
        
        logger.info(f"Found {len(json_files)} chunk files for embedding")
        
        try:
            process_contract_chunks_to_embeddings(
                chunks_dir=output_dir,
                collection_name=collection_name,
                db_path=self.db_path,
                model_name="qwen3-embedding:0.6b",
                batch_size=5
            )
            logger.info(f"✅ Successfully generated embeddings for {category}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings for {category}: {e}")
            return False
    
    def process_category(self, category: str, config: Dict) -> bool:
        """Process one category through the complete pipeline."""
        logger.info(f"\n{'='*20} PROCESSING {category.upper()} {'='*20}")
        
        # Stage 1: Parsing
        if not self.process_parsing_stage(category, config):
            logger.error(f"Parsing failed for {category}")
            return False
        
        # Stage 2: Chunking
        if not self.process_chunking_stage(category, config):
            logger.error(f"Chunking failed for {category}")
            return False
            
        # Stage 3: Embeddings
        if not self.process_embedding_stage(category, config):
            logger.error(f"Embedding generation failed for {category}")
            return False
            
        logger.info(f"🎉 {category} pipeline completed successfully!")
        return True
    
    def create_summary(self) -> None:
        """Create a summary of pipeline execution."""
        logger.info("\n" + "="*60)
        logger.info("📋 PIPELINE EXECUTION SUMMARY")
        logger.info("="*60)
        
        total_categories = len(PIPELINE_CONFIG)
        successful_categories = sum(1 for success in self.results.values() if success)
        
        for category, config in PIPELINE_CONFIG.items():
            status = "✅ SUCCESS" if self.results.get(category, False) else "❌ FAILED"
            logger.info(f"{category:12} | {config['description']:25} | {status}")
        
        logger.info(f"\nOverall: {successful_categories}/{total_categories} categories completed successfully")
        
        # Save summary to file
        summary_file = Path("../outputs/pipeline_summary.json")
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            "pipeline_config": PIPELINE_CONFIG,
            "execution_results": self.results,
            "success_rate": f"{successful_categories}/{total_categories}",
            "successful_categories": successful_categories,
            "total_categories": total_categories
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 Pipeline summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline for all categories."""
        logger.info("🚀 Starting Complete PDF Processing Pipeline")
        logger.info("="*60)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Dependencies check failed")
            return False
        
        # Display configuration
        logger.info("📋 Pipeline Configuration:")
        for category, config in PIPELINE_CONFIG.items():
            logger.info(f"  {category}: {config['description']}")
        
        # Process each category
        for category, config in PIPELINE_CONFIG.items():
            try:
                self.results[category] = self.process_category(category, config)
            except Exception as e:
                logger.error(f"Unexpected error processing {category}: {e}")
                self.results[category] = False
        
        # Create summary
        self.create_summary()
        
        # Final status
        all_successful = all(self.results.values())
        if all_successful:
            logger.info("\n🎊 ALL PIPELINES COMPLETED SUCCESSFULLY! 🎊")
        else:
            logger.warning("\n⚠️  SOME PIPELINES FAILED - CHECK LOGS ABOVE")
        
        return all_successful

def main():
    """Main entry point."""
    try:
        processor = PipelineProcessor()
        success = processor.run_pipeline()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
