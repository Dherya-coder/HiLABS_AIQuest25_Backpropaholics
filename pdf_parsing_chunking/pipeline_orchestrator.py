#!/usr/bin/env python3
"""
Complete PDF Processing Pipeline Orchestrator

Processes 12 PDFs across 4 categories:
- TN (5 PDFs) -> TNredacted collection
- WA (5 PDFs) -> WAredacted collection  
- TN Standard (1 PDF) -> TNstandard collection
- WA Standard (1 PDF) -> WAstandard collection

Pipeline: PDF -> Parsing -> Chunking -> Embeddings -> ChromaDB
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import json

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

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def process_parsing_stage(category: str, config: Dict) -> bool:
    """Process parsing stage for a category."""
    print(f"\n📄 PARSING STAGE: {category}")
    
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "input_file" in config:
        # Single file processing (standards)
        input_file = input_dir / config["input_file"]
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            return False
            
        cmd = [
            "python", "parsing.py",
            "--input-pdf", str(input_file),
            "--output-dir", str(output_dir)
        ]
        return run_command(cmd, f"Parse {config['input_file']}")
    else:
        # Multiple file processing
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {input_dir}")
            return False
            
        success_count = 0
        for pdf_file in pdf_files:
            cmd = [
                "python", "parsing.py", 
                "--input-pdf", str(pdf_file),
                "--output-dir", str(output_dir)
            ]
            if run_command(cmd, f"Parse {pdf_file.name}"):
                success_count += 1
                
        print(f"📊 Parsing completed: {success_count}/{len(pdf_files)} files processed")
        return success_count > 0

def process_chunking_stage(category: str, config: Dict) -> bool:
    """Process chunking stage for a category."""
    print(f"\n🔪 CHUNKING STAGE: {category}")
    
    output_dir = Path(config["output_dir"])
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return False
        
    md_files = list(output_dir.glob("*.md"))
    if not md_files:
        print(f"❌ No markdown files found in {output_dir}")
        return False
        
    cmd = [
        "python", "chunking.py",
        "--input-dir", str(output_dir),
        "--output-dir", str(output_dir)
    ]
    
    return run_command(cmd, f"Chunk markdown files in {category}")

def process_embedding_stage(category: str, config: Dict) -> bool:
    """Process embedding stage for a category."""
    print(f"\n🧠 EMBEDDING STAGE: {category}")
    
    output_dir = Path(config["output_dir"])
    collection_name = config["collection_name"]
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return False
        
    json_files = list(output_dir.glob("*_chunks.json"))
    if not json_files:
        print(f"❌ No chunk JSON files found in {output_dir}")
        return False
        
    cmd = [
        "python", "embedding.py",
        "--chunks-dir", str(output_dir),
        "--collection-name", collection_name,
        "--db-path", "../chroma_db_qwen"
    ]
    
    return run_command(cmd, f"Generate embeddings for {category}")

def create_pipeline_summary(results: Dict) -> None:
    """Create a summary of pipeline execution."""
    print("\n" + "="*60)
    print("📋 PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    total_categories = len(PIPELINE_CONFIG)
    successful_categories = sum(1 for success in results.values() if success)
    
    for category, config in PIPELINE_CONFIG.items():
        status = "✅ SUCCESS" if results.get(category, False) else "❌ FAILED"
        print(f"{category:12} | {config['description']:25} | {status}")
    
    print(f"\nOverall: {successful_categories}/{total_categories} categories completed successfully")
    
    # Save summary to file
    summary_file = Path("../outputs/pipeline_summary.json")
    summary_data = {
        "pipeline_config": PIPELINE_CONFIG,
        "execution_results": results,
        "success_rate": f"{successful_categories}/{total_categories}",
        "successful_categories": successful_categories,
        "total_categories": total_categories
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Pipeline summary saved to: {summary_file}")

def main():
    print("🚀 Starting Complete PDF Processing Pipeline")
    print("="*60)
    
    # Display configuration
    print("📋 Pipeline Configuration:")
    for category, config in PIPELINE_CONFIG.items():
        print(f"  {category}: {config['description']}")
    
    results = {}
    
    # Process each category through the complete pipeline
    for category, config in PIPELINE_CONFIG.items():
        print(f"\n{'='*20} PROCESSING {category.upper()} {'='*20}")
        
        category_success = True
        
        # Stage 1: Parsing
        if not process_parsing_stage(category, config):
            category_success = False
        
        # Stage 2: Chunking (only if parsing succeeded)
        if category_success and not process_chunking_stage(category, config):
            category_success = False
            
        # Stage 3: Embeddings (only if chunking succeeded)  
        if category_success and not process_embedding_stage(category, config):
            category_success = False
            
        results[category] = category_success
        
        if category_success:
            print(f"🎉 {category} pipeline completed successfully!")
        else:
            print(f"💥 {category} pipeline failed!")
    
    # Create summary
    create_pipeline_summary(results)
    
    # Final status
    if all(results.values()):
        print("\n🎊 ALL PIPELINES COMPLETED SUCCESSFULLY! 🎊")
        return 0
    else:
        print("\n⚠️  SOME PIPELINES FAILED - CHECK LOGS ABOVE")
        return 1

if __name__ == "__main__":
    sys.exit(main())
