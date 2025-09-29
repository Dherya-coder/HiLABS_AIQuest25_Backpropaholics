#!/usr/bin/env python3
"""
Simplified Pipeline Runner

Handles the complete PDF processing pipeline with dependency checking.
"""

import subprocess
import sys
from pathlib import Path
import json

# Pipeline configuration
CATEGORIES = {
    "TNredacted": {
        "input_dir": "../data/Contracts_data/TN",
        "output_dir": "../outputs/pdf_parsed/TN",
        "collection": "TNredacted"
    },
    "WAredacted": {
        "input_dir": "../data/Contracts_data/WA", 
        "output_dir": "../outputs/pdf_parsed/WA",
        "collection": "WAredacted"
    },
    "TNstandard": {
        "input_dir": "../data/Contracts_data/Standard Templates_data",
        "input_file": "TN_Standard_Template_Redacted.pdf",
        "output_dir": "../outputs/pdf_parsed/TN_standard",
        "collection": "TNstandard"
    },
    "WAstandard": {
        "input_dir": "../data/Contracts_data/Standard Templates_data",
        "input_file": "WA_Standard_Redacted.pdf",
        "output_dir": "../outputs/pdf_parsed/WA_standard", 
        "collection": "WAstandard"
    }
}

def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = ['pdfplumber', 'pytesseract', 'pdf2image', 'pypdf', 'tiktoken', 'langchain_text_splitters']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"❌ {module}")
    
    if missing:
        print(f"\n⚠️ Missing dependencies: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies available")
    return True

def run_cmd(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def process_category(category, config):
    """Process one category through the pipeline."""
    print(f"\n{'='*20} {category.upper()} {'='*20}")
    
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    collection = config["collection"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Step 1: Parsing
    if "input_file" in config:
        # Single file
        input_file = input_dir / config["input_file"]
        cmd = f'python3 parsing.py --input-pdf "{input_file}" --output-dir "{output_dir}"'
        if not run_cmd(cmd, f"Parse {config['input_file']}"):
            success = False
    else:
        # Multiple files
        for pdf_file in input_dir.glob("*.pdf"):
            cmd = f'python3 parsing.py --input-pdf "{pdf_file}" --output-dir "{output_dir}"'
            if not run_cmd(cmd, f"Parse {pdf_file.name}"):
                success = False
    
    if not success:
        return False
    
    # Step 2: Chunking
    cmd = f'python3 chunking.py --input-dir "{output_dir}" --output-dir "{output_dir}"'
    if not run_cmd(cmd, f"Chunk {category}"):
        return False
    
    # Step 3: Embeddings
    cmd = f'python3 embedding.py --chunks-dir "{output_dir}" --collection-name "{collection}"'
    if not run_cmd(cmd, f"Generate embeddings for {category}"):
        return False
    
    return True

def main():
    print("🚀 PDF Processing Pipeline")
    print("="*50)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    if not check_dependencies():
        print("❌ Please install missing dependencies first")
        return 1
    
    # Process each category
    results = {}
    for category, config in CATEGORIES.items():
        results[category] = process_category(category, config)
    
    # Summary
    print("\n" + "="*50)
    print("📊 PIPELINE SUMMARY")
    print("="*50)
    
    successful = sum(results.values())
    total = len(results)
    
    for category, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{category:12} | {status}")
    
    print(f"\nOverall: {successful}/{total} categories completed")
    
    # Save results
    summary_file = Path("../outputs/pipeline_results.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump({
            "categories": CATEGORIES,
            "results": results,
            "success_rate": f"{successful}/{total}"
        }, f, indent=2)
    
    print(f"📄 Results saved to: {summary_file}")
    
    if successful == total:
        print("\n🎉 ALL PIPELINES COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print("\n⚠️ SOME PIPELINES FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
