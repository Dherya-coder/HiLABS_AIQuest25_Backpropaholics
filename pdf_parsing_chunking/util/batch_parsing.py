#!/usr/bin/env python3
"""
Batch PDF Processing Script

Processes multiple PDFs from a directory and saves markdown outputs.
Handles the 4 categories: TN, WA, TN_standard, WA_standard

Usage:
    python batch_parsing.py --input-dir "../data/Contracts_data/TN" --output-dir "../outputs/pdf_parsed/TN"
"""

import argparse
import os
import time
from pathlib import Path
from typing import List
import subprocess
import sys

def process_single_pdf(input_pdf: Path, output_dir: Path) -> bool:
    """Process a single PDF using the parsing.py script."""
    cmd = [
        sys.executable, "parsing.py",
        "--input-pdf", str(input_pdf),
        "--output-dir", str(output_dir)
    ]
    
    try:
        print(f"🔄 Processing: {input_pdf.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Completed: {input_pdf.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {input_pdf.name}")
        print(f"Error: {e.stderr}")
        return False

def batch_process_pdfs(input_dir: Path, output_dir: Path, file_pattern: str = "*.pdf") -> None:
    """Process all PDFs in a directory."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(input_dir.glob(file_pattern))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files in {input_dir}")
    print(f"📤 Output directory: {output_dir}")
    
    start_time = time.time()
    success_count = 0
    
    # Process each PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file.name}")
        
        if process_single_pdf(pdf_file, output_dir):
            success_count += 1
    
    # Summary
    duration = time.time() - start_time
    print(f"\n" + "="*50)
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"="*50)
    print(f"Total files: {len(pdf_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(pdf_files) - success_count}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average: {duration/len(pdf_files):.2f} seconds per file")
    
    if success_count == len(pdf_files):
        print("🎉 All files processed successfully!")
    else:
        print(f"⚠️ {len(pdf_files) - success_count} files failed to process")

def main():
    parser = argparse.ArgumentParser(description="Batch PDF processing")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for markdown files")
    parser.add_argument("--file-pattern", type=str, default="*.pdf", help="File pattern to match (default: *.pdf)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    
    batch_process_pdfs(input_dir, output_dir, args.file_pattern)

if __name__ == "__main__":
    main()
