#!/usr/bin/env python3
"""
Simple attribute chunking - create one JSON chunk per attribute row.
Keep it simple like the contract chunks in chunked123 folder.

Usage:
    python simple_attribute_chunking.py --excel-file "AttributeDictionary.xlsx" --output-dir "../outputs/attributes_chunks"
    
    Or simply run without arguments to use defaults:
    python simple_attribute_chunking.py
"""

import argparse
import json
import pandas as pd
import tiktoken
from pathlib import Path
from typing import Dict, Any

# Initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(enc.encode(text))

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Remove white box characters (redaction marks)
    text = text.replace("█", "").replace("████", "").replace("█████", "")
    # Clean up extra spaces
    text = " ".join(text.split())
    return text

def process_excel_to_simple_chunks(excel_path: Path, output_dir: Path) -> None:
    """Convert Excel to simple JSON chunks - one per attribute."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read Excel file
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
        print(f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        raise Exception(f"Error reading Excel file: {e}")
    
    all_chunks = []
    
    # Process each row as one chunk
    for row_idx, row in df.iterrows():
        # Get attribute name (first column)
        attribute_name = clean_text(row.iloc[0]) if len(row) > 0 else f"Attribute_{row_idx + 1}"
        
        if not attribute_name:
            continue
        
        # Concatenate all column values without column names for cleaner embedding
        content_parts = []
        for col_name, value in row.items():
            clean_value = clean_text(value)
            if clean_value:
                content_parts.append(clean_value)
        
        # Create full content for embedding (just the values, no column labels)
        full_content = " ".join(content_parts)
        
        if not full_content.strip():
            continue
        
        # Create simple chunk structure (similar to contract chunks)
        chunk_data = {
            "chunk_id": f"attr_{row_idx + 1:03d}",
            "attribute_number": row_idx + 1,
            "attribute_name": attribute_name,
            "content": full_content,
            "token_count": count_tokens(full_content),
            "source_file": excel_path.name,
            "document_type": "attribute_definition"
        }
        
        all_chunks.append(chunk_data)
        print(f"Created chunk {row_idx + 1}: {attribute_name}")
    
    # Save all chunks in one JSON file (like contract chunks)
    output_file = output_dir / f"{excel_path.stem}_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created {len(all_chunks)} attribute chunks")
    print(f"📁 Saved to: {output_file}")
    
    # Show sample
    if all_chunks:
        print(f"\n📄 Sample chunk structure:")
        sample = all_chunks[0]
        for key, value in sample.items():
            if key == "content":
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"  {key}: {preview}")
            else:
                print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Simple attribute chunking")
    parser.add_argument("--excel-file", type=str, default="AttributeDictionary.xlsx", help="Path to Excel file (default: AttributeDictionary.xlsx)")
    parser.add_argument("--output-dir", type=str, default="../outputs/attributes_chunks", help="Output directory (default: ../outputs/attributes_chunks)")
    
    args = parser.parse_args()
    
    excel_path = Path(args.excel_file)
    output_dir = Path(args.output_dir)
    
    if not excel_path.exists():
        raise SystemExit(f"Excel file not found: {excel_path}")
    
    process_excel_to_simple_chunks(excel_path, output_dir)

if __name__ == "__main__":
    main()
