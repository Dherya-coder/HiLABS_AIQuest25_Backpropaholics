#!/usr/bin/env python3
"""
PDF Processing Module

Extracts text and tables from PDFs using pdfplumber and OCR fallback.
Converts to markdown format with proper clause structure.
"""

import os
import re
import time
import argparse
import concurrent.futures
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter
from pathlib import Path
import sys

# ---------- Configuration ----------
# Tesseract will use system PATH on Linux
MAX_CORES = 6
FOOTER_HEIGHT = 60  # points or pixels to skip from bottom

# ---------- Helper functions ----------
def format_for_markdown(text):
    lines = text.splitlines()
    md_lines = []
    buffer_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^(SECTION|ARTICLE)\s+[IVXLCDM]+', line, re.IGNORECASE):
            if buffer_text:
                md_lines.append("\n".join(buffer_text))
                buffer_text = []
            md_lines.append(f"# {line}")
        elif re.match(r'^(\d+(\.\d+))(\s+.)?', line):
            if buffer_text:
                md_lines.append("\n".join(buffer_text))
                buffer_text = []
            match = re.match(r'^(\d+(\.\d+)*)', line)
            clause_no = match.group(1)
            md_lines.append(f"## {clause_no}")
            rest = line[len(clause_no):].strip()
            if rest:
                buffer_text.append(rest)
        elif line.startswith("//"):
            if buffer_text:
                md_lines.append("\n".join(buffer_text))
                buffer_text = []
            md_lines.append(line)
        else:
            buffer_text.append(line)

    if buffer_text:
        md_lines.append("\n".join(buffer_text))

    return "\n".join(md_lines)


def format_table_to_md(table):
    if not table:
        return ""
    headers = table[0]
    md_table = ["| " + " | ".join(str(cell or "") for cell in headers) + " |"]
    md_table.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in table[1:]:
        md_table.append("| " + " | ".join(str(cell or "") for cell in row) + " |")
    return "\n".join(md_table)


# ---------- Top-level OCR worker ----------
def ocr_for_page(args):
    print("OCR On page: ", args[0] + 1)
    page_num, image, footer_height = args
    w, h = image.size
    cropped_img = image.crop((0, 0, w, h - footer_height))
    text = pytesseract.image_to_string(cropped_img, config="--oem 3 --psm 6")
    return f"\n\n--- Page {page_num + 1} (OCR) ---\n\n{format_for_markdown(text)}"


# ---------- Step 0: Create intermediate "no-footer" PDF ----------
def create_cropped_pdf(input_pdf: Path, output_pdf: Path, footer_height: int):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    for page in reader.pages:
        # Crop bottom FOOTER_HEIGHT points
        page.mediabox.lower_left = (
            page.mediabox.lower_left[0],
            page.mediabox.lower_left[1] + footer_height
        )
        writer.add_page(page)
    with open(output_pdf, "wb") as f:
        writer.write(f)
    print(f"✅ Cropped no-footer PDF saved as {output_pdf}")


# ---------- Main extraction ----------
def process_pdf(input_pdf: Path, output_dir: Path, footer_height: int = FOOTER_HEIGHT) -> None:
    """Process a single PDF file and save as markdown."""
    start_time = time.time()
    num_cores = min(os.cpu_count(), MAX_CORES)
    print(f"💡 Using {num_cores} CPU cores for OCR")
    print(f"📄 Processing: {input_pdf.name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define intermediate and output files
    cropped_pdf = output_dir / f"{input_pdf.stem}_cropped.pdf"
    output_md = output_dir / f"{input_pdf.stem}.md"
    
    # Step 0: Create cropped PDF
    create_cropped_pdf(input_pdf, cropped_pdf, footer_height)

    page_texts = []
    pages_for_ocr = []

    # Step 1: Extract text & tables from cropped PDF
    CROP_HEIGHT = 60  # points from bottom to remove

    with pdfplumber.open(cropped_pdf) as pdf:
        for i, page in enumerate(pdf.pages):
            page_content = []

            # PDF origin is bottom-left
            x0, y0, x1, y1 = page.bbox  # actual page bbox
            cropped_page = page.within_bbox((x0, y0 + CROP_HEIGHT, x1, y1))  # remove bottom CROP_HEIGHT

            # Extract text
            text = cropped_page.extract_text()
            if text and text.strip():
                page_content.append(format_for_markdown(text))
            else:
                pages_for_ocr.append(i)
                page_content.append(None)

            # Extract tables
            tables = cropped_page.extract_tables()
            for table in tables:
                page_content.append(format_table_to_md(table))

            page_texts.append(page_content)

    # Step 2: OCR for pages with no text
    if pages_for_ocr:
        print(f"🔍 Performing OCR on {len(pages_for_ocr)} pages: {', '.join(str(p + 1) for p in pages_for_ocr)}")
        images = []
        for pg_num in pages_for_ocr:
            # pdf2image uses 1-based page numbers
            img = convert_from_path(
                cropped_pdf,
                dpi=200,
                first_page=pg_num + 1,
                last_page=pg_num + 1
            )[0]  # convert_from_path always returns a list
            images.append(img)
        print("performed OCR")

        args_list = [(pages_for_ocr[i], images[i], FOOTER_HEIGHT) for i in range(len(pages_for_ocr))]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(ocr_for_page, args_list))

        for idx, page_num in enumerate(pages_for_ocr):
            page_texts[page_num] = [results[idx]]

    # Step 3: Write Markdown
    with open(output_md, "w", encoding="utf-8") as f:
        for i, content_list in enumerate(page_texts):
            f.write(f"\n\n--- Page {i + 1} ---\n\n")
            for content in content_list:
                if content:
                    f.write(content + "\n\n")

    print(f"✅ Extracted text saved to {output_md}")
    print(f"⏱ Total duration: {time.time() - start_time:.2f} seconds")
    
    # Step 4: Delete intermediate cropped PDF
    try:
        cropped_pdf.unlink()
        print(f"🗑️ Deleted intermediate file: {cropped_pdf}")
    except Exception as e:
        print(f"⚠️ Could not delete {cropped_pdf}: {e}")


def main():
    parser = argparse.ArgumentParser(description="PDF to Markdown converter with OCR support")
    parser.add_argument("--input-pdf", type=str, required=True, 
                       help="Path to input PDF file")
    parser.add_argument("--output-dir", type=str, required=True, 
                       help="Output directory for markdown files")
    parser.add_argument("--footer-height", type=int, default=FOOTER_HEIGHT, 
                       help="Footer height to crop (in points)")
    
    args = parser.parse_args()
    
    input_pdf = Path(args.input_pdf)
    output_dir = Path(args.output_dir)
    
    if not input_pdf.exists():
        raise SystemExit(f"Input PDF not found: {input_pdf}")
    
    process_pdf(input_pdf, output_dir, args.footer_height)


if __name__ == "__main__":
    main()
