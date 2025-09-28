
# Hybrid PDF Parser Script Documentation

This Python script implements a hybrid PDF parser that combines multiple tools to extract text, tables, and images from PDF documents.
It uses pdfplumber for text and table extraction, PyMuPDF (fitz) to detect images on textless pages, and PyTesseract for OCR processing when pdfplumber fails.
Additionally, it includes text normalization, header/footer removal, and clause extraction functionalities.

---

## Dependencies

- pdfplumber
- PyMuPDF (`fitz`)
- pytesseract
- Pillow (`PIL`)
- re (regular expressions)
- json, os, time, pathlib

---

## Function Overview

### 1. normalize_text(text: str) -> str

Normalizes input text by:
- Unifying line endings
- Fixing hyphenation across line breaks
- Collapsing multiple spaces and blank lines
- Standardizing punctuation marks
- Trimming whitespace

### 2. normalize_parsed_json(parsed_json)

Processes parsed JSON to add normalized text fields.

### 3. remove_headers_footers(parsed_json, threshold=0.5)

Removes repeated headers and footers from the parsed content.

### 4. extract_clauses_from_text(text: str) -> list

Extracts numbered and bullet-pointed clauses from text.

### 5. extract_all_clauses(parsed_json)

Extracts clauses from normalized JSON, excluding tables.

### 6. parse_pdf_hybrid_with_ocr(pdf_path: str) -> Dict[str, Any]

Main hybrid parsing combining pdfplumber, PyMuPDF, and pytesseract OCR.

### 7. save_parsed_json(data: Dict[str, Any], output_path: str) -> None

Saves JSON output to the specified path.

---

## Main Workflow (main())

- Parses sample PDF file.
- Saves parsed JSON.
- Normalizes text.
- Extracts clauses.
- Saves clauses to JSON.

---

## Usage Notes

- Tables are extracted and preserved separately.
- Clauses include numbered and bullet points.
- OCR improves text extraction on image pages.

---

## File Output Locations

- Parsed JSON: outputs/parsed/
- Normalized JSON: outputs/parsed_normalized/
- Clauses JSON: outputs/parsed/clauses.json

---
