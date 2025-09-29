from pathlib import Path
import json
import re
import argparse
import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ----------------------------- Configuration ---------------------------------
# NOTE: These constants define how markdown is segmented and how chunks are sized.
HEADERS_TO_SPLIT_ON = [("#", "section"), ("##", "clause"), ("###", "subclause")]
CHUNK_SIZE = 300         # Token budget per chunk (GPT tokens)
CHUNK_OVERLAP = 100      # Overlap between consecutive chunks
OUT_DIR_NAME = "pdf_parsed"  # Unused constant retained for compatibility
# -----------------------------------------------------------------------------


# Tokenizer setup for token-aware splitting
enc = tiktoken.get_encoding("cl100k_base")


def token_len(txt: str) -> int:
    """Return token length for a given string using the configured tokenizer."""
    return len(enc.encode(txt))


def parse_page_markers(text: str) -> list[dict]:
    """
    Split a single markdown string into page segments using markers:
        --- Page N ---
    Returns:
        List of dicts with keys:
            - page_number: int
            - content: str
    """
    pages = []
    pattern = re.compile(r"--- Page (\d+) ---")
    parts = pattern.split(text)

    # parts[0] is content before the first marker (often empty)
    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        page_text = parts[i + 1].strip()
        pages.append({"page_number": page_number, "content": page_text})

    return pages


def chunk_markdown_files(folder: Path, out_dir: Path) -> None:
    """
    Read all .md files in `folder`, segment by page markers, perform header-aware
    and token-aware splitting, and write one JSON file of chunks per input file
    into `out_dir`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Splitter that tracks markdown header hierarchy in metadata
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)

    # Token-aware splitter for final chunking
    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for md in sorted(folder.glob("*.md")):
        raw_markdown = md.read_text(encoding="utf-8")
        if not raw_markdown.strip():
            # Keep the same print message (including symbol) for compatibility
            print(f"⚠ {md.name} is empty, skipping")
            continue

        # Page-level split
        pages = parse_page_markers(raw_markdown)

        all_chunks = []
        last_header_context = {}  # Persist header context across pages

        for page in pages:
            page_number = page["page_number"]
            page_text = page["content"]

            # Lightweight cleanup to remove OCR markers and table artifacts
            cleaned_text = re.sub(r"--- Page \d+ \(OCR\) ---", "", page_text)
            cleaned_text = re.sub(
                r"\|\s*\|\s*\|\s*\n\|\s*---\s*\|\s*---\s*\|", "", cleaned_text
            ).strip()

            # 1) Header-aware segmentation -> list[Document]
            header_docs = header_splitter.split_text(cleaned_text)

            # 2) Token-aware chunking for each Document.page_content
            for doc in header_docs:
                # Current header metadata for this document
                current_headers = doc.metadata.copy()

                # Inherit from previous page when a page lacks headers
                if not current_headers and last_header_context:
                    current_headers = last_header_context.copy()

                # Update persistent context when headers are present
                if current_headers:
                    last_header_context.update(current_headers)

                # Build a readable header path for traceability
                header_path = (
                    " > ".join(f"{lvl}:{val}" for lvl, val in current_headers.items())
                    if current_headers
                    else "document_content"
                )

                # Final token-aware split
                splits = token_splitter.split_text(doc.page_content)

                for i, chunk in enumerate(splits, 1):
                    content = chunk.strip()
                    if not content:
                        continue

                    # Maintain exact chunk_id format and ordering
                    chunk_entry = {
                        "chunk_id": f"{md.stem}_p{page_number}_c{len(all_chunks) + 1}",
                        "page_number": page_number,
                        "header_path": header_path,
                        "chunk_index": i,
                        "content": content,
                        "token_count": token_len(content),
                        "source_file": md.name,
                        "document_type": "legal_contract",
                    }
                    all_chunks.append(chunk_entry)

        # Persist chunks per source file
        out_path = out_dir / f"{md.stem}_chunks.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        print(f"{md.name}: wrote {len(all_chunks)} chunks → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk markdown files into JSON")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../outputs/pdf_parsed",
        help="Directory containing markdown files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../outputs/pdf_parsed",
        help="Output directory for JSON chunks",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    print(f"Looking for markdown files in: {input_dir}")
    print(f"Output chunks will be saved to: {output_dir}")

    chunk_markdown_files(input_dir, output_dir)
