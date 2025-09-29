#!/usr/bin/env python3
"""
WA_5_Redacted.pdf Processing Pipeline

Single script to parse, chunk, and embed WA_5_Redacted.pdf (99MB) with memory optimization.
Stores results in the WAredacted ChromaDB collection.

Usage (from anywhere):
    python parsing&embedding/process_wa5_pipeline.py
    
Features:
- Memory-optimized processing for large PDF
- PDF parsing with OCR fallback
- Chunking with proper metadata
- Qwen embeddings via Ollama
- Direct storage in WAredacted collection
"""

import os
import re
import time
import json
import logging
import requests
import numpy as np
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any
import gc
import psutil

# PDF processing
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter

# ChromaDB
import chromadb
from chromadb.config import Settings

# Text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get repository root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Configuration
CONFIG = {
    "input_pdf": PROJECT_ROOT / "data/Contracts_data/WA/WA_5_Redacted.pdf",
    "output_dir": PROJECT_ROOT / "outputs/pdf_parsed/WA",
    "db_path": PROJECT_ROOT / "chroma_db_qwen",
    "collection_name": "WAredacted",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "batch_size": 5,
    "max_cores": 4,
    "footer_height": 60,
    "embedding_model": "qwen3-embedding:0.6b",
    "ollama_url": "http://localhost:11434"
}

class MemoryTracker:
    """Track memory usage during processing."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024
    
    def log_memory(self, stage: str):
        memory_mb = self.get_memory_mb()
        logger.info(f"Memory usage at {stage}: {memory_mb:.1f} MB")

def format_for_markdown(text: str) -> str:
    """Format text for markdown with proper clause structure."""
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

def ocr_for_page(args) -> str:
    """OCR processing for a single page - standalone function for multiprocessing."""
    page_num, image, footer_height = args
    logger.info(f"OCR processing page {page_num + 1}")
    w, h = image.size
    cropped_img = image.crop((0, 0, w, h - footer_height))
    text = pytesseract.image_to_string(cropped_img, config="--oem 3 --psm 6")
    return f"\n\n--- Page {page_num + 1} (OCR) ---\n\n{format_for_markdown(text)}"

class WA5Processor:
    """Process WA_5_Redacted.pdf with memory optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_tracker = MemoryTracker()
        self.pdf_path = Path(config["input_pdf"])
        self.output_dir = Path(config["output_dir"])
        self.output_md = self.output_dir / f"{self.pdf_path.stem}.md"
        

    def format_table_to_md(self, table) -> str:
        """Format table to markdown."""
        if not table:
            return ""
        headers = table[0]
        md_table = ["| " + " | ".join(str(cell or "") for cell in headers) + " |"]
        md_table.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in table[1:]:
            md_table.append("| " + " | ".join(str(cell or "") for cell in row) + " |")
        return "\n".join(md_table)


    def create_cropped_pdf(self, input_pdf: Path, output_pdf: Path, footer_height: int):
        """Create PDF with footer removed."""
        reader = PdfReader(input_pdf)
        writer = PdfWriter()
        for page in reader.pages:
            page.mediabox.lower_left = (
                page.mediabox.lower_left[0],
                page.mediabox.lower_left[1] + footer_height
            )
            writer.add_page(page)
        with open(output_pdf, "wb") as f:
            writer.write(f)
        logger.info(f"✅ Created cropped PDF: {output_pdf}")

    def step1_parse_pdf(self) -> bool:
        """Step 1: Parse PDF to markdown with memory optimization."""
        logger.info("🔄 Step 1: Parsing WA_5_Redacted.pdf to markdown")
        self.memory_tracker.log_memory("start_parsing")
        
        if not self.pdf_path.exists():
            logger.error(f"PDF not found: {self.pdf_path}")
            return False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cropped PDF
        cropped_pdf = self.output_dir / f"{self.pdf_path.stem}_cropped.pdf"
        self.create_cropped_pdf(self.pdf_path, cropped_pdf, self.config["footer_height"])
        
        page_texts = []
        pages_for_ocr = []
        
        # Step 1a: Extract text from cropped PDF
        logger.info("Extracting text from PDF...")
        with pdfplumber.open(cropped_pdf) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing {total_pages} pages")
            
            for i, page in enumerate(pdf.pages):
                if i % 50 == 0:  # Log progress every 50 pages
                    self.memory_tracker.log_memory(f"page_{i}")
                    gc.collect()  # Force garbage collection
                
                page_content = []
                
                # Crop bottom portion
                x0, y0, x1, y1 = page.bbox
                cropped_page = page.within_bbox((x0, y0 + self.config["footer_height"], x1, y1))
                
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
                    page_content.append(self.format_table_to_md(table))
                
                page_texts.append(page_content)
        
        self.memory_tracker.log_memory("after_text_extraction")
        
        # Step 1b: OCR for pages with no text (with memory management)
        if pages_for_ocr:
            logger.info(f"🔍 Performing OCR on {len(pages_for_ocr)} pages")
            
            # Process OCR in smaller batches to manage memory
            batch_size = min(10, len(pages_for_ocr))  # Process 10 pages at a time
            
            for batch_start in range(0, len(pages_for_ocr), batch_size):
                batch_end = min(batch_start + batch_size, len(pages_for_ocr))
                batch_pages = pages_for_ocr[batch_start:batch_end]
                
                logger.info(f"OCR batch {batch_start//batch_size + 1}: pages {batch_pages}")
                
                # Convert batch of pages to images
                images = []
                for pg_num in batch_pages:
                    img = convert_from_path(
                        cropped_pdf,
                        dpi=200,  # Reduced DPI for memory
                        first_page=pg_num + 1,
                        last_page=pg_num + 1
                    )[0]
                    images.append(img)
                
                # OCR processing
                args_list = [(batch_pages[i], images[i], self.config["footer_height"]) 
                           for i in range(len(batch_pages))]
                
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.config["max_cores"]) as executor:
                    results = list(executor.map(ocr_for_page, args_list))
                
                # Update page_texts
                for idx, page_num in enumerate(batch_pages):
                    page_texts[page_num] = [results[idx]]
                
                # Clean up images from memory
                del images
                gc.collect()
                self.memory_tracker.log_memory(f"ocr_batch_{batch_start//batch_size + 1}")
        
        # Step 1c: Write markdown
        logger.info("Writing markdown file...")
        with open(self.output_md, "w", encoding="utf-8") as f:
            for i, content_list in enumerate(page_texts):
                f.write(f"\n\n--- Page {i + 1} ---\n\n")
                for content in content_list:
                    if content:
                        f.write(content + "\n\n")
        
        # Clean up cropped PDF
        try:
            cropped_pdf.unlink()
            logger.info(f"🗑️ Deleted temporary file: {cropped_pdf}")
        except Exception as e:
            logger.warning(f"Could not delete {cropped_pdf}: {e}")
        
        logger.info(f"✅ Step 1 Complete: Parsed PDF to {self.output_md}")
        self.memory_tracker.log_memory("end_parsing")
        return True

    def step2_create_chunks(self) -> List[Dict[str, Any]]:
        """Step 2: Create chunks from markdown."""
        logger.info("🔄 Step 2: Creating chunks from markdown")
        
        if not self.output_md.exists():
            logger.error(f"Markdown file not found: {self.output_md}")
            return []
        
        # Read markdown content
        with open(self.output_md, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split into chunks
        chunks = text_splitter.split_text(content)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            # Extract page info from chunk
            page_match = re.search(r'--- Page (\d+) ---', chunk_text)
            page_num = int(page_match.group(1)) if page_match else 1
            
            # Extract section info
            section_match = re.search(r'^# (.+)$', chunk_text, re.MULTILINE)
            section = section_match.group(1) if section_match else "General"
            
            # Extract clause info
            clause_match = re.search(r'^## ([\d.]+)', chunk_text, re.MULTILINE)
            clause = clause_match.group(1) if clause_match else None
            
            chunk_obj = {
                "chunk_id": f"wa5_chunk_{i+1:04d}",
                "content": chunk_text.strip(),
                "page": page_num,
                "section": section,
                "clause": clause,
                "source_file": "WA_5_Redacted.pdf",
                "collection_key": "WAredacted",
                "document_type": "contract",
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            chunk_objects.append(chunk_obj)
        
        logger.info(f"✅ Step 2 Complete: Created {len(chunk_objects)} chunk objects")
        return chunk_objects

    def generate_embedding(self, text: str) -> List[float]:
        """Generate Qwen embedding for text."""
        try:
            payload = {
                "model": self.config["embedding_model"],
                "prompt": text
            }
            response = requests.post(
                f"{self.config['ollama_url']}/api/embeddings",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                if embedding:
                    # Normalize
                    embedding = np.array(embedding)
                    embedding = embedding / np.linalg.norm(embedding)
                    return embedding.tolist()
            else:
                logger.error(f"Embedding API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return []

    def step3_generate_embeddings(self, chunks: List[Dict[str, Any]]) -> bool:
        """Step 3: Generate embeddings and store in ChromaDB."""
        logger.info("🔄 Step 3: Generating embeddings and storing in ChromaDB")
        
        if not chunks:
            logger.error("No chunks to process")
            return False
        
        # Setup ChromaDB
        db_path = str(self.config["db_path"])
        collection_name = self.config["collection_name"]
        
        try:
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                collection = client.get_collection(collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except:
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"description": "WA redacted contracts with Qwen embeddings"}
                )
                logger.info(f"Created new collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"ChromaDB connection error: {e}")
            return False
        
        # Process chunks in batches
        batch_size = self.config["batch_size"]
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_chunks)} chunks)")
            
            # Generate embeddings for batch
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for chunk in batch_chunks:
                # Generate embedding
                embedding = self.generate_embedding(chunk["content"])
                if not embedding:
                    logger.warning(f"Failed to generate embedding for chunk {chunk['chunk_id']}")
                    continue
                
                # Prepare metadata
                metadata = {
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "source_file": chunk["source_file"],
                    "collection_key": chunk["collection_key"],
                    "document_type": chunk["document_type"],
                    "chunk_index": chunk["chunk_index"],
                    "embedding_model": self.config["embedding_model"]
                }
                if chunk["clause"]:
                    metadata["clause"] = chunk["clause"]
                
                embeddings.append(embedding)
                documents.append(chunk["content"])
                metadatas.append(metadata)
                ids.append(chunk["chunk_id"])
            
            # Add batch to ChromaDB
            if embeddings:
                try:
                    collection.add(
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    logger.info(f"✅ Added batch {batch_idx + 1} to ChromaDB ({len(embeddings)} chunks)")
                except Exception as e:
                    logger.error(f"Error adding batch {batch_idx + 1} to ChromaDB: {e}")
                    return False
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                gc.collect()
                self.memory_tracker.log_memory(f"batch_{batch_idx + 1}")
        
        # Test query
        try:
            test_results = collection.query(
                query_texts=["payment terms"],
                n_results=3
            )
            logger.info(f"🧪 Test query returned {len(test_results['documents'][0])} results")
        except Exception as e:
            logger.warning(f"Test query failed: {e}")
        
        logger.info(f"✅ Step 3 Complete: Stored {len(chunks)} chunks in {collection_name}")
        return True

    def run_pipeline(self) -> bool:
        """Run the complete WA5 processing pipeline."""
        logger.info("🚀 Starting WA_5_Redacted.pdf Processing Pipeline")
        logger.info("=" * 60)
        logger.info(f"Input PDF: {self.pdf_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"ChromaDB collection: {self.config['collection_name']}")
        
        start_time = time.time()
        self.memory_tracker.log_memory("pipeline_start")
        
        # Step 1: Parse PDF
        if not self.step1_parse_pdf():
            logger.error("❌ Pipeline failed at Step 1: PDF Parsing")
            return False
        
        # Step 2: Create chunks
        chunks = self.step2_create_chunks()
        if not chunks:
            logger.error("❌ Pipeline failed at Step 2: Chunking")
            return False
        
        # Step 3: Generate embeddings
        if not self.step3_generate_embeddings(chunks):
            logger.error("❌ Pipeline failed at Step 3: Embeddings")
            return False
        
        duration = time.time() - start_time
        self.memory_tracker.log_memory("pipeline_end")
        
        logger.info("=" * 60)
        logger.info("🎉 WA_5_Redacted.pdf Processing Pipeline Completed!")
        logger.info(f"⏱️ Total duration: {duration:.1f} seconds")
        logger.info(f"📄 Processed chunks: {len(chunks)}")
        logger.info(f"🗄️ Collection: {self.config['collection_name']}")
        
        return True

def main():
    """Main entry point."""
    logger.info("WA_5_Redacted.pdf Processing Pipeline")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # Validate input file
    if not CONFIG["input_pdf"].exists():
        logger.error(f"Input PDF not found: {CONFIG['input_pdf']}")
        return 1
    
    # Log configuration
    logger.info("📋 Configuration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    
    # Run pipeline
    processor = WA5Processor(CONFIG)
    success = processor.run_pipeline()
    
    if success:
        logger.info("✅ Pipeline completed successfully!")
        return 0
    else:
        logger.error("❌ Pipeline failed!")
        return 1

if __name__ == "__main__":
    exit(main())
