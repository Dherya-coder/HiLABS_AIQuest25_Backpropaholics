#!/usr/bin/env python3
"""
Complete Attribute Processing Pipeline

This pipeline handles the complete attribute processing workflow:
1. Chunking: Converts AttributeDictionary.xlsx to JSON chunks
2. Embedding: Generates embeddings and stores in ChromaDB

Works from any directory using absolute paths.

Usage:
    python attribute_pipeline.py
    python attribute_pipeline.py --replace-collection
    python attribute_pipeline.py --db-path /custom/path/to/chroma_db
"""

import argparse
import json
import logging
import requests
import numpy as np
import pandas as pd
import tiktoken
from pathlib import Path
from typing import Dict, List, Any
import warnings

import chromadb
from chromadb.config import Settings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Pipeline configuration with absolute paths
PIPELINE_CONFIG = {
    "excel_file": PROJECT_ROOT / "parsing&embedding/attributes_parsing_embedding/AttributeDictionary.xlsx",
    "chunks_output_dir": PROJECT_ROOT / "outputs/attributes_chunks",
    "chunks_file": PROJECT_ROOT / "outputs/attributes_chunks/AttributeDictionary_chunks.json",
    "db_path": PROJECT_ROOT / "chroma_db_qwen",
    "collection_name": "attributes_simple",
    "embedding_model": "qwen3-embedding:0.6b",
    "ollama_url": "http://localhost:11434"
}

# Initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")

class SimpleOllamaEmbedding:
    """Simple Ollama embedding generator."""
    
    def __init__(self, model_name: str = "qwen3-embedding:0.6b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.embed_url = f"{ollama_url.rstrip('/')}/api/embeddings"
        logger.info(f"Using model: {model_name}")
        logger.info(f"Ollama URL: {self.embed_url}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            payload = {"model": self.model_name, "prompt": text}
            response = requests.post(self.embed_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                if embedding:
                    # Normalize
                    embedding = np.array(embedding)
                    embedding = embedding / np.linalg.norm(embedding)
                    return embedding.tolist()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

class AttributePipeline:
    """Complete attribute processing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_generator = None
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(enc.encode(text))

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text):
            return ""
        text = str(text).strip()
        # Remove white box characters (redaction marks)
        text = text.replace("█", "").replace("████", "").replace("█████", "")
        # Clean up extra spaces
        text = " ".join(text.split())
        return text

    def step1_create_chunks(self) -> bool:
        """Step 1: Convert Excel to JSON chunks."""
        logger.info("🔄 Step 1: Creating attribute chunks from Excel")
        
        excel_path = Path(self.config["excel_file"])
        output_dir = Path(self.config["chunks_output_dir"])
        
        if not excel_path.exists():
            logger.error(f"Excel file not found: {excel_path}")
            return False
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read Excel file
        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
            logger.info(f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return False
        
        all_chunks = []
        
        # Process each row as one chunk
        for row_idx, row in df.iterrows():
            # Get attribute name (first column)
            attribute_name = self.clean_text(row.iloc[0]) if len(row) > 0 else f"Attribute_{row_idx + 1}"
            
            if not attribute_name:
                continue
            
            # Concatenate all column values without column names for cleaner embedding
            content_parts = []
            for col_name, value in row.items():
                clean_value = self.clean_text(value)
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
                "token_count": self.count_tokens(full_content),
                "source_file": excel_path.name,
                "document_type": "attribute_definition"
            }
            
            all_chunks.append(chunk_data)
            logger.info(f"Created chunk {row_idx + 1}: {attribute_name}")
        
        # Save all chunks in one JSON file
        output_file = Path(self.config["chunks_file"])
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Step 1 Complete: Created {len(all_chunks)} attribute chunks")
        logger.info(f"📁 Saved to: {output_file}")
        
        # Show sample
        if all_chunks:
            logger.info(f"📄 Sample chunk structure:")
            sample = all_chunks[0]
            for key, value in sample.items():
                if key == "content":
                    preview = value[:100] + "..." if len(value) > 100 else value
                    logger.info(f"  {key}: {preview}")
                else:
                    logger.info(f"  {key}: {value}")
        
        return True

    def prepare_simple_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Create simple metadata for ChromaDB."""
        return {
            # Core info
            "chunk_id": chunk["chunk_id"],
            "attribute_number": chunk["attribute_number"],
            "attribute_name": chunk["attribute_name"],
            "token_count": chunk["token_count"],
            "source_file": chunk["source_file"],
            
            # Simple content flags
            "has_medicaid": "medicaid" in chunk["content"].lower(),
            "has_medicare": "medicare" in chunk["content"].lower(),
            "has_timely_filing": "timely filing" in chunk["content"].lower(),
            "has_fee_schedule": "fee schedule" in chunk["content"].lower(),
            "has_claims": "claims" in chunk["content"].lower(),
            "has_provider": "provider" in chunk["content"].lower(),
            
            # Metadata
            "embedding_model": self.config["embedding_model"],
            "document_type": "attribute_definition"
        }

    def step2_generate_embeddings(self, replace_existing: bool = True) -> bool:
        """Step 2: Generate embeddings and store in ChromaDB."""
        logger.info("🔄 Step 2: Generating embeddings and storing in ChromaDB")
        
        chunks_file = Path(self.config["chunks_file"])
        if not chunks_file.exists():
            logger.error(f"Chunks file not found: {chunks_file}")
            return False
        
        # Load chunks
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} attribute chunks")
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return False
        
        # Initialize embedding generator
        self.embedding_generator = SimpleOllamaEmbedding(
            model_name=self.config["embedding_model"],
            ollama_url=self.config["ollama_url"]
        )
        
        # Setup ChromaDB
        db_path = str(self.config["db_path"])
        collection_name = self.config["collection_name"]
        
        try:
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Connected to ChromaDB at: {db_path}")
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {e}")
            return False
        
        # Handle existing collection
        if replace_existing:
            try:
                client.delete_collection(name=collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass  # Collection doesn't exist
        
        # Create new collection
        try:
            collection = client.create_collection(
                name=collection_name,
                metadata={"description": "Simple attribute definitions with Qwen embeddings"}
            )
            logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
        
        # Process chunks
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing {i+1}/{len(chunks)}: {chunk['attribute_name']}")
            
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(chunk["content"])
            if not embedding:
                logger.error(f"Failed to generate embedding for chunk {i+1}")
                continue
            
            # Prepare data
            embeddings.append(embedding)
            documents.append(chunk["content"])
            metadatas.append(self.prepare_simple_metadata(chunk))
            ids.append(chunk["chunk_id"])
        
        # Add to ChromaDB
        if embeddings:
            try:
                collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"✅ Step 2 Complete: Added {len(embeddings)} embeddings to collection")
                
                # Test query
                test_embedding = self.embedding_generator.generate_embedding("medicaid timely filing")
                if test_embedding:
                    results = collection.query(
                        query_embeddings=[test_embedding],
                        n_results=3
                    )
                    logger.info(f"🧪 Test query returned {len(results['documents'][0])} results")
                    
                    # Show top result
                    if results['documents'][0]:
                        top_doc = results['documents'][0][0]
                        top_meta = results['metadatas'][0][0]
                        logger.info(f"🎯 Top result: {top_meta['attribute_name']}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error adding to ChromaDB: {e}")
                return False
        else:
            logger.error("No embeddings generated")
            return False

    def run_pipeline(self, replace_existing: bool = True) -> bool:
        """Run the complete attribute processing pipeline."""
        logger.info("🚀 Starting Attribute Processing Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Create chunks
        if not self.step1_create_chunks():
            logger.error("❌ Pipeline failed at Step 1: Chunking")
            return False
        
        # Step 2: Generate embeddings
        if not self.step2_generate_embeddings(replace_existing):
            logger.error("❌ Pipeline failed at Step 2: Embeddings")
            return False
        
        logger.info("=" * 60)
        logger.info("🎉 Attribute Processing Pipeline Completed Successfully!")
        logger.info(f"📊 Collection: {self.config['collection_name']}")
        logger.info(f"🗄️ Database: {self.config['db_path']}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Complete Attribute Processing Pipeline")
    parser.add_argument("--db-path", type=str, help="Custom ChromaDB path (overrides default)")
    parser.add_argument("--collection-name", type=str, default="attributes_simple", help="Collection name")
    parser.add_argument("--replace-collection", action="store_true", help="Replace existing collection")
    parser.add_argument("--embedding-model", type=str, default="qwen3-embedding:0.6b", help="Embedding model name")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama URL")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = PIPELINE_CONFIG.copy()
    if args.db_path:
        config["db_path"] = Path(args.db_path)
    if args.collection_name:
        config["collection_name"] = args.collection_name
    if args.embedding_model:
        config["embedding_model"] = args.embedding_model
    if args.ollama_url:
        config["ollama_url"] = args.ollama_url
    
    # Ensure paths exist
    config["db_path"] = Path(config["db_path"])
    config["db_path"].mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info("📋 Pipeline Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Run pipeline
    pipeline = AttributePipeline(config)
    success = pipeline.run_pipeline(replace_existing=args.replace_collection)
    
    if success:
        logger.info("✅ Pipeline completed successfully!")
        return 0
    else:
        logger.error("❌ Pipeline failed!")
        return 1

if __name__ == "__main__":
    exit(main())
