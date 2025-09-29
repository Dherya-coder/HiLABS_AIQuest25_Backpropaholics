#!/usr/bin/env python3
"""
Simple attribute embedding generation - clean and minimal metadata.

Usage:
    python simple_attribute_embeddings.py --chunks-file "../outputs/attributes_chunks/AttributeDictionary_chunks.json"
    
    Or simply run without arguments to use defaults:
    python simple_attribute_embeddings.py
"""

import argparse
import json
import logging
import requests
import numpy as np
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

class SimpleOllamaEmbedding:
    """Simple Ollama embedding generator."""
    
    def __init__(self, model_name: str = "qwen3-embedding:0.6b"):
        self.model_name = model_name
        self.embed_url = "http://localhost:11434/api/embeddings"
        logger.info(f"Using model: {model_name}")
    
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
                logger.error(f"API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

def prepare_simple_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
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
        "embedding_model": "qwen3-embedding:0.6b",
        "document_type": "attribute_definition"
    }

def process_attribute_embeddings(chunks_file: Path, 
                                collection_name: str = "attributes_simple",
                                db_path: str = "./chroma_db_qwen",
                                replace_existing: bool = True) -> None:
    """Process attribute chunks and create embeddings."""
    
    # Load chunks
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} attribute chunks")
    except Exception as e:
        logger.error(f"Error loading chunks: {e}")
        return
    
    # Initialize components
    embedding_generator = SimpleOllamaEmbedding()
    
    # Setup ChromaDB
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
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
        return
    
    # Process chunks
    embeddings = []
    documents = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing {i+1}/{len(chunks)}: {chunk['attribute_name']}")
        
        # Generate embedding
        embedding = embedding_generator.generate_embedding(chunk["content"])
        if not embedding:
            logger.error(f"Failed to generate embedding for chunk {i+1}")
            continue
        
        # Prepare data
        embeddings.append(embedding)
        documents.append(chunk["content"])
        metadatas.append(prepare_simple_metadata(chunk))
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
            logger.info(f"✅ Added {len(embeddings)} embeddings to collection")
            
            # Test query
            test_embedding = embedding_generator.generate_embedding("medicaid timely filing")
            if test_embedding:
                results = collection.query(
                    query_embeddings=[test_embedding],
                    n_results=3
                )
                logger.info(f"Test query returned {len(results['documents'][0])} results")
                
                # Show top result
                if results['documents'][0]:
                    top_doc = results['documents'][0][0]
                    top_meta = results['metadatas'][0][0]
                    logger.info(f"Top result: {top_meta['attribute_name']}")
            
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")
    
    logger.info("🎉 Simple attribute embedding generation completed!")

def main():
    parser = argparse.ArgumentParser(description="Generate simple attribute embeddings")
    parser.add_argument("--chunks-file", type=str, default="../outputs/attributes_chunks/AttributeDictionary_chunks.json", help="Path to chunks JSON file (default: ../outputs/attributes_chunks/AttributeDictionary_chunks.json)")
    parser.add_argument("--collection-name", type=str, default="attributes_simple", help="Collection name")
    parser.add_argument("--db-path", type=str, default="../chroma_db_qwen", help="ChromaDB path")
    parser.add_argument("--replace", action="store_true", help="Replace existing collection")
    
    args = parser.parse_args()
    
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        raise SystemExit(f"Chunks file not found: {chunks_file}")
    
    process_attribute_embeddings(
        chunks_file=chunks_file,
        collection_name=args.collection_name,
        db_path=args.db_path,
        replace_existing=args.replace
    )

if __name__ == "__main__":
    main()
