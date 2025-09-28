#!/usr/bin/env python3
"""
Generate embeddings for contract JSON chunks using Qwen 3 embedding model and store in ChromaDB.

Requirements:
    pip install chromadb requests numpy

Usage:
    python generate_contract_embeddings.py --chunks-dir "./outputs/chunked123" --collection-name "redacted"
"""

import argparse
import json
import logging
import os
import uuid
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
import numpy as np

import chromadb
from chromadb.config import Settings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaEmbeddingGenerator:
    """Generate embeddings using Ollama Qwen model."""
    
    def __init__(self, model_name: str = "qwen3-embedding:0.6b", ollama_url: str = "http://localhost:11434"):
        """Initialize the Ollama embedding generator."""
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.embed_url = f"{ollama_url}/api/embeddings"
        
        logger.info(f"Using Ollama model: {model_name}")
        logger.info(f"Ollama URL: {ollama_url}")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                logger.info(f"Available Ollama models: {model_names}")
                
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in available models")
                    logger.info("You may need to pull the model first: ollama pull qwen3-embedding:0.6b")
                else:
                    logger.info(f"Model {self.model_name} is available")
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(
                self.embed_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                if embedding:
                    # Normalize the embedding
                    embedding = np.array(embedding)
                    embedding = embedding / np.linalg.norm(embedding)
                    return embedding.tolist()
                else:
                    logger.error("No embedding returned from Ollama")
                    return []
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = []
        
        for i, text in enumerate(texts):
            logger.info(f"Generating embedding {i+1}/{len(texts)}")
            embedding = self.generate_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                logger.error(f"Failed to generate embedding for text {i+1}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 768)  # Assuming 768-dim embeddings
        
        return embeddings

class ChromaDBManager:
    """Manage ChromaDB operations for storing embeddings."""
    
    def __init__(self, db_path: str = "../chroma_db_qwen", collection_name: str = "redacted"):
        """Initialize ChromaDB client and collection."""
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Retrieved existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Legal contract document chunks with Qwen embeddings"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_embeddings(self, 
                      embeddings: List[List[float]], 
                      documents: List[str], 
                      metadatas: List[Dict[str, Any]], 
                      ids: List[str]) -> None:
        """Add embeddings to the collection."""
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(embeddings)} embeddings to collection")
        except Exception as e:
            logger.error(f"Error adding embeddings to collection: {e}")
            raise
    
    def query_similar(self, 
                     query_embedding: List[float], 
                     n_results: int = 5,
                     where: Optional[Dict] = None) -> Dict:
        """Query similar embeddings."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return {}
    
    def get_collection_info(self) -> Dict:
        """Get collection information."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

def load_contract_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """Load all contract JSON chunks from directory."""
    chunks = []
    json_files = list(chunks_dir.glob("*_chunks.json"))
    
    logger.info(f"Found {len(json_files)} JSON chunk files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunk_list = json.load(f)
                # The file contains a list of chunks
                if isinstance(chunk_list, list):
                    chunks.extend(chunk_list)
                else:
                    chunks.append(chunk_list)
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(chunks)} chunks")
    return chunks

def prepare_contract_metadata_for_chroma(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare contract metadata for ChromaDB (only string, int, float, bool values)."""
    metadata = {
        # Core identifiers
        "chunk_id": chunk_data.get("chunk_id", "unknown"),
        "page_number": chunk_data.get("page_number", 0),
        "chunk_index": chunk_data.get("chunk_index", 0),
        "source_file": chunk_data.get("source_file", "unknown"),
        
        # Content metadata
        "header_path": chunk_data.get("header_path", ""),
        "token_count": chunk_data.get("token_count", 0),
        "document_type": chunk_data.get("document_type", "legal_contract"),
        
        # Embedding metadata
        "embedding_model": "qwen3-embedding:0.6b",
        "semantic_type": "contract_chunk",
        "domain": "legal_healthcare",
        
        # Content analysis flags
        "has_header": bool(chunk_data.get("header_path", "").strip() and chunk_data.get("header_path") != "document_content"),
        "is_definition": "definition" in chunk_data.get("header_path", "").lower(),
        "is_clause": "clause:" in chunk_data.get("header_path", "").lower(),
        "is_section": "section:" in chunk_data.get("header_path", "").lower(),
        
        # Content type detection
        "contains_compensation": any(term in chunk_data.get("content", "").lower() 
                                  for term in ["compensation", "payment", "reimbursement", "rate"]),
        "contains_provider": "provider" in chunk_data.get("content", "").lower(),
        "contains_member": "member" in chunk_data.get("content", "").lower(),
        "contains_medicaid": "medicaid" in chunk_data.get("content", "").lower(),
        "contains_medicare": "medicare" in chunk_data.get("content", "").lower(),
        "contains_claims": "claim" in chunk_data.get("content", "").lower(),
        "contains_network": "network" in chunk_data.get("content", "").lower(),
        
        # Length indicators
        "is_short_chunk": chunk_data.get("token_count", 0) < 50,
        "is_long_chunk": chunk_data.get("token_count", 0) > 200,
        
        # Page context
        "is_first_page": chunk_data.get("page_number", 0) == 1,
        "page_range": f"page_{chunk_data.get('page_number', 0)//10 * 10 + 1}-{(chunk_data.get('page_number', 0)//10 + 1) * 10}"
    }
    
    return metadata

def process_contract_chunks_to_embeddings(chunks_dir: Path, 
                                        collection_name: str = "redacted",
                                        db_path: str = "../chroma_db_qwen",
                                        model_name: str = "qwen3-embedding:0.6b",
                                        batch_size: int = 5) -> None:
    """Main function to process contract JSON chunks and generate embeddings using Ollama."""
    
    # Initialize components
    logger.info("Initializing Ollama embedding generator and ChromaDB manager...")
    embedding_generator = OllamaEmbeddingGenerator(model_name=model_name)
    chroma_manager = ChromaDBManager(db_path=db_path, collection_name=collection_name)
    
    # Load contract chunks
    logger.info(f"Loading contract chunks from {chunks_dir}")
    chunks = load_contract_chunks(chunks_dir)
    
    if not chunks:
        logger.error("No chunks found to process")
        return
    
    # Process chunks in smaller batches for Ollama
    logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_texts = []
        batch_documents = []
        batch_metadatas = []
        batch_ids = []
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        
        for chunk_data in batch_chunks:
            # Extract text for embedding
            content = chunk_data.get("content", "")
            header_path = chunk_data.get("header_path", "")
            
            # Create enhanced text for embedding (include header context)
            if header_path and header_path != "document_content":
                text_for_embedding = f"Section: {header_path} | Content: {content}"
            else:
                text_for_embedding = content
            
            batch_texts.append(text_for_embedding)
            
            # Document text (for storage and retrieval)
            batch_documents.append(content)
            
            # Prepare metadata
            metadata = prepare_contract_metadata_for_chroma(chunk_data)
            batch_metadatas.append(metadata)
            
            # Use chunk_id as the ID
            chunk_id = chunk_data.get("chunk_id", f"chunk_{i}_{len(batch_ids)}")
            batch_ids.append(chunk_id)
        
        # Generate embeddings for batch using Ollama
        logger.info(f"Generating embeddings for batch of {len(batch_texts)} texts using Ollama...")
        batch_embeddings = embedding_generator.generate_batch_embeddings(batch_texts)
        
        if not batch_embeddings or len(batch_embeddings) != len(batch_texts):
            logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}")
            continue
        
        # Add to ChromaDB
        logger.info(f"Adding batch to ChromaDB...")
        try:
            chroma_manager.add_embeddings(
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logger.info(f"Successfully processed batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"Failed to add batch {i//batch_size + 1} to ChromaDB: {e}")
            continue
    
    # Print collection info
    collection_info = chroma_manager.get_collection_info()
    logger.info(f"Collection Info: {collection_info}")
    
    # Test query
    logger.info("Testing similarity search with contract embeddings...")
    if chunks:
        test_text = "provider compensation and payment terms"
        test_embedding = embedding_generator.generate_embedding(test_text)
        
        if test_embedding:
            results = chroma_manager.query_similar(test_embedding, n_results=3)
            logger.info(f"Test query returned {len(results.get('documents', [[]])[0])} results")
            
            # Show test results
            if results.get('documents') and results['documents'][0]:
                logger.info("Top result previews:")
                for i, (doc, metadata) in enumerate(zip(results['documents'][0][:3], results['metadatas'][0][:3])):
                    header = metadata.get('header_path', 'No header')
                    page = metadata.get('page_number', 'Unknown')
                    preview = doc[:100] + "..." if len(doc) > 100 else doc
                    logger.info(f"  {i+1}. Page {page}, {header}: {preview}")
    
    logger.info("✅ Contract embedding generation completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for contract JSON chunks using Ollama Qwen model")
    parser.add_argument("--chunks-dir", type=str, default="./outputs/chunked123", help="Directory containing JSON chunk files")
    parser.add_argument("--collection-name", type=str, default="redacted", help="ChromaDB collection name")
    parser.add_argument("--db-path", type=str, default="../chroma_db_qwen", help="ChromaDB database path")
    parser.add_argument("--model-name", type=str, default="qwen3-embedding:0.6b", help="Ollama model name")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing (smaller for Ollama)")
    
    args = parser.parse_args()
    
    chunks_dir = Path(args.chunks_dir)
    
    if not chunks_dir.exists():
        raise SystemExit(f"Chunks directory not found: {chunks_dir}")
    
    process_contract_chunks_to_embeddings(
        chunks_dir=chunks_dir,
        collection_name=args.collection_name,
        db_path=args.db_path,
        model_name=args.model_name,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
