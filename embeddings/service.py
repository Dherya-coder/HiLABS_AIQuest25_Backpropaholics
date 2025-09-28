import os
import httpx
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np

app = FastAPI(
    title="Embeddings Service",
    description="FastAPI wrapper for Ollama embeddings using Qwen3-Embedding 0.6B",
    version="1.0.0"
)

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

# Pydantic models
class EmbedRequest(BaseModel):
    text: str
    model: Optional[str] = None

class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    text_length: int

class BatchEmbedRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None

class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    count: int

class HealthResponse(BaseModel):
    status: str
    service: str
    ollama_status: str
    available_models: List[str]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint that also verifies Ollama connectivity
    """
    try:
        async with httpx.AsyncClient() as client:
            # Check if Ollama is accessible
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10.0)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                ollama_status = "connected"
            else:
                available_models = []
                ollama_status = "error"
    except Exception as e:
        available_models = []
        ollama_status = f"connection_failed: {str(e)}"
    
    return HealthResponse(
        status="healthy",
        service="embeddings",
        ollama_status=ollama_status,
        available_models=available_models
    )

@app.post("/embed", response_model=EmbedResponse)
async def generate_embedding(request: EmbedRequest):
    """
    Generate embedding for a single text using Ollama
    """
    model = request.model or EMBEDDING_MODEL
    
    try:
        async with httpx.AsyncClient() as client:
            # Call Ollama embeddings API
            ollama_request = {
                "model": model,
                "prompt": request.text
            }
            
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json=ollama_request,
                timeout=60.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama API error: {response.text}"
                )
            
            result = response.json()
            
            if "embedding" not in result:
                raise HTTPException(
                    status_code=500,
                    detail="No embedding returned from Ollama"
                )
            
            embedding = result["embedding"]
            
            # Validate embedding
            if not isinstance(embedding, list) or len(embedding) == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Invalid embedding format from Ollama"
                )
            
            return EmbedResponse(
                embedding=embedding,
                model=model,
                text_length=len(request.text)
            )
            
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Timeout while calling Ollama API"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to Ollama: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.post("/embed-batch", response_model=BatchEmbedResponse)
async def generate_batch_embeddings(request: BatchEmbedRequest):
    """
    Generate embeddings for multiple texts in batch
    """
    model = request.model or EMBEDDING_MODEL
    
    if len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Too many texts (max 100)")
    
    embeddings = []
    
    try:
        async with httpx.AsyncClient() as client:
            # Process texts in parallel with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
            
            async def get_single_embedding(text: str):
                async with semaphore:
                    ollama_request = {
                        "model": model,
                        "prompt": text
                    }
                    
                    response = await client.post(
                        f"{OLLAMA_BASE_URL}/api/embeddings",
                        json=ollama_request,
                        timeout=60.0
                    )
                    
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Ollama API error for text: {text[:50]}..."
                        )
                    
                    result = response.json()
                    return result["embedding"]
            
            # Execute all requests concurrently
            tasks = [get_single_embedding(text) for text in request.texts]
            embeddings = await asyncio.gather(*tasks)
            
            return BatchEmbedResponse(
                embeddings=embeddings,
                model=model,
                count=len(embeddings)
            )
            
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Timeout while calling Ollama API"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to Ollama: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during batch processing: {str(e)}"
        )

@app.get("/models")
async def list_available_models():
    """
    List available models from Ollama
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10.0)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to fetch models from Ollama"
                )
            
            models_data = response.json()
            return {
                "models": models_data.get("models", []),
                "current_embedding_model": EMBEDDING_MODEL
            }
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to Ollama: {str(e)}"
        )

@app.post("/similarity")
async def compute_similarity(text1: str, text2: str, model: Optional[str] = None):
    """
    Compute cosine similarity between two texts
    """
    model = model or EMBEDDING_MODEL
    
    try:
        # Get embeddings for both texts
        batch_request = BatchEmbedRequest(texts=[text1, text2], model=model)
        batch_response = await generate_batch_embeddings(batch_request)
        
        if len(batch_response.embeddings) != 2:
            raise HTTPException(status_code=500, detail="Failed to get embeddings for both texts")
        
        # Compute cosine similarity
        emb1 = np.array(batch_response.embeddings[0])
        emb2 = np.array(batch_response.embeddings[1])
        
        # Cosine similarity formula
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm1 * norm2)
        
        return {
            "text1": text1,
            "text2": text2,
            "similarity": float(similarity),
            "model": model
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute similarity: {str(e)}"
        )

@app.get("/")
async def root():
    """
    Root endpoint with service information
    """
    return {
        "service": "Embeddings Service",
        "version": "1.0.0",
        "description": "FastAPI wrapper for Ollama embeddings",
        "endpoints": {
            "health": "/health",
            "embed": "/embed",
            "embed_batch": "/embed-batch",
            "models": "/models",
            "similarity": "/similarity"
        },
        "current_model": EMBEDDING_MODEL,
        "ollama_url": OLLAMA_BASE_URL
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
