import os
import json
from typing import List, Dict, Any, Optional
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Contract Processing Backend",
    description="Orchestrates PDF parsing, chunking, embedding, and storage",
    version="1.0.0"
)

# Configuration
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://chromadb:8000")
EMBEDDINGS_SERVICE_URL = os.getenv("EMBEDDINGS_SERVICE_URL", "http://embeddings:8000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
SLM_MODEL = os.getenv("SLM_MODEL", "phi3:mini")

# Pydantic models
class EmbedRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ProcessingStatus(BaseModel):
    status: str
    message: str
    file_id: Optional[str] = None

class UploadResponse(BaseModel):
    status: str
    message: str
    file_id: str

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int

# Global storage for processing status
processing_status = {}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "backend", "version": "1.0.0"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Contract Processing Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/upload-pdf", "/embed-clauses", "/query", "/status/{file_id}", "/files"]
    }

@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF and start processing pipeline (STUB)
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique file ID
    file_id = f"{file.filename}_{abs(hash(file.filename))}"
    
    # Store processing status (stub)
    processing_status[file_id] = {
        "status": "completed",
        "message": f"PDF {file.filename} processed successfully (STUB)",
        "file_path": f"/app/uploads/{file.filename}"
    }
    
    return UploadResponse(
        status="completed",
        message=f"PDF {file.filename} uploaded and processed (STUB)",
        file_id=file_id
    )

@app.post("/embed-clauses")
async def embed_clauses(file_id: str):
    """
    Generate embeddings for processed clauses (STUB)
    """
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "message": f"Embeddings generated for file {file_id} (STUB)",
        "file_id": file_id,
        "embeddings_count": 42  # Stub value
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using natural language (STUB)
    """
    # Stub response
    stub_results = [
        {
            "content": f"This is a stub result for query: {request.query}",
            "metadata": {"source": "stub_document.pdf", "page": 1},
            "similarity_score": 0.95
        },
        {
            "content": f"Another stub result matching: {request.query}",
            "metadata": {"source": "stub_document.pdf", "page": 2},
            "similarity_score": 0.87
        }
    ]
    
    return QueryResponse(
        query=request.query,
        results=stub_results[:request.top_k],
        total_results=len(stub_results)
    )

@app.get("/status/{file_id}")
async def get_processing_status(file_id: str):
    """
    Get processing status for a file (STUB)
    """
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    return processing_status[file_id]

@app.get("/files")
async def list_processed_files():
    """
    List all processed files and their status (STUB)
    """
    return {
        "files": [
            {"file_id": file_id, **status}
            for file_id, status in processing_status.items()
        ],
        "total_files": len(processing_status)
    }

# Service connectivity check endpoints
@app.get("/check-services")
async def check_services():
    """
    Check connectivity to other services
    """
    services_status = {}
    
    # Check ChromaDB
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{CHROMADB_URL}/api/v1/heartbeat", timeout=5.0)
            services_status["chromadb"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        services_status["chromadb"] = "unreachable"
    
    # Check Embeddings Service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{EMBEDDINGS_SERVICE_URL}/health", timeout=5.0)
            services_status["embeddings"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        services_status["embeddings"] = "unreachable"
    
    # Check Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            services_status["ollama"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        services_status["ollama"] = "unreachable"
    
    return {
        "backend": "healthy",
        "services": services_status,
        "config": {
            "chromadb_url": CHROMADB_URL,
            "embeddings_url": EMBEDDINGS_SERVICE_URL,
            "ollama_url": OLLAMA_BASE_URL,
            "slm_model": SLM_MODEL
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
