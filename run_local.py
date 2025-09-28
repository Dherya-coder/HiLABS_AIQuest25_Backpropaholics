#!/usr/bin/env python3
"""
Local development runner for Contract Processing Services
Run this if you don't have Docker installed yet.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def install_requirements():
    """Install Python requirements for local development"""
    print("📦 Installing Python requirements...")
    
    # Install backend requirements
    backend_req = Path("backend/requirements.txt")
    if backend_req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(backend_req)], check=True)
    
    # Install embeddings requirements  
    embeddings_req = Path("embeddings/requirements.txt")
    if embeddings_req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(embeddings_req)], check=True)
    
    print("✅ Requirements installed!")

def start_local_services():
    """Start services locally for development"""
    print("🚀 Starting local development services...")
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("""
🔧 LOCAL DEVELOPMENT MODE

Since Docker isn't available, here's how to run the services locally:

1. **Install Ollama locally**:
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve &
   ollama pull qwen2.5-coder:0.5b
   ollama pull phi3:mini

2. **Install ChromaDB**:
   pip install chromadb
   # Run: python -c "import chromadb; chromadb.HttpClient(host='localhost', port=8001)"

3. **Start Embeddings Service**:
   cd embeddings/
   export OLLAMA_BASE_URL=http://localhost:11434
   uvicorn service:app --host 0.0.0.0 --port 8002 &

4. **Start Backend Service**:
   cd backend/
   export CHROMADB_URL=http://localhost:8001
   export EMBEDDINGS_SERVICE_URL=http://localhost:8002
   export OLLAMA_BASE_URL=http://localhost:11434
   uvicorn main:app --host 0.0.0.0 --port 8000 &

5. **Test the setup**:
   python test_services.py

📚 **Alternative: Use the Docker setup once installed**
   sudo systemctl start docker
   sudo usermod -aG docker $USER
   # Logout and login again
   docker compose up -d
""")

def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("🔍 Checking system setup...")
    
    if check_docker():
        print("✅ Docker is available! Use: docker compose up -d")
        return
    
    print("⚠️  Docker not found. Setting up local development environment...")
    
    try:
        install_requirements()
        start_local_services()
    except Exception as e:
        print(f"❌ Error setting up local environment: {e}")
        print("\n💡 Recommendation: Install Docker for the best experience:")
        print("   sudo apt update && sudo apt install docker.io docker-compose-v2 -y")
        print("   sudo systemctl start docker")
        print("   sudo usermod -aG docker $USER")

if __name__ == "__main__":
    main()
