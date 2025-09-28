#!/bin/bash

# Setup script for Contract Processing Microservices
echo "🚀 Setting up Contract Processing Microservices..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models uploads outputs

# Pull required Ollama models
echo "🤖 Pulling Ollama models..."
echo "This will download the required models. This may take a while..."

# Start Ollama service first
echo "Starting Ollama service..."
docker compose up -d ollama

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
sleep 30

# Pull embedding model (Qwen2.5-Coder 0.5B)
echo "Pulling embedding model: qwen2.5-coder:0.5b"
docker exec ollama ollama pull qwen2.5-coder:0.5b

# Pull SLM model (Phi3 Mini)
echo "Pulling SLM model: phi3:mini"
docker exec ollama ollama pull phi3:mini

echo "✅ Models downloaded successfully!"

# Start all services
echo "🐳 Starting all services..."
docker compose up -d

echo "⏳ Waiting for all services to be ready..."
sleep 60

# Check service health
echo "🔍 Checking service health..."

echo "Checking ChromaDB..."
curl -f http://localhost:8001/api/v1/heartbeat || echo "❌ ChromaDB not ready"

echo "Checking Ollama..."
curl -f http://localhost:11434/api/tags || echo "❌ Ollama not ready"

echo "Checking Embeddings Service..."
curl -f http://localhost:8002/health || echo "❌ Embeddings Service not ready"

echo "Checking Backend Service..."
curl -f http://localhost:8000/health || echo "❌ Backend Service not ready"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Service URLs:"
echo "  - Backend API: http://localhost:8000"
echo "  - Embeddings API: http://localhost:8002"
echo "  - ChromaDB: http://localhost:8001"
echo "  - Ollama: http://localhost:11434"
echo ""
echo "📚 API Documentation:"
echo "  - Backend: http://localhost:8000/docs"
echo "  - Embeddings: http://localhost:8002/docs"
echo ""
echo "🧪 Test the setup:"
echo "  curl -X POST 'http://localhost:8000/upload-pdf' -F 'file=@your-contract.pdf'"
echo ""
