#!/bin/bash

echo "🚀 Starting Contract Processing Microservices..."

# Stop any existing Ollama processes to avoid port conflicts
echo "🛑 Stopping any existing Ollama processes..."
sudo pkill ollama 2>/dev/null || true

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models uploads outputs

# Start services
echo "🐳 Starting Docker services..."
sudo docker compose down 2>/dev/null || true
sudo docker compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
./check_status.sh

echo ""
echo "🎉 Services started!"
echo ""
echo "📋 Service URLs:"
echo "  - Backend API: http://localhost:8000"
echo "  - Backend Docs: http://localhost:8000/docs"
echo "  - Embeddings API: http://localhost:8002"
echo "  - Embeddings Docs: http://localhost:8002/docs"
echo "  - ChromaDB: http://localhost:8001"
echo "  - Ollama: http://localhost:11434"
echo ""
echo "🧪 Next steps:"
echo "  1. Initialize models: ./init-models.sh"
echo "  2. Test services: python test_services.py"
echo "  3. Upload a PDF: curl -X POST 'http://localhost:8000/upload-pdf' -F 'file=@contract.pdf'"
