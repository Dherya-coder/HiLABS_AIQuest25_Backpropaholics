#!/bin/bash

echo "🔧 Fixing Docker and Ollama issues..."

# Stop system Ollama service if it exists
echo "🛑 Stopping system Ollama service..."
sudo systemctl stop ollama 2>/dev/null || true
sudo systemctl disable ollama 2>/dev/null || true

# Kill any remaining Ollama processes
echo "🛑 Killing any remaining Ollama processes..."
sudo pkill -f ollama 2>/dev/null || true

# Wait a moment
sleep 2

# Check if port is free now
if sudo lsof -i :11434 >/dev/null 2>&1; then
    echo "❌ Port 11434 is still in use. Let's find what's using it:"
    sudo lsof -i :11434
    echo "Please manually stop the process above and run this script again."
    exit 1
fi

echo "✅ Port 11434 is now free!"

# Start Docker services
echo "🐳 Starting Docker services..."
sudo docker compose down 2>/dev/null || true
sudo docker compose up -d

echo "⏳ Waiting for services to start..."
sleep 20

# Check status
echo "🔍 Checking service status..."
sudo docker compose ps

echo ""
echo "🧪 Testing ChromaDB directly:"
curl -s http://localhost:8001/api/v1/heartbeat && echo "✅ ChromaDB is healthy!" || echo "❌ ChromaDB not responding"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Check all services: sudo docker compose ps"
echo "  2. View logs if needed: sudo docker compose logs [service_name]"
echo "  3. Test APIs:"
echo "     - Backend: curl http://localhost:8000/health"
echo "     - Embeddings: curl http://localhost:8002/health"
echo "     - ChromaDB: curl http://localhost:8001/api/v1/heartbeat"
