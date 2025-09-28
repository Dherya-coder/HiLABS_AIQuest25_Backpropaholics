#!/bin/bash

echo "🔍 Checking Docker Compose Services Status..."
echo "=============================================="

# Check if services are running
sudo docker compose ps

echo ""
echo "📊 Service Health Checks:"
echo "------------------------"

# Check ChromaDB
echo -n "ChromaDB (port 8001): "
if curl -s -f http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; then
    echo "✅ Healthy"
else
    echo "❌ Not responding"
fi

# Check Ollama
echo -n "Ollama (port 11434): "
if curl -s -f http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Healthy"
else
    echo "❌ Not responding"
fi

# Check Embeddings Service
echo -n "Embeddings (port 8002): "
if curl -s -f http://localhost:8002/health > /dev/null 2>&1; then
    echo "✅ Healthy"
else
    echo "❌ Not responding"
fi

# Check Backend Service
echo -n "Backend (port 8000): "
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Healthy"
else
    echo "❌ Not responding"
fi

echo ""
echo "📋 Quick Commands:"
echo "  View logs: sudo docker compose logs -f [service_name]"
echo "  Restart: sudo docker compose restart [service_name]"
echo "  Stop all: sudo docker compose down"
echo "  Start all: sudo docker compose up -d"
