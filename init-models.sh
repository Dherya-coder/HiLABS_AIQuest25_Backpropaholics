#!/bin/bash

echo "🤖 Initializing Ollama models..."

# Wait for Ollama to be ready
echo "Waiting for Ollama service to be ready..."
until curl -f http://localhost:11434/api/tags >/dev/null 2>&1; do
    echo "Waiting for Ollama..."
    sleep 5
done

echo "✅ Ollama is ready!"

# Pull the embedding model
echo "📥 Pulling Qwen3-Embedding model..."
docker exec ollama ollama pull qwen3-embedding:0.6b

# Pull the SLM model
echo "📥 Pulling Phi3-Mini model..."
docker exec ollama ollama pull phi3:mini

echo "✅ All models downloaded successfully!"

# List available models
echo "📋 Available models:"
docker exec ollama ollama list
