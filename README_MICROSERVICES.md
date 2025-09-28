# Contract Processing Microservices

A comprehensive microservices architecture for processing legal contracts with PDF parsing, chunking, embedding generation, and semantic search capabilities.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Backend API   │    │ Embeddings API  │    │    ChromaDB     │
│   (Port 8000)   │◄──►│   (Port 8002)   │    │   (Port 8001)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        ▲
         │                        │                        │
         ▼                        ▼                        │
┌─────────────────┐    ┌─────────────────┐                │
│     Ollama      │    │  PDF Processing │                │
│  (Port 11434)   │    │    Pipeline     │────────────────┘
└─────────────────┘    └─────────────────┘
```

## 🚀 Services

### 1. **Backend Service** (Port 8000)
- **Purpose**: Main orchestrator for the entire pipeline
- **Routes**:
  - `POST /upload-pdf` - Upload and process PDF files
  - `POST /query` - Search documents with natural language
  - `GET /status/{file_id}` - Check processing status
  - `GET /files` - List all processed files
  - `GET /health` - Health check

### 2. **Embeddings Service** (Port 8002)
- **Purpose**: FastAPI wrapper for Ollama embeddings
- **Model**: Qwen2.5-Coder 0.5B for embeddings
- **Routes**:
  - `POST /embed` - Generate single embedding
  - `POST /embed-batch` - Generate batch embeddings
  - `POST /similarity` - Compute text similarity
  - `GET /models` - List available models
  - `GET /health` - Health check

### 3. **ChromaDB** (Port 8001)
- **Purpose**: Vector database for storing embeddings
- **Features**: Persistent storage, similarity search
- **Collections**: `contract_embeddings`

### 4. **Ollama** (Port 11434)
- **Purpose**: Local LLM inference
- **Models**:
  - `qwen2.5-coder:0.5b` - For embeddings
  - `phi3:mini` - For SLM tasks

## 📋 Prerequisites

- Docker and Docker Compose
- At least 8GB RAM (16GB recommended)
- NVIDIA GPU (optional, for faster inference)

## 🛠️ Setup

### Quick Start

1. **Clone and navigate to the repository**:
   ```bash
   cd /home/jyotiraditya/preprocess
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

### Manual Setup

1. **Create necessary directories**:
   ```bash
   mkdir -p models uploads outputs
   ```

2. **Start services**:
   ```bash
   docker-compose up -d
   ```

3. **Pull required models** (after Ollama is running):
   ```bash
   docker exec ollama ollama pull qwen2.5-coder:0.5b
   docker exec ollama ollama pull phi3:mini
   ```

## 📖 Usage

### 1. Upload and Process a PDF

```bash
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@your-contract.pdf"
```

Response:
```json
{
  "status": "processing",
  "message": "PDF uploaded and processing started",
  "file_id": "contract.pdf_123456789"
}
```

### 2. Check Processing Status

```bash
curl "http://localhost:8000/status/contract.pdf_123456789"
```

### 3. Query Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the payment terms?",
    "top_k": 5
  }'
```

### 4. Direct Embedding Generation

```bash
curl -X POST "http://localhost:8002/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sample contract clause about payment terms."
  }'
```

## 🔄 Processing Pipeline

1. **PDF Upload** → Backend receives PDF file
2. **PDF Parsing** → Extract text using pdfplumber + PyMuPDF + PyTesseract
3. **Markdown Conversion** → Convert to structured markdown with clauses
4. **Chunking** → Split into semantic chunks using LangChain
5. **Embedding Generation** → Generate embeddings using Qwen2.5-Coder
6. **Storage** → Store embeddings and metadata in ChromaDB
7. **Query Ready** → Documents available for semantic search

## 🐳 Docker Services

### Service Health Checks

```bash
# Check all services
docker-compose ps

# Individual service logs
docker-compose logs backend
docker-compose logs embeddings
docker-compose logs chromadb
docker-compose logs ollama
```

### Scaling Services

```bash
# Scale embeddings service for higher throughput
docker-compose up -d --scale embeddings=3
```

## 📊 API Documentation

- **Backend API**: http://localhost:8000/docs
- **Embeddings API**: http://localhost:8002/docs

## 🔧 Configuration

### Environment Variables

**Backend Service**:
- `CHROMADB_URL`: ChromaDB connection URL
- `EMBEDDINGS_SERVICE_URL`: Embeddings service URL
- `OLLAMA_BASE_URL`: Ollama API URL
- `SLM_MODEL`: Small language model name

**Embeddings Service**:
- `OLLAMA_BASE_URL`: Ollama API URL
- `EMBEDDING_MODEL`: Embedding model name

### Model Configuration

Edit `docker-compose.yml` to change models:

```yaml
environment:
  - EMBEDDING_MODEL=qwen2.5-coder:0.5b  # Change embedding model
  - SLM_MODEL=phi3:mini                 # Change SLM model
```

## 🧪 Testing

### Test Embedding Service

```bash
curl -X POST "http://localhost:8002/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test embedding generation"}'
```

### Test Similarity

```bash
curl -X POST "http://localhost:8002/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "Payment shall be made within 30 days",
    "text2": "Invoice payment due in one month"
  }'
```

### Test ChromaDB

```bash
curl "http://localhost:8001/api/v1/heartbeat"
```

## 🚨 Troubleshooting

### Common Issues

1. **Ollama models not downloading**:
   ```bash
   docker exec -it ollama bash
   ollama pull qwen2.5-coder:0.5b
   ```

2. **ChromaDB connection issues**:
   ```bash
   docker-compose restart chromadb
   ```

3. **Memory issues**:
   - Increase Docker memory limit
   - Use smaller models
   - Reduce batch sizes

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
```

## 📈 Performance

### Optimization Tips

1. **GPU Acceleration**: Ensure NVIDIA Docker runtime for GPU support
2. **Batch Processing**: Use batch embedding endpoints for multiple texts
3. **Caching**: ChromaDB provides built-in caching
4. **Scaling**: Scale embeddings service for higher throughput

### Expected Performance

- **PDF Processing**: 1-5 minutes per document
- **Embedding Generation**: ~100ms per chunk
- **Query Response**: <1 second

## 🔒 Security

- Services communicate over internal Docker network
- No external API keys required (all local models)
- File uploads stored in isolated Docker volumes

## 📝 File Structure

```
preprocess/
├── docker-compose.yml          # Main orchestration
├── setup.sh                   # Setup script
├── backend/
│   ├── Dockerfile
│   ├── main.py                # FastAPI backend
│   └── requirements.txt
├── embeddings/
│   ├── Dockerfile
│   ├── service.py             # FastAPI embeddings service
│   └── requirements.txt
├── preprocess/                # Existing PDF processing code
│   ├── pdf_parser_to_footer_removal_markdown.py
│   └── chunking.py
├── models/                    # Ollama models (mounted)
├── uploads/                   # PDF uploads
└── outputs/                   # Processing outputs
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This project is licensed under the MIT License.
