# HiLABS Complete Setup Guide

Complete setup instructions for the HiLABS preprocessing pipeline, FastAPI backend, chatbot, and all components.

## 🚀 Quick Start (Docker - Recommended)

### Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM and 20GB disk space
- Internet connection for downloading models

### 1. Clone and Setup
```bash
git clone <repository-url>
cd preprocess

# Ensure data directory exists with your PDF files
mkdir -p data/Contracts_data/WA/
# Place WA_5_Redacted.pdf in data/Contracts_data/WA/
```

### 2. Start Services
```bash
# Start all services (Ollama, ChromaDB, Pipeline)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f hilabs-pipeline
```

### 3. Access Services
- **FastAPI Chatbot**: http://localhost:8000
- **ChromaDB**: http://localhost:8001
- **Ollama**: http://localhost:11434

### 4. Run Complete Pipeline
```bash
# Run complete pipeline inside container
docker-compose exec hilabs-pipeline python main.py

# Or run specific steps
docker-compose exec hilabs-pipeline python main.py --step 4
```

## 🔧 Manual Setup (Local Development)

### System Dependencies

#### Ubuntu/Debian
```bash
# Install system packages
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils curl wget git

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &
```

#### macOS
```bash
# Install system packages
brew install tesseract poppler

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &
```

#### Windows
```bash
# Install via Chocolatey
choco install tesseract poppler

# Install Ollama manually from https://ollama.ai/download
# Start Ollama service
ollama serve
```

### Python Environment Setup

#### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv hilabs-env

# Activate environment
# Linux/macOS:
source hilabs-env/bin/activate
# Windows:
hilabs-env\Scripts\activate
```

#### 2. Install Python Dependencies
```bash
# Install all requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

#### 3. Pull Ollama Models
```bash
# Pull required models (this may take 10-15 minutes)
ollama pull qwen3-embedding:0.6b
ollama pull phi3:mini

# Verify models are available
ollama list
```

### 4. Verify Installation
```bash
# Check system dependencies
tesseract --version
pdftoppm -h
ollama --version

# Check Python packages
python -c "import chromadb, sentence_transformers, transformers, spacy; print('All packages installed successfully')"

# Check spaCy model
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded successfully')"
```

## 📋 Pipeline Execution Guide

### Complete Pipeline
```bash
# Run all 5 steps sequentially
python main.py

# Expected duration: 50-120 minutes
# Steps: PDF Parsing → Ranking → Similarity → Classification → Chatbot
```

### Step-by-Step Execution

#### Step 1: PDF Parsing & Embedding (15-30 min)
```bash
python main.py --step 1

# What it does:
# - Parses WA_5_Redacted.pdf (99MB) with OCR
# - Creates markdown output
# - Generates initial ChromaDB embeddings
# - Output: outputs/pdf_parsed/WA/WA_5_Redacted.md
```

#### Step 2: Attribute Ranking (10-20 min)
```bash
python main.py --step 2

# What it does:
# - Generates similarity rankings for all attributes
# - Processes TNredacted, WAredacted, TNstandard, WAstandard
# - Output: outputs/precise_similarity/*.json (4 files)
```

#### Step 3: Similarity Preprocessing & Embedding (5-15 min)
```bash
python main.py --step 3

# What it does:
# - Preprocesses similarity results
# - Generates Qwen + Paraphrase embeddings
# - Output: outputs/precise_similarity/processed_datasets/ (~12 files)
```

#### Step 4: Standard Classification (20-40 min)
```bash
python main.py --step 4

# What it does:
# - Multi-step classification pipeline
# - Exact match → Semantic → NLI → Rule flags
# - Adds isStandard field (0 or 1) to datasets
# - Output: Updated datasets + classification reports
```

#### Step 5: Contract Analysis Chatbot (Continuous)
```bash
python main.py --step 5

# What it does:
# - Starts FastAPI server on port 8000
# - Loads classified datasets
# - Provides intelligent contract analysis API
# - Access: http://localhost:8000
```

### Selective Execution
```bash
# Skip PDF processing (use existing data)
python main.py --skip-steps 1

# Run only classification and chatbot
python main.py --skip-steps 1,2,3

# Check pipeline status
python main.py --status
```

## 🌐 FastAPI Backend & Chatbot Setup

### 1. Start the API Server
```bash
# Method 1: Via main pipeline
python main.py --step 5

# Method 2: Direct backend start
cd backend
python main.py

# Method 3: Via uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Contract Summary
```bash
curl http://localhost:8000/chatbot/contracts/summary
```

#### Chat Interface
```bash
curl -X POST http://localhost:8000/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How many non-standard clauses are in TN Contract 1?",
    "session_id": "test-session"
  }'
```

### 3. Frontend Demo
```bash
# Open the demo HTML file
open backend/chatbot_frontend_example.html
# Or visit: http://localhost:8000/static/chatbot_frontend_example.html
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Ollama Not Running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve &

# Pull missing models
ollama pull qwen3-embedding:0.6b
ollama pull phi3:mini
```

#### 2. Memory Issues
```bash
# Monitor memory usage
python main.py --status

# Run steps individually if needed
python main.py --step 1
python main.py --step 2
# ... continue step by step
```

#### 3. ChromaDB Issues
```bash
# Check ChromaDB directory
ls -la chroma_db_qwen/

# Reset ChromaDB if corrupted
rm -rf chroma_db_qwen/
python main.py --step 1  # Rebuild from scratch
```

#### 4. PDF Processing Issues
```bash
# Check Tesseract installation
tesseract --version

# Check PDF file exists
ls -la data/Contracts_data/WA/WA_5_Redacted.pdf

# Check poppler installation
pdftoppm -h
```

#### 5. Classification Taking Too Long
```bash
# Monitor progress
tail -f pipeline.log

# Check GPU availability for transformers
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Log Files
- **Main pipeline**: `pipeline.log`
- **Individual steps**: Check respective directories
- **Docker logs**: `docker-compose logs -f hilabs-pipeline`

## 🚀 Production Deployment

### Docker Production Setup
```bash
# Production docker-compose
docker-compose -f docker-compose.yml up -d

# Scale services if needed
docker-compose up -d --scale hilabs-pipeline=2

# Monitor services
docker-compose ps
docker-compose logs -f
```

### Environment Variables
```bash
# Create .env file
cat > .env << EOF
OLLAMA_URL=http://localhost:11434
CHROMADB_URL=http://localhost:8001
LOG_LEVEL=INFO
PYTHONPATH=/app
EOF
```

### Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Monitoring
```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/api/v1/heartbeat
curl http://localhost:11434/api/tags

# Resource monitoring
docker stats
```

## 📊 Performance Optimization

### Hardware Recommendations
- **CPU**: 8+ cores (for parallel processing)
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 50GB+ SSD
- **GPU**: Optional (speeds up transformers)

### Optimization Tips
```bash
# Use GPU for transformers (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Increase batch sizes for better throughput
python main.py --step 3  # Uses optimized batch processing

# Monitor resource usage
htop
nvidia-smi  # If using GPU
```

## 🔐 Security Considerations

### API Security
```python
# Add authentication to FastAPI endpoints
# See backend/routes/chatbot_routes.py for implementation
```

### Data Security
```bash
# Secure data directory
chmod 700 data/
chmod 600 data/Contracts_data/WA/WA_5_Redacted.pdf

# Use environment variables for sensitive config
export OLLAMA_API_KEY="your-api-key"
```

## 🤝 Development Setup

### Code Quality Tools
```bash
# Install development dependencies
pip install black isort flake8 pytest

# Format code
black .
isort .

# Lint code
flake8 .

# Run tests
pytest
```

### IDE Setup
```bash
# VS Code settings
cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./hilabs-env/bin/python",
    "python.linting.enabled": true,
    "python.formatting.provider": "black"
}
EOF
```

## 📞 Support

### Getting Help
1. Check logs: `tail -f pipeline.log`
2. Verify status: `python main.py --status`
3. Test individual components
4. Check Docker logs: `docker-compose logs`

### Common Commands Reference
```bash
# Pipeline management
python main.py                    # Complete pipeline
python main.py --step 4           # Specific step
python main.py --skip-steps 1,2   # Skip steps
python main.py --status           # Check status

# Docker management
docker-compose up -d              # Start services
docker-compose down               # Stop services
docker-compose logs -f            # View logs
docker-compose exec hilabs-pipeline bash  # Shell access

# Service checks
curl http://localhost:8000/health  # API health
curl http://localhost:11434/api/tags  # Ollama models
curl http://localhost:8001/api/v1/heartbeat  # ChromaDB
```

This setup guide provides everything needed to deploy and run the complete HiLABS preprocessing pipeline on any computer!
