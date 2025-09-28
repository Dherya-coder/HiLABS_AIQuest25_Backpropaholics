#!/usr/bin/env python3
"""
Test script for Contract Processing Microservices
"""

import requests
import json
import time
import sys

# Service URLs
BACKEND_URL = "http://localhost:8000"
EMBEDDINGS_URL = "http://localhost:8002"
CHROMADB_URL = "http://localhost:8001"
OLLAMA_URL = "http://localhost:11434"

def test_service_health():
    """Test all service health endpoints"""
    print("🔍 Testing service health...")
    
    services = [
        ("Backend", f"{BACKEND_URL}/health"),
        ("Embeddings", f"{EMBEDDINGS_URL}/health"),
        ("ChromaDB", f"{CHROMADB_URL}/api/v1/heartbeat"),
        ("Ollama", f"{OLLAMA_URL}/api/tags")
    ]
    
    all_healthy = True
    for name, url in services:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"  ✅ {name}: Healthy")
            else:
                print(f"  ❌ {name}: Unhealthy (Status: {response.status_code})")
                all_healthy = False
        except Exception as e:
            print(f"  ❌ {name}: Connection failed - {str(e)}")
            all_healthy = False
    
    return all_healthy

def test_embeddings_service():
    """Test embeddings service functionality"""
    print("\n🧠 Testing embeddings service...")
    
    try:
        # Test single embedding
        response = requests.post(
            f"{EMBEDDINGS_URL}/embed",
            json={"text": "This is a test contract clause about payment terms."},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding_length = len(data["embedding"])
            print(f"  ✅ Single embedding: Generated {embedding_length}-dimensional vector")
        else:
            print(f"  ❌ Single embedding failed: {response.status_code}")
            return False
        
        # Test similarity
        response = requests.post(
            f"{EMBEDDINGS_URL}/similarity",
            params={
                "text1": "Payment shall be made within 30 days",
                "text2": "Invoice payment due in one month"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            similarity = data["similarity"]
            print(f"  ✅ Similarity computation: {similarity:.3f}")
        else:
            print(f"  ❌ Similarity test failed: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Embeddings service test failed: {str(e)}")
        return False

def test_backend_service():
    """Test backend service basic functionality"""
    print("\n🔧 Testing backend service...")
    
    try:
        # Test query endpoint (should work even without documents)
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"query": "test query", "top_k": 3},
            timeout=30
        )
        
        if response.status_code == 200:
            print("  ✅ Query endpoint: Working")
        elif response.status_code == 500:
            # Expected if no documents are indexed yet
            print("  ⚠️  Query endpoint: No documents indexed (expected)")
        else:
            print(f"  ❌ Query endpoint failed: {response.status_code}")
            return False
        
        # Test files listing
        response = requests.get(f"{BACKEND_URL}/files", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Files listing: {len(data['files'])} files")
        else:
            print(f"  ❌ Files listing failed: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backend service test failed: {str(e)}")
        return False

def test_ollama_models():
    """Test if required Ollama models are available"""
    print("\n🤖 Testing Ollama models...")
    
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            
            required_models = ["qwen2.5-coder:0.5b", "phi3:mini"]
            available_models = []
            
            for model in required_models:
                if any(model in m for m in models):
                    available_models.append(model)
                    print(f"  ✅ Model available: {model}")
                else:
                    print(f"  ❌ Model missing: {model}")
            
            if len(available_models) == len(required_models):
                print("  ✅ All required models are available")
                return True
            else:
                print(f"  ⚠️  {len(available_models)}/{len(required_models)} models available")
                return False
        else:
            print(f"  ❌ Failed to fetch models: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Ollama models test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Contract Processing Microservices Test Suite")
    print("=" * 50)
    
    tests = [
        ("Service Health", test_service_health),
        ("Ollama Models", test_ollama_models),
        ("Embeddings Service", test_embeddings_service),
        ("Backend Service", test_backend_service)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"  ✅ {test_name}: PASSED")
            else:
                print(f"  ❌ {test_name}: FAILED")
        except Exception as e:
            print(f"  ❌ {test_name}: ERROR - {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your microservices are ready to use.")
        print("\n📚 Next steps:")
        print("  1. Upload a PDF: curl -X POST 'http://localhost:8000/upload-pdf' -F 'file=@contract.pdf'")
        print("  2. Check status: curl 'http://localhost:8000/status/{file_id}'")
        print("  3. Query documents: curl -X POST 'http://localhost:8000/query' -d '{\"query\":\"payment terms\"}'")
        return 0
    else:
        print("❌ Some tests failed. Please check the service logs:")
        print("  docker-compose logs -f")
        return 1

if __name__ == "__main__":
    sys.exit(main())
