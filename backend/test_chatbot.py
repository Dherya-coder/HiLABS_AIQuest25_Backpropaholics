#!/usr/bin/env python3
"""
Test script for the chatbot functionality
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_chatbot():
    """Test the chatbot endpoints"""
    
    async with httpx.AsyncClient() as client:
        print("🤖 Testing Chatbot Functionality\n")
        
        # Test health endpoint
        print("1. Testing chatbot health...")
        try:
            response = await client.get(f"{BASE_URL}/chatbot/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        print("\n" + "="*50 + "\n")
        
        # Test contracts summary
        print("2. Testing contracts summary...")
        try:
            response = await client.get(f"{BASE_URL}/chatbot/contracts/summary")
            if response.status_code == 200:
                data = response.json()
                print("✅ Contracts summary:")
                for contract, stats in data.items():
                    print(f"  📄 {contract}:")
                    print(f"    - Total clauses: {stats['total_clauses']}")
                    print(f"    - Standard: {stats['standard_clauses']}")
                    print(f"    - Non-standard: {stats['non_standard_clauses']}")
            else:
                print(f"❌ Summary failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Summary error: {e}")
        
        print("\n" + "="*50 + "\n")
        
        # Test chat queries
        test_queries = [
            "How many non-standard clauses are in TN Contract 1?",
            "What are the non-standard clauses in TN Contract 1?",
            "Find clauses related to Medicaid in TN contracts",
            "Show me page numbers for non-standard clauses in Contract 1"
        ]
        
        session_id = "test_session_123"
        
        for i, query in enumerate(test_queries, 1):
            print(f"{i}. Testing query: '{query}'")
            try:
                response = await client.post(
                    f"{BASE_URL}/chatbot/chat",
                    json={
                        "message": query,
                        "session_id": session_id
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Response received:")
                    print(f"   📝 Answer: {data['response'][:200]}...")
                    print(f"   📊 Data found: {len(data['data_found'])} items")
                    if data['data_found']:
                        print(f"   📄 First item: {data['data_found'][0].get('attribute_name', 'N/A')}")
                else:
                    print(f"❌ Chat failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
            except Exception as e:
                print(f"❌ Chat error: {e}")
            
            print("\n" + "-"*30 + "\n")
        
        # Test chat history
        print("5. Testing chat history...")
        try:
            response = await client.get(f"{BASE_URL}/chatbot/sessions/{session_id}/history")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Chat history retrieved: {len(data['messages'])} messages")
            else:
                print(f"❌ History failed: {response.status_code}")
        except Exception as e:
            print(f"❌ History error: {e}")

if __name__ == "__main__":
    print("Starting chatbot tests...")
    print("Make sure the backend server is running on localhost:8000")
    print("And that Ollama is running with phi3:mini model\n")
    
    asyncio.run(test_chatbot())
