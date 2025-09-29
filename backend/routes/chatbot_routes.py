import os
import json
import httpx
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import glob

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
SLM_MODEL = os.getenv("SLM_MODEL", "phi3:mini")
DATA_PATH = "/home/jyotiraditya/preprocess/outputs/precise_similarity/processed_datasets"

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    data_found: List[Dict[str, Any]] = []

# In-memory chat history storage (in production, use Redis or database)
chat_sessions: Dict[str, List[ChatMessage]] = {}

# Global data storage
contract_data: Dict[str, List[Dict[str, Any]]] = {}

def load_contract_data():
    """Load all contract JSON files into memory"""
    global contract_data
    
    if contract_data:  # Already loaded
        return
    
    json_files = glob.glob(os.path.join(DATA_PATH, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                filename = os.path.basename(file_path)
                contract_data[filename] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

def search_contracts(query: str, contract_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search through contract data based on query"""
    results = []
    
    # Normalize query for better matching
    query_lower = query.lower()
    
    for filename, data in contract_data.items():
        # Apply contract filter if specified
        if contract_filter and contract_filter.lower() not in filename.lower():
            continue
            
        for item in data:
            # Check if this matches the query context
            content_match = (
                query_lower in item.get('attribute_name', '').lower() or
                query_lower in item.get('full_content_raw', '').lower() or
                query_lower in item.get('preprocessed_final_content', '').lower() or
                query_lower in item.get('section', '').lower()
            )
            
            if content_match:
                # Add filename for context
                item_copy = item.copy()
                item_copy['source_filename'] = filename
                results.append(item_copy)
    
    return results

def count_non_standard_clauses(contract_filter: Optional[str] = None) -> Dict[str, Any]:
    """Count non-standard clauses (isStandard = 0) in contracts"""
    results = {}
    
    for filename, data in contract_data.items():
        if contract_filter and contract_filter.lower() not in filename.lower():
            continue
            
        non_standard_clauses = [item for item in data if item.get('isStandard') == 0]
        
        if non_standard_clauses:
            results[filename] = {
                'count': len(non_standard_clauses),
                'clauses': non_standard_clauses
            }
    
    return results

def generate_llm_prompt(user_query: str, relevant_data: List[Dict[str, Any]], chat_history: List[ChatMessage]) -> str:
    """Generate a comprehensive prompt for the LLM"""
    
    # Build chat history context
    history_context = ""
    if chat_history:
        recent_history = chat_history[-6:]  # Last 3 exchanges
        for msg in recent_history:
            history_context += f"{msg.role.capitalize()}: {msg.content}\n"
    
    # Build data context
    data_context = ""
    if relevant_data:
        data_context = "RELEVANT CONTRACT DATA:\n"
        for i, item in enumerate(relevant_data[:10]):  # Limit to 10 items
            data_context += f"""
Item {i+1}:
- Source: {item.get('source_filename', 'Unknown')}
- Attribute: {item.get('attribute_name', 'N/A')}
- Page: {item.get('page', 'N/A')}
- Section: {item.get('section', 'N/A')}
- Is Standard: {'Yes' if item.get('isStandard') == 1 else 'No'}
- Content: {item.get('full_content_raw', '')[:200]}...
- Preprocessed: {item.get('preprocessed_final_content', '')[:100]}...
"""
    
    prompt = f"""You are a helpful contract analysis assistant. You have access to preprocessed contract data and can answer questions about contract clauses, standards compliance, and specific contract details.

CHAT HISTORY:
{history_context}

{data_context}

USER QUERY: {user_query}

INSTRUCTIONS:
1. Answer the user's question based on the provided contract data
2. If asked about non-standard clauses, count items where "isStandard" = 0
3. If asked about specific contracts, filter by filename (e.g., "TN_Contract1" for TNredacted_TN_Contract1_Redacted_dataset.json)
4. Always provide specific details like page numbers, sections, and clause content when available
5. Maintain context from the chat history
6. If no relevant data is found, clearly state that
7. Be concise but informative

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide specific counts, page numbers, and sections when relevant
- Include brief excerpts of clause content when helpful
- End with a summary or offer to help with follow-up questions

Please provide a helpful response based on the available data."""

    return prompt

async def call_ollama_llm(prompt: str) -> str:
    """Call Ollama LLM with the generated prompt"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": SLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Sorry, I couldn't generate a response.")
            else:
                return f"Error calling LLM: HTTP {response.status_code}"
                
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# Load contract data when module is imported
load_contract_data()

@router.get("/health")
async def chatbot_health():
    """Health check for chatbot service"""
    return {
        "status": "healthy",
        "service": "chatbot",
        "data_loaded": len(contract_data),
        "llm_model": SLM_MODEL
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    
    # Ensure data is loaded
    if not contract_data:
        load_contract_data()
    
    if not contract_data:
        raise HTTPException(status_code=500, detail="Contract data not available")
    
    # Get or create chat session
    session_id = request.session_id
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # Add user message to history
    user_message = ChatMessage(role="user", content=request.message)
    chat_sessions[session_id].append(user_message)
    
    # Search for relevant data
    relevant_data = []
    
    # Check if query is about counting non-standard clauses
    if "non-standard" in request.message.lower() or "non standard" in request.message.lower():
        # Extract contract filter if mentioned
        contract_filter = None
        message_lower = request.message.lower()
        
        if "tn contract 1" in message_lower or "contract1" in message_lower:
            contract_filter = "TN_Contract1"
        elif "tn contract 2" in message_lower or "contract2" in message_lower:
            contract_filter = "TN_Contract2"
        elif "tn contract 3" in message_lower or "contract3" in message_lower:
            contract_filter = "TN_Contract3"
        elif "tn contract 4" in message_lower or "contract4" in message_lower:
            contract_filter = "TN_Contract4"
        elif "tn contract 5" in message_lower or "contract5" in message_lower:
            contract_filter = "TN_Contract5"
        elif "wa" in message_lower:
            contract_filter = "WA"
        elif "tn" in message_lower:
            contract_filter = "TN"
        
        non_standard_data = count_non_standard_clauses(contract_filter)
        for filename, data in non_standard_data.items():
            relevant_data.extend(data['clauses'])
    else:
        # General search
        relevant_data = search_contracts(request.message)
    
    # Generate LLM prompt
    prompt = generate_llm_prompt(
        request.message, 
        relevant_data, 
        chat_sessions[session_id][:-1]  # Exclude current message
    )
    
    # Call LLM
    llm_response = await call_ollama_llm(prompt)
    
    # Add assistant response to history
    assistant_message = ChatMessage(role="assistant", content=llm_response)
    chat_sessions[session_id].append(assistant_message)
    
    # Limit chat history to last 20 messages
    if len(chat_sessions[session_id]) > 20:
        chat_sessions[session_id] = chat_sessions[session_id][-20:]
    
    return ChatResponse(
        response=llm_response,
        session_id=session_id,
        timestamp=datetime.now(),
        data_found=relevant_data[:5]  # Return first 5 relevant items
    )

@router.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in chat_sessions:
        return {"messages": []}
    
    return {"messages": chat_sessions[session_id]}

@router.delete("/sessions/{session_id}")
async def clear_chat_session(session_id: str):
    """Clear chat history for a session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    
    return {"message": f"Session {session_id} cleared"}

@router.get("/contracts/summary")
async def get_contracts_summary():
    """Get summary of loaded contracts"""
    if not contract_data:
        load_contract_data()
    
    summary = {}
    for filename, data in contract_data.items():
        total_clauses = len(data)
        standard_clauses = sum(1 for item in data if item.get('isStandard') == 1)
        non_standard_clauses = total_clauses - standard_clauses
        
        summary[filename] = {
            'total_clauses': total_clauses,
            'standard_clauses': standard_clauses,
            'non_standard_clauses': non_standard_clauses
        }
    
    return summary

@router.get("/contracts/{contract_name}/non-standard")
async def get_non_standard_clauses(contract_name: str):
    """Get all non-standard clauses for a specific contract"""
    if not contract_data:
        load_contract_data()
    
    # Find matching contract file
    matching_files = [f for f in contract_data.keys() if contract_name.lower() in f.lower()]
    
    if not matching_files:
        raise HTTPException(status_code=404, detail=f"Contract {contract_name} not found")
    
    results = {}
    for filename in matching_files:
        data = contract_data[filename]
        non_standard_clauses = [item for item in data if item.get('isStandard') == 0]
        results[filename] = {
            'count': len(non_standard_clauses),
            'clauses': non_standard_clauses
        }
    
    return results
