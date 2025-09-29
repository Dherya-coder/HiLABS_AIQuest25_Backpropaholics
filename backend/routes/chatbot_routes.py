import os
import json
import httpx
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from datetime import datetime
import glob
import re

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
SLM_MODEL = os.getenv("SLM_MODEL", "phi3:mini")
DATA_PATH = "/app/outputs/precise_similarity/processed_datasets"

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
        print(f"📊 Contract data already loaded: {len(contract_data)} files")
        return
    
    print(f"📂 Loading contract data from: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data path does not exist: {DATA_PATH}")
        return
    
    json_files = glob.glob(os.path.join(DATA_PATH, "*.json"))
    print(f"📄 Found {len(json_files)} JSON files")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                filename = os.path.basename(file_path)
                contract_data[filename] = data
                print(f"✅ Loaded {filename}: {len(data)} items")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    print(f"🎉 Total contract data loaded: {len(contract_data)} files")

def extract_clause_number(section: Optional[str]) -> str:
    """Extract clause number like 3.1 from section string when present."""
    if not section:
        return "N/A"
    m = re.search(r"clause\s*:\s*([^>\s]+)", section, re.IGNORECASE)
    return m.group(1) if m else "N/A"

def search_contracts(query: str, contract_filter: Optional[str] = None, is_standard: Optional[int] = None) -> List[Dict[str, Any]]:
    """Search through contract data based on query - optimized for speed with optional standard filter"""
    results = []
    query_lower = query.lower()

    # Quick keyword extraction for faster matching
    keywords = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]

    for filename, data in contract_data.items():
        # Apply contract filter if specified
        if contract_filter and contract_filter.lower() not in filename.lower():
            continue

        # Limit search to first 30 items per file for speed
        for item in data[:30]:
            # Apply standard filter if specified
            if is_standard in (0, 1) and item.get('isStandard') != is_standard:
                continue

            # Quick keyword matching instead of full text search
            item_text = f"{item.get('attribute_name', '')} {item.get('preprocessed_final_content', '')}".lower()

            if any(keyword in item_text for keyword in keywords) or not keywords:
                section = item.get('section')
                item_copy = {
                    'source_filename': filename,
                    'attribute_name': item.get('attribute_name'),
                    'attribute_number': item.get('attribute_number'),
                    'page': item.get('page'),
                    'section': section,
                    'clause_number': extract_clause_number(section),
                    'isStandard': item.get('isStandard'),
                    'full_content_raw': item.get('full_content_raw', '')[:220]  # Truncate for speed
                }
                results.append(item_copy)

                # Limit results for faster processing
                if len(results) >= 12:
                    return results

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
    """Generate a strict prompt that forces a single-line answer."""
    # Provide up to 6 compact evidence items
    items = []
    for item in relevant_data[:6]:
        items.append(
            f"Contract={item.get('source_filename','N/A')}; Attr={item.get('attribute_name','N/A')}; Clause={item.get('clause_number','N/A')}; Page={item.get('page','N/A')}; Standard={item.get('isStandard','N/A')}"
        )
    evidence = " | ".join(items) if items else "(no items)"

    prompt = (
        "You are Phi-3 Mini. Answer the user in ONE SHORT LINE of plain text (<= 20 words).\n"
        "Do NOT output JSON, lists, markdown, or multiple lines.\n"
        "Use only the provided evidence. If counting non-standard, isStandard=0.\n"
        f"Question: {user_query}\n"
        f"Evidence: {evidence}\n"
        "Return exactly one concise line."
    )
    return prompt

async def call_ollama_llm(prompt: str, model: Optional[str] = None) -> str:
    """Call Ollama LLM with the generated prompt"""
    try:
        model_to_use = model or SLM_MODEL
        print(f"🤖 Calling Ollama at {OLLAMA_BASE_URL} with model {model_to_use}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # First check if model is available
            try:
                models_response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if models_response.status_code == 200:
                    models = models_response.json()
                    available_models = [model['name'] for model in models.get('models', [])]
                    print(f"📋 Available models: {available_models}")
                    
                    if model_to_use not in available_models:
                        return f"❌ Model {model_to_use} not available. Available models: {available_models}. Please run: docker exec ollama ollama pull {model_to_use}"
            except Exception as e:
                print(f"⚠️ Could not check available models: {e}")
            
            # Make the generation request with optimized parameters
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.8,
                        "num_predict": 120,
                        "num_ctx": 512,      # lower KV cache
                        "stop": ["\n"]      # stop at first newline for single-line
                    }
                }
            )
            
            print(f"📡 Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get("response", "Sorry, I couldn't generate a response.")
                print(f"✅ LLM Response received: {llm_response[:100]}...")
                return llm_response
            else:
                error_text = response.text
                print(f"❌ Ollama error: {response.status_code} - {error_text}")
                # Auto fallback for memory errors: try a lighter quantized model
                if "requires more system memory" in error_text.lower():
                    fallback_model = "phi3:mini:Q2_K"
                    if fallback_model == model_to_use:
                        return f"Error calling LLM: HTTP {response.status_code} - {error_text}"
                    # Ensure fallback model exists
                    try:
                        tags = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                        if tags.status_code == 200 and fallback_model not in [m['name'] for m in tags.json().get('models', [])]:
                            return ("❌ Model requires more memory and fallback model not present. "
                                    f"Please run: docker exec ollama ollama pull {fallback_model}")
                    except Exception:
                        pass
                    print(f"↩️ Retrying with fallback model: {fallback_model}")
                    # Retry with smaller context and predict
                    retry = await client.post(
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": fallback_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "top_p": 0.8,
                                "num_predict": 80,
                                "num_ctx": 256,
                                "stop": ["\n"]
                            }
                        }
                    )
                    if retry.status_code == 200:
                        return retry.json().get("response", "")
                    return f"Error calling LLM (fallback {fallback_model}): HTTP {retry.status_code} - {retry.text}"
                return f"Error calling LLM: HTTP {response.status_code} - {error_text}"
                
    except Exception as e:
        print(f"💥 Exception calling LLM: {str(e)}")
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

@router.post("/chat", response_class=PlainTextResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - always uses the SLM (phi-3 mini) and returns one-line plain text."""
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

    # Build filters from message (contract hint + standard hint)
    message_lower = request.message.lower()
    contract_filter = None
    if "tn contract 1" in message_lower or "contract1" in message_lower:
        contract_filter = "TN_Contract1"
    elif "tn contract 2" in message_lower:
        contract_filter = "TN_Contract2"
    elif "tn contract 3" in message_lower:
        contract_filter = "TN_Contract3"
    elif "tn contract 4" in message_lower:
        contract_filter = "TN_Contract4"
    elif "tn contract 5" in message_lower:
        contract_filter = "TN_Contract5"
    elif " wa " in f" {message_lower} ":
        contract_filter = "WA"
    elif " tn " in f" {message_lower} ":
        contract_filter = "TN"

    is_standard = None
    if "non-standard" in message_lower or "non standard" in message_lower:
        is_standard = 0
    elif "standard" in message_lower:
        is_standard = 1

    # Retrieve relevant items (always)
    relevant_data = search_contracts(request.message, contract_filter=contract_filter, is_standard=is_standard)

    # Generate LLM prompt and call the model
    prompt = generate_llm_prompt(request.message, relevant_data, chat_sessions[session_id][:-1])
    llm_response = await call_ollama_llm(prompt)

    # Post-process to ensure a single line; fall back to data-derived one-liner on LLM error
    def to_single_line(text: str) -> str:
        text = text.replace("Answer:", "").strip()
        text = text.split("\n")[0]
        return " ".join(text.split())

    single = to_single_line(llm_response)

    if single.lower().startswith("error calling llm") or "memory" in single.lower():
        # Fallback concise line from data
        if is_standard == 0 and relevant_data:
            # Count non-standard in filtered scope
            count = len(relevant_data)
            hint = relevant_data[0]
            contract = hint.get('source_filename', 'contract')
            single = f"{contract}: {count} non-standard clauses"
        elif relevant_data:
            item = relevant_data[0]
            single = (
                f"Clause {item.get('clause_number','N/A')} page {item.get('page','N/A')} - {item.get('attribute_name','N/A')}"
            )
        else:
            single = "No relevant information found"
    
    # Add assistant response to history
    assistant_message = ChatMessage(role="assistant", content=single)
    chat_sessions[session_id].append(assistant_message)
    
    # Limit chat history to last 20 messages
    if len(chat_sessions[session_id]) > 20:
        chat_sessions[session_id] = chat_sessions[session_id][-20:]
    
    return single

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
