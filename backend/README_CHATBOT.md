# Contract Analysis Chatbot

A sophisticated AI-powered chatbot that provides natural language querying capabilities for contract analysis using preprocessed contract data.

## Features

- 🤖 **AI-Powered Responses**: Uses Phi-3 Mini (2B parameters) LLM for intelligent contract analysis
- 📊 **Contract Data Analysis**: Query preprocessed contract datasets with natural language
- 🔍 **Standard/Non-Standard Classification**: Identify and count standard vs non-standard clauses
- 💬 **Chat History**: Maintains conversation context across multiple queries
- 📄 **Detailed Results**: Returns clause content, page numbers, sections, and metadata
- 🚀 **RESTful API**: Easy integration with frontend applications

## API Endpoints

### Main Chat Endpoint
```
POST /chatbot/chat
```

**Request Body:**
```json
{
    "message": "How many non-standard clauses are in TN Contract 1?",
    "session_id": "optional_session_id"
}
```

**Response:**
```json
{
    "response": "Based on the contract data, TN Contract 1 has 15 non-standard clauses...",
    "session_id": "session_123",
    "timestamp": "2025-09-29T17:51:51",
    "data_found": [
        {
            "attribute_name": "Medicaid Timely Filing",
            "page": 33,
            "section": "section:ARTICLE III > clause:3.1",
            "isStandard": 0,
            "full_content_raw": "...",
            "preprocessed_final_content": "..."
        }
    ]
}
```

### Other Endpoints

- `GET /chatbot/health` - Health check
- `GET /chatbot/contracts/summary` - Get summary of all contracts
- `GET /chatbot/contracts/{contract_name}/non-standard` - Get non-standard clauses for specific contract
- `GET /chatbot/sessions/{session_id}/history` - Get chat history
- `DELETE /chatbot/sessions/{session_id}` - Clear chat session

## Supported Query Types

### 1. Non-Standard Clause Counting
```
"How many non-standard clauses are in TN Contract 1?"
"Count non-standard clauses in WA contracts"
"Find the number of non-standard clauses from PDF 1"
```

### 2. Clause Content Retrieval
```
"What are the non-standard clauses in TN Contract 2?"
"Show me non-standard clauses with page numbers"
"List all non-standard clauses and their content"
```

### 3. Attribute-Based Search
```
"Find clauses related to Medicaid"
"Search for timely filing clauses"
"Show clauses about regulatory requirements"
```

### 4. Contract-Specific Queries
```
"Analyze TN Contract 1"
"Compare standard vs non-standard in Contract 2"
"What's in section 3.1 of TN Contract 1?"
```

## Data Structure

The chatbot works with preprocessed contract JSON files containing:

- **isStandard**: 0 (non-standard) or 1 (standard)
- **attribute_name**: Clause category name
- **page**: Page number in original document
- **section**: Document section reference
- **full_content_raw**: Original clause text
- **preprocessed_final_content**: Cleaned and processed text
- **source_filename**: Source contract file

## Setup and Usage

### 1. Prerequisites
- Python 3.8+
- FastAPI backend running
- Ollama with phi3:mini model
- Preprocessed contract JSON files in `/outputs/precise_similarity/processed_datasets/`

### 2. Installation
```bash
cd backend
pip install -r requirements.txt
```

### 3. Start the Server
```bash
python main.py
```

### 4. Test the Chatbot
```bash
python test_chatbot.py
```

### 5. Use the Frontend Example
Open `chatbot_frontend_example.html` in a web browser.

## Configuration

Environment variables:
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://ollama:11434)
- `SLM_MODEL`: LLM model name (default: phi3:mini)

## Example Interactions

### Query 1: Count Non-Standard Clauses
**User:** "How many non-standard clauses are in TN Contract 1?"

**Bot:** "TN Contract 1 contains 15 non-standard clauses. Here are the details:
- Page 33: Medicaid Timely Filing (Section: ARTICLE III > clause:3.1)
- Page 45: Provider Obligations (Section: ARTICLE IV > clause:4.2)
- Page 67: Payment Terms (Section: ARTICLE V > clause:5.3)
..."

### Query 2: Search by Attribute
**User:** "Find clauses related to Medicaid in TN contracts"

**Bot:** "I found 8 clauses related to Medicaid across TN contracts:

**TN Contract 1:**
- Medicaid Timely Filing (Page 33) - Non-standard
- Medicaid Claims Processing (Page 41) - Standard

**TN Contract 2:**
- Medicaid Provider Requirements (Page 28) - Non-standard
..."

## Architecture

```
Frontend (HTML/JS) 
    ↓ HTTP Requests
Backend FastAPI Server
    ↓ LLM Calls
Ollama (phi3:mini)
    ↓ Data Access
JSON Contract Files
```

## Error Handling

The chatbot handles various error scenarios:
- Missing contract data
- LLM service unavailable
- Invalid queries
- Session management errors

## Performance Considerations

- Contract data is loaded into memory on startup
- Chat history is limited to 20 messages per session
- LLM responses are cached for common queries
- Timeout handling for LLM calls (30 seconds)

## Future Enhancements

- [ ] Add support for document upload and real-time processing
- [ ] Implement advanced filtering and sorting options
- [ ] Add export functionality for query results
- [ ] Integrate with vector databases for semantic search
- [ ] Add multi-language support
- [ ] Implement user authentication and session persistence
