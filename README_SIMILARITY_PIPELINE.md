# Attribute Similarity Pipeline

This pipeline uses **RRF (Reciprocal Rank Fusion)** to find the most similar contract clauses for each attribute across all contract collections.

## 🎯 **What It Does**

For each attribute in the `attributes_simple` collection, finds:
- **Top 10 matches** from `TNstandard` collection
- **Top 10 matches** from `WAstandard` collection  
- **Top 20 matches** from `TNredacted` collection
- **Top 20 matches** from `WAredacted` collection

Uses the proven `rrf_attribute_matcher.py` which combines:
- **Dense similarity** (cosine similarity with embeddings)
- **BM25 sparse retrieval** (keyword matching)
- **RRF fusion** for optimal ranking

## 📁 **Files Structure**

```
├── attribute_similarity_pipeline.py    # Main pipeline
├── run_similarity_analysis.py         # Simple runner script
├── view_similarity_results.py         # Results viewer
├── Ranker/
│   └── rrf_attribute_matcher.py       # Core RRF matching logic
└── outputs/similarity/                # Results directory
    ├── TNstandard_attribute_similarities.json
    ├── WAstandard_attribute_similarities.json
    ├── TNredacted_attribute_similarities.json
    ├── WAredacted_attribute_similarities.json
    └── similarity_pipeline_summary.json
```

## 🚀 **How to Run**

### **Option 1: Simple One-Command**
```bash
python3 run_similarity_analysis.py
```

### **Option 2: Run Pipeline Directly**
```bash
python3 attribute_similarity_pipeline.py
```

### **Option 3: Step by Step**
```bash
# 1. Check collections
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_db_qwen')
for name in ['attributes_simple', 'TNstandard', 'WAstandard', 'TNredacted', 'WAredacted']:
    try:
        collection = client.get_collection(name)
        print(f'{name}: {collection.count()} documents')
    except:
        print(f'{name}: Not found')
"

# 2. Run pipeline
python3 attribute_similarity_pipeline.py

# 3. View results
python3 view_similarity_results.py --interactive
```

## 📊 **Expected Output**

### **Console Output:**
```
🚀 STARTING ATTRIBUTE SIMILARITY PIPELINE
==============================================================
🔍 Verifying collections availability...
✅ attributes_simple: 45 attributes
✅ TNstandard: 156 documents
✅ WAstandard: 142 documents  
✅ TNredacted: 2404 documents
✅ WAredacted: 1876 documents
📊 Will process 4 collections

==================== TNSTANDARD ====================
🔄 Processing TNstandard (TNstandard)
   Target: Top 10 matches per attribute
✅ TNstandard: 45 attributes, 450 total matches

==================== WASTANDARD ====================
🔄 Processing WAstandard (WAstandard)
   Target: Top 10 matches per attribute
✅ WAstandard: 45 attributes, 450 total matches

🎉 ALL COLLECTIONS PROCESSED SUCCESSFULLY!
📁 Results saved in: outputs/similarity
```

### **Generated Files:**
- `TNstandard_attribute_similarities.json` (~2-5 MB)
- `WAstandard_attribute_similarities.json` (~2-5 MB)
- `TNredacted_attribute_similarities.json` (~8-15 MB)
- `WAredacted_attribute_similarities.json` (~8-15 MB)
- `similarity_pipeline_summary.json` (~50 KB)

## 🔍 **Viewing Results**

### **Quick Overview:**
```bash
python3 view_similarity_results.py
```

### **Interactive Exploration:**
```bash
python3 view_similarity_results.py --interactive
```

**Interactive Commands:**
- `overview` - Show results summary
- `collections` - List available collections
- `collection TNredacted` - Show TNredacted details
- `attributes` - List all attributes
- `attributes TNredacted` - List attributes in TNredacted
- `matches TNredacted Provider Network Requirements` - Show matches for specific attribute

### **Specific Collection:**
```bash
python3 view_similarity_results.py --collection TNredacted
```

### **Specific Attribute Matches:**
```bash
python3 view_similarity_results.py --collection TNredacted --attribute "Provider Network Requirements"
```

## 📋 **Result Structure**

Each result file contains:

```json
{
  "collection_info": {
    "collection_name": "TNredacted",
    "description": "TN Redacted Contracts",
    "top_k": 20,
    "processed_at": "2025-09-29T01:45:00"
  },
  "summary": {
    "method": "RRF (Dense + BM25)",
    "total_attributes": 45,
    "total_contracts": 2404,
    "rrf_k": 60
  },
  "matches": {
    "Provider Network Requirements": {
      "matches": [
        {
          "rank": 1,
          "rrf_score": 0.0234,
          "page": 15,
          "section": "clause:network_participation",
          "chunk_id": "TN_Contract1_p15_c3",
          "content_preview": "Provider agrees to participate in...",
          "score_breakdown": {
            "dense_similarity": 0.8456,
            "bm25_score": 12.34,
            "dense_rank": 2,
            "bm25_rank": 5,
            "rrf_contribution_dense": 0.0161,
            "rrf_contribution_bm25": 0.0154
          }
        }
      ]
    }
  }
}
```

## ⚡ **Performance Notes**

- **Processing time**: ~2-5 minutes per collection
- **Memory usage**: ~2-4 GB peak
- **Output size**: ~20-40 MB total
- **RRF scoring**: Combines dense + sparse for optimal results

## 🎯 **Use Cases**

1. **Contract Analysis**: Find similar clauses across different contract types
2. **Compliance Checking**: Identify how attributes are handled in different contracts
3. **Template Creation**: Extract best practices from similar clauses
4. **Legal Research**: Discover patterns in contract language
5. **Quality Assurance**: Verify consistency across contract collections

## 🔧 **Troubleshooting**

### **Collection Not Found:**
```bash
# Check available collections
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_db_qwen')
print([c.name for c in client.list_collections()])
"
```

### **Memory Issues:**
- Reduce batch size in `rrf_attribute_matcher.py`
- Process collections individually
- Use smaller `top_k` values

### **No Results:**
- Check if embeddings exist in collections
- Verify attribute collection has content
- Check ChromaDB path is correct

## 📈 **Expected Results Quality**

- **High RRF scores (>0.02)**: Very relevant matches
- **Medium RRF scores (0.01-0.02)**: Moderately relevant  
- **Low RRF scores (<0.01)**: Potentially relevant but lower confidence

The RRF method typically provides better results than pure similarity or pure keyword matching alone.
