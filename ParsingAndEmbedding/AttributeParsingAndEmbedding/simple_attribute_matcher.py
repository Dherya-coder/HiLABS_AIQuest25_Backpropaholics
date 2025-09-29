#!/usr/bin/env python3
"""
Simple attribute-contract matcher using the simplified collections.

Usage:
    python simple_attribute_matcher.py --attribute-collection "attributes_simple" --contract-collection "redacted"
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import warnings

import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAttributeMatcher:
    """Simple matcher for attributes and contracts."""
    
    def __init__(self, db_path: str = "./chroma_db_qwen"):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
    
    def match_attributes_to_contracts(self, 
                                    attribute_collection: str = "attributes_simple",
                                    contract_collection: str = "redacted",
                                    top_k: int = 10) -> Dict[str, Any]:
        """Match each attribute to top K contract clauses."""
        
        # Get collections
        try:
            attr_collection = self.client.get_collection(attribute_collection)
            contract_collection = self.client.get_collection(contract_collection)
            
            logger.info(f"Attribute collection: {attr_collection.count()} items")
            logger.info(f"Contract collection: {contract_collection.count()} items")
            
        except Exception as e:
            logger.error(f"Error loading collections: {e}")
            return {}
        
        # Get all attributes
        try:
            attr_results = attr_collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )
            logger.info(f"Retrieved {len(attr_results['documents'])} attributes")
        except Exception as e:
            logger.error(f"Error getting attributes: {e}")
            return {}
        
        results = {
            "summary": {
                "total_attributes": len(attr_results['documents']),
                "total_contracts": contract_collection.count(),
                "top_k": top_k
            },
            "matches": {}
        }
        
        # Process each attribute
        for i, (doc, metadata, embedding) in enumerate(zip(
            attr_results['documents'],
            attr_results['metadatas'], 
            attr_results['embeddings']
        )):
            attr_name = metadata.get('attribute_name', f'Attribute_{i+1}')
            attr_num = metadata.get('attribute_number', i+1)
            
            logger.info(f"Processing {i+1}/{len(attr_results['documents'])}: {attr_name}")
            
            # Find top contract matches
            try:
                contract_matches = contract_collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                matches = []
                if contract_matches['documents'] and contract_matches['documents'][0]:
                    for j, (contract_doc, contract_meta, distance) in enumerate(zip(
                        contract_matches['documents'][0],
                        contract_matches['metadatas'][0],
                        contract_matches['distances'][0]
                    )):
                        similarity = 1 - distance
                        
                        matches.append({
                            "rank": j + 1,
                            "similarity": round(similarity, 4),
                            "page": contract_meta.get('page_number', 0),
                            "section": contract_meta.get('header_path', ''),
                            "chunk_id": contract_meta.get('chunk_id', ''),
                            "content_preview": contract_doc[:150] + "..." if len(contract_doc) > 150 else contract_doc,
                            "full_content": contract_doc
                        })
                
                # Calculate stats
                similarities = [m['similarity'] for m in matches]
                stats = {
                    "avg_similarity": round(np.mean(similarities), 4) if similarities else 0,
                    "max_similarity": round(max(similarities), 4) if similarities else 0,
                    "high_confidence_matches": len([s for s in similarities if s > 0.7])
                }
                
                results["matches"][attr_name] = {
                    "attribute_info": {
                        "number": attr_num,
                        "name": attr_name,
                        "content_preview": doc[:100] + "..." if len(doc) > 100 else doc
                    },
                    "statistics": stats,
                    "top_matches": matches
                }
                
            except Exception as e:
                logger.error(f"Error matching attribute {attr_name}: {e}")
                continue
        
        logger.info("✅ Matching completed!")
        return results
    
    def generate_simple_report(self, results: Dict[str, Any]) -> str:
        """Generate a simple readable report."""
        
        summary = results.get('summary', {})
        matches = results.get('matches', {})
        
        report = f"""# Attribute-Contract Matching Report

## Summary
- **Total Attributes**: {summary.get('total_attributes', 0)}
- **Total Contract Chunks**: {summary.get('total_contracts', 0)}
- **Top Matches Per Attribute**: {summary.get('top_k', 0)}

## Attribute Matches

"""
        
        for attr_name, attr_data in matches.items():
            info = attr_data.get('attribute_info', {})
            stats = attr_data.get('statistics', {})
            top_matches = attr_data.get('top_matches', [])[:3]  # Show top 3
            
            report += f"""### {info.get('number', 'N/A')}. {attr_name}

**Attribute Content**: {info.get('content_preview', 'N/A')}

**Match Statistics**:
- Average Similarity: {stats.get('avg_similarity', 0)}
- Best Match: {stats.get('max_similarity', 0)}
- High Confidence Matches: {stats.get('high_confidence_matches', 0)}

**Top 3 Contract Matches**:
"""
            
            for match in top_matches:
                section = match.get('section', 'No header')
                if not section:
                    section = 'No header'
                
                report += f"""
{match.get('rank', 'N/A')}. **Page {match.get('page', 'N/A')}** - Similarity: {match.get('similarity', 0)}
   - Section: {section}
   - Content: {match.get('content_preview', 'N/A')}
"""
            
            report += "\n---\n\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Simple attribute-contract matching")
    parser.add_argument("--attribute-collection", type=str, default="attributes_simple", help="Attribute collection name")
    parser.add_argument("--contract-collection", type=str, default="redacted", help="Contract collection name")
    parser.add_argument("--db-path", type=str, default="./chroma_db_qwen", help="ChromaDB path")
    parser.add_argument("--top-k", type=int, default=10, help="Top K matches per attribute")
    parser.add_argument("--output", type=str, default="simple_matches.json", help="Output JSON file")
    parser.add_argument("--report", type=str, default="simple_report.md", help="Output report file")
    
    args = parser.parse_args()
    
    # Initialize matcher
    matcher = SimpleAttributeMatcher(db_path=args.db_path)
    
    # Perform matching
    results = matcher.match_attributes_to_contracts(
        attribute_collection=args.attribute_collection,
        contract_collection=args.contract_collection,
        top_k=args.top_k
    )
    
    if results:
        # Save JSON results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 Results saved to: {args.output}")
        
        # Generate report
        report = matcher.generate_simple_report(results)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📊 Report saved to: {args.report}")
        
        # Print summary
        summary = results.get('summary', {})
        print(f"\n🎉 Matching Complete!")
        print(f"📈 Processed {summary.get('total_attributes', 0)} attributes")
        print(f"🔍 Searched {summary.get('total_contracts', 0)} contract chunks")
        print(f"📁 Results: {args.output}")
        print(f"📋 Report: {args.report}")

if __name__ == "__main__":
    main()
