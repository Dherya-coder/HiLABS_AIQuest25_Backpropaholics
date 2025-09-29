#!/usr/bin/env python3
"""
Attribute Similarity Pipeline

Uses RRF (Reciprocal Rank Fusion) to find similar clauses across all contract collections
for each attribute in the attributes_simple collection.

Collections:
- attributes_simple: Source attributes
- TNstandard: Top 10 matches per attribute
- WAstandard: Top 10 matches per attribute  
- TNredacted: Top 20 matches per attribute
- WAredacted: Top 20 matches per attribute

Output: Structured results in /outputs/similarity/
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import traceback

# Add Ranker directory to path
sys.path.append(str(Path(__file__).parent / "Ranker"))

from rrf_attribute_matcher import RRFAttributeMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttributeSimilarityPipeline:
    """Pipeline to find attribute similarities across all contract collections."""
    
    def __init__(self, db_path: str = "chroma_db_qwen"):
        self.db_path = db_path
        self.output_dir = Path("outputs/similarity")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collection configurations
        self.collections_config = {
            "TNstandard": {
                "collection_name": "TNstandard",
                "description": "TN Standard Template",
                "top_k": 10
            },
            "WAstandard": {
                "collection_name": "WAstandard", 
                "description": "WA Standard Template",
                "top_k": 10
            },
            "TNredacted": {
                "collection_name": "TNredacted",
                "description": "TN Redacted Contracts",
                "top_k": 20
            },
            "WAredacted": {
                "collection_name": "WAredacted",
                "description": "WA Redacted Contracts", 
                "top_k": 20
            }
        }
        
        self.attribute_collection = "attributes_simple"
        
    def verify_collections(self) -> bool:
        """Verify all required collections exist."""
        logger.info("🔍 Verifying collections availability...")
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Check attribute collection
            try:
                attr_collection = client.get_collection(self.attribute_collection)
                attr_count = attr_collection.count()
                logger.info(f"✅ {self.attribute_collection}: {attr_count} attributes")
            except Exception as e:
                logger.error(f"❌ {self.attribute_collection}: Not found - {e}")
                return False
            
            # Check contract collections
            available_collections = {}
            for collection_key, config in self.collections_config.items():
                try:
                    collection = client.get_collection(config["collection_name"])
                    count = collection.count()
                    available_collections[collection_key] = count
                    logger.info(f"✅ {config['collection_name']}: {count} documents")
                except Exception as e:
                    logger.warning(f"⚠️ {config['collection_name']}: Not available - {e}")
                    available_collections[collection_key] = 0
            
            # Update collections config to only include available ones
            self.collections_config = {
                k: v for k, v in self.collections_config.items() 
                if available_collections.get(k, 0) > 0
            }
            
            if not self.collections_config:
                logger.error("❌ No contract collections available")
                return False
            
            logger.info(f"📊 Will process {len(self.collections_config)} collections")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying collections: {e}")
            return False
    
    def process_collection_similarities(self, collection_key: str, config: Dict) -> Dict[str, Any]:
        """Process similarities for one collection."""
        
        collection_name = config["collection_name"]
        top_k = config["top_k"]
        
        logger.info(f"🔄 Processing {collection_key} ({collection_name})")
        logger.info(f"   Target: Top {top_k} matches per attribute")
        
        try:
            # Initialize RRF matcher
            matcher = RRFAttributeMatcher(db_path=self.db_path)
            
            # Run RRF matching
            results = matcher.rrf_match_attributes_to_contracts(
                attribute_collection=self.attribute_collection,
                contract_collection=collection_name,
                top_k=top_k
            )
            
            if not results or not results.get("matches"):
                logger.warning(f"⚠️ No results for {collection_key}")
                return {}
            
            # Add collection metadata
            results["collection_info"] = {
                "collection_key": collection_key,
                "collection_name": collection_name,
                "description": config["description"],
                "top_k": top_k,
                "processed_at": datetime.now().isoformat()
            }
            
            # Log summary
            total_attributes = len(results.get("matches", {}))
            total_matches = sum(len(matches.get("matches", [])) for matches in results.get("matches", {}).values())
            
            logger.info(f"✅ {collection_key}: {total_attributes} attributes, {total_matches} total matches")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing {collection_key}: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def save_results(self, collection_key: str, results: Dict[str, Any]) -> bool:
        """Save results to JSON file."""
        
        if not results:
            logger.warning(f"⚠️ No results to save for {collection_key}")
            return False
        
        try:
            output_file = self.output_dir / f"{collection_key}_attribute_similarities.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"💾 Saved {collection_key} results: {output_file} ({file_size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving {collection_key} results: {e}")
            return False
    
    def create_summary_report(self, all_results: Dict[str, Dict]) -> None:
        """Create a summary report of all results."""
        
        logger.info("📊 Creating summary report...")
        
        summary = {
            "pipeline_info": {
                "name": "Attribute Similarity Pipeline",
                "method": "RRF (Reciprocal Rank Fusion)",
                "attribute_collection": self.attribute_collection,
                "processed_at": datetime.now().isoformat(),
                "total_collections": len(all_results)
            },
            "collections_summary": {},
            "attribute_statistics": {},
            "top_performing_attributes": []
        }
        
        # Collection summaries
        total_attributes_processed = 0
        total_matches_found = 0
        
        for collection_key, results in all_results.items():
            if not results:
                continue
                
            collection_info = results.get("collection_info", {})
            matches = results.get("matches", {})
            
            collection_summary = {
                "collection_name": collection_info.get("collection_name", collection_key),
                "description": collection_info.get("description", ""),
                "top_k": collection_info.get("top_k", 0),
                "attributes_processed": len(matches),
                "total_matches": sum(len(attr_matches.get("matches", [])) for attr_matches in matches.values()),
                "avg_rrf_score": 0.0,
                "high_quality_matches": 0
            }
            
            # Calculate averages
            all_rrf_scores = []
            high_quality_count = 0
            
            for attr_matches in matches.values():
                for match in attr_matches.get("matches", []):
                    rrf_score = match.get("rrf_score", 0.0)
                    all_rrf_scores.append(rrf_score)
                    if rrf_score > 0.01:  # High quality threshold
                        high_quality_count += 1
            
            if all_rrf_scores:
                collection_summary["avg_rrf_score"] = sum(all_rrf_scores) / len(all_rrf_scores)
                collection_summary["high_quality_matches"] = high_quality_count
            
            summary["collections_summary"][collection_key] = collection_summary
            
            total_attributes_processed += collection_summary["attributes_processed"]
            total_matches_found += collection_summary["total_matches"]
        
        # Overall statistics
        summary["overall_statistics"] = {
            "total_attributes_processed": total_attributes_processed,
            "total_matches_found": total_matches_found,
            "avg_matches_per_attribute": total_matches_found / max(total_attributes_processed, 1),
            "collections_processed": len([r for r in all_results.values() if r])
        }
        
        # Save summary
        try:
            summary_file = self.output_dir / "similarity_pipeline_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Summary report saved: {summary_file}")
            
            # Log key statistics
            logger.info("📊 PIPELINE SUMMARY:")
            logger.info(f"   Total attributes processed: {summary['overall_statistics']['total_attributes_processed']}")
            logger.info(f"   Total matches found: {summary['overall_statistics']['total_matches_found']}")
            logger.info(f"   Collections processed: {summary['overall_statistics']['collections_processed']}")
            
            for collection_key, col_summary in summary["collections_summary"].items():
                logger.info(f"   {collection_key}: {col_summary['attributes_processed']} attrs, "
                          f"{col_summary['total_matches']} matches, "
                          f"avg RRF: {col_summary['avg_rrf_score']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error saving summary: {e}")
    
    def run_pipeline(self) -> bool:
        """Run the complete attribute similarity pipeline."""
        
        logger.info("🚀 STARTING ATTRIBUTE SIMILARITY PIPELINE")
        logger.info("="*60)
        
        # Step 1: Verify collections
        if not self.verify_collections():
            logger.error("❌ Collection verification failed")
            return False
        
        # Step 2: Process each collection
        all_results = {}
        successful_collections = 0
        
        for collection_key, config in self.collections_config.items():
            logger.info(f"\n{'='*20} {collection_key.upper()} {'='*20}")
            
            try:
                # Process similarities
                results = self.process_collection_similarities(collection_key, config)
                
                if results:
                    # Save results
                    if self.save_results(collection_key, results):
                        all_results[collection_key] = results
                        successful_collections += 1
                    else:
                        logger.warning(f"⚠️ Failed to save results for {collection_key}")
                else:
                    logger.warning(f"⚠️ No results generated for {collection_key}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {collection_key}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Step 3: Create summary report
        if all_results:
            self.create_summary_report(all_results)
        
        # Step 4: Final status
        total_collections = len(self.collections_config)
        
        logger.info("\n" + "="*60)
        logger.info("🎯 PIPELINE COMPLETION STATUS")
        logger.info("="*60)
        
        if successful_collections == total_collections:
            logger.info("🎉 ALL COLLECTIONS PROCESSED SUCCESSFULLY!")
            logger.info(f"📁 Results saved in: {self.output_dir}")
            logger.info("📋 Files created:")
            for collection_key in all_results.keys():
                logger.info(f"   - {collection_key}_attribute_similarities.json")
            logger.info("   - similarity_pipeline_summary.json")
            return True
        else:
            logger.warning(f"⚠️ PARTIAL SUCCESS: {successful_collections}/{total_collections} collections processed")
            return False

def main():
    """Main entry point."""
    
    try:
        pipeline = AttributeSimilarityPipeline()
        success = pipeline.run_pipeline()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
