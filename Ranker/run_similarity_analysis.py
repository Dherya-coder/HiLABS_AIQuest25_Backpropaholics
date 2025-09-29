#!/usr/bin/env python3
"""
Run Similarity Analysis

Simple script to run the attribute similarity pipeline and analyze results.
"""

import sys
import subprocess
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_collections():
    """Check if all required collections exist."""
    logger.info("🔍 Checking collection availability...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path="chroma_db_qwen",
            settings=Settings(anonymized_telemetry=False)
        )
        
        required_collections = [
            "attributes_simple",
            "TNstandard", 
            "WAstandard",
            "TNredacted",
            "WAredacted"
        ]
        
        available = {}
        for collection_name in required_collections:
            try:
                collection = client.get_collection(collection_name)
                count = collection.count()
                available[collection_name] = count
                status = "✅" if count > 0 else "⚠️"
                logger.info(f"{status} {collection_name}: {count} documents")
            except Exception:
                available[collection_name] = 0
                logger.info(f"❌ {collection_name}: Not found")
        
        # Check minimum requirements
        if available.get("attributes_simple", 0) == 0:
            logger.error("❌ attributes_simple collection is required but not found")
            return False
        
        contract_collections = sum(1 for k, v in available.items() 
                                 if k != "attributes_simple" and v > 0)
        
        if contract_collections == 0:
            logger.error("❌ No contract collections available")
            return False
        
        logger.info(f"✅ Ready to process with {contract_collections} contract collections")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking collections: {e}")
        return False

def run_pipeline():
    """Run the similarity pipeline."""
    logger.info("🚀 Running attribute similarity pipeline...")
    
    try:
        result = subprocess.run(
            [sys.executable, "attribute_similarity_pipeline.py"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("✅ Pipeline completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error("❌ Pipeline failed")
        logger.error(f"Error: {e.stderr}")
        return False

def analyze_results():
    """Analyze the generated results."""
    logger.info("📊 Analyzing results...")
    
    output_dir = Path("outputs/similarity")
    if not output_dir.exists():
        logger.error("❌ Results directory not found")
        return False
    
    # Check generated files
    result_files = list(output_dir.glob("*_attribute_similarities.json"))
    summary_file = output_dir / "similarity_pipeline_summary.json"
    
    logger.info(f"📁 Found {len(result_files)} result files:")
    for file in result_files:
        file_size_mb = file.stat().st_size / (1024 * 1024)
        logger.info(f"   - {file.name} ({file_size_mb:.1f} MB)")
    
    # Load and display summary
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            logger.info("📋 SUMMARY STATISTICS:")
            overall = summary.get("overall_statistics", {})
            logger.info(f"   Total attributes processed: {overall.get('total_attributes_processed', 0)}")
            logger.info(f"   Total matches found: {overall.get('total_matches_found', 0)}")
            logger.info(f"   Average matches per attribute: {overall.get('avg_matches_per_attribute', 0):.1f}")
            
            collections = summary.get("collections_summary", {})
            for collection_key, stats in collections.items():
                logger.info(f"   {collection_key}: {stats.get('total_matches', 0)} matches, "
                          f"avg RRF: {stats.get('avg_rrf_score', 0):.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error reading summary: {e}")
    
    return True

def show_sample_results():
    """Show sample results from one collection."""
    logger.info("🔍 Showing sample results...")
    
    output_dir = Path("outputs/similarity")
    result_files = list(output_dir.glob("*_attribute_similarities.json"))
    
    if not result_files:
        logger.warning("⚠️ No result files found")
        return
    
    # Load first result file
    sample_file = result_files[0]
    logger.info(f"📄 Sample from: {sample_file.name}")
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches = data.get("matches", {})
        if not matches:
            logger.warning("⚠️ No matches found in sample file")
            return
        
        # Show first attribute's results
        first_attr = list(matches.keys())[0]
        first_matches = matches[first_attr].get("matches", [])
        
        logger.info(f"🎯 Sample attribute: {first_attr}")
        logger.info(f"   Found {len(first_matches)} matches")
        
        # Show top 3 matches
        for i, match in enumerate(first_matches[:3], 1):
            rrf_score = match.get("rrf_score", 0)
            page = match.get("page", 0)
            section = match.get("section", "")
            preview = match.get("content_preview", "")
            
            logger.info(f"   {i}. RRF: {rrf_score:.4f} | Page: {page} | Section: {section}")
            logger.info(f"      Preview: {preview[:100]}...")
        
    except Exception as e:
        logger.error(f"❌ Error reading sample results: {e}")

def main():
    """Main function."""
    
    logger.info("🎯 ATTRIBUTE SIMILARITY ANALYSIS")
    logger.info("="*50)
    
    # Step 1: Check collections
    if not check_collections():
        logger.error("❌ Collection check failed")
        return 1
    
    # Step 2: Run pipeline
    if not run_pipeline():
        logger.error("❌ Pipeline execution failed")
        return 1
    
    # Step 3: Analyze results
    if not analyze_results():
        logger.error("❌ Results analysis failed")
        return 1
    
    # Step 4: Show sample
    show_sample_results()
    
    logger.info("\n🎉 SIMILARITY ANALYSIS COMPLETED!")
    logger.info("📁 Check outputs/similarity/ for detailed results")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
