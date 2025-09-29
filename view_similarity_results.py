#!/usr/bin/env python3
"""
Similarity Results Viewer

Interactive viewer to explore attribute similarity results across collections.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimilarityResultsViewer:
    """Interactive viewer for similarity results."""
    
    def __init__(self, results_dir: str = "outputs/similarity"):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.summary = {}
        
    def load_results(self) -> bool:
        """Load all similarity results."""
        
        if not self.results_dir.exists():
            logger.error(f"❌ Results directory not found: {self.results_dir}")
            return False
        
        # Load summary
        summary_file = self.results_dir / "similarity_pipeline_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    self.summary = json.load(f)
                logger.info("✅ Loaded summary file")
            except Exception as e:
                logger.error(f"❌ Error loading summary: {e}")
        
        # Load individual result files
        result_files = list(self.results_dir.glob("*_attribute_similarities.json"))
        
        for file in result_files:
            collection_key = file.stem.replace("_attribute_similarities", "")
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    self.results[collection_key] = json.load(f)
                logger.info(f"✅ Loaded {collection_key} results")
            except Exception as e:
                logger.error(f"❌ Error loading {collection_key}: {e}")
        
        if not self.results:
            logger.error("❌ No result files found")
            return False
        
        logger.info(f"📊 Loaded results for {len(self.results)} collections")
        return True
    
    def show_overview(self):
        """Show overview of all results."""
        
        print("\n" + "="*60)
        print("📊 ATTRIBUTE SIMILARITY RESULTS OVERVIEW")
        print("="*60)
        
        if self.summary:
            overall = self.summary.get("overall_statistics", {})
            print(f"Total attributes processed: {overall.get('total_attributes_processed', 0)}")
            print(f"Total matches found: {overall.get('total_matches_found', 0)}")
            print(f"Average matches per attribute: {overall.get('avg_matches_per_attribute', 0):.1f}")
            print(f"Collections processed: {overall.get('collections_processed', 0)}")
        
        print(f"\n📁 Available Collections:")
        for collection_key, data in self.results.items():
            collection_info = data.get("collection_info", {})
            matches = data.get("matches", {})
            
            collection_name = collection_info.get("collection_name", collection_key)
            description = collection_info.get("description", "")
            top_k = collection_info.get("top_k", 0)
            
            total_matches = sum(len(attr_matches.get("matches", [])) for attr_matches in matches.values())
            
            print(f"  {collection_key:12} | {collection_name:15} | Top {top_k:2} | {len(matches):3} attrs | {total_matches:4} matches | {description}")
    
    def show_collection_details(self, collection_key: str):
        """Show detailed results for a specific collection."""
        
        if collection_key not in self.results:
            print(f"❌ Collection '{collection_key}' not found")
            return
        
        data = self.results[collection_key]
        collection_info = data.get("collection_info", {})
        matches = data.get("matches", {})
        
        print(f"\n" + "="*60)
        print(f"📋 COLLECTION DETAILS: {collection_key.upper()}")
        print("="*60)
        
        print(f"Collection Name: {collection_info.get('collection_name', 'Unknown')}")
        print(f"Description: {collection_info.get('description', 'No description')}")
        print(f"Top K per attribute: {collection_info.get('top_k', 0)}")
        print(f"Processed at: {collection_info.get('processed_at', 'Unknown')}")
        print(f"Total attributes: {len(matches)}")
        
        # Calculate statistics
        all_rrf_scores = []
        high_quality_matches = 0
        
        for attr_matches in matches.values():
            for match in attr_matches.get("matches", []):
                rrf_score = match.get("rrf_score", 0.0)
                all_rrf_scores.append(rrf_score)
                if rrf_score > 0.01:
                    high_quality_matches += 1
        
        if all_rrf_scores:
            print(f"Average RRF Score: {sum(all_rrf_scores) / len(all_rrf_scores):.4f}")
            print(f"Max RRF Score: {max(all_rrf_scores):.4f}")
            print(f"High Quality Matches (>0.01): {high_quality_matches}")
        
        # Show top attributes by RRF score
        print(f"\n🏆 TOP 5 ATTRIBUTES BY BEST RRF SCORE:")
        
        attr_best_scores = []
        for attr_name, attr_matches in matches.items():
            best_score = 0.0
            if attr_matches.get("matches"):
                best_score = max(match.get("rrf_score", 0.0) for match in attr_matches["matches"])
            attr_best_scores.append((attr_name, best_score))
        
        attr_best_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (attr_name, best_score) in enumerate(attr_best_scores[:5], 1):
            print(f"  {i}. {attr_name}: {best_score:.4f}")
    
    def show_attribute_matches(self, collection_key: str, attribute_name: str):
        """Show matches for a specific attribute in a collection."""
        
        if collection_key not in self.results:
            print(f"❌ Collection '{collection_key}' not found")
            return
        
        matches = self.results[collection_key].get("matches", {})
        
        if attribute_name not in matches:
            print(f"❌ Attribute '{attribute_name}' not found in {collection_key}")
            available_attrs = list(matches.keys())[:10]
            print(f"Available attributes (first 10): {available_attrs}")
            return
        
        attr_data = matches[attribute_name]
        attr_matches = attr_data.get("matches", [])
        
        print(f"\n" + "="*80)
        print(f"🎯 ATTRIBUTE MATCHES: {attribute_name}")
        print(f"Collection: {collection_key}")
        print("="*80)
        
        if not attr_matches:
            print("❌ No matches found for this attribute")
            return
        
        print(f"Found {len(attr_matches)} matches:\n")
        
        for i, match in enumerate(attr_matches, 1):
            rrf_score = match.get("rrf_score", 0.0)
            page = match.get("page", 0)
            section = match.get("section", "")
            chunk_id = match.get("chunk_id", "")
            preview = match.get("content_preview", "")
            
            score_breakdown = match.get("score_breakdown", {})
            dense_score = score_breakdown.get("dense_similarity", 0.0)
            bm25_score = score_breakdown.get("bm25_score", 0.0)
            
            print(f"🏆 Rank {i} | RRF Score: {rrf_score:.4f}")
            print(f"   📄 Page: {page} | Section: {section}")
            print(f"   🆔 Chunk ID: {chunk_id}")
            print(f"   📊 Dense: {dense_score:.4f} | BM25: {bm25_score:.4f}")
            print(f"   📝 Preview: {preview}")
            print("-" * 80)
    
    def list_attributes(self, collection_key: str = None):
        """List all attributes, optionally for a specific collection."""
        
        if collection_key:
            if collection_key not in self.results:
                print(f"❌ Collection '{collection_key}' not found")
                return
            
            matches = self.results[collection_key].get("matches", {})
            attributes = list(matches.keys())
            
            print(f"\n📋 ATTRIBUTES IN {collection_key.upper()} ({len(attributes)} total):")
            for i, attr in enumerate(attributes, 1):
                match_count = len(matches[attr].get("matches", []))
                print(f"  {i:2}. {attr} ({match_count} matches)")
        
        else:
            # Show attributes across all collections
            all_attributes = set()
            for data in self.results.values():
                all_attributes.update(data.get("matches", {}).keys())
            
            attributes = sorted(list(all_attributes))
            
            print(f"\n📋 ALL ATTRIBUTES ({len(attributes)} total):")
            for i, attr in enumerate(attributes, 1):
                print(f"  {i:2}. {attr}")
    
    def interactive_mode(self):
        """Interactive exploration mode."""
        
        print("\n🔍 INTERACTIVE MODE")
        print("Commands:")
        print("  overview - Show results overview")
        print("  collections - List available collections")
        print("  collection <name> - Show collection details")
        print("  attributes [collection] - List attributes")
        print("  matches <collection> <attribute> - Show matches for attribute")
        print("  quit - Exit")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                
                if not command:
                    continue
                
                if command[0] == "quit":
                    break
                elif command[0] == "overview":
                    self.show_overview()
                elif command[0] == "collections":
                    print(f"\nAvailable collections: {list(self.results.keys())}")
                elif command[0] == "collection" and len(command) > 1:
                    self.show_collection_details(command[1])
                elif command[0] == "attributes":
                    collection = command[1] if len(command) > 1 else None
                    self.list_attributes(collection)
                elif command[0] == "matches" and len(command) > 2:
                    self.show_attribute_matches(command[1], " ".join(command[2:]))
                else:
                    print("❌ Invalid command or missing arguments")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")

def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description="View attribute similarity results")
    parser.add_argument("--results-dir", default="outputs/similarity", help="Results directory")
    parser.add_argument("--collection", help="Show specific collection details")
    parser.add_argument("--attribute", help="Show matches for specific attribute (requires --collection)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize viewer
    viewer = SimilarityResultsViewer(args.results_dir)
    
    if not viewer.load_results():
        return 1
    
    # Show overview by default
    viewer.show_overview()
    
    # Handle specific requests
    if args.collection and args.attribute:
        viewer.show_attribute_matches(args.collection, args.attribute)
    elif args.collection:
        viewer.show_collection_details(args.collection)
    
    # Interactive mode
    if args.interactive:
        viewer.interactive_mode()
    
    return 0

if __name__ == "__main__":
    exit(main())
