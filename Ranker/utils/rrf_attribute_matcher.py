#!/usr/bin/env python3
"""
RRF (Reciprocal Rank Fusion) Attribute-Contract Matcher
Combines BM25 (sparse) + Dense Similarity Search + RRF for optimal results.

Usage:
    python rrf_attribute_matcher.py --attribute-collection "attributes_simple" --contract-collection "redacted"
"""

import argparse
import json
import logging
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import warnings
import re
import traceback

import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BM25Scorer:
    """BM25 scoring implementation for sparse retrieval."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.term_frequencies = []
        self.document_frequencies = {}
        self.vocabulary = set()
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization - convert to lowercase and split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def fit(self, documents: List[str]) -> None:
        """Fit BM25 on document corpus."""
        self.documents = documents
        self.doc_lengths = []
        self.term_frequencies = []
        
        # Calculate term frequencies for each document
        for doc in documents:
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies in this document
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
                self.vocabulary.add(token)
            
            self.term_frequencies.append(dict(tf))
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate document frequencies
        self.document_frequencies = defaultdict(int)
        for tf in self.term_frequencies:
            for term in tf.keys():
                self.document_frequencies[term] += 1
        
        logger.info(f"BM25 fitted on {len(documents)} documents, vocabulary size: {len(self.vocabulary)}")
    
    def score_query(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Score all documents against query and return ranked results."""
        query_tokens = self.tokenize(query)
        scores = []
        
        for doc_idx in range(len(self.documents)):
            score = self._calculate_bm25_score(query_tokens, doc_idx)
            scores.append((doc_idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            scores = scores[:top_k]
        
        return scores
    
    def _calculate_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document given query tokens."""
        score = 0.0
        doc_length = self.doc_lengths[doc_idx]
        tf_doc = self.term_frequencies[doc_idx]
        
        for token in query_tokens:
            if token not in self.vocabulary:
                continue
            
            # Term frequency in document
            tf = tf_doc.get(token, 0)
            
            # Document frequency
            df = self.document_frequencies[token]
            
            # IDF calculation (smoothed)
            idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5) + 1e-9)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / (self.avg_doc_length + 1e-9)))
            
            score += idf * (numerator / (denominator + 1e-9))
        
        return score

class RRFAttributeMatcher:
    """RRF-based matcher combining BM25 and dense similarity search."""
    
    def __init__(self, db_path: str = "./chroma_db_qwen", rrf_k: int = 20):
        """Initialize RRF matcher."""
        self.db_path = db_path
        self.rrf_k = rrf_k  # RRF parameter
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize BM25
        self.bm25 = BM25Scorer()
        self.contract_documents = []
        self.contract_metadatas = []
        
    def load_collections(self, 
                        attribute_collection: str = "attributes_simple",
                        contract_collection: str = "redacted") -> None:
        """Load both collections and prepare BM25 index."""
        
        try:
            # Load collections
            self.attr_collection = self.client.get_collection(attribute_collection)
            self.contract_collection = self.client.get_collection(contract_collection)
            
            logger.info(f"Loaded attribute collection: {self.attr_collection.count()} items")
            logger.info(f"Loaded contract collection: {self.contract_collection.count()} items")
            
            # Get all contract documents for BM25
            contract_results = self.contract_collection.get(
                include=['documents', 'metadatas']
            )
            
            self.contract_documents = contract_results.get('documents', [])
            self.contract_metadatas = contract_results.get('metadatas', [])
            
            # Fit BM25 on contract documents
            logger.info("Fitting BM25 on contract documents...")
            if not self.contract_documents:
                logger.warning("No contract documents available for BM25 fitting.")
            self.bm25.fit(self.contract_documents)
            
        except Exception as e:
            logger.error(f"Error loading collections: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_dense_similarity_scores(self, 
                                   attribute_embedding: List[float]) -> List[Tuple[int, float]]:
        """Get similarity scores using ChromaDB's built-in query method (like simple_attribute_matcher)."""
        try:
            # Use ChromaDB's query method to get ALL documents with similarity scores
            # Query with a large n_results to get all documents
            total_docs = self.contract_collection.count()
            
            results = self.contract_collection.query(
                query_embeddings=[attribute_embedding],
                n_results=total_docs,  # Get all documents
                include=['distances', 'documents']
            )
            
            if not results['distances'] or not results['distances'][0]:
                logger.error("No distances returned from ChromaDB query")
                return []
            
            # Convert distances to similarities (ChromaDB returns distances, we want similarities)
            distances = results['distances'][0]
            query_documents = results['documents'][0]
            similarities = [1 - distance for distance in distances]
            
            # Map query results back to original document indices
            # We need to find the index of each returned document in our original document list
            dense_scores = []
            for i, (query_doc, similarity) in enumerate(zip(query_documents, similarities)):
                # Find the index of this document in our original contract_documents list
                try:
                    original_idx = self.contract_documents.index(query_doc)
                    dense_scores.append((original_idx, float(similarity)))
                except ValueError:
                    # Document not found in original list (shouldn't happen)
                    logger.warning(f"Query result document {i} not found in original document list")
                    continue
            
            # Sort by similarity descending
            dense_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"ChromaDB query returned {len(dense_scores)} similarity scores")
            logger.info(f"Similarity range: [{min(similarities):.6f}, {max(similarities):.6f}]")
            logger.info(f"Top 5 similarities: {[(idx, f'{sim:.6f}') for idx, sim in dense_scores[:5]]}")
            
            return dense_scores
            
        except Exception as e:
            logger.error(f"Error getting ChromaDB similarity scores: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def get_bm25_scores(self, query_text: str) -> List[Tuple[int, float]]:
        """Get BM25 scores for ALL documents."""
        try:
            # Get BM25 scores for all documents (no top_k limit)
            bm25_scores = self.bm25.score_query(query_text, top_k=None)
            logger.info(f"Calculated BM25 scores for {len(bm25_scores)} documents")
            return bm25_scores
        except Exception as e:
            logger.error(f"Error getting BM25 scores: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def apply_rrf(self, 
                  dense_scores: List[Tuple[int, float]], 
                  bm25_scores: List[Tuple[int, float]], 
                  top_k: int = 10) -> List[Tuple[int, float, Dict[str, float]]]:
        """Apply Reciprocal Rank Fusion to combine dense and BM25 scores."""
        
        if not dense_scores:
            logger.warning("Dense scores empty — dense component will contribute default minimal RRF values.")
        if not bm25_scores:
            logger.warning("BM25 scores empty — BM25 component will contribute default minimal RRF values.")

        # Create rank dictionaries
        dense_ranks = {doc_idx: rank + 1 for rank, (doc_idx, score) in enumerate(dense_scores)}
        bm25_ranks = {doc_idx: rank + 1 for rank, (doc_idx, score) in enumerate(bm25_scores)}
        
        # Create score dictionaries for reference
        dense_score_dict = {doc_idx: score for doc_idx, score in dense_scores}
        bm25_score_dict = {doc_idx: score for doc_idx, score in bm25_scores}
        
        # Get all unique document indices
        all_doc_indices = set(dense_ranks.keys()) | set(bm25_ranks.keys())

        # If both are empty, return empty
        if not all_doc_indices:
            logger.error("Both dense and BM25 score lists are empty. RRF cannot proceed.")
            return []
        
        # Calculate RRF scores
        rrf_scores = []
        default_dense_rank = len(dense_scores) + 1 if dense_scores else (len(bm25_scores) + 1 if bm25_scores else 1)
        default_bm25_rank = len(bm25_scores) + 1 if bm25_scores else (len(dense_scores) + 1 if dense_scores else 1)

        for doc_idx in all_doc_indices:
            # RRF formula: 1/(k + rank)
            dense_rank_val = dense_ranks.get(doc_idx, default_dense_rank)
            bm25_rank_val = bm25_ranks.get(doc_idx, default_bm25_rank)

            dense_rrf = 1.0 / (self.rrf_k + dense_rank_val)
            bm25_rrf = 1.0 / (self.rrf_k + bm25_rank_val)
            
            combined_rrf = dense_rrf + bm25_rrf
            
            # Store detailed scores for analysis
            score_details = {
                'dense_score': dense_score_dict.get(doc_idx, 0.0),
                'bm25_score': bm25_score_dict.get(doc_idx, 0.0),
                'dense_rank': dense_rank_val,
                'bm25_rank': bm25_rank_val,
                'dense_rrf': dense_rrf,
                'bm25_rrf': bm25_rrf,
                'combined_rrf': combined_rrf
            }
            
            rrf_scores.append((doc_idx, combined_rrf, score_details))
        
        # Sort by RRF score descending
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        
        return rrf_scores[:top_k]
    
    def rrf_match_attributes_to_contracts(self, 
                                         attribute_collection: str = "attributes_simple",
                                         contract_collection: str = "redacted",
                                         top_k: int = 10,
                                         retrieval_k: int=None) -> Dict[str, Any]:
        """Main RRF matching function."""
        
        # Load collections
        self.load_collections(attribute_collection, contract_collection)
        
        # Get all attributes
        try:
            attr_results = self.attr_collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )
            logger.info(f"Retrieved {len(attr_results.get('documents', []))} attributes for RRF matching")
        except Exception as e:
            logger.error(f"Error getting attributes: {e}")
            logger.error(traceback.format_exc())
            return {}
        
        results = {
            "summary": {
                "method": "RRF (Dense + BM25)",
                "total_attributes": len(attr_results.get('documents', [])),
                "total_contracts": len(self.contract_documents),
                "top_k": top_k,
                "retrieval_k": retrieval_k,
                "rrf_k": self.rrf_k
            },
            "matches": {}
        }
        
        # Process each attribute
        for i, (attr_doc, attr_metadata, attr_embedding) in enumerate(zip(
            attr_results.get('documents', []),
            attr_results.get('metadatas', []), 
            attr_results.get('embeddings', [])
        )):
            attr_name = (attr_metadata or {}).get('attribute_name', f'Attribute_{i+1}')
            attr_num = (attr_metadata or {}).get('attribute_number', i+1)
            
            logger.info(f"Processing {i+1}/{len(attr_results.get('documents', []))}: {attr_name}")
            
            try:
                # Log basic embedding diagnostics
                try:
                    logger.info("Attribute embedding length/type: %s / %s", 
                                (len(attr_embedding) if attr_embedding is not None else None),
                                type(attr_embedding[0]) if (attr_embedding and len(attr_embedding) > 0) else None)
                except Exception:
                    logger.debug("Could not log attribute embedding details.")

                # Get dense similarity scores (cosine similarity for ALL documents)
                dense_scores = self.get_dense_similarity_scores(attr_embedding)

                # Get BM25 scores using attribute content as query (for ALL documents)
                bm25_scores = self.get_bm25_scores(attr_doc)
                
                # Apply RRF
                rrf_results = self.apply_rrf(dense_scores, bm25_scores, top_k=top_k)
                
                # Format results
                matches = []
                for rank, (doc_idx, rrf_score, score_details) in enumerate(rrf_results):
                    if doc_idx < len(self.contract_documents):
                        contract_doc = self.contract_documents[doc_idx]
                        contract_meta = self.contract_metadatas[doc_idx] if doc_idx < len(self.contract_metadatas) else {}
                        
                        matches.append({
                            "rank": rank + 1,
                            "rrf_score": rrf_score,
                            "page": contract_meta.get('page_number', 0),
                            "section": contract_meta.get('header_path', ''),
                            "chunk_id": contract_meta.get('chunk_id', ''),
                            "content_preview": contract_doc[:150] + "..." if len(contract_doc) > 150 else contract_doc,
                            "full_content": contract_doc,
                            "score_breakdown": {
                                "dense_similarity": score_details.get('dense_score', 0.0),
                                "bm25_score": score_details.get('bm25_score', 0.0),
                                "dense_rank": score_details.get('dense_rank', None),
                                "bm25_rank": score_details.get('bm25_rank', None),
                                "rrf_contribution_dense": score_details.get('dense_rrf', 0.0),
                                "rrf_contribution_bm25": score_details.get('bm25_rrf', 0.0)
                            }
                        })
                
                # Calculate statistics
                rrf_scores_list = [m['rrf_score'] for m in matches]
                dense_scores_list = [m['score_breakdown']['dense_similarity'] for m in matches]
                bm25_scores_list = [m['score_breakdown']['bm25_score'] for m in matches]
                
                stats = {
                    "avg_rrf_score": float(np.mean(rrf_scores_list)) if rrf_scores_list else 0.0,
                    "max_rrf_score": float(max(rrf_scores_list)) if rrf_scores_list else 0.0,
                    "avg_dense_score": float(np.mean(dense_scores_list)) if dense_scores_list else 0.0,
                    "avg_bm25_score": float(np.mean(bm25_scores_list)) if bm25_scores_list else 0.0,
                    "high_rrf_matches": int(len([s for s in rrf_scores_list if s > 0.01]))
                }
                
                results["matches"][attr_name] = {
                    "attribute_info": {
                        "number": attr_num,
                        "name": attr_name,
                        "content_preview": attr_doc[:100] + "..." if len(attr_doc) > 100 else attr_doc
                    },
                    "statistics": stats,
                    "top_matches": matches
                }
                
            except Exception as e:
                logger.error(f"Error processing attribute {attr_name}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        logger.info("✅ RRF matching completed!")
        return results
    
    def generate_rrf_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed RRF analysis report."""
        
        summary = results.get('summary', {})
        matches = results.get('matches', {})
        
        report = f"""# RRF Attribute-Contract Matching Report

## Method: {summary.get('method', 'RRF')}
- **Total Attributes**: {summary.get('total_attributes', 0)}
- **Total Contract Chunks**: {summary.get('total_contracts', 0)}
- **Top Matches Per Attribute**: {summary.get('top_k', 0)}
- **Retrieval Pool Size**: {summary.get('retrieval_k', 0)}
- **RRF Parameter (k)**: {summary.get('rrf_k', 60)}

## RRF Methodology
RRF combines Dense Similarity Search (semantic) with BM25 (lexical) using:
- **Dense**: Vector similarity using Qwen embeddings (or your chosen model)
- **BM25**: Traditional keyword-based scoring
- **RRF Formula**: 1/(k + rank) for each method, then sum scores

## Attribute Analysis

"""
        
        for attr_name, attr_data in matches.items():
            info = attr_data.get('attribute_info', {})
            stats = attr_data.get('statistics', {})
            top_matches = attr_data.get('top_matches', [])[:3]
            
            report += f"""### {info.get('number', 'N/A')}. {attr_name}

**Attribute Content**: {info.get('content_preview', 'N/A')}

**RRF Statistics**:
- Average RRF Score: {stats.get('avg_rrf_score', 0)}
- Best RRF Score: {stats.get('max_rrf_score', 0)}
- Average Dense Score: {stats.get('avg_dense_score', 0)}
- Average BM25 Score: {stats.get('avg_bm25_score', 0)}
- High Quality Matches: {stats.get('high_rrf_matches', 0)}

**Top 3 RRF Matches**:
"""
            
            for match in top_matches:
                section = match.get('section', 'No header')
                if not section:
                    section = 'No header'
                
                breakdown = match.get('score_breakdown', {})
                
                report += f"""
{match.get('rank', 'N/A')}. **Page {match.get('page', 'N/A')}** - RRF Score: {match.get('rrf_score', 0)}
   - Section: {section}
   - Dense Similarity: {breakdown.get('dense_similarity', 0)} (Rank: {breakdown.get('dense_rank', 'N/A')})
   - BM25 Score: {breakdown.get('bm25_score', 0)} (Rank: {breakdown.get('bm25_rank', 'N/A')})
   - Content: {match.get('content_preview', 'N/A')}
"""
            
            report += "\n---\n\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description="RRF Attribute-Contract Matching (Dense + BM25)")
    parser.add_argument("--attribute-collection", type=str, default="attributes_simple", help="Attribute collection name")
    parser.add_argument("--contract-collection", type=str, default="redacted", help="Contract collection name")
    parser.add_argument("--db-path", type=str, default="./chroma_db_qwen", help="ChromaDB path")
    parser.add_argument("--top-k", type=int, default=10, help="Top K final matches per attribute")
    parser.add_argument("--retrieval-k", type=int, default=100, help="Retrieval pool size for each method")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF parameter k")
    parser.add_argument("--output", type=str, default="rrf_matches.json", help="Output JSON file")
    parser.add_argument("--report", type=str, default="rrf_report.md", help="Output report file")
    
    args = parser.parse_args()
    
    # Initialize RRF matcher
    matcher = RRFAttributeMatcher(db_path=args.db_path, rrf_k=args.rrf_k)
    
    # Perform RRF matching
    results = matcher.rrf_match_attributes_to_contracts(
        attribute_collection=args.attribute_collection,
        contract_collection=args.contract_collection,
        top_k=args.top_k,
        retrieval_k=args.retrieval_k
    )
    
    if results:
        # Save JSON results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 RRF results saved to: {args.output}")
        
        # Generate report
        report = matcher.generate_rrf_report(results)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📊 RRF report saved to: {args.report}")
        
        # Print summary
        summary = results.get('summary', {})
        print(f"\n🎉 RRF Matching Complete!")
        print(f"📈 Method: {summary.get('method', 'RRF')}")
        print(f"🔍 Processed {summary.get('total_attributes', 0)} attributes")
        print(f"📚 Searched {summary.get('total_contracts', 0)} contract chunks")
        print(f"⚡ RRF Parameter k: {summary.get('rrf_k', 60)}")
        print(f"📁 Results: {args.output}")
        print(f"📋 Report: {args.report}")

if __name__ == "__main__":
    main()
