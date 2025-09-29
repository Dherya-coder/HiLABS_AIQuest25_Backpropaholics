#!/usr/bin/env python3
"""
Multi-step Standard vs Non-Standard clause classifier implementing Steps 2-7.

This script processes clauses that were NOT marked as Standard in Step 1 (exact match)
and applies semantic similarity, paraphrase similarity, NLI, negation analysis, 
and rule flags to make final classifications.

Pipeline:
Step 1: Already done (exact match) - isStandard=1 clauses skip further processing
Step 2: Semantic Similarity (Qwen embeddings, τ=0.80)
Step 3: Paraphrase Similarity (paraphrase embeddings, τ=0.75) 
Step 4: Natural Language Inference (two-way entailment)
Step 5: Negation Scope Analysis (dependency parsing)
Step 6: Rule Flags (forbidden patterns/modifications)
Step 7: Final Aggregation

Run:
  python classifystandard/multi_step_classifier.py \
    --input-dir outputs/precise_similarity/processed_datasets \
    --db-path chroma_db_qwen \
    --output-dir classifystandard/standard_final
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np

# Core dependencies
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable debug logging for classification details
logger.setLevel(logging.DEBUG)

# Thresholds
SEMANTIC_THRESHOLD = 0.80
PARAPHRASE_THRESHOLD = 0.7
HIGH_PARAPHRASE_THRESHOLD = 0.90

# Collection mappings
TEMPLATE_COLLECTIONS = {
    "qwen": "similarityTemplate_qwen",
    "para": "similarityTemplate_para"
}
REDACTED_COLLECTIONS = {
    "qwen": "SimilarityRedacted_qwen", 
    "para": "SimilarityRedacted_para"
}

STATE_MAPPINGS = {"TNredacted": "TN", "WAredacted": "WA", "TNstandard": "TN", "WAstandard": "WA"}

class MultiStepClassifier:
    def __init__(self, db_path: Path):
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.template_qwen = self.client.get_collection(TEMPLATE_COLLECTIONS["qwen"])
        self.template_para = self.client.get_collection(TEMPLATE_COLLECTIONS["para"])
        self.redacted_qwen = self.client.get_collection(REDACTED_COLLECTIONS["qwen"])
        self.redacted_para = self.client.get_collection(REDACTED_COLLECTIONS["para"])
        
        # Initialize embedding models (lazy loading)
        self._qwen_embedder = None
        self._para_embedder = None
        
        # Initialize NLI model (lazy loading)
        self._nli_model = None
        self._nli_tokenizer = None
        
        # Initialize spaCy for negation analysis (lazy loading)
        self._nlp = None
        
    def _get_nli_model(self):
        """Lazy load NLI model"""
        if self._nli_model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                model_name = "microsoft/deberta-large-mnli"
                self._nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._nli_model.to(self._device)
                logger.info(f"Loaded NLI model: {model_name} on {self._device}")
            except Exception as e:
                logger.warning(f"Failed to load NLI model: {e}")
                self._nli_model = "failed"
        return self._nli_model if self._nli_model != "failed" else None
    
    def _get_nlp(self):
        """Lazy load spaCy model"""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self._nlp = "failed"
        return self._nlp if self._nlp != "failed" else None
    
    def _get_qwen_embedder(self):
        """Lazy load Qwen embedding function"""
        if self._qwen_embedder is None:
            try:
                import requests
                self._qwen_embedder = self._embed_with_qwen
                logger.info("Qwen embedder initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Qwen embedder: {e}")
                self._qwen_embedder = "failed"
        return self._qwen_embedder if self._qwen_embedder != "failed" else None
    
    def _get_para_embedder(self):
        """Lazy load paraphrase embedding function"""
        if self._para_embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                self._para_embedder = lambda texts: model.encode(texts, normalize_embeddings=True).tolist()
                logger.info("Paraphrase embedder initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize paraphrase embedder: {e}")
                self._para_embedder = "failed"
        return self._para_embedder if self._para_embedder != "failed" else None
    
    def _embed_with_qwen(self, texts):
        """Generate Qwen embeddings via Ollama API"""
        try:
            import requests
            import numpy as np
            
            embeddings = []
            for text in texts:
                response = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "qwen3-embedding:0.6b", "prompt": text},
                    timeout=30
                )
                if response.status_code == 200:
                    embedding = response.json()["embedding"]
                    # L2 normalize
                    embedding = np.array(embedding)
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding.tolist())
                else:
                    logger.error(f"Qwen embedding failed: {response.status_code}")
                    return None
            return embeddings
        except Exception as e:
            logger.error(f"Qwen embedding error: {e}")
            return None

    def step2_semantic_similarity(self, candidate_text: str, candidate_attr: int, candidate_state: str) -> Tuple[bool, float, str]:
        """
        Step 2: Semantic similarity using Qwen embeddings
        Returns: (is_standard, max_similarity, reasoning)
        """
        try:
            # Generate Qwen embedding for candidate text
            qwen_embedder = self._get_qwen_embedder()
            if qwen_embedder is None:
                return False, 0.0, "Qwen embedder not available"
            
            candidate_embeddings = qwen_embedder([candidate_text])
            if not candidate_embeddings:
                return False, 0.0, "Failed to generate candidate embedding"
            
            # Query template embeddings using the generated embedding
            template_results = self.template_qwen.query(
                query_embeddings=[candidate_embeddings[0]],
                n_results=10,
                where={
                    "$and": [
                        {"attribute_number": candidate_attr},
                        {"collection_key": f"{candidate_state}standard"}
                    ]
                }
            )
            
            if not template_results['distances'] or not template_results['distances'][0]:
                return False, 0.0, "No template embeddings found"
            
            # Convert distance to similarity (cosine distance = 1 - cosine similarity)
            max_similarity = 1.0 - min(template_results['distances'][0])
            
            is_standard = max_similarity >= SEMANTIC_THRESHOLD
            reasoning = f"Semantic similarity: {max_similarity:.3f} (threshold: {SEMANTIC_THRESHOLD})"
            
            logger.debug(f"Step 2 - State: {candidate_state}, Attr: {candidate_attr}, Similarity: {max_similarity:.3f}")
            
            return is_standard, max_similarity, reasoning
            
        except Exception as e:
            logger.error(f"Step 2 error: {e}")
            return False, 0.0, f"Error: {e}"

    def step3_paraphrase_similarity(self, candidate_text: str, candidate_attr: int, candidate_state: str) -> Tuple[bool, float, str]:
        """
        Step 3: Paraphrase similarity check
        Returns: (proceed_to_nli, max_similarity, reasoning)
        """
        try:
            # Generate paraphrase embedding for candidate text
            para_embedder = self._get_para_embedder()
            if para_embedder is None:
                return False, 0.0, "Paraphrase embedder not available"
            
            candidate_embeddings = para_embedder([candidate_text])
            if not candidate_embeddings:
                return False, 0.0, "Failed to generate candidate paraphrase embedding"
            
            # Query template paraphrase embeddings using the generated embedding
            template_results = self.template_para.query(
                query_embeddings=[candidate_embeddings[0]],
                n_results=10,
                where={
                    "$and": [
                        {"attribute_number": candidate_attr},
                        {"collection_key": f"{candidate_state}standard"}
                    ]
                }
            )
            
            if not template_results['distances'] or not template_results['distances'][0]:
                return False, 0.0, "No template paraphrase embeddings found"
            
            # Convert distance to similarity
            max_similarity = 1.0 - min(template_results['distances'][0])
            
            proceed_to_nli = max_similarity >= PARAPHRASE_THRESHOLD
            reasoning = f"Paraphrase similarity: {max_similarity:.3f} (threshold: {PARAPHRASE_THRESHOLD})"
            
            logger.debug(f"Step 3 - State: {candidate_state}, Attr: {candidate_attr}, Similarity: {max_similarity:.3f}")
            
            return proceed_to_nli, max_similarity, reasoning
            
        except Exception as e:
            logger.error(f"Step 3 error: {e}")
            return False, 0.0, f"Error: {e}"

    def step4_nli_check(self, candidate_text: str, template_text: str, paraphrase_sim: float) -> Tuple[bool, str]:
        """
        Step 4: Natural Language Inference check
        Returns: (is_standard, reasoning)
        """
        nli_model = self._get_nli_model()
        if nli_model is None:
            # Fallback: use high paraphrase similarity
            if paraphrase_sim >= HIGH_PARAPHRASE_THRESHOLD:
                return True, f"NLI unavailable, high paraphrase similarity: {paraphrase_sim:.3f}"
            return False, "NLI unavailable, moderate paraphrase similarity"
        
        try:
            import torch
            
            # Two-way entailment checks
            # Template ⇒ Candidate
            inputs1 = self._nli_tokenizer(template_text, candidate_text, return_tensors="pt", truncation=True, max_length=512)
            inputs1 = {k: v.to(self._device) for k, v in inputs1.items()}
            
            # Candidate ⇒ Template  
            inputs2 = self._nli_tokenizer(candidate_text, template_text, return_tensors="pt", truncation=True, max_length=512)
            inputs2 = {k: v.to(self._device) for k, v in inputs2.items()}
            
            with torch.no_grad():
                outputs1 = nli_model(**inputs1)
                outputs2 = nli_model(**inputs2)
            
            # Get predictions (0=entailment, 1=neutral, 2=contradiction)
            pred1 = torch.argmax(outputs1.logits, dim=-1).item()
            pred2 = torch.argmax(outputs2.logits, dim=-1).item()
            
            # Interpret results
            if pred1 == 0 and pred2 == 0:  # Both entailments
                return True, "Both-way entailment: meanings aligned"
            elif pred1 == 2 or pred2 == 2:  # Any contradiction
                return False, "Contradiction detected"
            elif (pred1 == 1 and pred2 == 1) and paraphrase_sim >= HIGH_PARAPHRASE_THRESHOLD:
                return True, f"Neutral NLI but high paraphrase similarity: {paraphrase_sim:.3f}"
            else:
                return False, f"One-way or neutral entailment (pred1={pred1}, pred2={pred2})"
                
        except Exception as e:
            logger.error(f"Step 4 NLI error: {e}")
            # Fallback to paraphrase similarity
            if paraphrase_sim >= HIGH_PARAPHRASE_THRESHOLD:
                return True, f"NLI failed, high paraphrase similarity: {paraphrase_sim:.3f}"
            return False, f"NLI failed: {e}"

    def step5_negation_analysis(self, candidate_text: str, template_text: str) -> Tuple[bool, str]:
        """
        Step 5: Negation scope analysis
        Returns: (maintains_classification, reasoning)
        """
        nlp = self._get_nlp()
        if nlp is None:
            return True, "Negation analysis unavailable (spaCy not loaded)"
        
        try:
            # Process both texts
            candidate_doc = nlp(candidate_text)
            template_doc = nlp(template_text)
            
            # Extract negations and their scopes
            def extract_negations(doc):
                negations = []
                for token in doc:
                    if token.dep_ == "neg" or token.lemma_.lower() in ["not", "no", "without", "except"]:
                        # Find the head that this negation modifies
                        head = token.head
                        negations.append((token.text.lower(), head.lemma_.lower()))
                return negations
            
            candidate_negs = extract_negations(candidate_doc)
            template_negs = extract_negations(template_doc)
            
            # Compare negation patterns
            if len(candidate_negs) != len(template_negs):
                return False, f"Different negation counts: candidate={len(candidate_negs)}, template={len(template_negs)}"
            
            # If both have no negations, maintain classification
            if not candidate_negs and not template_negs:
                return True, "No negations in either text"
            
            # Simple heuristic: if negated concepts are similar, maintain classification
            candidate_concepts = {neg[1] for neg in candidate_negs}
            template_concepts = {neg[1] for neg in template_negs}
            
            overlap = len(candidate_concepts & template_concepts)
            total = len(candidate_concepts | template_concepts)
            
            if total == 0:
                return True, "No negated concepts identified"
            
            similarity_ratio = overlap / total
            maintains = similarity_ratio >= 0.5
            
            return maintains, f"Negation scope similarity: {similarity_ratio:.2f}"
            
        except Exception as e:
            logger.error(f"Step 5 negation analysis error: {e}")
            return True, f"Negation analysis failed: {e}"

    def step6_rule_flags(self, candidate_text: str) -> Tuple[bool, List[str]]:
        """
        Step 6: Rule flags for forbidden modifications
        Returns: (has_forbidden_patterns, triggered_flags)
        """
        text_lower = candidate_text.lower()
        triggered_flags = []
        
        # Exception/Carve-out patterns
        exception_patterns = [
            r'\bexcept\b', r'\bunless\b', r'\bprovided that\b', r'\bsubject to\b', 
            r'\bnotwithstanding\b', r'\bhowever\b', r'\bbut\b'
        ]
        
        # Methodology shift patterns
        methodology_patterns = [
            r'\bmedicare\b', r'\bucr\b', r'\brbrvs\b', r'\bper diem\b', 
            r'\bcapitation\b', r'\bbundled\b', r'\bstop.loss\b'
        ]
        
        # Re-anchoring patterns
        reanchoring_patterns = [
            r'\blesser of charges\b', r'\bpayer allowed amount\b', 
            r'\bmost favored nation\b', r'\blowest rate\b'
        ]
        
        # Forbidden conditions
        forbidden_patterns = [
            r'\bindemnify\b', r'\bhold harmless\b', r'\bsole discretion\b',
            r'\bat will\b', r'\bunilateral\b'
        ]
        
        # Check each pattern category
        pattern_groups = [
            ("exceptions", exception_patterns),
            ("methodology_shift", methodology_patterns), 
            ("reanchoring", reanchoring_patterns),
            ("forbidden_conditions", forbidden_patterns)
        ]
        
        for group_name, patterns in pattern_groups:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    triggered_flags.append(f"{group_name}: {pattern}")
        
        has_forbidden = len(triggered_flags) > 0
        return has_forbidden, triggered_flags

    def get_best_template_match(self, candidate_text: str, candidate_attr: int, candidate_state: str) -> str:
        """Get the best template match for NLI comparison"""
        try:
            # Generate paraphrase embedding for candidate text
            para_embedder = self._get_para_embedder()
            if para_embedder is None:
                return ""
            
            candidate_embeddings = para_embedder([candidate_text])
            if not candidate_embeddings:
                return ""
            
            # Query template paraphrase embeddings to find best match
            template_results = self.template_para.query(
                query_embeddings=[candidate_embeddings[0]],
                n_results=1,
                where={
                    "$and": [
                        {"attribute_number": candidate_attr},
                        {"collection_key": f"{candidate_state}standard"}
                    ]
                }
            )
            
            if template_results['documents'] and template_results['documents'][0]:
                return template_results['documents'][0][0]
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting template match: {e}")
            return ""

    def classify_clause(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the full multi-step classification pipeline to a single clause
        """
        # Skip if already marked as Standard in Step 1
        if row.get("isStandard") == 1:
            row["classification_step"] = "step1_exact_match"
            row["classification_reasoning"] = "Exact structural match with template"
            return row
        
        candidate_text = row.get("preprocessed_final_content", "")
        if not candidate_text:
            row["isStandard"] = 0
            row["classification_step"] = "error"
            row["classification_reasoning"] = "No preprocessed content"
            return row
        
        candidate_attr = row.get("attribute_number")
        collection_key = row.get("collection_key", "")
        candidate_state = STATE_MAPPINGS.get(collection_key)
        
        if not candidate_state:
            row["isStandard"] = 0
            row["classification_step"] = "error"
            row["classification_reasoning"] = f"Unknown state for collection: {collection_key}"
            return row
        
        try:
            logger.debug(f"Classifying clause - State: {candidate_state}, Attr: {candidate_attr}, Collection: {collection_key}")
            
            # Step 2: Semantic Similarity
            semantic_standard, semantic_sim, semantic_reason = self.step2_semantic_similarity(
                candidate_text, candidate_attr, candidate_state
            )
            
            if semantic_standard:
                row["isStandard"] = 1
                row["classification_step"] = "step2_semantic"
                row["classification_reasoning"] = semantic_reason
                row["semantic_similarity"] = semantic_sim
                return row
            
            # Step 3: Paraphrase Similarity
            proceed_nli, para_sim, para_reason = self.step3_paraphrase_similarity(
                candidate_text, candidate_attr, candidate_state
            )
            
            if not proceed_nli:
                row["isStandard"] = 0
                row["classification_step"] = "step3_paraphrase_failed"
                row["classification_reasoning"] = para_reason
                row["paraphrase_similarity"] = para_sim
                row["semantic_similarity"] = semantic_sim
                return row
            
            # Get best template for NLI
            best_template = self.get_best_template_match(candidate_text, candidate_attr, candidate_state)
            
            # Step 4: NLI Check
            nli_standard, nli_reason = self.step4_nli_check(candidate_text, best_template, para_sim)
            
            # Step 5: Negation Analysis
            negation_maintains, negation_reason = self.step5_negation_analysis(candidate_text, best_template)
            
            # Step 6: Rule Flags
            has_forbidden, rule_flags = self.step6_rule_flags(candidate_text)
            
            # Step 7: Final Classification
            if has_forbidden:
                final_classification = 0
                final_reason = f"Rule flags triggered: {rule_flags}"
                final_step = "step6_rule_flags"
            elif not negation_maintains:
                final_classification = 0
                final_reason = f"Negation scope mismatch: {negation_reason}"
                final_step = "step5_negation"
            elif nli_standard:
                final_classification = 1
                final_reason = f"NLI passed: {nli_reason}"
                final_step = "step4_nli"
            else:
                final_classification = 0
                final_reason = f"NLI failed: {nli_reason}"
                final_step = "step4_nli"
            
            # Update row with all analysis results
            row["isStandard"] = final_classification
            row["classification_step"] = final_step
            row["classification_reasoning"] = final_reason
            row["semantic_similarity"] = semantic_sim
            row["paraphrase_similarity"] = para_sim
            row["nli_reasoning"] = nli_reason
            row["negation_reasoning"] = negation_reason
            row["rule_flags"] = rule_flags
            
            return row
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            row["isStandard"] = 0
            row["classification_step"] = "error"
            row["classification_reasoning"] = f"Classification failed: {e}"
            return row


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Load JSON dataset"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Save JSON dataset"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description="Multi-step Standard vs Non-Standard classifier")
    parser.add_argument("--input-dir", default=str(repo_root / "outputs/precise_similarity/processed_datasets"), 
                       help="Input directory with preprocessed datasets")
    parser.add_argument("--db-path", default=str(repo_root / "chroma_db_qwen"), 
                       help="ChromaDB path with embeddings")
    parser.add_argument("--output-dir", default=str(repo_root / "classifystandard/standard_final"), 
                       help="Output directory for final classifications")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    
    if not db_path.exists():
        raise SystemExit(f"ChromaDB not found: {db_path}")
    
    # Initialize classifier
    classifier = MultiStepClassifier(db_path)
    
    # Process all redacted datasets
    redacted_files = [f for f in input_dir.glob("*.json") if "redacted" in f.name.lower()]
    
    total_processed = 0
    total_standard = 0
    
    for file_path in sorted(redacted_files):
        logger.info(f"Processing: {file_path.name}")
        
        rows = load_rows(file_path)
        if not rows:
            continue
        
        # Classify each row
        classified_rows = []
        file_standard = 0
        
        for row in rows:
            classified_row = classifier.classify_clause(row)
            classified_rows.append(classified_row)
            
            if classified_row.get("isStandard") == 1:
                file_standard += 1
        
        # Save back to original location (with classification updates)
        save_rows(file_path, classified_rows)
        
        # Save standard-only rows to final output
        standard_rows = [r for r in classified_rows if r.get("isStandard") == 1]
        output_path = output_dir / file_path.name
        save_rows(output_path, standard_rows)
        
        total_processed += len(rows)
        total_standard += file_standard
        
        logger.info(f"{file_path.name}: {file_standard}/{len(rows)} classified as Standard")
    
    logger.info(f"\nFinal Summary: {total_standard}/{total_processed} clauses classified as Standard")
    logger.info(f"Updated datasets saved to: {input_dir}")
    logger.info(f"Standard-only outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
