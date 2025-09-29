# Healthcare Contract Classification System

This directory contains a complete 7-step pipeline for classifying healthcare contract clauses as **Standard** or **Non-Standard** compared to template clauses.

## Pipeline Overview

### Step 1: Exact Structural Match ✅ COMPLETED
- **Script**: `exact_structure_classifier.py`
- **Purpose**: Mark clauses with exact `preprocessed_final_content` matches as Standard
- **Status**: Already run - `isStandard` flags added to all datasets

### Steps 2-7: Multi-Step Analysis
- **Script**: `multi_step_classifier.py`
- **Purpose**: Advanced classification using semantic similarity, NLI, negation analysis, and rule flags

## Quick Start

### 1. Install Dependencies
```bash
pip install -r classifystandard/requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run Complete Pipeline
```bash
# Run the full 7-step classification pipeline
python classifystandard/run_full_pipeline.py
```

### 3. Generate Analysis Report
```bash
# Create detailed classification summary
python classifystandard/analysis_summary.py
```

## Individual Scripts

### Exact Structure Classifier (Step 1)
```bash
python classifystandard/exact_structure_classifier.py \
  --input-dir outputs/precise_similarity/processed_datasets \
  --output-dir classifystandard/standard
```

### Multi-Step Classifier (Steps 2-7)
```bash
python classifystandard/multi_step_classifier.py \
  --input-dir outputs/precise_similarity/processed_datasets \
  --db-path chroma_db_qwen \
  --output-dir classifystandard/standard_final
```

## Classification Steps Detail

| Step | Method | Threshold | Action |
|------|--------|-----------|---------|
| 1 | Exact Match | 100% | Mark as Standard if preprocessed text exactly matches template |
| 2 | Semantic Similarity (Qwen) | ≥0.80 | Mark as Standard if high semantic similarity |
| 3 | Paraphrase Similarity | ≥0.75 | Proceed to NLI if above threshold, else Non-Standard |
| 4 | Natural Language Inference | Two-way entailment | Standard if meanings align |
| 5 | Negation Scope Analysis | Scope consistency | Override to Non-Standard if scope differs |
| 6 | Rule Flags | Pattern detection | Force Non-Standard if forbidden patterns found |
| 7 | Final Aggregation | Combined logic | Final classification decision |

## Embeddings Used

The system uses pre-generated embeddings from ChromaDB:

**Template Collections:**
- `similarityTemplate_qwen` - TN/WA standard templates (Qwen 1024-d)
- `similarityTemplate_para` - TN/WA standard templates (Paraphrase 384-d)

**Contract Collections:**
- `SimilarityRedacted_qwen` - TN/WA redacted contracts (Qwen 1024-d)
- `SimilarityRedacted_para` - TN/WA redacted contracts (Paraphrase 384-d)

## Output Structure

### Input
- `outputs/precise_similarity/processed_datasets/*.json` - Preprocessed contract datasets

### Outputs
1. **Updated source files** - Original datasets with classification metadata added
2. **classifystandard/standard/** - Step 1 results (exact matches only)
3. **classifystandard/standard_final/** - Final results (all Standard clauses)
4. **classifystandard/classification_summary.json** - Detailed analysis
5. **classifystandard/classification_report.txt** - Human-readable report

## Classification Metadata Added

Each clause gets these additional fields:
- `isStandard`: 1 (Standard) or 0 (Non-Standard)
- `classification_step`: Which step determined the result
- `classification_reasoning`: Explanation of the decision
- `semantic_similarity`: Qwen similarity score (if calculated)
- `paraphrase_similarity`: Paraphrase similarity score (if calculated)
- `nli_reasoning`: NLI analysis result (if performed)
- `negation_reasoning`: Negation scope analysis (if performed)
- `rule_flags`: List of triggered rule patterns (if any)

## Rule Flags Detected

The system detects these forbidden patterns:
- **Exceptions**: except, unless, provided that, subject to, notwithstanding
- **Methodology Shifts**: Medicare, UCR, RBRVS, per diem, capitation, bundled
- **Re-anchoring**: lesser of charges, payer allowed amount, most favored nation
- **Forbidden Conditions**: indemnify, hold harmless, sole discretion

## System Requirements

- Python 3.8+
- ChromaDB with pre-generated embeddings
- Transformers library for NLI (microsoft/deberta-large-mnli)
- spaCy for negation analysis (en_core_web_sm)
- CUDA-compatible GPU (optional, for faster NLI processing)

## Performance Notes

- **Step 1**: Instant (exact string matching)
- **Steps 2-3**: Fast (vector similarity using existing embeddings)
- **Step 4**: Moderate (transformer-based NLI inference)
- **Steps 5-6**: Fast (rule-based pattern matching)

Total processing time: ~2-5 minutes for all contracts depending on hardware.
