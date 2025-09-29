#!/usr/bin/env python3
"""
Analysis summary script to generate detailed reports on classification results.

Run:
  python classifystandard/analysis_summary.py \
    --input-dir outputs/precise_similarity/processed_datasets \
    --output-file classifystandard/classification_summary.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Load JSON dataset"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def analyze_classifications(input_dir: Path) -> Dict[str, Any]:
    """Analyze classification results across all datasets"""
    
    analysis = {
        "overall_stats": {
            "total_clauses": 0,
            "standard_clauses": 0,
            "non_standard_clauses": 0,
            "standard_percentage": 0.0
        },
        "by_step": defaultdict(int),
        "by_attribute": defaultdict(lambda: {"total": 0, "standard": 0}),
        "by_state": defaultdict(lambda: {"total": 0, "standard": 0}),
        "by_contract": {},
        "rule_flags_analysis": Counter(),
        "similarity_distributions": {
            "semantic": [],
            "paraphrase": []
        }
    }
    
    # Process all redacted datasets
    redacted_files = [f for f in input_dir.glob("*.json") if "redacted" in f.name.lower()]
    
    for file_path in sorted(redacted_files):
        rows = load_rows(file_path)
        if not rows:
            continue
        
        contract_stats = {
            "total": len(rows),
            "standard": 0,
            "by_step": defaultdict(int),
            "by_attribute": defaultdict(lambda: {"total": 0, "standard": 0})
        }
        
        for row in rows:
            # Overall stats
            analysis["overall_stats"]["total_clauses"] += 1
            
            is_standard = row.get("isStandard", 0)
            if is_standard == 1:
                analysis["overall_stats"]["standard_clauses"] += 1
                contract_stats["standard"] += 1
            else:
                analysis["overall_stats"]["non_standard_clauses"] += 1
            
            # By classification step
            step = row.get("classification_step", "unknown")
            analysis["by_step"][step] += 1
            contract_stats["by_step"][step] += 1
            
            # By attribute
            attr = row.get("attribute_number", "unknown")
            analysis["by_attribute"][str(attr)]["total"] += 1
            contract_stats["by_attribute"][str(attr)]["total"] += 1
            if is_standard == 1:
                analysis["by_attribute"][str(attr)]["standard"] += 1
                contract_stats["by_attribute"][str(attr)]["standard"] += 1
            
            # By state
            collection_key = row.get("collection_key", "")
            state = "TN" if "TN" in collection_key else "WA" if "WA" in collection_key else "unknown"
            analysis["by_state"][state]["total"] += 1
            if is_standard == 1:
                analysis["by_state"][state]["standard"] += 1
            
            # Rule flags analysis
            rule_flags = row.get("rule_flags", [])
            for flag in rule_flags:
                analysis["rule_flags_analysis"][flag] += 1
            
            # Similarity distributions
            semantic_sim = row.get("semantic_similarity")
            if semantic_sim is not None:
                analysis["similarity_distributions"]["semantic"].append(semantic_sim)
            
            paraphrase_sim = row.get("paraphrase_similarity")
            if paraphrase_sim is not None:
                analysis["similarity_distributions"]["paraphrase"].append(paraphrase_sim)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        contract_stats["by_step"] = dict(contract_stats["by_step"])
        contract_stats["by_attribute"] = {k: dict(v) for k, v in contract_stats["by_attribute"].items()}
        
        analysis["by_contract"][file_path.stem] = contract_stats
    
    # Calculate percentages
    total = analysis["overall_stats"]["total_clauses"]
    if total > 0:
        analysis["overall_stats"]["standard_percentage"] = (
            analysis["overall_stats"]["standard_clauses"] / total
        ) * 100
    
    # Add percentage calculations for attributes and states
    for attr_data in analysis["by_attribute"].values():
        if attr_data["total"] > 0:
            attr_data["standard_percentage"] = (attr_data["standard"] / attr_data["total"]) * 100
    
    for state_data in analysis["by_state"].values():
        if state_data["total"] > 0:
            state_data["standard_percentage"] = (state_data["standard"] / state_data["total"]) * 100
    
    # Convert defaultdicts to regular dicts for JSON serialization
    analysis["by_step"] = dict(analysis["by_step"])
    analysis["by_attribute"] = {k: dict(v) for k, v in analysis["by_attribute"].items()}
    analysis["by_state"] = {k: dict(v) for k, v in analysis["by_state"].items()}
    analysis["rule_flags_analysis"] = dict(analysis["rule_flags_analysis"])
    
    return analysis

def generate_readable_report(analysis: Dict[str, Any]) -> str:
    """Generate a human-readable report"""
    
    report_lines = [
        "=" * 80,
        "HEALTHCARE CONTRACT CLAUSE CLASSIFICATION ANALYSIS",
        "=" * 80,
        "",
        "OVERALL SUMMARY",
        "-" * 40,
        f"Total Clauses Analyzed: {analysis['overall_stats']['total_clauses']:,}",
        f"Standard Clauses: {analysis['overall_stats']['standard_clauses']:,} ({analysis['overall_stats']['standard_percentage']:.1f}%)",
        f"Non-Standard Clauses: {analysis['overall_stats']['non_standard_clauses']:,}",
        "",
        "CLASSIFICATION BY STEP",
        "-" * 40
    ]
    
    for step, count in sorted(analysis["by_step"].items()):
        percentage = (count / analysis["overall_stats"]["total_clauses"]) * 100
        report_lines.append(f"{step}: {count:,} ({percentage:.1f}%)")
    
    report_lines.extend([
        "",
        "CLASSIFICATION BY ATTRIBUTE",
        "-" * 40
    ])
    
    for attr, data in sorted(analysis["by_attribute"].items()):
        report_lines.append(
            f"Attribute {attr}: {data['standard']}/{data['total']} Standard ({data.get('standard_percentage', 0):.1f}%)"
        )
    
    report_lines.extend([
        "",
        "CLASSIFICATION BY STATE",
        "-" * 40
    ])
    
    for state, data in sorted(analysis["by_state"].items()):
        report_lines.append(
            f"{state}: {data['standard']}/{data['total']} Standard ({data.get('standard_percentage', 0):.1f}%)"
        )
    
    if analysis["rule_flags_analysis"]:
        report_lines.extend([
            "",
            "MOST COMMON RULE FLAGS",
            "-" * 40
        ])
        
        # Sort rule flags by count (descending) and take top 10
        sorted_flags = sorted(analysis["rule_flags_analysis"].items(), key=lambda x: x[1], reverse=True)[:10]
        for flag, count in sorted_flags:
            report_lines.append(f"{flag}: {count}")
    
    # Similarity statistics
    semantic_sims = analysis["similarity_distributions"]["semantic"]
    paraphrase_sims = analysis["similarity_distributions"]["paraphrase"]
    
    if semantic_sims:
        avg_semantic = sum(semantic_sims) / len(semantic_sims)
        report_lines.extend([
            "",
            "SIMILARITY STATISTICS",
            "-" * 40,
            f"Average Semantic Similarity: {avg_semantic:.3f} (n={len(semantic_sims)})"
        ])
    
    if paraphrase_sims:
        avg_paraphrase = sum(paraphrase_sims) / len(paraphrase_sims)
        report_lines.append(f"Average Paraphrase Similarity: {avg_paraphrase:.3f} (n={len(paraphrase_sims)})")
    
    report_lines.extend([
        "",
        "=" * 80
    ])
    
    return "\n".join(report_lines)

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description="Generate classification analysis summary")
    parser.add_argument("--input-dir", default=str(repo_root / "outputs/precise_similarity/processed_datasets"),
                       help="Input directory with classified datasets")
    parser.add_argument("--output-file", default=str(script_dir / "classification_summary.json"),
                       help="Output file for analysis summary")
    parser.add_argument("--report-file", default=str(script_dir / "classification_report.txt"),
                       help="Output file for readable report")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    report_file = Path(args.report_file)
    
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    
    logger.info("Analyzing classification results...")
    analysis = analyze_classifications(input_dir)
    
    # Save JSON analysis
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Analysis saved to: {output_file}")
    
    # Generate and save readable report
    report = generate_readable_report(analysis)
    with report_file.open("w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_file}")
    
    # Print summary
    print("\n" + report)

if __name__ == "__main__":
    main()
