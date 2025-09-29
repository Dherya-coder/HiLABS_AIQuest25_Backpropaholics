#!/usr/bin/env python3
"""
Convert similarity JSON results to comprehensive Markdown.

- Reads all *_attribute_similarities.json files from outputs/similarity/
- Writes a corresponding *.md file with FULL content mirrored from JSON
- Preserves all fields, including score breakdowns and full_content
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/precise_similarity")


def md_escape(text: str) -> str:
    if text is None:
        return ""
    # Keep markdown formatting mostly intact; escape only common fence sequences
    return str(text).replace("```", "``` ")


def write_collection_md(json_path: Path) -> Path:
    with open(json_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    md_path = json_path.with_suffix('.md')

    collection_info = data.get("collection_info", {})
    summary = data.get("summary", {})
    matches = data.get("matches", {})

    with open(md_path, 'w', encoding='utf-8') as out:
        out.write(f"# Attribute Similarity Results — {md_escape(collection_info.get('collection_name', json_path.stem))}\n\n")
        out.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Collection info
        out.write("## Collection Info\n\n")
        out.write("```json\n" + json.dumps(collection_info, indent=2, ensure_ascii=False) + "\n```\n\n")

        # Summary
        out.write("## Summary\n\n")
        out.write("```json\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n```\n\n")

        # Matches per attribute (FULL CONTENT)
        out.write("## Matches\n\n")
        for attr_name, attr_data in matches.items():
            out.write(f"### Attribute: {md_escape(attr_name)}\n\n")

            # Write attribute block (include everything provided under this key)
            out.write("```json\n" + json.dumps(attr_data, indent=2, ensure_ascii=False) + "\n```\n\n")

            # Also pretty render the top matches section for readability
            top = attr_data.get("top_matches") or attr_data.get("matches") or []
            if top:
                out.write("#### Top Matches (detailed)\n\n")
                for m in top:
                    out.write(f"- **Rank**: {m.get('rank')}\n")
                    out.write(f"  - **RRF Score**: {m.get('rrf_score')}\n")
                    out.write(f"  - **Page**: {m.get('page')}\n")
                    out.write(f"  - **Section**: {md_escape(m.get('section'))}\n")
                    out.write(f"  - **Chunk ID**: {md_escape(m.get('chunk_id'))}\n")
                    out.write(f"  - **Content Preview**: {md_escape(m.get('content_preview'))}\n")
                    # Full content in a collapsible section
                    out.write("  - **Full Content:**\n\n")
                    out.write("    ```\n" + md_escape(m.get('full_content', '')) + "\n```\n\n")
                    # Score breakdown
                    out.write("  - **Score Breakdown:**\n\n")
                    out.write("```json\n" + json.dumps(m.get('score_breakdown', {}), indent=2, ensure_ascii=False) + "\n```\n\n")

        # Footer
        out.write("---\n")
        out.write(f"End of report for {md_escape(collection_info.get('collection_name', json_path.stem))}\n")

    logger.info(f"Wrote Markdown: {md_path}")
    return md_path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(OUTPUT_DIR.glob("*_attribute_similarities.json"))

    if not json_files:
        logger.error(f"No similarity JSON files found in {OUTPUT_DIR}")
        return 1

    for jf in json_files:
        try:
            write_collection_md(jf)
        except Exception as e:
            logger.error(f"Failed to convert {jf.name} -> MD: {e}")
            continue

    logger.info("Conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
