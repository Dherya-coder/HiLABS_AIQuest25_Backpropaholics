from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from pathlib import Path
import json
import numpy as np

router = APIRouter(tags=["Charts"])

# Default base path inside the container where metrics.json is mounted
DEFAULT_BASE_PATH = "/app"

# ---------- Helpers ----------

def _load_metrics(base_path: str) -> Dict[str, Any]:
    metrics_path = Path(base_path) / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"metrics.json not found at {metrics_path}")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics.json: {e}")


def _create_histogram_data(values: List[float], bins: int = 20) -> List[Dict[str, Any]]:
    if len(values) == 0:
        return []
    hist, bin_edges = np.histogram(values, bins=bins)
    total = int(hist.sum()) if int(hist.sum()) else 1
    out = []
    for i in range(len(hist)):
        out.append({
            "bin_start": round(float(bin_edges[i]), 1),
            "bin_end": round(float(bin_edges[i + 1]), 1),
            "bin_center": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 1),
            "count": int(hist[i]),
            "percentage": round(float(hist[i]) / total * 100, 1)
        })
    return out


def _get_attribute_count_by_state(metrics: Dict[str, Any], attribute: str, state: str) -> int:
    count = 0
    for clause in metrics.get("clauses", []):
        if clause.get("assigned_attribute") == attribute:
            contract = next((c for c in metrics.get("contracts", []) if c.get("contract_id") == clause.get("contract_id")), None)
            if contract and contract.get("state") == state:
                count += 1
    return count


# ---------- 1. Pie: Standard vs Non-Standard ----------
@router.get("/charts/corpus-standard-vs-nonstandard")
def corpus_standard_vs_nonstandard(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    std = m["corpus"]["standard_count"]
    nonstd = m["corpus"]["nonstandard_count"]
    total = std + nonstd or 1
    return {
        "chart_type": "pie",
        "title": "Standard vs Non-Standard Clauses Distribution",
        "data": [
            {"label": "Standard", "value": std, "percentage": round(std / total * 100, 1), "color": "#2E86AB"},
            {"label": "Non-Standard", "value": nonstd, "percentage": round(nonstd / total * 100, 1), "color": "#A23B72"},
        ],
        "total": total,
    }


# ---------- 2. Histogram: Chunk Token Lengths ----------
@router.get("/charts/corpus-chunk-size-hist")
def corpus_chunk_size_hist(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    token_lengths = [clause.get("token_length", 0) for clause in m.get("clauses", [])]
    hist_data = _create_histogram_data(token_lengths, bins=20) if token_lengths else []
    stats = {
        "mean": round(float(np.mean(token_lengths)), 1) if token_lengths else 0.0,
        "median": round(float(np.median(token_lengths)), 1) if token_lengths else 0.0,
        "min": min(token_lengths) if token_lengths else 0,
        "max": max(token_lengths) if token_lengths else 0,
        "std": round(float(np.std(token_lengths)), 1) if token_lengths else 0.0,
    }
    return {
        "chart_type": "histogram",
        "title": "Distribution of Chunk Token Lengths",
        "data": hist_data,
        "stats": stats,
        "x_axis": "Token Count",
        "y_axis": "Frequency",
    }


# ---------- 3. Heatmap: Attribute vs State (avg similarity + counts) ----------
@router.get("/charts/heatmap-attr-vs-state")
def heatmap_attr_vs_state(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    states = list({c.get("state") for c in m.get("contracts", [])})
    attributes = list(m.get("attributes", {}).keys())
    data_rows = []
    for attr in attributes:
        row_values = []
        per_state_avg = m["attributes"][attr].get("per_state_avg_similarity", {})
        for st in states:
            avg_sim = float(per_state_avg.get(st, 0))
            row_values.append({
                "state": st,
                "value": round(avg_sim, 3),
                "count": _get_attribute_count_by_state(m, attr, st)
            })
        data_rows.append({"attribute": attr, "values": row_values})
    return {
        "chart_type": "heatmap",
        "title": "Attribute Coverage by State",
        "data": data_rows,
        "states": states,
        "attributes": attributes,
        "x_axis": "State",
        "y_axis": "Attribute",
    }


# ---------- 4. Grouped Bar: Standard vs Non-Standard by Contract ----------
@router.get("/charts/contract-standard-by-contract")
def contract_standard_by_contract(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    items = []
    for c in m.get("contracts", []):
        sc = c.get("standard_count", 0)
        nsc = c.get("nonstandard_count", 0)
        total = sc + nsc
        pct = round(sc / total * 100, 1) if total else 0.0
        items.append({
            "contract_id": c.get("contract_id"),
            "contract_name": str(c.get("contract_id", "")).split("_")[-1],
            "state": c.get("state"),
            "standard": sc,
            "non_standard": nsc,
            "total": total,
            "standard_percentage": pct,
        })
    return {
        "chart_type": "grouped_bar",
        "title": "Standard vs Non-Standard Clauses by Contract",
        "data": items,
        "x_axis": "Contract",
        "y_axis": "Count",
    }


# ---------- 5. Grouped Bar: Attribute Distribution by Classification ----------
@router.get("/charts/attribute-vs-standard-grouped")
def attribute_vs_standard_grouped(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    items = []
    for attr, data in m.get("attributes", {}).items():
        items.append({
            "attribute": attr,
            "standard": data.get("standard_count", 0),
            "non_standard": data.get("nonstandard_count", 0),
            "total": data.get("occurrences", 0),
            "avg_similarity": round(float(data.get("avg_similarity", 0.0)), 3),
        })
    return {
        "chart_type": "grouped_bar",
        "title": "Attribute Distribution by Classification",
        "data": items,
        "x_axis": "Attribute",
        "y_axis": "Count",
    }


# ---------- 6. Boxplot: Similarity Score Distribution by Attribute ----------
@router.get("/charts/attribute-similarity-boxplot")
def attribute_similarity_boxplot(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    # Group similarity by attribute from clauses
    attr_sims: Dict[str, List[float]] = {}
    for clause in m.get("clauses", []):
        attr = clause.get("assigned_attribute")
        if attr is None:
            continue
        attr_sims.setdefault(attr, []).append(float(clause.get("similarity", 0.0)))
    series = []
    for attr, sims in attr_sims.items():
        if not sims:
            continue
        sims_sorted = sorted(sims)
        n = len(sims_sorted)
        q1 = sims_sorted[n // 4]
        med = sims_sorted[n // 2]
        q3 = sims_sorted[3 * n // 4]
        iqr = q3 - q1
        outliers = [round(s, 3) for s in sims_sorted if s < q1 - 1.5 * iqr or s > q3 + 1.5 * iqr]
        series.append({
            "attribute": attr,
            "min": round(float(min(sims_sorted)), 3),
            "q1": round(float(q1), 3),
            "median": round(float(med), 3),
            "q3": round(float(q3), 3),
            "max": round(float(max(sims_sorted)), 3),
            "mean": round(float(np.mean(sims_sorted)), 3),
            "count": n,
            "outliers": outliers,
        })
    return {
        "chart_type": "boxplot",
        "title": "Similarity Score Distribution by Attribute",
        "data": series,
        "x_axis": "Attribute",
        "y_axis": "Similarity Score",
    }


# ---------- 7. Scatter: Clause Token Length vs Similarity ----------
@router.get("/charts/clause-length-vs-similarity-scatter")
def clause_length_vs_similarity_scatter(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    standard = []
    non_standard = []
    for clause in m.get("clauses", []):
        point = {
            "x": clause.get("token_length", 0),
            "y": round(float(clause.get("similarity", 0.0)), 3),
            "clause_id": clause.get("chunk_id"),
            "contract_id": clause.get("contract_id"),
            "attribute": clause.get("assigned_attribute"),
        }
        if clause.get("classification") == "standard":
            standard.append(point)
        else:
            non_standard.append(point)
    return {
        "chart_type": "scatter",
        "title": "Clause Token Length vs Similarity Score",
        "data": {"standard": standard, "non_standard": non_standard},
        "x_axis": "Token Length",
        "y_axis": "Similarity Score",
    }


# ---------- 8. Scatter: Embedding 2D Projection (mock) ----------
@router.get("/charts/embedding-2d-projection")
def embedding_2d_projection(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    # Mock 2D projection like generator (random normal with fixed seed)
    np.random.seed(42)
    standard = []
    non_standard = []
    for i, clause in enumerate(m.get("clauses", [])[:500]):  # sample limit
        x = float(np.random.normal(0, 1))
        y = float(np.random.normal(0, 1))
        point = {
            "x": round(x, 3),
            "y": round(y, 3),
            "clause_id": clause.get("chunk_id"),
            "contract_id": clause.get("contract_id"),
            "attribute": clause.get("assigned_attribute"),
            "similarity": round(float(clause.get("similarity", 0.0)), 3),
        }
        if clause.get("classification") == "standard":
            standard.append(point)
        else:
            non_standard.append(point)
    return {
        "chart_type": "scatter",
        "title": "2D UMAP Projection of Clause Embeddings",
        "data": {"standard": standard, "non_standard": non_standard},
        "x_axis": "UMAP Component 1",
        "y_axis": "UMAP Component 2",
    }


# ---------- 9. Bar: Processing Time Timeline ----------
@router.get("/charts/processing-time-timeline")
def processing_time_timeline(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    items = []
    timings: Dict[str, float] = m.get("processing", {}).get("timings", {})
    for contract_id, sec in timings.items():
        cinfo = next((c for c in m.get("contracts", []) if c.get("contract_id") == contract_id), None)
        items.append({
            "contract_id": contract_id,
            "contract_name": str(contract_id).split("_")[-1],
            "state": cinfo.get("state") if cinfo else "Unknown",
            "processing_time": round(float(sec), 2),
            "num_chunks": cinfo.get("num_chunks", 0) if cinfo else 0,
        })
    # Sort descending by processing_time
    items.sort(key=lambda x: x["processing_time"], reverse=True)
    return {
        "chart_type": "bar",
        "title": "Processing Time by Contract",
        "data": items,
        "x_axis": "Contract",
        "y_axis": "Processing Time (seconds)",
    }


# ---------- 10. Wordcloud Data ----------
@router.get("/charts/corpus-wordcloud")
def corpus_wordcloud(base_path: str = Query(DEFAULT_BASE_PATH)):
    m = _load_metrics(base_path)
    all_words: List[str] = []
    stop = set(['the','and','for','are','with','this','that','from','they','have','will','been','said','each','which','their'])
    for clause in m.get("clauses", []):
        words = str(clause.get("text_snippet", "")).lower().split()
        filtered = [w for w in words if len(w) > 3 and w not in stop]
        all_words.extend(filtered)
    # Frequencies
    freq: Dict[str, int] = {}
    for w in all_words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:50]
    data = [{"text": w, "value": c, "size": min(c * 2, 100)} for w, c in top]
    return {
        "chart_type": "wordcloud",
        "title": "Most Common Terms Across All Contracts",
        "data": data,
    }
