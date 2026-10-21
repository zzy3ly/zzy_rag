"""Preselected, paired reranking subset; source-only public evidence sent to Cohere."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cohere
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "petroleum_retrieval"))
from run_experiment import RecordedReranker
sys.path.insert(0, str(ROOT))
from evaluate import read_jsonl, metrics, aggregate, paired


def select():
    questions = read_jsonl(ROOT / "data/queries.jsonl")
    families = sorted({q["family"] for q in questions if q["split"] == "test" and q["source_type"] == "public_ocr"})
    ids = []
    for family in families:
        # One paraphrase and one two-part query per held-out source report.
        candidates = [q for q in questions if q["family"] == family and q["category"] == "paraphrase"]
        candidates.sort(key=lambda q: hashlib.sha256(q["id"].encode()).hexdigest())
        ids.append(candidates[0]["id"])
        ids.extend(q["id"] for q in questions if q["family"] == family and q["category"] == "multi_evidence")
    path = ROOT / "data/rerank_subset.json"
    payload = {"rule": "One hash-selected paraphrase + multi-evidence question per test public report",
               "query_ids": ids, "methods": ["token_dense", "bm25", "hybrid"],
               "model": "rerank-english-v3.0", "top_n": 5, "threshold": 0.01}
    if path.exists():
        assert json.loads(path.read_text()) == payload
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    subset = select()
    if "--select-only" in sys.argv:
        print(f"FROZEN rerank subset: {len(subset['query_ids'])} public queries", flush=True)
        return
    from run_local import check_frozen
    check_frozen()
    qs = {q["id"]: q for q in read_jsonl(ROOT / "data/queries.jsonl")}
    chunks = {c["id"]: c for c in read_jsonl(ROOT / "data/chunks_token.jsonl")}
    ranks = {r["query_id"]: r for r in read_jsonl(ROOT / "results/local_rankings.jsonl")}
    key = os.environ.get("COHERE_API_KEY") or (ROOT.parents[1] / "backend/cohere.txt").read_text().strip()
    reranker = RecordedReranker(cohere.Client(key), ROOT / "cache/rerank")
    rows = []
    path = ROOT / "results/rerank_rankings.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i, qid in enumerate(subset["query_ids"]):
            q = qs[qid]
            for method in subset["methods"]:
                ids = ranks[qid]["rankings"][method]
                kwargs = {"model": subset["model"], "query": q["query"],
                          "documents": [chunks[c]["text"] for c in ids], "top_n": 5}
                # One request returned a transient 403 during the recorded run while
                # the same credential and exact payload succeeded immediately after.
                # Retry once; persistent authorization failures still fail loudly.
                try:
                    response = reranker.rerank(**kwargs)
                except RuntimeError as exc:
                    if "status=403" not in str(exc):
                        raise
                    time.sleep(5)
                    response = reranker.rerank(**kwargs)
                final = [ids[r.index] for r in response.results if r.relevance_score >= subset["threshold"]]
                row = {"query_id": qid, "family": q["family"], "method": method, "ids": final,
                       "candidate_count": len(ids), "metrics": {k:v for k,v in metrics(q, final, chunks).items() if k.endswith("@5")},
                       "elapsed_seconds": reranker.elapsed, "cached": reranker.cached}
                rows.append(row)
                f.write(json.dumps(row)+"\n")
                f.flush()
            print(f"RERANK {i+1}/{len(subset['query_ids'])}", flush=True)
    result = {"subset": subset, "aggregates": {m: aggregate([r for r in rows if r["method"] == m]) for m in subset["methods"]},
              "paired": [paired(rows, a, "hybrid", "evidence_recall@5") for a in ["token_dense", "bm25"]]}
    (ROOT / "results/rerank_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
