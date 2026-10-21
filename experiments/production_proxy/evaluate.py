"""Evidence-span coverage, family bootstrap, and separately reported policy checks."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def span_covered(span, hits):
    intervals = sorted((max(h["start"], span["start"]), min(h["end"], span["end"]))
                       for h in hits if h["document_id"] == span["document_id"]
                       and h["start"] < span["end"] and h["end"] > span["start"])
    cursor = span["start"]
    for start, end in intervals:
        if start > cursor:
            return False
        cursor = max(cursor, end)
    return cursor >= span["end"]


def coverage(groups, hits):
    if not groups:
        raise ValueError("Unanswerable/unauthorized questions have no ordinary recall denominator")
    return sum(any(span_covered(span, hits) for span in alternatives) for alternatives in groups) / len(groups)


def metrics(q, ids, chunks):
    answer = {}
    for k in [5, 10, 20, 100]:
        hits = [chunks[c] for c in ids[:k]]
        recall = coverage(q["evidence_groups"], hits)
        answer[f"evidence_recall@{k}"] = recall
        answer[f"all_evidence@{k}"] = float(recall == 1)
        answer[f"hit@{k}"] = float(recall > 0)
    return answer


def aggregate(rows):
    keys = list(rows[0]["metrics"]) if rows else []
    return {"n": len(rows), **{k: float(np.mean([r["metrics"][k] for r in rows])) for k in keys}}


def paired(rows, a, b, key):
    lookup = {(r["query_id"], r["method"]): r for r in rows}
    deltas, families = [], defaultdict(list)
    for (qid, method), row in lookup.items():
        if method != a or (qid, b) not in lookup:
            continue
        delta = lookup[qid, b]["metrics"][key] - row["metrics"][key]
        deltas.append(delta)
        families[row["family"]].append(delta)
    group_values = list(families.values())
    if not deltas:
        return {}
    rng = np.random.default_rng(20261020)
    bootstrap = [np.mean([v for i in rng.integers(0, len(group_values), len(group_values)) for v in group_values[i]])
                 for _ in range(5000)]
    return {"baseline": a, "challenger": b, "metric": key, "n": len(deltas),
            "independent_source_families": len(families), "delta_pp": 100*float(np.mean(deltas)),
            "family_bootstrap_95ci_pp": (100*np.percentile(bootstrap, [2.5, 97.5])).tolist(),
            "improved": sum(v > 0 for v in deltas), "regressed": sum(v < 0 for v in deltas),
            "unchanged": sum(v == 0 for v in deltas)}


def main():
    from run_local import check_frozen, allowed_document
    check_frozen()
    qs = {q["id"]: q for q in read_jsonl(ROOT / "data/queries.jsonl")}
    docs = {d["id"]: d for d in read_jsonl(ROOT / "data/documents.jsonl")}
    chunks = {c["id"]: c for kind in ["token", "legacy1000", "page"]
              for c in read_jsonl(ROOT / f"data/chunks_{kind}.jsonl")}
    ranking_rows = read_jsonl(ROOT / "results/local_rankings.jsonl")
    detailed, safety = [], []
    for row in ranking_rows:
        q = qs[row["query_id"]]
        for method, ids in row["rankings"].items():
            if q["evidence_groups"]:
                detailed.append({"query_id": q["id"], "family": q["family"], "split": q["split"],
                                 "source_type": q["source_type"], "category": q["category"], "method": method,
                                 "metrics": metrics(q, ids, chunks)})
            if q["source_type"] == "synthetic_policy":
                top = [docs[chunks[c]["document_id"]] for c in ids[:5]]
                unauthorized = [d for d in top if "public" not in d["roles"] and
                                not set(d["roles"]) & set(q["context"]["roles"])]
                stale = [d for d in top if not d["valid_from"] <= q["context"]["as_of"] <= d["valid_to"]]
                safety.append({"query_id": q["id"], "method": method, "split": q["split"],
                               "category": q["category"], "any_unauthorized_top5": bool(unauthorized),
                               "any_wrong_date_top5": bool(stale),
                               "any_out_of_scope_top5": any(not allowed_document(d, q["context"]) for d in top),
                               "returned_candidates": len(ids),
                               "clarification_triggered": row["action"] == "clarify_site" and method != "ungated_hybrid"})
    summary = {}
    for split in ["dev", "test"]:
        for source in ["public_ocr", "synthetic_policy"]:
            selected = [r for r in detailed if r["split"] == split and r["source_type"] == source]
            summary[f"{split}/{source}"] = {m: aggregate([r for r in selected if r["method"] == m])
                                           for m in sorted({r["method"] for r in selected})}
            for category in sorted({r["category"] for r in selected}):
                summary[f"{split}/{source}/{category}"] = {
                    m: aggregate([r for r in selected if r["method"] == m and r["category"] == category])
                    for m in sorted({r["method"] for r in selected})}
    primary = [r for r in detailed if r["split"] == "test" and r["source_type"] == "public_ocr"]
    comparisons = [paired(primary, a, b, key) for a, b in [
        ("page_dense", "token_dense"), ("legacy1000_dense", "token_dense"),
        ("token_dense", "hybrid"), ("bm25", "hybrid")]
                   for key in ["evidence_recall@5", "evidence_recall@100"]]
    policy_summary = {}
    for method in sorted({r["method"] for r in safety}):
        rs = [r for r in safety if r["method"] == method and r["split"] == "test"]
        policy_summary[method] = {"n": len(rs),
            "queries_exposing_unauthorized_top5": sum(r["any_unauthorized_top5"] for r in rs),
            "queries_exposing_wrong_date_top5": sum(r["any_wrong_date_top5"] for r in rs),
            "queries_exposing_out_of_scope_top5": sum(r["any_out_of_scope_top5"] for r in rs),
            "clarifications_triggered": sum(r["clarification_triggered"] for r in rs),
            "unanswerable_queries_returning_candidates": sum(r["category"] == "unanswerable" and r["returned_candidates"] > 0 for r in rs)}
    result = {"aggregates": summary, "paired_comparisons": comparisons, "policy_checks": policy_summary,
              "metric_definition": "Known-positive evidence-group recall. A group counts only if all characters of at least one annotated source span are covered by retrieved chunks, allowing their union. Not exhaustive relevance or answer accuracy."}
    (ROOT / "results/metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results/per_query_metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in detailed), encoding="utf-8")
    print(json.dumps(summary["test/public_ocr"], indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
