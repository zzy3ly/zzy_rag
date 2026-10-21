"""Metrics and paired uncertainty; no hyperparameter selection."""
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def metrics(ranking, relevant, k):
    ranking = ranking[:k]
    if not relevant:
        raise ValueError("No-answer questions must not enter relevance recall")
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate IDs in ranking")
    gains = [int(cid in relevant) for cid in ranking]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {"recall": sum(gains) / len(relevant), "hit": float(any(gains)),
            "mrr": next((1 / (i+1) for i, g in enumerate(gains) if g), 0),
            "ndcg": dcg / ideal, "all_evidence": float(set(relevant).issubset(ranking))}


def paired_interval(deltas, queries):
    # Group queries sharing judged evidence, including the overlapping Well-Log pair.
    parents = list(range(len(queries)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i in range(len(queries)):
        for j in range(i):
            if set(queries[i]["relevant_ids"]) & set(queries[j]["relevant_ids"]):
                parents[find(i)] = find(j)
    groups = {}
    for i in range(len(queries)):
        groups.setdefault(find(i), []).append(i)
    groups = list(groups.values())
    rng = np.random.default_rng(20261020)
    means = []
    for _ in range(10000):
        ix = [i for g in rng.integers(0, len(groups), len(groups)) for i in groups[g]]
        means.append(float(np.mean(np.asarray(deltas)[ix])))
    return {"mean_delta": float(np.mean(deltas)),
            "cluster_bootstrap_95_ci": np.quantile(means, [0.025, 0.975]).tolist(),
            "clusters": len(groups)}


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "full_test"
    output = ROOT / "results" / run
    manifest = json.loads((output / "run_manifest.json").read_text())
    split = run.split("_")[-1]
    queries = [json.loads(l) for l in (ROOT / f"data/queries_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    predictions = [json.loads(l) for l in (output / "predictions.jsonl").read_text().splitlines()]
    by_key = {(p["query_id"], p["method"]): p for p in predictions}
    assert len(by_key) == len(predictions), "Duplicate prediction records"
    methods = ["legacy_dense_rerank", "hybrid_rrf_rerank", "bm25_only"]
    if any(p["method"] == "bm25_rerank" for p in predictions):
        methods.append("bm25_rerank")
    per_query, summaries = [], {}
    for method in methods:
        rows = []
        for q in queries:
            p = by_key[q["id"], method]  # Missing/failed runs must fail, not silently disappear.
            gold = set(q["relevant_ids"])
            final = metrics(p["final_ids"], gold, 5)
            candidate = metrics(p["candidate_ids"], gold, 100)
            initial5 = metrics(p["candidate_ids"], gold, 5)
            row = {"query_id": q["id"], "review_index": q["review_index"],
                   "question": q["question"], "topic": q["topic"], "method": method,
                   **{k+"_at_5": v for k,v in final.items()},
                   "candidate_recall_at_100": candidate["recall"],
                   "candidate_hit_at_100": candidate["hit"],
                   "first_stage_recall_at_5": initial5["recall"],
                   "candidate_count": len(p["candidate_ids"]),
                   "relevant_count": len(gold),
                   "latency_seconds": p.get("elapsed_seconds_excluding_pacing"),
                   "cached": p.get("rerank_cached"),
                   "final_ids": p["final_ids"], "relevant_ids": q["relevant_ids"]}
            rows.append(row)
        keys = ["recall_at_5","hit_at_5","mrr_at_5","ndcg_at_5","all_evidence_at_5",
                "candidate_recall_at_100","candidate_hit_at_100","first_stage_recall_at_5","candidate_count"]
        summary = {key: float(np.mean([r[key] for r in rows])) for key in keys}
        live_latencies = [r["latency_seconds"] for r in rows if r["latency_seconds"] is not None and not r["cached"]]
        summary["live_latency_sample_count"] = len(live_latencies)
        summary["p50_seconds"] = float(np.median(live_latencies)) if live_latencies else None
        summary["p95_seconds"] = float(np.quantile(live_latencies, .95)) if live_latencies else None
        summaries[method] = summary
        per_query.extend(rows)
    keyed = {(r["query_id"], r["method"]): r for r in per_query}
    deltas = [keyed[q["id"],"hybrid_rrf_rerank"]["recall_at_5"] -
              keyed[q["id"],"legacy_dense_rerank"]["recall_at_5"] for q in queries]
    comparison = paired_interval(deltas, queries)
    comparison.update({"improved_queries": sum(d > 1e-9 for d in deltas),
                       "regressed_queries": sum(d < -1e-9 for d in deltas),
                       "unchanged_queries": sum(abs(d) <= 1e-9 for d in deltas)})
    by_topic = {}
    for topic in sorted({q["topic"] for q in queries}):
        by_topic[topic] = {method: float(np.mean([r["recall_at_5"] for r in per_query
                                                if r["topic"] == topic and r["method"] == method]))
                           for method in methods}
    result = {"run": run, "manifest": manifest, "metrics": summaries,
              "comparison": comparison, "by_topic": by_topic}
    if "bm25_rerank" in methods:
        control_deltas = [keyed[q["id"],"hybrid_rrf_rerank"]["recall_at_5"] -
                          keyed[q["id"],"bm25_rerank"]["recall_at_5"] for q in queries]
        result["hybrid_minus_bm25_rerank"] = paired_interval(control_deltas, queries)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "per_query.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in per_query),encoding="utf-8")
    with (output / "per_query.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    lines = [f"# {run} 实验结果", "", f"语料片段：{manifest['corpus_count']}；测试问题：{len(queries)}。",
             "", "| 方法 | 候选 Recall@100 | 最终 Recall@5 | Hit@5 | MRR@5 | NDCG@5 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method,s in summaries.items():
        lines.append(f"| {method} | {s['candidate_recall_at_100']:.2%} | {s['recall_at_5']:.2%} | "
                     f"{s['hit_at_5']:.2%} | {s['mrr_at_5']:.4f} | {s['ndcg_at_5']:.4f} |")
    lo,hi=comparison["cluster_bootstrap_95_ci"]
    lines += ["", f"混合减原始：Recall@5 {comparison['mean_delta']*100:+.2f} 个百分点；",
              f"按共享证据聚类的配对 bootstrap 95% 区间：[{lo*100:.2f}, {hi*100:.2f}] 个百分点。",
              f"改善 / 退步 / 持平：{comparison['improved_queries']} / {comparison['regressed_queries']} / {comparison['unchanged_queries']}。",
              "", "BM25-only 未经过重排，只作第一阶段诊断，不与两条完整链路混称同配置。",
              "这是英文公开油气报告上的小样本、AI 核对、非穷尽证据标注试验；不是中文企业线上指标。",
              "延迟是单次串行 API 测量，不是压测；排除主动限速等待，缓存结果不进入实时延迟统计。",
              "", "## 逐题变化", "", "| 题号 | 主题 | 原始 Recall@5 | 混合 Recall@5 | 变化 |",
              "|---|---|---:|---:|---:|"]
    for q,d in zip(queries,deltas):
        a=keyed[q["id"],"legacy_dense_rerank"]["recall_at_5"]
        b=keyed[q["id"],"hybrid_rrf_rerank"]["recall_at_5"]
        lines.append(f"| {q['review_index']} | {q['topic']} | {a:.2%} | {b:.2%} | {d*100:+.2f} pp |")
    (output / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"metrics":summaries,"comparison":comparison},indent=2))


if __name__ == "__main__":
    main()
