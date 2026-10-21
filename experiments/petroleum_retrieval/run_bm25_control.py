"""Additional diagnostic: BM25 Top100 with the SAME reranker.

Added after the primary comparison because BM25-only was strong. No tuning.
Run sequentially after the other experiments to respect account-wide API pacing.
"""
import json
import os
import time

import cohere

from run_experiment import ROOT, PROJECT, read_jsonl, RecordedReranker
from HybridRetrieval import BM25Index


def main():
    corpus = read_jsonl(ROOT / "data/corpus.jsonl")
    queries = read_jsonl(ROOT / "data/queries_test.jsonl")
    output = ROOT / "results/full_test"
    predictions = read_jsonl(output / "predictions.jsonl")
    done = {r["query_id"] for r in predictions if r["method"] == "bm25_rerank"}
    key = os.environ.get("COHERE_API_KEY") or (PROJECT / "backend/cohere.txt").read_text().strip()
    reranker = RecordedReranker(cohere.Client(key, timeout=90), ROOT / "cache/rerank")
    index = BM25Index(corpus)
    for i,q in enumerate(queries):
        if q["id"] in done:
            continue
        start = time.perf_counter()
        hits = index.search(q["question"],100)
        response = reranker.rerank(model="rerank-english-v3.0", query=q["question"],
                                  documents=[h.text for h in hits],top_n=5)
        elapsed = time.perf_counter()-start
        row = {"query_id":q["id"],"method":"bm25_rerank","candidate_ids":[h.id for h in hits],
               "final_ids":[hits[r.index].id for r in response.results if r.relevance_score>=.01],
               "elapsed_seconds_excluding_pacing":elapsed-reranker.waited,
               "rerank_seconds":reranker.elapsed,"rerank_cached":reranker.cached,
               "note":"Post-primary diagnostic, fixed BM25 Top100 and same rerank/threshold; no test-set parameter tuning."}
        with (output/"predictions.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row)+"\n")
        print(f"BM25 CONTROL {i+1}/{len(queries)}",flush=True)


if __name__ == "__main__":
    main()
