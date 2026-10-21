"""Paired experiment against the UNMODIFIED legacy retrieval method.

Only public passages are sent to Cohere. Credentials are read locally, never saved.
Run: python -X utf8 experiments/petroleum_retrieval/run_experiment.py
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import chromadb
from chromadb.config import Settings
import cohere

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "backend"))
sys.path.insert(0, str(ROOT))
from legacy_embeddings import LegacyEmbeddingHandler
from HybridRetrieval import HybridRetriever


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RecordedReranker:
    def __init__(self, client, cache):
        self.client, self.cache = client, cache
        self.last_call = 0.0
        self.elapsed = self.waited = 0.0
        self.last_documents = []
        self.cached = False
        self.cache.mkdir(parents=True, exist_ok=True)

    def rerank(self, **kwargs):
        self.last_documents = kwargs["documents"]
        signature = hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{signature}.json"
        self.waited = 0.0
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.elapsed, self.cached = stored["elapsed_seconds"], True
        else:
            self.cached = False
            wait = max(0, 6.3 - (time.perf_counter() - self.last_call))
            time.sleep(wait)
            self.waited = wait
            for attempt in range(4):
                self.last_call = time.perf_counter()
                start = time.perf_counter()
                try:
                    response = self.client.rerank(**kwargs)
                    self.elapsed = time.perf_counter() - start
                    stored = {"elapsed_seconds": self.elapsed,
                              "model": kwargs["model"],
                              "results": [{"index": r.index, "relevance_score": r.relevance_score}
                                          for r in response.results]}
                    path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
                    break
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    if attempt == 3 or status not in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"Rerank failed: {type(exc).__name__}, status={status}") from None
                    delay = 15 * (attempt + 1)
                    self.waited += time.perf_counter() - start + delay
                    time.sleep(delay)
        return SimpleNamespace(results=[SimpleNamespace(**r) for r in stored["results"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-corpus", action="store_true")
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    args = parser.parse_args()
    data = ROOT / "data"
    frozen = json.loads((data / "frozen_hashes.json").read_text())
    for filename, path in [
        ("corpus.jsonl", data / "corpus.jsonl"),
        ("queries_reviewed.jsonl", data / "queries_reviewed.jsonl"),
        ("annotations.json", ROOT / "annotations.json"),
        ("legacy_embeddings.py", ROOT / "legacy_embeddings.py")
    ]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen[filename], f"Frozen input changed: {filename}"
    corpus_name = "corpus_small.jsonl" if args.small_corpus else "corpus.jsonl"
    corpus = read_jsonl(data / corpus_name)
    queries = read_jsonl(data / f"queries_{args.split}.jsonl")
    name = ("small" if args.small_corpus else "full") + "_" + args.split
    output = ROOT / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache"
    cache.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(cache / "chroma"), settings=Settings(anonymized_telemetry=False))
    corpus_hash = hashlib.sha256((data / corpus_name).read_bytes()).hexdigest()
    collection = client.get_or_create_collection(
        name="petroleum_" + corpus_hash[:16], metadata={"hnsw:space": "l2", "hnsw:search_ef": 100})
    existing = set(collection.get(include=[])["ids"])
    pending = [doc for doc in corpus if doc["id"] not in existing]
    for start in range(0, len(pending), 32):
        batch = pending[start:start+32]
        collection.add(ids=[d["id"] for d in batch], documents=[d["text"] for d in batch],
                       metadatas=[{"filename": d["document_names"][0], "chunk_id": d["id"],
                                   "document_id": d["document_ids"][0]} for d in batch])
        print(f"INDEX {min(start+32,len(pending))}/{len(pending)}", flush=True)
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        key = (PROJECT / "backend/cohere.txt").read_text().strip()
    reranker = RecordedReranker(cohere.Client(key, timeout=90), cache / "rerank")
    # Bypass only constructor side effects (absolute credential path/upload parser).
    # The actual legacy retrieve_documents body is invoked verbatim.
    legacy = LegacyEmbeddingHandler.__new__(LegacyEmbeddingHandler)
    legacy.collection, legacy.co, legacy.similarity_score = collection, reranker, 0.01
    hybrid = HybridRetriever(collection, [
        {"id": d["id"], "text": d["text"], "metadata": {
            "filename": d["document_names"][0], "chunk_id": d["id"],
            "document_id": d["document_ids"][0]}} for d in corpus], reranker)
    text_to_id = {d["text"]: d["id"] for d in corpus}
    manifest = {
        "run": name, "corpus_count": len(corpus), "query_count": len(queries),
        "corpus_sha256": corpus_hash, "frozen_inputs": frozen,
        "embedding": "Chroma default ONNX all-MiniLM-L6-v2; 384 dimensions; default truncation",
        "distance": "L2", "reranker": "rerank-english-v3.0", "threshold": 0.01,
        "dense_candidates": 100, "hybrid_dense": 50, "hybrid_bm25": 50,
        "rrf_constant": 60, "bm25_k1": 1.5, "bm25_b": 0.75, "final_k": 5,
        "legacy_code": "unmodified EmbeddingHandler.retrieve_documents; constructor bypass only",
        "segmentation": "Shared released source passages, whitespace normalized; no re-chunking",
        "tuning": "None. All parameters fixed before first retrieval run.",
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["chromadb", "cohere", "onnxruntime", "numpy"]},
        "timing_note": "API pacing excluded; cached runs retain original reranker duration but are not live latency samples.",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    previous = read_jsonl(output / "predictions.jsonl") if (output / "predictions.jsonl").exists() else []
    done = {(r["query_id"], r["method"]) for r in previous}
    with (output / "legacy_stdout.log").open("a", encoding="utf-8") as log:
        for qi, q in enumerate(queries):
            sparse = hybrid.bm25.search(q["question"], 100)
            sparse_record = {"query_id": q["id"], "method": "bm25_only",
                             "candidate_ids": [h.id for h in sparse], "final_ids": [h.id for h in sparse[:5]],
                             "note": "No reranking: diagnostic first-stage baseline, not the paired final comparison."}
            if (q["id"], "bm25_only") not in done:
                with (output / "predictions.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(sparse_record) + "\n")
            methods = [("legacy_dense_rerank", legacy), ("hybrid_rrf_rerank", hybrid)]
            if qi % 2:
                methods.reverse()
            for method, handler in methods:
                if (q["id"], method) in done:
                    continue
                started = time.perf_counter()
                with contextlib.redirect_stdout(log):
                    status, error, docs = handler.retrieve_documents(q["question"])
                elapsed = time.perf_counter() - started
                if status != "success":
                    raise RuntimeError(f"{method} failed for {q['id']}: {error}")
                record = {"query_id": q["id"], "method": method,
                          "candidate_ids": [text_to_id[t] for t in reranker.last_documents],
                          "final_ids": [d["metadata"]["chunk_id"] for d in docs],
                          "elapsed_seconds_excluding_pacing": elapsed - reranker.waited,
                          "rerank_seconds": reranker.elapsed, "rerank_cached": reranker.cached}
                with (output / "predictions.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"QUERY {qi+1}/{len(queries)} {method} candidates={len(record['candidate_ids'])}", flush=True)
    print(f"DONE {output}", flush=True)


if __name__ == "__main__":
    main()
