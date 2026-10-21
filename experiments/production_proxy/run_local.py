"""Exact, reproducible first-stage retrieval; no LLM answer generation.

L2 ranking of normalized MiniLM vectors is equivalent to descending dot product.
Exact search removes ANN tuning as a confound; this is not a service load test.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "backend"))
from HybridRetrieval import BM25Index, Hit, reciprocal_rank_fusion
from TokenChunking import token_windows


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def save_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")


def check_frozen():
    for path, expected in json.loads((ROOT / "data/frozen_hashes.json").read_text()).items():
        assert hashlib.sha256((PROJECT / path).read_bytes()).hexdigest() == expected, path


def make_chunks(documents, tokenizer, kind):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, add_start_index=True)
    for d in documents:
        prefix = d["title"] + "\n"
        if kind == "token":
            windows = list(token_windows(d["text"], tokenizer, prefix, 256, 32))
        elif kind == "legacy1000":
            windows = [{"start": c.metadata["start_index"],
                        "end": c.metadata["start_index"]+len(c.page_content), "text": c.page_content}
                       for c in splitter.create_documents([d["text"]])]
        else:
            windows = [{"start": 0, "end": len(d["text"]), "text": d["text"]}] if d["text"].strip() else []
        for i, w in enumerate(windows):
            assert d["text"][w["start"]:w["end"]] == w["text"]
            chunks.append({"id": f"{d['id']}-{kind}-{i:03d}", "document_id": d["id"],
                           "start": w["start"], "end": w["end"], "text": prefix+w["text"],
                           "metadata": {k: d[k] for k in ("family", "roles", "valid_from", "valid_to", "source_type")},
                           "token_count": len(tokenizer.encode(prefix+w["text"]).ids)})
    return chunks


def encode_cached(texts, encoder, name):
    signature = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    cache = ROOT / "cache" / f"{name}-{signature[:16]}"
    cache.mkdir(parents=True, exist_ok=True)
    batches = []
    for start in range(0, len(texts), 32):
        path = cache / f"{start:06d}.npy"
        if not path.exists():
            vectors = np.asarray(encoder(texts[start:start+32]), dtype=np.float32)
            np.save(path, vectors)
        batches.append(np.load(path))
        if start % 320 == 0 or start+32 >= len(texts):
            print(f"EMBED {name} {min(start+32,len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(batches)


def allowed_document(d, context):
    role_ok = "public" in d["roles"] or bool(set(d["roles"]) & set(context["roles"]))
    date = context["as_of"]
    date_ok = d["valid_from"] <= date <= d["valid_to"]
    site_ok = d.get("site") is None or context.get("site") is None or d["site"] == context["site"]
    return role_ok and date_ok and site_ok


def needs_clarification(context, documents):
    if context.get("asset") and not context.get("site"):
        sites = {d.get("site") for d in documents if d.get("asset") == context["asset"]
                 and allowed_document(d, context)}
        return len(sites) > 1
    return False


def dense_hits(chunks, vectors, query_vector, eligible, limit=100):
    scores = vectors @ query_vector
    indices = np.flatnonzero(eligible)
    ranked = sorted(indices, key=lambda i: (-float(scores[i]), chunks[i]["id"]))[:limit]
    return [Hit(chunks[i]["id"], chunks[i]["text"], chunks[i]["metadata"], float(scores[i])) for i in ranked]


def main():
    check_frozen()
    documents = read_jsonl(ROOT / "data/documents.jsonl")
    questions = read_jsonl(ROOT / "data/queries.jsonl")
    model_dir = Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    encoder = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    encoder.model = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=options,
                                        providers=["CPUExecutionProvider"])
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    chunks_by_kind, vectors_by_kind, diagnostics = {}, {}, {}
    for kind in ["token", "legacy1000", "page"]:
        chunks = make_chunks(documents, tokenizer, kind)
        chunks_by_kind[kind] = chunks
        save_jsonl(ROOT / "data" / f"chunks_{kind}.jsonl", chunks)
        counts = [c["token_count"] for c in chunks]
        diagnostics[kind] = {"chunks": len(chunks), "over_256": sum(n > 256 for n in counts),
                             "median_tokens": float(np.median(counts)), "max_tokens": max(counts)}
        if kind == "token":
            assert max(counts) <= 256
        print("CHUNKS", kind, diagnostics[kind], flush=True)
        vectors_by_kind[kind] = encode_cached([c["text"] for c in chunks], encoder, kind)
    qvectors = encode_cached([q["query"] for q in questions], encoder, "queries")
    bm25 = BM25Index(chunks_by_kind["token"])
    lookup = {d["id"]: d for d in documents}
    results = []
    for qi, q in enumerate(questions):
        context = q["context"]
        clarified = needs_clarification(context, documents)
        rankings, elapsed = {}, {}
        all_dense = None
        for kind in ["token", "legacy1000", "page"]:
            chunks = chunks_by_kind[kind]
            valid = np.array([allowed_document(lookup[c["document_id"]], context) for c in chunks])
            if clarified:
                valid[:] = False
            start = time.perf_counter()
            dense = dense_hits(chunks, vectors_by_kind[kind], qvectors[qi], valid)
            method = kind + "_dense"
            elapsed[method] = time.perf_counter() - start
            rankings[method] = dense
            if kind == "token":
                start = time.perf_counter()
                sparse = bm25.search(q["query"], 100, {c["id"] for c, keep in zip(chunks, valid) if keep})
                elapsed["bm25"] = time.perf_counter() - start
                rankings["bm25"] = sparse
                start = time.perf_counter()
                rankings["hybrid"] = reciprocal_rank_fusion([dense[:50], sparse[:50]], 100, 60)
                elapsed["hybrid"] = elapsed[method] + elapsed["bm25"] + time.perf_counter()-start
                # Explicit unsafe control; never used as the authorized branch.
                if q["source_type"] == "synthetic_policy":
                    all_dense = dense_hits(chunks, vectors_by_kind[kind], qvectors[qi], np.ones(len(chunks), dtype=bool), 50)
                    all_sparse = bm25.search(q["query"], 50)
                    rankings["ungated_hybrid"] = reciprocal_rank_fusion([all_dense, all_sparse], 100, 60)
        results.append({"query_id": q["id"], "action": "clarify_site" if clarified else "retrieval_only",
                        "rankings": {m: [h.id for h in hits] for m, hits in rankings.items()},
                        "retrieval_seconds_excluding_embedding_and_scope": elapsed})
        if qi % 40 == 0:
            print(f"RETRIEVE {qi+1}/{len(questions)}", flush=True)
    save_jsonl(output / "local_rankings.jsonl", results)
    diagnostics.update({"model": "all-MiniLM-L6-v2 (Chroma ONNX)", "dimension": int(qvectors.shape[1]),
                        "model_sha256": hashlib.sha256((model_dir / "model.onnx").read_bytes()).hexdigest(),
                        "distance": "exact normalized dot product; L2-equivalent ranking",
                        "query_embedding_max_tokens": max(len(tokenizer.encode(q["query"]).ids) for q in questions),
                        "title_prefix_for_all_methods": True, "overlap_tokens": 32,
                        "scope_and_clarification_rules": "same for all gated methods; context supplied explicitly",
                        "hybrid": {"dense_k": 50, "bm25_k": 50, "rrf_constant": 60},
                        "candidate_budget": 100})
    (output / "index_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print("DONE local retrieval", flush=True)


if __name__ == "__main__":
    main()
