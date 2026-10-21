"""Download a pinned public dataset and freeze a result-independent review sample."""
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
REVISION = "627ecd293877d943aac18914c0ce86053d6c1b97"
BASE = f"https://huggingface.co/datasets/porestar/diskos_qa/resolve/{REVISION}"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    raw = ROOT / "raw"
    data = ROOT / "data"
    raw.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    for remote, local in [("FORCE_QA_public_final.csv", "diskos_qa.csv"), ("README.md", "DATASET_CARD.md")]:
        path = raw / local
        if not path.exists():
            response = requests.get(f"{BASE}/{remote}", timeout=90)
            response.raise_for_status()
            path.write_bytes(response.content)
    rows = list(csv.DictReader((raw / "diskos_qa.csv").open(encoding="utf-8-sig")))
    corpus = {}
    aliases = {}
    questions = []
    for row in rows:
        texts, ids, docids, names = [ast.literal_eval(row[k]) for k in
                                   ["Content", "Chunk_IDs", "Document_id", "Document_name"]]
        assert len(texts) == len(ids) == len(docids) == len(names)
        context_ids = []
        for text, cid, docid, name in zip(texts, ids, docids, names):
            # Exact normalized-text deduplication; no questions/answers in the index.
            text = " ".join(text.split())
            canonical = hashlib.sha256(text.encode()).hexdigest()[:24]
            if str(cid) in aliases:
                assert aliases[str(cid)] == canonical, "Source chunk ID has conflicting text"
            aliases[str(cid)] = canonical
            item = corpus.setdefault(canonical, {"id": canonical, "text": text,
                                                "source_chunk_ids": [], "document_ids": [], "document_names": []})
            for field, value in [("source_chunk_ids", str(cid)), ("document_ids", docid), ("document_names", name)]:
                if value not in item[field]:
                    item[field].append(value)
            context_ids.append(canonical)
        questions.append({"id": row["Question_ID"], "question": row["Question"],
                          "answer": row["Answer"], "topic": row["Topic"],
                          "entailed": row["Entailed"].lower() == "true",
                          "source_context_ids": list(dict.fromkeys(context_ids))})
    groups = collections.defaultdict(list)
    for q in questions:
        if q["entailed"]:
            groups[q["topic"]].append(q)
    sample = []
    for topic in sorted(groups):
        ordered = sorted(groups[topic], key=lambda q: hashlib.sha256(
            ("petroleum-v1-20261020:" + q["id"]).encode()).hexdigest())
        for i, q in enumerate(ordered[:5]):
            sample.append({**q, "split": "dev" if i == 0 else "test",
                           "review_index": len(sample)})
    write_jsonl(data / "corpus.jsonl", sorted(corpus.values(), key=lambda x: x["id"]))
    write_jsonl(data / "questions_all.jsonl", questions)
    write_jsonl(data / "review_sample.jsonl", sample)
    manifest = {"source_url": f"{BASE}/FORCE_QA_public_final.csv", "revision": REVISION,
                "source_sha256": hashlib.sha256((raw / "diskos_qa.csv").read_bytes()).hexdigest(),
                "source_rows": len(rows), "source_unique_chunk_ids": len(aliases),
                "deduplicated_passages": len(corpus), "sample_questions": len(sample),
                "sampling": "SHA256(petroleum-v1-20261020:Question_ID); first 5 entailed per topic; first dev, remaining test",
                "license": "NLOD 2.0; upstream FORCE source CC BY 4.0",
                "label_warning": "Source context lists are NOT relevance judgments; reviewed qrels are separate."}
    (data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
