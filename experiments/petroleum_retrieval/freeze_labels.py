import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = ROOT / "data"
sample = [json.loads(l) for l in (data / "review_sample.jsonl").read_text(encoding="utf-8").splitlines()]
annotations = json.loads((ROOT / "annotations.json").read_text(encoding="utf-8"))
accepted, excluded = [], []
for annotation, q in zip(annotations["items"], sample, strict=True):
    assert annotation["index"] == q["review_index"]
    if "exclude" in annotation:
        excluded.append({**q, "reason": annotation["exclude"]})
        continue
    gold = [q["source_context_ids"][i] for i in annotation["positive"]]
    assert gold
    accepted.append({**q, "relevant_ids": gold, "annotation_note": annotation["note"],
                     "label_origin": annotations["reviewer"]})
for filename, rows in [("queries_reviewed.jsonl", accepted), ("excluded.jsonl", excluded),
                       ("queries_test.jsonl", [q for q in accepted if q["split"] == "test"]),
                       ("queries_dev.jsonl", [q for q in accepted if q["split"] == "dev"])]:
    (data / filename).write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows), encoding="utf-8")
# Self-contained smoke-test corpus; primary experiment always searches the full corpus.
ids = {cid for q in accepted for cid in q["source_context_ids"]}
corpus = [json.loads(l) for l in (data / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
(data / "corpus_small.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n"
    for c in corpus if c["id"] in ids), encoding="utf-8")
(data / "small_dataset.json").write_text(json.dumps({
    "description": "Public DISKOS-QA-derived pilot; AI-reviewed evidence, not human gold; 8 dev and 30 test queries",
    "corpus": [c for c in corpus if c["id"] in ids], "queries": accepted
}, ensure_ascii=False, indent=2), encoding="utf-8")
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    data / "corpus.jsonl", data / "queries_reviewed.jsonl", ROOT / "annotations.json",
    ROOT.parents[1] / "backend" / "Embeddings.py"]}
manifest.update({"accepted": len(accepted), "test": sum(q["split"] == "test" for q in accepted),
                 "excluded": len(excluded), "small_corpus": len(ids)})
(data / "frozen_hashes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
