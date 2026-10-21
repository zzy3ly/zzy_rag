"""Print query/answer and evidence windows BEFORE any retrieval results are run."""
import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
corpus = {r["id"]: r for r in map(json.loads, (root / "data/corpus.jsonl").read_text(encoding="utf-8").splitlines())}
sample = list(map(json.loads, (root / "data/review_sample.jsonl").read_text(encoding="utf-8").splitlines()))
start = int(sys.argv[1])
end = int(sys.argv[2])
for q in sample[start:end]:
    print(f'\nITEM {q["review_index"]} [{q["topic"]}] {q["question"]}\nANSWER: {q["answer"]}')
    words = set(re.findall(r"[a-z0-9]+", q["answer"].lower())) - set("the a an of in on and to is was were with from at by for it as be this that are".split())
    for i, cid in enumerate(q["source_context_ids"]):
        text = corpus[cid]["text"]
        sentences = re.split(r"(?<=[.;])\s+", text)
        ranked = sorted(enumerate(sentences), key=lambda pair:
                        -sum(min(len(w), 8) for w in words.intersection(re.findall(r"[a-z0-9]+", pair[1].lower()))))
        selected = sorted(ranked[:3])
        print(f'PASSAGE {i} [{cid}] {text[:140]} ... ' +
              " ... ".join(s[:800] for _, s in selected))
