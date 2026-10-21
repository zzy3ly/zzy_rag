import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "backend"))
from evaluate import coverage
from run_local import allowed_document, needs_clarification, read_jsonl
from TokenChunking import token_windows


class EvidenceTests(unittest.TestCase):
    def test_fragment_union_and_gap(self):
        groups = [[{"document_id": "x", "start": 10, "end": 30}]]
        hits = [{"document_id": "x", "start": 0, "end": 20}, {"document_id": "x", "start": 20, "end": 40}]
        self.assertEqual(coverage(groups, hits), 1)
        hits[1]["start"] = 21
        self.assertEqual(coverage(groups, hits), 0)
        hits[1]["start"] = 20
        hits[1]["document_id"] = "y"
        self.assertEqual(coverage(groups, hits), 0)

    def test_all_evidence_and_alternative(self):
        groups = [[{"document_id": "x", "start": 0, "end": 5}, {"document_id": "y", "start": 0, "end": 5}],
                  [{"document_id": "z", "start": 0, "end": 5}]]
        self.assertEqual(coverage(groups, [{"document_id": "y", "start": 0, "end": 10}]), .5)
        with self.assertRaises(ValueError):
            coverage([], [])

    def test_token_limit_offsets_and_tail(self):
        from tokenizers import Tokenizer
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        path = Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / "onnx/tokenizer.json"
        tok = Tokenizer.from_file(str(path))
        tok.no_truncation()
        tok.no_padding()
        text = ("Pressure 34/8-A-9; heterogeneous reservoir. " * 100) + "TAIL_MARKER"
        prefix = "Document PL1070\n"
        chunks = list(token_windows(text, tok, prefix))
        self.assertEqual(chunks[-1]["end"], len(text))
        for c in chunks:
            self.assertLessEqual(len(tok.encode(prefix+c["text"]).ids), 256)
            self.assertEqual(text[c["start"]:c["end"]], c["text"])
        self.assertTrue(all(a["end"] >= b["start"] for a, b in zip(chunks, chunks[1:])))

    def test_scope_independent_conditions(self):
        d = {"roles": ["finance"], "valid_from": "2025-01-01", "valid_to": "9999-12-31", "site": "A"}
        ctx = {"roles": ["operator"], "as_of": "2026-01-01", "site": "A"}
        self.assertFalse(allowed_document(d, ctx))
        ctx["roles"] = ["finance"]
        self.assertTrue(allowed_document(d, ctx))
        ctx["as_of"] = "2024-01-01"
        self.assertFalse(allowed_document(d, ctx))
        ctx["as_of"], ctx["site"] = "2026-01-01", "B"
        self.assertFalse(allowed_document(d, ctx))

    def test_dataset_groups_and_sources(self):
        qs = read_jsonl(ROOT / "data/queries.jsonl")
        docs = {d["id"]:d for d in read_jsonl(ROOT / "data/documents.jsonl")}
        self.assertEqual(len(qs), 220)
        train = {q["family"] for q in qs if q["split"] == "dev"}
        test = {q["family"] for q in qs if q["split"] == "test"}
        self.assertFalse(train & test)
        for q in qs:
            for group in q["evidence_groups"]:
                self.assertTrue(group)
                for span in group:
                    d = docs[span["document_id"]]
                    self.assertTrue(0 <= span["start"] < span["end"] <= len(d["text"]))
                    self.assertEqual(d["family"], q["family"])
                    self.assertTrue(allowed_document(d, q["context"]))
            if q["category"] == "clarification":
                self.assertTrue(needs_clarification(q["context"], list(docs.values())))

    def test_completed_result_shapes(self):
        local = read_jsonl(ROOT / "results/local_rankings.jsonl")
        rerank = read_jsonl(ROOT / "results/rerank_rankings.jsonl")
        self.assertEqual(len(local), 220)
        self.assertEqual(len({r["query_id"] for r in local}), 220)
        self.assertEqual(len(rerank), 96)
        self.assertEqual(len({r["query_id"] for r in rerank}), 32)
        self.assertEqual({r["method"] for r in rerank}, {"token_dense", "bm25", "hybrid"})


if __name__ == "__main__":
    unittest.main()
