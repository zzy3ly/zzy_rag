import math
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "backend"))
sys.path.insert(0, str(ROOT))
from HybridRetrieval import BM25Index, Hit, HybridRetriever, reciprocal_rank_fusion, tokenize
from evaluate import metrics


class RetrievalTests(unittest.TestCase):
    def test_bm25_numeric_reference_and_scope(self):
        docs = [{"id":"a","text":"pump pump"},{"id":"b","text":"valve"}]
        result = BM25Index(docs).search("pump")
        expected = math.log(1+(2-1+.5)/(1+.5))*2*2.5/(2+1.5*(.25+.75*2/1.5))
        self.assertAlmostEqual(result[0].score, expected)
        self.assertEqual(BM25Index(docs).search("pump", allowed_ids={"b"}), [])

    def test_identifier_preservation(self):
        self.assertIn("34/8-a-9", tokenize("Well 34/8-A-9"))

    def test_rrf_cross_branch_deduplication(self):
        a,b,c=[Hit(i,i,{},0) for i in "abc"]
        result=reciprocal_rank_fusion([[a,b],[b,c]],constant=60)
        self.assertEqual(result[0].id,"b")
        self.assertEqual(len(result),3)
        self.assertAlmostEqual(result[0].score,1/62+1/61)

    def test_recall_distinct_from_hit(self):
        result=metrics(["a","x","b"],{"a","b","c"},5)
        self.assertAlmostEqual(result["recall"],2/3)
        self.assertEqual(result["hit"],1)
        self.assertEqual(result["all_evidence"],0)
        with self.assertRaises(ValueError):
            metrics(["a","a"],{"a"},5)

    def test_empty_authorized_scope_never_queries_global(self):
        class Collection:
            def get(self, **kwargs):
                return {"ids":[]}
            def query(self, **kwargs):
                raise AssertionError("must not query outside empty scope")
        retriever=HybridRetriever(Collection(),[],None)
        self.assertEqual(retriever.candidates("pump",{"document_id":"missing"}),[])

    def test_metadata_filter_applies_to_both_branches(self):
        class Collection:
            def get(self, **kwargs):
                return {"ids":["allowed"]}
            def query(self, **kwargs):
                assert kwargs["where"] == {"document_id":"d1"}
                return {"ids":[["allowed"]],"documents":[["pressure"]],
                        "metadatas":[[{"document_id":"d1"}]],"distances":[[0.2]]}
        docs=[{"id":"allowed","text":"pressure"},{"id":"secret","text":"pump pump"}]
        retriever=HybridRetriever(Collection(),docs,None)
        self.assertEqual([h.id for h in retriever.candidates("pump",{"document_id":"d1"})],["allowed"])

    def test_frozen_data_integrity_and_no_missing_gold(self):
        data = ROOT / "data"
        frozen = json.loads((data / "frozen_hashes.json").read_text())
        old = ROOT / "legacy_embeddings.py"
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), frozen["legacy_embeddings.py"])
        corpus = [json.loads(l) for l in (data / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        queries = [json.loads(l) for l in (data / "queries_reviewed.jsonl").read_text(encoding="utf-8").splitlines()]
        ids = {c["id"] for c in corpus}
        self.assertEqual(len(ids), len(corpus))
        for q in queries:
            self.assertTrue(set(q["relevant_ids"]).issubset(ids))
            self.assertTrue(set(q["relevant_ids"]).issubset(q["source_context_ids"]))
        self.assertEqual(sum(q["split"] == "test" for q in queries), 30)
        self.assertEqual(sum(q["split"] == "dev" for q in queries), 8)


if __name__ == "__main__":
    unittest.main()
