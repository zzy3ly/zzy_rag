import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Embeddings import EmbeddingHandler
from HybridRetrieval import HybridRetriever, tokenize


class FakeCollection:
    def __init__(self, documents):
        self.documents = {item["id"]: item for item in documents}

    def count(self):
        return len(self.documents)

    def get(self, where=None, include=None):
        items = list(self.documents.values())
        if where:
            items = [item for item in items if all(
                item.get("metadata", {}).get(key) == value
                for key, value in where.items())]
        return {
            "ids": [item["id"] for item in items],
            "documents": [item["text"] for item in items],
            "metadatas": [item.get("metadata", {}) for item in items],
        }

    def query(self, query_texts, n_results, where=None):
        items = list(self.documents.values())
        if where:
            items = [item for item in items if all(
                item.get("metadata", {}).get(key) == value
                for key, value in where.items())]
        items = items[:n_results]
        return {
            "ids": [[item["id"] for item in items]],
            "documents": [[item["text"] for item in items]],
            "metadatas": [[item.get("metadata", {}) for item in items]],
            "distances": [[float(i) for i in range(len(items))]],
        }

    def add(self, documents, metadatas, ids):
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            self.documents[chunk_id] = {
                "id": chunk_id, "text": text, "metadata": metadata}

    def delete(self, where):
        doomed = [chunk_id for chunk_id, item in self.documents.items()
                  if all(item["metadata"].get(key) == value
                         for key, value in where.items())]
        for chunk_id in doomed:
            del self.documents[chunk_id]


class FakeReranker:
    def rerank(self, documents, top_n, **kwargs):
        order = sorted(range(len(documents)),
                       key=lambda i: ("pl127b" not in documents[i].lower(), i))[:top_n]
        return SimpleNamespace(results=[
            SimpleNamespace(index=i, relevance_score=0.9) for i in order])


class HybridIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {"id": "dense", "text": "general reservoir discussion",
             "metadata": {"document_id": "d1", "filename": "a.txt"}},
            {"id": "sparse", "text": "PL 127 B pump seal failure analysis",
             "metadata": {"document_id": "d1", "filename": "b.txt"}},
            {"id": "other", "text": "PL127B pump seal unrelated scope",
             "metadata": {"document_id": "d2", "filename": "c.txt"}},
        ]

    def test_identifier_and_chinese_tokens(self):
        self.assertIn("pl127b", tokenize("PL 127 B"))
        self.assertIn("pl127b", tokenize("PL127B"))
        self.assertIn("检索", tokenize("混合检索系统"))

    def test_online_hybrid_retrieval_and_scope(self):
        retriever = HybridRetriever(
            FakeCollection(self.documents), self.documents, FakeReranker(),
            dense_k=2, sparse_k=2, candidate_k=3)
        candidates = retriever.candidates(
            "PL127B pump seal", {"document_id": "d1"})
        self.assertIn("sparse", [hit.id for hit in candidates])
        self.assertNotIn("other", [hit.id for hit in candidates])
        status, _, results = retriever.retrieve_documents(
            "PL127B pump seal", metadata_filters={"document_id": "d1"})
        self.assertEqual(status, "success")
        self.assertEqual(results[0]["metadata"]["document_id"], "d1")
        self.assertEqual(
            retriever.candidates("anything", {"document_id": {"$in": []}}), [])

    def test_embedding_handler_refreshes_after_mutations(self):
        handler = EmbeddingHandler.__new__(EmbeddingHandler)
        handler.collection = FakeCollection(self.documents[:1])
        handler.hybrid_retriever = SimpleNamespace(replace_documents=lambda docs: None)
        self.assertEqual(handler.add_processed_documents(
            ["PL127B new evidence"], ["new"],
            [{"document_id": "new-doc", "filename": "new.txt"}]),
            ["success", ""])
        handler.refresh_sparse_index()
        self.assertEqual(len(handler._collection_documents()), 2)
        self.assertEqual(handler.remove_documents(["new-doc"]), ["success", ""])
        self.assertEqual(len(handler._collection_documents()), 1)


if __name__ == "__main__":
    unittest.main()
