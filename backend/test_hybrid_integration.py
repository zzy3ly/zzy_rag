import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Embeddings import EmbeddingHandler
from OpenSearchBackend import OpenSearchBackend


class FakeEncoder:
    dimension = 3

    def encode(self, texts):
        vectors = []
        for text in texts:
            value = text.lower()
            vector = [float("pl127b" in value or "pl 127 b" in value),
                      float("reservoir" in value), 1.0]
            norm = math.sqrt(sum(item * item for item in vector))
            vectors.append([item / norm for item in vector])
        return vectors


class FakeIndices:
    def __init__(self, owner):
        self.owner = owner

    def exists(self, index):
        return self.owner.created

    def create(self, index, body):
        self.owner.created = True
        self.owner.mapping = body
        return {"acknowledged": True}


class FakeTransport:
    def __init__(self, owner):
        self.owner = owner

    def perform_request(self, method, path, body=None):
        self.owner.pipeline = body
        return {"acknowledged": True}


def _matches(source, query):
    if "term" in query:
        field, value = next(iter(query["term"].items()))
        return source.get(field) == value
    if "terms" in query:
        field, values = next(iter(query["terms"].items()))
        return source.get(field) in values
    if "match_none" in query:
        return False
    if "bool" in query:
        if "filter" in query["bool"]:
            return all(_matches(source, item) for item in query["bool"]["filter"])
        if "should" in query["bool"]:
            return any(_matches(source, item) for item in query["bool"]["should"])
    return True


class FakeOpenSearch:
    def __init__(self):
        self.created = False
        self.mapping = None
        self.pipeline = None
        self.docs = {}
        self.last_search = None
        self.last_bulk_params = None
        self.indices = FakeIndices(self)
        self.transport = FakeTransport(self)

    def bulk(self, body, params=None):
        self.last_bulk_params = params
        for position in range(0, len(body), 2):
            action, payload = body[position], body[position + 1]
            operation, header = next(iter(action.items()))
            chunk_id = header["_id"]
            if operation == "index":
                self.docs[chunk_id] = dict(payload)
            elif operation == "update":
                self.docs[chunk_id].update(payload["doc"])
        return {"errors": False, "items": []}

    def search(self, index, body, params=None):
        self.last_search = {"index": index, "body": body, "params": params}
        query = body["query"]
        if "hybrid" in query:
            condition = query["hybrid"].get("filter")
            items = [(key, value) for key, value in self.docs.items()
                     if not condition or _matches(value, condition)]
            text = query["hybrid"]["queries"][0]["match"]["text"]["query"].lower()
            items.sort(key=lambda item: (text not in item[1]["text"].lower(), item[0]))
        else:
            items = [(key, value) for key, value in self.docs.items()
                     if _matches(value, query)]
        hits = [{"_id": key, "_source": dict(value), "_score": 1.0}
                for key, value in items[:body.get("size", 10)]]
        return {"hits": {"hits": hits}}

    def delete_by_query(self, index, body, params=None):
        doomed = [key for key, value in self.docs.items()
                  if _matches(value, body["query"])]
        for key in doomed:
            del self.docs[key]
        return {"deleted": len(doomed)}


class FakeReranker:
    def rerank(self, documents, top_n, **kwargs):
        order = sorted(range(len(documents)),
                       key=lambda i: ("pl127b" not in documents[i].lower(), i))[:top_n]
        return SimpleNamespace(results=[
            SimpleNamespace(index=i, relevance_score=0.9) for i in order])


class OpenSearchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeOpenSearch()
        self.encoder = FakeEncoder()
        self.handler = EmbeddingHandler(
            collection="test-rag", opensearch_client=self.client,
            encoder=self.encoder, reranker=FakeReranker(),
            dense_k=10, candidate_k=20)

    def test_schema_and_native_rrf_pipeline(self):
        self.assertTrue(self.client.mapping["settings"]["index.knn"])
        vector = self.client.mapping["mappings"]["properties"]["embedding"]
        self.assertEqual(vector["type"], "knn_vector")
        processor = self.client.pipeline["phase_results_processors"][0]
        self.assertEqual(
            processor["score-ranker-processor"]["combination"]["technique"], "rrf")

    def test_online_hybrid_retrieval_scope_and_parent_expansion(self):
        self.handler.add_processed_documents(
            ["general reservoir discussion", "PL127B pump seal failure", "PL127B other"],
            ["dense", "sparse", "other"],
            [{"document_id": "d1", "filename": "a.txt"},
             {"document_id": "d1", "filename": "b.txt",
              "parent_id": "p1", "parent_text": "complete failure procedure"},
             {"document_id": "d2", "filename": "c.txt"}])
        self.assertEqual(self.client.last_bulk_params, {"refresh": "wait_for"})
        status, _, results = self.handler.retrieve_documents(
            "PL127B", metadata_filters={"document_id": {"$in": ["d1"]}})
        self.assertEqual(status, "success")
        self.assertEqual(results[0]["text"], "complete failure procedure")
        self.assertEqual(results[0]["metadata"]["document_id"], "d1")
        hybrid = self.client.last_search["body"]["query"]["hybrid"]
        self.assertEqual(len(hybrid["queries"]), 2)
        self.assertEqual(self.client.last_search["params"]["search_pipeline"],
                         "rag-hybrid-rrf-v1")
        self.assertEqual(self.handler.retrieve_documents(
            "anything", metadata_filters={"document_id": {"$in": []}})[2], [])

    def test_version_staging_validation_activation_and_removal(self):
        chunk = {"chunk_id": "c1", "embedding_text": "new policy",
                 "parent_id": "p1", "parent_text": "new complete policy",
                 "title_path": "Rules", "start": 0, "end": 10,
                 "token_count": 4}
        self.handler.stage_version([chunk], "doc", "v1", "collection", "a.txt")
        self.assertTrue(self.handler.validate_version("v1", 1))
        self.handler.activate_version("doc", "v1")
        self.assertEqual(next(iter(self.client.docs.values()))["lifecycle_state"], "ACTIVE")
        with self.assertRaises(RuntimeError):
            self.handler.remove_version("v1")
        self.handler.deactivate_version("v1")
        self.handler.remove_version("v1")
        self.assertFalse(self.client.docs)


if __name__ == "__main__":
    unittest.main()
