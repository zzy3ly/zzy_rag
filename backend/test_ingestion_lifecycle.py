import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Ingestion import IngestionManager
from Embeddings import EmbeddingHandler
from StructureChunking import load_minilm_tokenizer, structure_parent_child_chunks


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        matches = list(re.finditer(r"\S", text))
        return SimpleNamespace(
            ids=list(range(len(matches))),
            offsets=[match.span() for match in matches])


class FakeVersionIndex:
    def __init__(self):
        self.versions = {}
        self.active = {}
        self.fail_next_stage = False

    def stage_version(self, chunks, document_id, version_id, collection_id, filename):
        if self.fail_next_stage:
            self.fail_next_stage = False
            raise RuntimeError("injected embedding outage")
        self.versions[version_id] = {
            "document_id": document_id, "state": "STAGING",
            "chunks": list(chunks)}
        return len(chunks)

    def validate_version(self, version_id, expected_chunks):
        value = self.versions[version_id]
        if len(value["chunks"]) != expected_chunks:
            raise RuntimeError("bad count")
        return True

    def activate_version(self, document_id, version_id):
        for value in self.versions.values():
            if value["document_id"] == document_id and value["state"] == "ACTIVE":
                value["state"] = "ARCHIVED"
        self.versions[version_id]["state"] = "ACTIVE"
        self.active[document_id] = version_id

    def remove_version(self, version_id):
        if self.versions.get(version_id, {}).get("state") != "ACTIVE":
            self.versions.pop(version_id, None)


class StableReranker:
    def rerank(self, documents, top_n, **kwargs):
        return SimpleNamespace(results=[
            SimpleNamespace(index=index, relevance_score=1.0)
            for index in range(min(top_n, len(documents)))])


class StructureChunkingTests(unittest.TestCase):
    def test_heading_path_parent_expansion_and_token_limit(self):
        text = ("第一章 井控要求\n\n适用条件：进入目的层前完成检查。\n\n"
                "1.1 检查步骤\n\n" + "逐项检查防喷器并记录压力。" * 30)
        result = structure_parent_child_chunks(
            text, CharacterTokenizer(), "井控管理规定",
            child_max_tokens=80, parent_max_tokens=180, overlap_tokens=10)
        self.assertGreater(len(result["parents"]), 1)
        self.assertGreater(len(result["children"]), 1)
        self.assertTrue(all(item["token_count"] <= 80
                            for item in result["children"]))
        self.assertTrue(any("1.1 检查步骤" in item["title_path"]
                            for item in result["children"]))
        self.assertTrue(all(item["parent_text"] for item in result["children"]))


class IngestionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.index = FakeVersionIndex()
        root = Path(self.temp.name)
        self.manager = IngestionManager(
            self.index, root / "lifecycle.db", root / "blobs",
            tokenizer=CharacterTokenizer(), max_attempts=1)

    def tearDown(self):
        self.manager.shutdown()
        self.temp.cleanup()

    def submit_wait(self, body):
        submitted = self.manager.submit(
            "engineering", "well-control.txt", body.encode("utf-8"),
            document_key="well-control-policy")
        return submitted, self.manager.wait(submitted["job_id"], 5)

    def test_async_idempotency_version_switch_and_rollback(self):
        first, first_job = self.submit_wait(
            "第一章 总则\n\n旧版要求：每周检查一次防喷器。")
        self.assertEqual(first_job["status"], "ACTIVE")
        duplicate = self.manager.submit(
            "engineering", "renamed.txt",
            "第一章 总则\n\n旧版要求：每周检查一次防喷器。".encode("utf-8"),
            document_key="well-control-policy")
        self.assertTrue(duplicate["deduplicated"])
        self.assertEqual(duplicate["job_id"], first["job_id"])

        second, second_job = self.submit_wait(
            "第一章 总则\n\n新版要求：每班检查一次防喷器并记录压力。")
        self.assertEqual(second_job["status"], "ACTIVE")
        self.assertEqual(self.index.versions[first["version_id"]]["state"], "ARCHIVED")
        self.assertEqual(self.index.active[first["document_id"]], second["version_id"])
        history = self.manager.store.history(first["document_id"])
        self.assertEqual([item["status"] for item in history], ["ACTIVE", "ARCHIVED"])

        self.manager.rollback(first["document_id"], first["version_id"])
        self.assertEqual(self.index.active[first["document_id"]], first["version_id"])
        self.assertEqual(self.manager.store.version(first["version_id"])["status"], "ACTIVE")

    def test_failed_job_can_be_retried_without_replacing_active_version(self):
        first, _ = self.submit_wait("第一章 总则\n\n当前有效内容。")
        self.index.fail_next_stage = True
        failed, failed_job = self.submit_wait("第一章 总则\n\n准备更新的内容。")
        self.assertEqual(failed_job["status"], "FAILED")
        self.assertEqual(self.index.active[first["document_id"]], first["version_id"])

        self.manager.retry(failed["job_id"])
        retried = self.manager.wait(failed["job_id"], 5)
        self.assertEqual(retried["status"], "ACTIVE")
        self.assertEqual(self.index.active[first["document_id"]], failed["version_id"])

    def test_pending_job_is_recovered_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = FakeVersionIndex()
            stopped = IngestionManager(
                index, root / "recover.db", root / "blobs",
                tokenizer=CharacterTokenizer(), start_workers=False)
            submitted = stopped.submit(
                "engineering", "recovery.txt", "重启恢复任务。".encode())
            self.assertEqual(stopped.status(submitted["job_id"])["status"], "PENDING")
            restarted = IngestionManager(
                index, root / "recover.db", root / "blobs",
                tokenizer=CharacterTokenizer())
            self.assertEqual(restarted.wait(submitted["job_id"], 5)["status"], "ACTIVE")
            restarted.shutdown()


class RealChromaLifecycleTests(unittest.TestCase):
    def test_staging_activation_parent_expansion_and_visibility(self):
        # Chroma's Windows HNSW handle can outlive the client object briefly.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            handler = EmbeddingHandler(
                persist_directory=str(root / "chroma"),
                collection="lifecycle-test", reranker=StableReranker(),
                similarity_score=0.0, dense_k=10, sparse_k=10)
            manager = IngestionManager(
                handler, root / "lifecycle.db", root / "blobs",
                tokenizer=load_minilm_tokenizer(), max_attempts=1)
            first = manager.submit(
                "engineering", "policy.txt",
                "第一章 旧规程\n\n旧版识别码 OLD-SEAL-771，每周检查一次。".encode(),
                document_key="policy")
            self.assertEqual(manager.wait(first["job_id"], 10)["status"], "ACTIVE")
            status, _, results = handler.retrieve_documents(
                "OLD-SEAL-771", 3,
                {"document_id": {"$in": [first["document_id"]]}})
            self.assertEqual(status, "success")
            self.assertTrue(results)
            self.assertIn("旧版识别码", results[0]["text"])
            self.assertIn("matched_child_text", results[0]["metadata"])

            second = manager.submit(
                "engineering", "policy.txt",
                "第一章 新规程\n\n新版识别码 NEW-SEAL-992，每班检查并记录压力。".encode(),
                document_key="policy")
            self.assertEqual(manager.wait(second["job_id"], 10)["status"], "ACTIVE")
            _, _, old_results = handler.retrieve_documents("OLD-SEAL-771", 3)
            self.assertTrue(all("旧版识别码" not in item["text"] for item in old_results))
            _, _, new_results = handler.retrieve_documents("NEW-SEAL-992", 3)
            self.assertTrue(any("新版识别码" in item["text"] for item in new_results))
            manager.shutdown()


if __name__ == "__main__":
    unittest.main()
