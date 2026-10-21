"""Document ingestion and the online hybrid retrieval entry point."""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import threading
import uuid

import chromadb
import cohere

from HybridRetrieval import HybridRetriever
from UploadHandler import UploadHandler


def _load_cohere_key(explicit_key=None):
    if explicit_key:
        return explicit_key
    environment_key = os.environ.get("COHERE_API_KEY", "").strip()
    if environment_key:
        return environment_key
    local_file = Path(__file__).with_name("cohere.txt")
    if local_file.exists():
        return local_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "Cohere credential missing: set COHERE_API_KEY or create backend/cohere.txt")


class EmbeddingHandler:
    """Maintain Chroma vectors and a synchronized BM25 + dense RRF retriever."""

    def __init__(self, persist_directory="./chromadb", similarity_score=0.01,
                 collection="crisischatbot", chunk_size=1000, chunk_overlap=0,
                 reranker=None, cohere_api_key=None, dense_k=50, sparse_k=50,
                 candidate_k=100, rrf_constant=60):
        self.db = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.db.get_or_create_collection(collection)
        self._lifecycle_lock = threading.RLock()
        self.upload_handler = UploadHandler(chunk_size=chunk_size)
        self._migrate_legacy_chunks()
        self.co = reranker or cohere.Client(_load_cohere_key(cohere_api_key))
        self.similarity_score = similarity_score
        self.hybrid_retriever = HybridRetriever(
            collection=self.collection,
            documents=self._collection_documents(),
            reranker=self.co,
            dense_k=dense_k,
            sparse_k=sparse_k,
            candidate_k=candidate_k,
            rrf_constant=rrf_constant,
            threshold=similarity_score,
        )

    def _migrate_legacy_chunks(self):
        """Keep indexes created before lifecycle metadata queryable."""
        snapshot = self.collection.get(include=["documents", "metadatas"])
        ids = snapshot.get("ids") or []
        documents = snapshot.get("documents") or []
        metadatas = snapshot.get("metadatas") or []
        changed_ids, changed_metadatas = [], []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            if (metadata or {}).get("lifecycle_state"):
                continue
            value = dict(metadata or {})
            value["lifecycle_state"] = "ACTIVE"
            value.setdefault("version_id", f"legacy-{value.get('document_id', chunk_id)}")
            value.setdefault("parent_id", chunk_id)
            value.setdefault("parent_text", text)
            changed_ids.append(chunk_id)
            changed_metadatas.append(value)
        if changed_ids:
            self.collection.update(ids=changed_ids, metadatas=changed_metadatas)

    def _collection_documents(self):
        snapshot = self.collection.get(include=["documents", "metadatas"])
        documents = snapshot.get("documents") or []
        metadatas = snapshot.get("metadatas") or [{} for _ in documents]
        return [
            {"id": chunk_id, "text": text, "metadata": metadata or {}}
            for chunk_id, text, metadata in zip(
                snapshot.get("ids", []), documents, metadatas)
        ]

    def refresh_sparse_index(self):
        """Rebuild BM25 after a successful Chroma mutation."""
        self.hybrid_retriever.replace_documents(self._collection_documents())

    def fully_process_file(self, file_content: bytes, filename: str,
                           document_id: str) -> Tuple[str, str]:
        result = self.upload_handler.handle_file(file_content, filename)
        if result[0] == "error":
            return ["error", result[1]]
        documents, ids, metadatas = [], [], []
        for chunk in result[1]:
            documents.append(chunk)
            ids.append(str(uuid.uuid4()))
            metadatas.append({"document_id": document_id, "filename": filename})
        return self.add_processed_documents(documents, ids, metadatas)

    def fully_process_url(self, url: str, document_id: str) -> Tuple[str, str]:
        result = self.upload_handler.handle_urls([url], "url")
        if result[0] != "success":
            return ["error", result[1]]
        documents, ids, metadatas = [], [], []
        for chunk in result[1]:
            documents.append(chunk)
            ids.append(str(uuid.uuid4()))
            metadatas.append({"document_id": document_id, "filename": url})
        return self.add_processed_documents(documents, ids, metadatas)

    def read_zip_contents(self, file_content: bytes, filename: str):
        try:
            return ["zip", self.upload_handler.extract_zip_contents(file_content)]
        except Exception as exc:
            return ["error", f"Failed to read zip contents for {filename}: {str(exc)}"]

    def add_processed_documents(self, documents: List[str], ids: List[str],
                                metadatas: List[Dict[str, str]]) -> Tuple[str, str]:
        if not (len(documents) == len(ids) == len(metadatas)):
            return ["error", "documents, ids and metadatas must have equal lengths"]
        if not documents:
            return ["success", ""]
        try:
            normalized = []
            for chunk_id, document, metadata in zip(ids, documents, metadatas):
                value = dict(metadata)
                value.setdefault("lifecycle_state", "ACTIVE")
                value.setdefault("version_id", f"legacy-{value.get('document_id', chunk_id)}")
                value.setdefault("parent_id", chunk_id)
                value.setdefault("parent_text", document)
                normalized.append(value)
            self.collection.add(documents=documents, metadatas=normalized, ids=ids)
            self.refresh_sparse_index()
            return ["success", ""]
        except Exception as exc:
            filename = metadatas[0].get("filename", "Unknown")
            return ["error", f"An error occurred while embedding {filename}: {type(exc).__name__}"]

    def stage_version(self, chunks, document_id: str, version_id: str,
                      collection_id: str, filename: str):
        """Embed a complete version while keeping it invisible to queries."""
        documents, ids, metadatas = [], [], []
        for chunk in chunks:
            ids.append(f"{version_id}:{chunk['chunk_id']}")
            documents.append(chunk["embedding_text"])
            metadatas.append({
                "document_id": document_id,
                "version_id": version_id,
                "collection_id": collection_id,
                "filename": filename,
                "lifecycle_state": "STAGING",
                "chunk_type": "child",
                "parent_id": f"{version_id}:{chunk['parent_id']}",
                "parent_text": chunk["parent_text"],
                "title_path": chunk["title_path"],
                "source_start": int(chunk["start"]),
                "source_end": int(chunk["end"]),
                "token_count": int(chunk["token_count"]),
            })
        result = self.add_processed_documents(documents, ids, metadatas)
        if result[0] != "success":
            raise RuntimeError(result[1])
        return len(ids)

    def validate_version(self, version_id: str, expected_chunks: int):
        snapshot = self.collection.get(
            where={"version_id": version_id}, include=["metadatas"])
        ids = snapshot.get("ids") or []
        metadatas = snapshot.get("metadatas") or []
        valid = (len(ids) == expected_chunks and expected_chunks > 0 and
                 all(item.get("lifecycle_state") == "STAGING" for item in metadatas) and
                 len(ids) == len(set(ids)))
        if not valid:
            raise RuntimeError(
                f"staged version validation failed: expected {expected_chunks}, found {len(ids)}")
        return True

    def activate_version(self, document_id: str, version_id: str):
        """Switch one logical document under the same lock used by retrieval.

        Chroma has no cross-record transaction.  The process-level lock prevents
        online readers from observing the archive/activate metadata updates, and
        compensation restores the previous metadata if an update fails.
        """
        with self._lifecycle_lock:
            snapshot = self.collection.get(
                where={"document_id": document_id}, include=["metadatas"])
            ids = snapshot.get("ids") or []
            original = snapshot.get("metadatas") or []
            target_ids = [cid for cid, meta in zip(ids, original)
                          if meta.get("version_id") == version_id]
            if not target_ids:
                raise RuntimeError("version contains no indexed chunks")
            updated = []
            for metadata in original:
                value = dict(metadata)
                if value.get("version_id") == version_id:
                    value["lifecycle_state"] = "ACTIVE"
                elif value.get("lifecycle_state") == "ACTIVE":
                    value["lifecycle_state"] = "ARCHIVED"
                updated.append(value)
            try:
                self.collection.update(ids=ids, metadatas=updated)
                self.refresh_sparse_index()
            except Exception:
                try:
                    self.collection.update(ids=ids, metadatas=original)
                    self.refresh_sparse_index()
                finally:
                    raise

    def remove_version(self, version_id: str):
        with self._lifecycle_lock:
            snapshot = self.collection.get(
                where={"version_id": version_id}, include=["metadatas"])
            if any(item.get("lifecycle_state") == "ACTIVE"
                   for item in (snapshot.get("metadatas") or [])):
                raise RuntimeError("refusing to remove an active version")
            self.collection.delete(where={"version_id": version_id})
            self.refresh_sparse_index()

    def deactivate_version(self, version_id: str):
        """Compensate a metadata-transaction failure before first activation."""
        with self._lifecycle_lock:
            snapshot = self.collection.get(
                where={"version_id": version_id}, include=["metadatas"])
            ids = snapshot.get("ids") or []
            metadatas = []
            for item in snapshot.get("metadatas") or []:
                value = dict(item)
                value["lifecycle_state"] = "STAGING"
                metadatas.append(value)
            if ids:
                self.collection.update(ids=ids, metadatas=metadatas)
                self.refresh_sparse_index()

    def remove_documents(self, document_ids: List[str]) -> Tuple[str, str]:
        try:
            for document_id in document_ids:
                self.collection.delete(where={"document_id": document_id})
            self.refresh_sparse_index()
            return ["success", ""]
        except Exception as exc:
            return ["error", f"An error occurred while removing documents: {type(exc).__name__}"]

    def retrieve_documents(self, query: str, num_docs: int = 5,
                           metadata_filters: Dict = None):
        """Dense Top50 + BM25 Top50 + RRF(60), followed by Cohere reranking."""
        active_filter = {"lifecycle_state": "ACTIVE"}
        scoped_filter = ({"$and": [metadata_filters, active_filter]}
                         if metadata_filters else active_filter)
        with self._lifecycle_lock:
            status, error, results = self.hybrid_retriever.retrieve_documents(
                query=query, num_docs=max(num_docs * 3, num_docs),
                metadata_filters=scoped_filter)
        if status != "success":
            return [status, error, results]
        expanded, seen = [], set()
        for result in results:
            metadata = dict(result["metadata"])
            parent_id = metadata.get("parent_id") or result["text"]
            if parent_id in seen:
                continue
            seen.add(parent_id)
            matched_child = result["text"]
            parent_text = metadata.get("parent_text", matched_child)
            metadata["matched_child_text"] = matched_child
            expanded.append({"text": parent_text, "metadata": metadata})
            if len(expanded) == num_docs:
                break
        return ["success", "", expanded]
