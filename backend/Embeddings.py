"""Document ingestion and OpenSearch-native hybrid retrieval entry point."""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import uuid

import cohere

from MiniLMEncoder import MiniLMEncoder
from OpenSearchBackend import OpenSearchBackend, create_opensearch_client
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
    """Use one OpenSearch index for BM25, vectors, filters, and versions."""

    def __init__(self, persist_directory=None, similarity_score=0.01,
                 collection="rag-chunks-v1", chunk_size=1000, chunk_overlap=0,
                 reranker=None, cohere_api_key=None, dense_k=50, sparse_k=50,
                 candidate_k=100, rrf_constant=60, opensearch_client=None,
                 encoder=None, pipeline_name="rag-hybrid-rrf-v1",
                 initialize=True):
        # Kept for compatibility with callers from the Chroma implementation.
        del persist_directory, chunk_overlap, sparse_k
        self.upload_handler = UploadHandler(chunk_size=chunk_size)
        self.encoder = encoder or MiniLMEncoder()
        self.co = reranker or cohere.Client(_load_cohere_key(cohere_api_key))
        self.similarity_score = similarity_score
        self.backend = OpenSearchBackend(
            client=opensearch_client or create_opensearch_client(),
            encoder=self.encoder,
            index_name=collection,
            pipeline_name=pipeline_name,
            dimension=self.encoder.dimension,
            dense_k=dense_k,
            candidate_k=candidate_k,
            rrf_constant=rrf_constant,
            shards=int(os.environ.get("OPENSEARCH_INDEX_SHARDS", "1")),
            replicas=int(os.environ.get("OPENSEARCH_INDEX_REPLICAS", "0")),
            initialize=initialize,
        )

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
            indexed = []
            for chunk_id, text, metadata in zip(ids, documents, metadatas):
                value = dict(metadata)
                value.setdefault("lifecycle_state", "ACTIVE")
                value.setdefault("version_id", f"legacy-{value.get('document_id', chunk_id)}")
                value.setdefault("parent_id", chunk_id)
                value.setdefault("parent_text", text)
                value.setdefault("chunk_type", "child")
                indexed.append({"id": chunk_id, "text": text, "metadata": value})
            self.backend.index_documents(indexed)
            return ["success", ""]
        except Exception as exc:
            filename = metadatas[0].get("filename", "Unknown")
            return ["error", f"An error occurred while indexing {filename}: {type(exc).__name__}"]

    def stage_version(self, chunks, document_id: str, version_id: str,
                      collection_id: str, filename: str):
        indexed = []
        for chunk in chunks:
            indexed.append({
                "id": f"{version_id}:{chunk['chunk_id']}",
                "text": chunk["embedding_text"],
                "metadata": {
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
                },
            })
        return self.backend.index_documents(indexed)

    def validate_version(self, version_id: str, expected_chunks: int):
        return self.backend.validate_version(version_id, expected_chunks)

    def activate_version(self, document_id: str, version_id: str):
        return self.backend.activate_version(document_id, version_id)

    def remove_version(self, version_id: str):
        return self.backend.remove_version(version_id)

    def deactivate_version(self, version_id: str):
        return self.backend.deactivate_version(version_id)

    def remove_documents(self, document_ids: List[str]) -> Tuple[str, str]:
        try:
            self.backend.remove_documents(document_ids)
            return ["success", ""]
        except Exception as exc:
            return ["error", f"An error occurred while removing documents: {type(exc).__name__}"]

    def retrieve_documents(self, query: str, num_docs: int = 5,
                           metadata_filters: Dict = None):
        """OpenSearch BM25+k-NN native RRF, Cohere rerank, parent expansion."""
        active_filter = {"lifecycle_state": "ACTIVE"}
        scoped_filter = ({"$and": [metadata_filters, active_filter]}
                         if metadata_filters else active_filter)
        try:
            with self.backend.lifecycle_lock:
                hits = self.backend.hybrid_search(query, scoped_filter)
            if not hits:
                return ["success", "", []]
            response = self.co.rerank(
                model=os.environ.get(
                    "COHERE_RERANK_MODEL", "rerank-multilingual-v3.0"),
                query=query,
                documents=[hit["_source"]["text"] for hit in hits],
                top_n=min(max(num_docs * 3, num_docs), len(hits)))
            expanded, seen = [], set()
            for result in response.results:
                if result.relevance_score < self.similarity_score:
                    continue
                source = dict(hits[result.index]["_source"])
                matched_child = source.pop("text")
                parent_id = source.get("parent_id") or hits[result.index]["_id"]
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                parent_text = source.pop("parent_text", matched_child)
                source["matched_child_text"] = matched_child
                expanded.append({"text": parent_text, "metadata": source})
                if len(expanded) == num_docs:
                    break
            return ["success", "", expanded]
        except Exception as exc:
            return ["error", type(exc).__name__, []]
