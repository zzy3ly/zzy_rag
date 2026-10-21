"""Document ingestion and the online hybrid retrieval entry point."""

import os
from pathlib import Path
from typing import Dict, List, Tuple
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
        self.upload_handler = UploadHandler(chunk_size=chunk_size)
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
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            self.refresh_sparse_index()
            return ["success", ""]
        except Exception as exc:
            filename = metadatas[0].get("filename", "Unknown")
            return ["error", f"An error occurred while embedding {filename}: {type(exc).__name__}"]

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
        return self.hybrid_retriever.retrieve_documents(
            query=query,
            num_docs=num_docs,
            metadata_filters=metadata_filters,
        )
