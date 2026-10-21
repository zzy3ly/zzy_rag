"""OpenSearch storage, native BM25+k-NN RRF, and version lifecycle operations."""

import os
import threading
from urllib.parse import urlparse


def _bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_opensearch_client(url=None, username=None, password=None,
                             verify_certs=None):
    from opensearchpy import OpenSearch

    target = url or os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
    parsed = urlparse(target)
    user = username if username is not None else os.environ.get("OPENSEARCH_USERNAME")
    secret = password if password is not None else os.environ.get("OPENSEARCH_PASSWORD")
    verify = (_bool_env("OPENSEARCH_VERIFY_CERTS", parsed.scheme == "https")
              if verify_certs is None else verify_certs)
    options = {
        "hosts": [{"host": parsed.hostname or "localhost",
                   "port": parsed.port or (443 if parsed.scheme == "https" else 9200)}],
        "use_ssl": parsed.scheme == "https",
        "verify_certs": verify,
        "http_compress": True,
        "timeout": 30,
        "max_retries": 3,
        "retry_on_timeout": True,
    }
    if user or secret:
        if not (user and secret):
            raise ValueError("both OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD are required")
        options["http_auth"] = (user, secret)
    return OpenSearch(**options)


class OpenSearchBackend:
    """One OpenSearch index for lexical, vector, metadata, and lifecycle data."""

    def __init__(self, client, encoder, index_name="rag-chunks-v1",
                 pipeline_name="rag-hybrid-rrf-v1", dimension=384,
                 dense_k=50, candidate_k=100, rrf_constant=60,
                 shards=1, replicas=0, initialize=True):
        self.client = client
        self.encoder = encoder
        self.index_name = index_name
        self.pipeline_name = pipeline_name
        self.dimension = dimension
        self.dense_k = dense_k
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant
        self.shards = shards
        self.replicas = replicas
        self.lifecycle_lock = threading.RLock()
        if initialize:
            self.ensure_schema()

    def ensure_schema(self):
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body={
                "settings": {
                    "index.knn": True,
                    "number_of_shards": self.shards,
                    "number_of_replicas": self.replicas,
                },
                "mappings": {"dynamic": "strict", "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector", "dimension": self.dimension,
                        "method": {
                            "name": "hnsw", "engine": "lucene",
                            "space_type": "cosinesimil",
                            "parameters": {"ef_construction": 128, "m": 16},
                        },
                    },
                    "document_id": {"type": "keyword"},
                    "version_id": {"type": "keyword"},
                    "collection_id": {"type": "keyword"},
                    "filename": {"type": "keyword", "ignore_above": 1024},
                    "lifecycle_state": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},
                    "parent_text": {"type": "text", "index": False},
                    "title_path": {"type": "text",
                                   "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}}},
                    "source_start": {"type": "integer"},
                    "source_end": {"type": "integer"},
                    "token_count": {"type": "integer"},
                }},
            })
        self.client.transport.perform_request(
            "PUT", f"/_search/pipeline/{self.pipeline_name}", body={
                "description": "Native BM25 and vector reciprocal-rank fusion",
                "phase_results_processors": [{
                    "score-ranker-processor": {"combination": {
                        "technique": "rrf",
                        "rank_constant": self.rrf_constant,
                    }}
                }],
            })

    @staticmethod
    def _metadata_filter(value):
        if not value:
            return None
        if "$and" in value:
            parts = [OpenSearchBackend._metadata_filter(item)
                     for item in value["$and"]]
            return {"bool": {"filter": [item for item in parts if item]}}
        if "$or" in value:
            parts = [OpenSearchBackend._metadata_filter(item)
                     for item in value["$or"]]
            return {"bool": {"should": [item for item in parts if item],
                             "minimum_should_match": 1}}
        filters = []
        for field, expected in value.items():
            if isinstance(expected, dict) and "$in" in expected:
                if not expected["$in"]:
                    return {"match_none": {}}
                filters.append({"terms": {field: expected["$in"]}})
            elif isinstance(expected, dict) and "$eq" in expected:
                filters.append({"term": {field: expected["$eq"]}})
            elif isinstance(expected, dict):
                raise ValueError(f"unsupported metadata operator for {field}")
            else:
                filters.append({"term": {field: expected}})
        if len(filters) == 1:
            return filters[0]
        return {"bool": {"filter": filters}}

    def _bulk(self, operations):
        if not operations:
            return
        response = self.client.bulk(
            body=operations, params={"refresh": "wait_for"})
        if response.get("errors"):
            failures = []
            for item in response.get("items", []):
                result = next(iter(item.values()))
                if result.get("error"):
                    failures.append(result["error"].get("reason", "bulk operation failed"))
            raise RuntimeError("; ".join(failures[:3]) or "OpenSearch bulk operation failed")

    def index_documents(self, documents):
        if not documents:
            return 0
        vectors = self.encoder.encode([item["text"] for item in documents])
        operations = []
        for item, vector in zip(documents, vectors):
            source = dict(item["metadata"])
            source["text"] = item["text"]
            source["embedding"] = vector
            operations.extend([
                {"index": {"_index": self.index_name, "_id": item["id"]}},
                source,
            ])
        self._bulk(operations)
        return len(documents)

    def _search_sources(self, query, size=10000):
        response = self.client.search(
            index=self.index_name,
            body={"size": size, "query": query,
                  "_source": {"excludes": ["embedding"]}})
        return response.get("hits", {}).get("hits", [])

    def version_documents(self, version_id):
        return self._search_sources({"term": {"version_id": version_id}})

    def document_versions(self, document_id):
        return self._search_sources({"term": {"document_id": document_id}})

    def validate_version(self, version_id, expected_chunks):
        hits = self.version_documents(version_id)
        valid = (expected_chunks > 0 and len(hits) == expected_chunks and
                 len({hit["_id"] for hit in hits}) == expected_chunks and
                 all(hit["_source"].get("lifecycle_state") == "STAGING"
                     for hit in hits))
        if not valid:
            raise RuntimeError(
                f"staged version validation failed: expected {expected_chunks}, found {len(hits)}")
        return True

    def _update_states(self, changes):
        operations = []
        for chunk_id, state in changes:
            operations.extend([
                {"update": {"_index": self.index_name, "_id": chunk_id}},
                {"doc": {"lifecycle_state": state}},
            ])
        self._bulk(operations)

    def activate_version(self, document_id, version_id):
        with self.lifecycle_lock:
            hits = self.document_versions(document_id)
            if not any(hit["_source"].get("version_id") == version_id for hit in hits):
                raise RuntimeError("version contains no indexed chunks")
            original = [(hit["_id"], hit["_source"].get("lifecycle_state", "STAGING"))
                        for hit in hits]
            changes = []
            for hit in hits:
                source = hit["_source"]
                if source.get("version_id") == version_id:
                    changes.append((hit["_id"], "ACTIVE"))
                elif source.get("lifecycle_state") == "ACTIVE":
                    changes.append((hit["_id"], "ARCHIVED"))
            try:
                self._update_states(changes)
            except Exception:
                self._update_states(original)
                raise

    def deactivate_version(self, version_id):
        with self.lifecycle_lock:
            hits = self.version_documents(version_id)
            self._update_states([(hit["_id"], "STAGING") for hit in hits])

    def remove_version(self, version_id):
        with self.lifecycle_lock:
            hits = self.version_documents(version_id)
            if any(hit["_source"].get("lifecycle_state") == "ACTIVE" for hit in hits):
                raise RuntimeError("refusing to remove an active version")
            self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"version_id": version_id}}},
                params={"refresh": "true", "conflicts": "proceed"})

    def remove_documents(self, document_ids):
        if not document_ids:
            return
        self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"terms": {"document_id": document_ids}}},
            params={"refresh": "true", "conflicts": "proceed"})

    def hybrid_search(self, query_text, metadata_filters=None):
        query_filter = self._metadata_filter(metadata_filters)
        if query_filter == {"match_none": {}}:
            return []
        vector = self.encoder.encode([query_text])[0]
        hybrid = {
            "pagination_depth": self.candidate_k,
            "queries": [
                {"match": {"text": {"query": query_text}}},
                {"knn": {"embedding": {"vector": vector, "k": self.dense_k}}},
            ],
        }
        if query_filter:
            hybrid["filter"] = query_filter
        response = self.client.search(
            index=self.index_name,
            params={"search_pipeline": self.pipeline_name},
            body={
                "size": self.candidate_k,
                "_source": {"excludes": ["embedding"]},
                "query": {"hybrid": hybrid},
            })
        return response.get("hits", {}).get("hits", [])
