"""Small-corpus BM25 + vector RRF retriever; leaves Embeddings.py unchanged.

This in-memory BM25 implementation is for a reproducible prototype. For a larger
or multi-process deployment, replace its backend with a maintained search service.
The index is an immutable snapshot: rebuild after a corpus change.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
import unicodedata

import numpy as np

STOPWORDS = frozenset(
    "a an the and or of to in on at for from by with is are was were be been being "
    "what which how do does did this that these those as it its than then".split()
)


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    # Preserve identifiers such as 34/8-A-9, and also index their components.
    compounds = re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)+", text)
    words = re.findall(r"[a-z0-9]+", text)
    return [w for w in words if w not in STOPWORDS] + compounds


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    metadata: dict
    score: float


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        self.lengths = np.zeros(len(self.documents), dtype=np.float64)
        for i, doc in enumerate(self.documents):
            terms = Counter(tokenize(doc["text"]))
            self.lengths[i] = sum(terms.values())
            for term, frequency in terms.items():
                self.postings[term].append((i, frequency))
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 1.0

    def search(self, query, limit=50, allowed_ids=None):
        scores = np.zeros(len(self.documents), dtype=np.float64)
        n = len(self.documents)
        for term in set(tokenize(query)):
            posting = self.postings.get(term, [])
            if not posting:
                continue
            df = len(posting)
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            for i, tf in posting:
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avg_length, 1))
                scores[i] += idf * tf * (self.k1 + 1) / denominator
        eligible = [i for i in range(n) if scores[i] > 0 and
                    (allowed_ids is None or self.documents[i]["id"] in allowed_ids)]
        eligible.sort(key=lambda i: (-scores[i], self.documents[i]["id"]))
        return [Hit(self.documents[i]["id"], self.documents[i]["text"],
                    self.documents[i].get("metadata", {}), float(scores[i])) for i in eligible[:limit]]


def reciprocal_rank_fusion(rankings, limit=100, constant=60):
    if constant < 0 or limit < 0:
        raise ValueError("constant and limit must be nonnegative")
    scores, docs = defaultdict(float), {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            if hit.id in seen:
                continue
            seen.add(hit.id)
            docs[hit.id] = hit
            scores[hit.id] += 1.0 / (constant + rank)
    ids = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
    return [Hit(cid, docs[cid].text, docs[cid].metadata, scores[cid]) for cid in ids]


class HybridRetriever:
    def __init__(self, collection, documents, reranker, dense_k=50, sparse_k=50,
                 candidate_k=100, rrf_constant=60, threshold=0.01):
        self.collection = collection
        self.bm25 = BM25Index(documents)
        self.reranker = reranker
        self.dense_k, self.sparse_k = dense_k, sparse_k
        self.candidate_k, self.rrf_constant = candidate_k, rrf_constant
        self.threshold = threshold

    def candidates(self, query, metadata_filters=None):
        kwargs = {"where": metadata_filters} if metadata_filters else {}
        # Resolve the SAME explicit scope for both branches. Never post-filter only.
        allowed = None
        if metadata_filters:
            allowed = set(self.collection.get(where=metadata_filters, include=[])["ids"])
            if not allowed:
                return []
        count = len(allowed) if allowed is not None else self.collection.count()
        if not count:
            return []
        result = self.collection.query(query_texts=[query], n_results=min(self.dense_k, count), **kwargs)
        dense = [Hit(cid, text, meta, -distance) for cid, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])]
        sparse = self.bm25.search(query, self.sparse_k, allowed)
        return reciprocal_rank_fusion([dense, sparse], self.candidate_k, self.rrf_constant)

    def retrieve_documents(self, query, num_docs=5, metadata_filters=None):
        """Same success/error tuple shape as the legacy handler."""
        try:
            hits = self.candidates(query, metadata_filters)
            if not hits:
                return ["success", "", []]
            response = self.reranker.rerank(
                model="rerank-english-v3.0", query=query,
                documents=[h.text for h in hits], top_n=num_docs)
            return ["success", "", [
                {"text": hits[r.index].text, "metadata": hits[r.index].metadata}
                for r in response.results if r.relevance_score >= self.threshold
            ]]
        except Exception as exc:
            # Do not return credential-bearing exception bodies.
            return ["error", type(exc).__name__, []]
