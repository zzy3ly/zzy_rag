"""Frozen pure-vector retrieval baseline used by the original experiment.

The production EmbeddingHandler can evolve independently. The experiment bypasses
this class's constructor and injects its isolated Chroma collection and reranker.
"""


class LegacyEmbeddingHandler:
    def retrieve_documents(self, query, num_docs=5, metadata_filters=None):
        try:
            print(f"query received: {query}")
            print(f"number of documents requested: {num_docs}")
            print(f"metadata filters: {metadata_filters}")
            if metadata_filters is None:
                result = self.collection.query(query_texts=[query], n_results=100)
                print("querying without metadata filters.")
            else:
                result = self.collection.query(
                    query_texts=[query], n_results=100, where=metadata_filters)
                print("querying with metadata filters.")
            ranked = sorted(zip(result["documents"][0], result["distances"][0],
                                result["metadatas"][0]), key=lambda item: item[1])
            documents = [{"text": text, "metadata": metadata}
                         for text, _, metadata in ranked]
            if not documents:
                print("no documents to rerank.")
                return ["success", "", []]
            response = self.co.rerank(
                model="rerank-english-v3.0", query=query,
                documents=[doc["text"] for doc in documents], top_n=num_docs)
            return ["success", "", [documents[item.index] for item in response.results
                    if item.relevance_score >= self.similarity_score]]
        except Exception as exc:
            print(f"an error occurred: {str(exc)}")
            return ["error", f"an error occurred retrieving documents for query {query}: {str(exc)}", []]
