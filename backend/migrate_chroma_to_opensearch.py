"""One-time migration of legacy local Chroma chunks into OpenSearch."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb

from MiniLMEncoder import MiniLMEncoder
from OpenSearchBackend import OpenSearchBackend, create_opensearch_client


ALLOWED_METADATA = {
    "document_id", "version_id", "collection_id", "filename",
    "lifecycle_state", "chunk_type", "parent_id", "parent_text",
    "title_path", "source_start", "source_end", "token_count",
}


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def migrate(persist_directory, source_collection, target_index, batch_size=128):
    source = chromadb.PersistentClient(path=persist_directory).get_collection(
        source_collection)
    snapshot = source.get(include=["documents", "metadatas"])
    rows = []
    for chunk_id, text, metadata in zip(
            snapshot.get("ids") or [], snapshot.get("documents") or [],
            snapshot.get("metadatas") or []):
        cleaned = {key: value for key, value in (metadata or {}).items()
                   if key in ALLOWED_METADATA and value is not None}
        cleaned.setdefault("lifecycle_state", "ACTIVE")
        cleaned.setdefault("version_id", f"legacy-{cleaned.get('document_id', chunk_id)}")
        cleaned.setdefault("parent_id", chunk_id)
        cleaned.setdefault("parent_text", text)
        cleaned.setdefault("chunk_type", "child")
        rows.append({"id": chunk_id, "text": text, "metadata": cleaned})

    encoder = MiniLMEncoder()
    backend = OpenSearchBackend(
        create_opensearch_client(), encoder, index_name=target_index,
        dimension=encoder.dimension)
    for group in batches(rows, batch_size):
        backend.index_documents(group)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-directory", default="./chromadb")
    parser.add_argument("--source-collection", default="crisischatbot")
    parser.add_argument("--target-index", default="rag-chunks-v1")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    count = migrate(args.persist_directory, args.source_collection,
                    args.target_index, args.batch_size)
    print(f"Migrated {count} chunks to OpenSearch index {args.target_index}")


if __name__ == "__main__":
    main()
