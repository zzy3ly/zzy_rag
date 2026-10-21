"""Local MiniLM embedding and tokenizer adapters shared by ingestion/search."""

import os
from types import SimpleNamespace
import logging
import threading


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HuggingFaceTokenizerAdapter:
    """Expose the tokenizers-style API required by TokenChunking."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text, add_special_tokens=False):
        encoded = self.tokenizer(
            text, add_special_tokens=add_special_tokens,
            return_offsets_mapping=True, truncation=False)
        return SimpleNamespace(ids=encoded["input_ids"],
                               offsets=encoded["offset_mapping"])


class MiniLMEncoder:
    dimension = 384

    def __init__(self, model_name=None):
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", DEFAULT_MODEL)
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    logging.getLogger(__name__).info(
                        "Loading embedding model %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def tokenizer(self):
        return HuggingFaceTokenizerAdapter(self._ensure_model().tokenizer)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = self._ensure_model().encode(
            list(texts), normalize_embeddings=True,
            show_progress_bar=False)
        return vectors.tolist()


def load_minilm_tokenizer(model_name=None):
    return MiniLMEncoder(model_name).tokenizer
