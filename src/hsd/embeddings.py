"""Offline E5 embeddings from explicitly supplied, integrity-checked local files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

MODEL_ID = "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_SHA256 = "4654c156f3e4171abc9c716cdb771bf9116455d15ac1aab364aeeede0e3205b0"
TOKENIZER_SHA256 = "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
DIMENSIONS = 384
MAX_TOKENS = 512


class ModelError(ValueError):
    """Local model files or inference could not produce an embedding."""


def _require_ml():
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ModelError("Install HSD with the ml extra to use local embeddings.") from error
    return np, ort, Tokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_pool_normalize(hidden, attention_mask):
    """Apply the E5 model card's masked mean pooling and L2 normalization."""
    np, _, _ = _require_ml()
    hidden = np.asarray(hidden, dtype=np.float32)
    mask = np.asarray(attention_mask, dtype=np.float32)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ModelError("Invalid model output dimensions.")
    weights = mask[:, :, None]
    totals = np.sum(hidden * weights, axis=1)
    counts = np.sum(weights, axis=1)
    if np.any(counts == 0):
        raise ModelError("Model input has no tokens.")
    pooled = totals / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if not np.all(np.isfinite(pooled)) or np.any(norms == 0):
        raise ModelError("Model produced an invalid embedding.")
    return pooled / norms


class LocalEmbedder:
    """Use pinned multilingual E5 weights and tokenizer without network access."""

    model_id = MODEL_ID
    dimensions = DIMENSIONS

    def __init__(self, model_dir: Path):
        np, ort, Tokenizer = _require_ml()
        directory = Path(model_dir)
        model = directory / "model.onnx"
        tokenizer_path = directory / "tokenizer.json"
        try:
            if not model.is_file() or not tokenizer_path.is_file():
                raise ModelError("Local model.onnx and tokenizer.json files are required.")
            if _sha256(model) != MODEL_SHA256 or _sha256(tokenizer_path) != TOKENIZER_SHA256:
                raise ModelError("Local model files failed SHA-256 verification.")
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
            self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            self._session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        except (OSError, RuntimeError) as error:
            raise ModelError("Cannot open local model files.") from error
        self._np = np
        inputs = {item.name for item in self._session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(inputs) or not inputs.issubset(
            {"input_ids", "attention_mask", "token_type_ids"}
        ):
            raise ModelError("Local model has unsupported inputs.")
        self._inputs = inputs

    def _embed(self, texts: Sequence[str], prefix: str, batch_size: int) -> list[tuple[float, ...]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding text must be nonempty")
        result: list[tuple[float, ...]] = []
        for offset in range(0, len(texts), batch_size):
            encoded = self._tokenizer.encode_batch([prefix + text for text in texts[offset:offset + batch_size]])
            input_ids = self._np.asarray([item.ids for item in encoded], dtype=self._np.int64)
            attention_mask = self._np.asarray([item.attention_mask for item in encoded], dtype=self._np.int64)
            values = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in self._inputs:
                values["token_type_ids"] = self._np.asarray([item.type_ids for item in encoded], dtype=self._np.int64)
            try:
                hidden = self._session.run(None, values)[0]
            except Exception as error:
                raise ModelError("Local model inference failed.") from error
            vectors = mean_pool_normalize(hidden, attention_mask)
            if vectors.shape != (len(encoded), DIMENSIONS):
                raise ModelError("Local model returned the wrong embedding dimension.")
            result.extend(tuple(float(value) for value in row) for row in vectors)
        return result

    def embed_passages(self, texts: Sequence[str], batch_size: int = 16) -> list[tuple[float, ...]]:
        return self._embed(texts, "passage: ", batch_size)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed([text], "query: ", 1)[0]
