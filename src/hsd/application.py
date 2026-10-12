"""Shared local operations for the CLI and future desktop window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import _core
from .search import save_index


@dataclass(frozen=True)
class SearchHit:
    text: str
    source_path: str
    start_byte: int
    end_byte: int
    lexical_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0


class SearchSession:
    """Hold one unlocked index in memory without retaining its password."""

    def __init__(self, chunks: list, embeddings: list, model, root: str):
        self._lexical = _core.LexicalIndex()
        self._lexical.add_chunks(chunks)
        self._vector = None
        self._hybrid = None
        self._model = model
        self.root = root
        if model is not None:
            self._vector = _core.VectorIndex()
            self._vector.add_chunks(chunks, embeddings)
            self._hybrid = _core.HybridIndex()
            self._hybrid.add_chunks(chunks, embeddings)

    @property
    def size(self) -> int:
        if self._lexical is None:
            raise ValueError("search session is closed")
        return self._lexical.size

    @property
    def has_model(self) -> bool:
        return self._model is not None

    def search(self, query: str, mode: str = "lexical", limit: int = 10) -> list[SearchHit]:
        if self._lexical is None:
            raise ValueError("search session is closed")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if mode == "lexical":
            results = self._lexical.search(query, limit)
        elif mode in ("vector", "hybrid"):
            if self._model is None:
                raise ValueError("vector and hybrid search require a matching local model")
            vector = self._model.embed_query(query)
            results = (self._vector.search(vector, limit) if mode == "vector" else
                       self._hybrid.search(query, vector, limit))
        else:
            raise ValueError("unsupported search mode")
        return [SearchHit(
            text=result.chunk.text,
            source_path=result.chunk.source_path,
            start_byte=result.chunk.start_byte,
            end_byte=result.chunk.end_byte,
            lexical_score=getattr(result, "lexical_score", 0.0),
            vector_score=getattr(result, "vector_score", 0.0),
            hybrid_score=getattr(result, "hybrid_score", 0.0),
        ) for result in results]

    def close(self) -> None:
        self._model = None
        self._hybrid = None
        self._vector = None
        self._lexical = None


def create_or_update(root: Path, index_path: Path, password: str,
                     model_dir: Path | None = None) -> int:
    return save_index(Path(root), Path(index_path), password,
                      Path(model_dir) if model_dir is not None else None)


def open_session(index_path: Path, password: str,
                 model_dir: Path | None = None) -> SearchSession:
    path = Path(index_path)
    if path.suffix != ".hsd" or path.is_symlink() or not path.is_file():
        raise ValueError("existing .hsd index file required")
    store = _core.EncryptedStore(str(path), password, False)
    chunks = store.read_chunks()
    root, _ = store.read_file_hashes()
    model = None
    embeddings = []
    if model_dir is not None:
        model_id, embeddings = store.read_embeddings()
        if not model_id:
            raise ValueError("index has no embeddings; rebuild it with a local model")
        from .embeddings import LocalEmbedder

        model = LocalEmbedder(Path(model_dir))
        if model.model_id != model_id:
            raise ValueError("local model does not match the index model")
        if len(chunks) != len(embeddings):
            raise _core.StoreError("Encrypted embeddings are damaged.")
    return SearchSession(chunks, embeddings, model, root)


def delete_index(index_path: Path) -> None:
    path = Path(index_path)
    if path.suffix != ".hsd" or path.is_symlink() or not path.is_file():
        raise ValueError("existing .hsd index file required")
    sidecars = [Path(str(path) + suffix) for suffix in ("-journal", "-wal", "-shm")]
    if any(sidecar.is_symlink() or (sidecar.exists() and not sidecar.is_file())
           for sidecar in sidecars):
        raise ValueError("index sidecar is not a regular file")
    path.unlink()
    for sidecar in sidecars:
        sidecar.unlink(missing_ok=True)
