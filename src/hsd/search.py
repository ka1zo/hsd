"""Application operations that connect local documents to the C++ core."""

from itertools import islice
import hashlib
from pathlib import Path

from . import _core
from .chunking import chunk_text
from .ingest import iter_chunks, iter_supported_files, read_text


def build_index(root: Path, batch_size: int = 256) -> _core.LexicalIndex:
    """Build a temporary in-memory lexical index from a chosen folder."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    index = _core.LexicalIndex()
    chunks = iter(iter_chunks(root))
    while batch := list(islice(chunks, batch_size)):
        native_chunks = [
            _core.Chunk(chunk.source_path, chunk.text, chunk.start_byte, chunk.end_byte)
            for chunk in batch
        ]
        index.add_chunks(native_chunks)
    return index


def save_index(root: Path, index_path: Path, password: str, model_dir: Path | None = None) -> int:
    """Update changed documents atomically while retaining unchanged fragments."""
    if root.is_symlink():
        raise ValueError("selected folder must not be a symlink")
    root = root.resolve(strict=True)
    index_path = index_path.absolute()
    if index_path.suffix != ".hsd":
        raise ValueError("index filename must end in .hsd")
    if index_path.is_symlink():
        raise ValueError("index path cannot be a symbolic link")
    model_id = ""
    model = None
    if model_dir is not None:
        from .embeddings import LocalEmbedder

        model = LocalEmbedder(model_dir)
        model_id = model.model_id
    store = _core.EncryptedStore(str(index_path), password, True) if index_path.exists() else None
    indexed_root, old_hashes = store.read_file_hashes() if store else ("", {})
    if indexed_root and indexed_root != str(root):
        raise ValueError("index belongs to a different folder")
    stored_model_id = store.read_model_id() if store else ""
    if (indexed_root or stored_model_id) and stored_model_id != model_id:
        raise ValueError("index uses a different embedding model; create a new index")
    files = []
    chunks = []
    prepared_paths = []
    for path in iter_supported_files(root):
        text = read_text(path)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_path = str(path)
        files.append((source_path, digest))
        if old_hashes.get(source_path) != digest:
            prepared_paths.append(source_path)
            chunks.extend(
                _core.Chunk(chunk.source_path, chunk.text, chunk.start_byte, chunk.end_byte)
                for chunk in chunk_text(text, source_path)
            )
    embeddings = model.embed_passages([chunk.text for chunk in chunks]) if model and chunks else []
    if store is None:
        store = _core.EncryptedStore(str(index_path), password, True)
    return store.apply_update(str(root), files, chunks, embeddings, model_id, prepared_paths)


def open_index(index_path: Path, password: str) -> _core.LexicalIndex:
    """Unlock an encrypted index and rebuild its lexical search structures in memory."""
    if index_path.is_symlink():
        raise ValueError("index path cannot be a symbolic link")
    store = _core.EncryptedStore(str(index_path), password, False)
    index = _core.LexicalIndex()
    index.add_chunks(store.read_chunks())
    return index


def _open_embedded_index(index_path: Path, password: str, model_dir: Path):
    if index_path.is_symlink():
        raise ValueError("index path cannot be a symbolic link")
    store = _core.EncryptedStore(str(index_path), password, False)
    model_id, embeddings = store.read_embeddings()
    if not model_id:
        raise ValueError("index has no embeddings; rebuild it with --model-dir")

    from .embeddings import LocalEmbedder

    model = LocalEmbedder(model_dir)
    if model.model_id != model_id:
        raise ValueError("local model does not match the index model")
    chunks = store.read_chunks()
    if len(chunks) != len(embeddings):
        raise _core.StoreError("Encrypted embeddings are damaged.")
    return chunks, embeddings, model


def search_encrypted_vectors(index_path: Path, password: str, query: str,
                             model_dir: Path, limit: int = 10) -> tuple[int, list]:
    """Search encrypted fragments with a query embedding from the index's model."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    chunks, embeddings, model = _open_embedded_index(index_path, password, model_dir)
    index = _core.VectorIndex()
    index.add_chunks(chunks, embeddings)
    return index.size, index.search(model.embed_query(query), limit)


def search_encrypted_hybrid(index_path: Path, password: str, query: str,
                            model_dir: Path, limit: int = 10) -> tuple[int, list]:
    """Fuse lexical and vector rankings from an encrypted index in C++."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    chunks, embeddings, model = _open_embedded_index(index_path, password, model_dir)
    index = _core.HybridIndex()
    index.add_chunks(chunks, embeddings)
    return index.size, index.search(query, model.embed_query(query), limit)
