"""Discover and read local text documents without modifying them."""

import os
import stat
from collections.abc import Iterator
from pathlib import Path

from .chunking import chunk_text
from .models import Chunk

MAX_FILE_BYTES = 8 * 1024 * 1024
SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown"})


class DocumentReadError(Exception):
    """A supported document could not be read safely."""


class DocumentTooLarge(DocumentReadError):
    """A document exceeded the configured byte limit."""


class InvalidDocumentEncoding(DocumentReadError):
    """A document was not valid UTF-8."""


def iter_supported_files(root: Path) -> Iterator[Path]:
    """Yield regular TXT/Markdown files under root in stable walk order."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("selected folder must not be a symlink")
    try:
        selected = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("selected folder does not exist") from error
    if not selected.is_dir():
        raise ValueError("selected path is not a folder")

    def failed_walk(error: OSError) -> None:
        raise DocumentReadError("cannot traverse selected folder") from error

    for directory, directories, files in os.walk(selected, followlinks=False, onerror=failed_walk):
        base = Path(directory)
        directories[:] = sorted(name for name in directories if not (base / name).is_symlink())
        for name in sorted(files):
            path = base / name
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or path.is_symlink():
                continue
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise DocumentReadError(f"cannot inspect document: {path}") from error
            if resolved.is_relative_to(selected) and resolved.is_file():
                yield path


def read_text(path: Path, max_bytes: int = MAX_FILE_BYTES) -> str:
    """Read one regular UTF-8 file, bounded before and during the read."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    path = Path(path)
    if path.is_symlink():
        raise DocumentReadError(f"symbolic links are not supported: {path}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise DocumentReadError(f"not a regular file: {path}")
            if metadata.st_size > max_bytes:
                raise DocumentTooLarge(f"document exceeds {max_bytes} bytes: {path}")
            data = stream.read(max_bytes + 1)
    except OSError as error:
        raise DocumentReadError(f"cannot read document: {path}") from error

    if len(data) > max_bytes:
        raise DocumentTooLarge(f"document exceeds {max_bytes} bytes: {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidDocumentEncoding(f"document is not valid UTF-8: {path}") from error


def iter_chunks(
    root: Path,
    max_chars: int = 800,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> Iterator[Chunk]:
    """Stream chunks from supported documents inside a chosen folder."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")

    for path in iter_supported_files(root):
        text = read_text(path, max_bytes=max_file_bytes)
        yield from chunk_text(text, str(path), max_chars=max_chars)
