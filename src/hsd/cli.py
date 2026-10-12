"""Command-line entry point for HSD."""

import argparse
import getpass
import sys
import unicodedata
from importlib.metadata import version
from pathlib import Path

from .ingest import DocumentReadError
from . import _core


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return number


def _terminal_text(value: str) -> str:
    """Keep untrusted paths and excerpts from controlling the terminal."""
    escaped = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    parts = []
    for character in value:
        codepoint = ord(character)
        if character in escaped:
            parts.append(escaped[character])
        elif codepoint < 32 or 0x7F <= codepoint <= 0x9F or unicodedata.category(character) == "Cf":
            escape = f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}"
            parts.append(escape)
        else:
            parts.append(character)
    return "".join(parts)


def _password(from_stdin: bool) -> str:
    if from_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("Index password: ")
    if not password:
        raise ValueError("index password cannot be empty")
    return password


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="hsd",
        description="HSD — local lexical, vector and hybrid search for documents",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"HSD {version('hsd')}",
    )
    commands = parser.add_subparsers(dest="command")
    index_command = commands.add_parser("index", help="count fragments in a temporary index")
    index_command.add_argument("folder", help="folder containing TXT and Markdown files")
    index_command.add_argument("--output", help="save an encrypted .hsd index at this path")
    index_command.add_argument("--model-dir", help="local E5 model directory for encrypted embeddings")
    index_command.add_argument("--password-stdin", action="store_true", help="read the password from stdin")
    search_command = commands.add_parser("search", help="search a folder or encrypted index")
    search_command.add_argument("folder", help="folder or encrypted .hsd index")
    search_command.add_argument("query", help="words to find")
    search_command.add_argument("--limit", type=_positive_int, default=10)
    search_command.add_argument("--mode", choices=("lexical", "vector", "hybrid"), default="lexical")
    search_command.add_argument("--model-dir", help="local E5 model directory for vector search")
    search_command.add_argument("--password-stdin", action="store_true", help="read the password from stdin")
    delete_command = commands.add_parser("delete-index", help="delete an encrypted index, leaving documents alone")
    delete_command.add_argument("path", help="encrypted .hsd index to remove")

    options = parser.parse_args(argv)
    if options.command is None:
        parser.print_help()
        return 0

    from .search import build_index
    from .application import create_or_update, delete_index, open_session
    from .embeddings import ModelError

    try:
        if options.command == "delete-index":
            delete_index(Path(options.path))
            print("Encrypted index deleted; source documents were not changed.")
            return 0
        if options.command == "index":
            if options.model_dir and not options.output:
                raise ValueError("--model-dir requires --output")
            if options.output:
                count = create_or_update(Path(options.folder), Path(options.output),
                                         _password(options.password_stdin),
                                         model_dir=Path(options.model_dir) if options.model_dir else None)
                noun = "fragment" if count == 1 else "fragments"
                print(f"Saved {count} {noun} in encrypted index.")
                return 0
            index = build_index(Path(options.folder))
            noun = "fragment" if index.size == 1 else "fragments"
            print(f"Indexed {index.size} {noun} in memory; nothing was saved.")
            return 0
        if options.mode in ("vector", "hybrid"):
            if Path(options.folder).suffix != ".hsd":
                raise ValueError("vector and hybrid search require an encrypted .hsd index")
            if not options.model_dir:
                raise ValueError("vector and hybrid search require --model-dir")
        if Path(options.folder).suffix == ".hsd":
            if options.mode == "lexical" and options.model_dir:
                raise ValueError("--model-dir requires --mode vector or hybrid")
            session = open_session(
                Path(options.folder), _password(options.password_stdin),
                Path(options.model_dir) if options.model_dir else None)
            try:
                count = session.size
                results = session.search(options.query, options.mode, options.limit)
            finally:
                session.close()
            persistent = True
        else:
            if options.model_dir:
                raise ValueError("--model-dir requires --mode vector or hybrid")
            index = build_index(Path(options.folder))
            results = index.search(options.query, options.limit)
            count = index.size
            persistent = False
    except (DocumentReadError, ModelError, ValueError, _core.StoreError, OSError) as error:
        print(f"hsd: {_terminal_text(str(error))}", file=sys.stderr)
        return 2

    if count == 0:
        print("No supported text fragments found.")
    elif not results:
        print("No matching fragments.")
    else:
        for rank, result in enumerate(results, start=1):
            chunk = result if persistent else result.chunk
            location = (f"{rank}. {_terminal_text(chunk.source_path)}:"
                        f"{chunk.start_byte}-{chunk.end_byte} ")
            if options.mode == "hybrid":
                scores = (f"bm25={result.lexical_score:.4f} "
                          f"cosine={result.vector_score:.4f} rrf={result.hybrid_score:.4f}")
            else:
                score = result.vector_score if options.mode == "vector" else result.lexical_score
                scores = f"score={score:.4f}"
            print(location + scores)
            print(f"   {_terminal_text(chunk.text)}")
    return 0
