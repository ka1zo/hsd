"""Pure text splitting with offsets into the original UTF-8 text."""

from collections.abc import Iterator

from .models import Chunk


def chunk_text(text: str, source_path: str, max_chars: int = 800) -> Iterator[Chunk]:
    """Yield non-overlapping chunks, preferring whitespace near each limit."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    cursor = 0
    byte_cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            byte_cursor += len(text[cursor].encode("utf-8"))
            cursor += 1
        if cursor == len(text):
            break

        limit = min(cursor + max_chars, len(text))
        end = limit
        if limit < len(text):
            minimum_break = cursor + max(1, max_chars // 2)
            for index in range(limit - 1, minimum_break - 1, -1):
                if text[index].isspace():
                    end = index
                    break

        while end > cursor and text[end - 1].isspace():
            end -= 1

        part = text[cursor:end]
        end_byte = byte_cursor + len(part.encode("utf-8"))
        yield Chunk(source_path, part, byte_cursor, end_byte)
        cursor = end
        byte_cursor = end_byte
