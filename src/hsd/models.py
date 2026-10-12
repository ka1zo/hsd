"""Small values exchanged between document ingestion and the search core."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    source_path: str
    text: str
    start_byte: int
    end_byte: int
