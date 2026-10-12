import unittest

from hsd.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_unicode_chunks_preserve_exact_utf8_positions(self) -> None:
        original = "  Привет мир  café\n"
        chunks = list(chunk_text(original, "notes.txt", max_chars=9))

        self.assertEqual(
            [(chunk.text, chunk.start_byte, chunk.end_byte) for chunk in chunks],
            [("Привет", 2, 14), ("мир", 15, 21), ("café", 23, 28)],
        )
        self.assertTrue(all(chunk.source_path == "notes.txt" for chunk in chunks))
        for chunk in chunks:
            self.assertEqual(
                original.encode("utf-8")[chunk.start_byte : chunk.end_byte].decode("utf-8"),
                chunk.text,
            )

    def test_empty_and_whitespace_text_yield_no_chunks(self) -> None:
        self.assertEqual(list(chunk_text("", "empty.txt")), [])
        self.assertEqual(list(chunk_text(" \n\t ", "blank.md")), [])

    def test_long_word_is_split_without_losing_text(self) -> None:
        chunks = list(chunk_text("abcdefghij", "long.txt", max_chars=4))
        self.assertEqual([chunk.text for chunk in chunks], ["abcd", "efgh", "ij"])
        self.assertEqual([(chunk.start_byte, chunk.end_byte) for chunk in chunks], [(0, 4), (4, 8), (8, 10)])

    def test_non_positive_chunk_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            list(chunk_text("abc", "a.txt", max_chars=0))


if __name__ == "__main__":
    unittest.main()
