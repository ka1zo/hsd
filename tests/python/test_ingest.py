import tempfile
import unittest
from pathlib import Path

from hsd.ingest import (
    DocumentReadError,
    DocumentTooLarge,
    InvalidDocumentEncoding,
    iter_chunks,
    iter_supported_files,
    read_text,
)


class IngestTests(unittest.TestCase):
    def test_discovery_is_sorted_and_limits_file_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "z.txt").write_text("z", encoding="utf-8")
            (root / "skip.pdf").write_text("ignore", encoding="utf-8")
            nested = root / "a"
            nested.mkdir()
            (nested / "b.MD").write_text("b", encoding="utf-8")
            (nested / "a.markdown").write_text("a", encoding="utf-8")

            self.assertEqual(
                [path.relative_to(root.resolve()).as_posix() for path in iter_supported_files(root)],
                ["z.txt", "a/a.markdown", "a/b.MD"],
            )

    def test_symlinked_files_and_directories_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as selected, tempfile.TemporaryDirectory() as outside:
            root = Path(selected)
            external = Path(outside)
            (external / "secret.txt").write_text("secret", encoding="utf-8")
            (root / "safe.txt").write_text("safe", encoding="utf-8")
            try:
                (root / "linked.txt").symlink_to(external / "secret.txt")
                (root / "linked-dir").symlink_to(external, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")

            self.assertEqual([path.name for path in iter_supported_files(root)], ["safe.txt"])
            with self.assertRaises(DocumentReadError):
                read_text(root / "linked.txt")

    def test_bad_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "not-a-directory.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(iter_supported_files(root / "missing"))
            with self.assertRaises(ValueError):
                list(iter_supported_files(root / "not-a-directory.txt"))

    def test_reader_rejects_oversized_and_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            large = root / "large.txt"
            large.write_bytes(b"hello")
            with self.assertRaises(DocumentTooLarge):
                read_text(large, max_bytes=4)

            invalid = root / "invalid.md"
            invalid.write_bytes(b"\xff")
            with self.assertRaises(InvalidDocumentEncoding):
                read_text(invalid)

    def test_pipeline_yields_sources_and_positions_without_modifying_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "notes.txt"
            original = "  Привет мир  café\n".encode("utf-8")
            document.write_bytes(original)

            chunks = list(iter_chunks(root, max_chars=9))

            self.assertEqual(
                [(chunk.text, chunk.start_byte, chunk.end_byte) for chunk in chunks],
                [("Привет", 2, 14), ("мир", 15, 21), ("café", 23, 28)],
            )
            self.assertTrue(all(chunk.source_path == str(document.resolve()) for chunk in chunks))
            self.assertEqual(document.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
