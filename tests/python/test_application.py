import os
import tempfile
import unittest
from pathlib import Path

from hsd import application
from hsd import _core


class ApplicationTests(unittest.TestCase):
    def test_session_searches_twice_and_locks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("private lighthouse", encoding="utf-8")
            index = root / "search.hsd"
            self.assertEqual(application.create_or_update(root, index, "correct"), 1)
            session = application.open_session(index, "correct")
            self.assertEqual(session.size, 1)
            self.assertEqual(session.search("lighthouse")[0].text, "private lighthouse")
            self.assertEqual(session.search("private")[0].source_path,
                             str((root / "note.txt").resolve()))
            session.close()
            with self.assertRaisesRegex(ValueError, "closed"):
                session.search("lighthouse")
            with self.assertRaises(_core.StoreError):
                application.open_session(index, "wrong")

    def test_delete_only_removes_index_and_safe_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "note.txt"
            document.write_text("original")
            index = root / "search.hsd"
            application.create_or_update(root, index, "correct")
            sidecar = Path(str(index) + "-journal")
            sidecar.write_bytes(b"placeholder")
            application.delete_index(index)
            self.assertFalse(index.exists())
            self.assertFalse(sidecar.exists())
            self.assertEqual(document.read_text(), "original")
            with self.assertRaises(ValueError):
                application.delete_index(document)

    def test_vector_mode_requires_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("private lighthouse")
            index = root / "search.hsd"
            application.create_or_update(root, index, "correct")
            session = application.open_session(index, "correct")
            with self.assertRaisesRegex(ValueError, "model|embedding"):
                session.search("lighthouse", "vector")
            with self.assertRaisesRegex(ValueError, "mode"):
                session.search("lighthouse", "invalid")

    @unittest.skipUnless(os.environ.get("HSD_TEST_MODEL_DIR"), "set HSD_TEST_MODEL_DIR")
    def test_session_supports_all_three_modes_with_pinned_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "маяк.md").write_text("Маяк стоит на берегу.", encoding="utf-8")
            index = root / "search.hsd"
            model_dir = Path(os.environ["HSD_TEST_MODEL_DIR"])
            application.create_or_update(root, index, "correct", model_dir)
            session = application.open_session(index, "correct", model_dir)
            for mode in ("lexical", "vector", "hybrid"):
                hit = session.search("маяк у моря", mode)[0]
                self.assertIn("Маяк", hit.text)
                self.assertEqual(hit.start_byte, 0)
            session.close()


if __name__ == "__main__":
    unittest.main()
