import os
import importlib.util
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if importlib.util.find_spec("PySide6") is None:
    QApplication = None
else:
    from PySide6.QtWidgets import QApplication

from hsd.application import create_or_update


@unittest.skipIf(QApplication is None, "install HSD with the desktop extra")
class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from hsd.desktop import DesktopWindow

        self.window = DesktopWindow()
        self.window.show()

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def wait_idle(self) -> None:
        deadline = time.monotonic() + 10
        while self.window.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertFalse(self.window.busy, self.window.status_label.text())

    def test_unlock_search_lock_and_delete_keeps_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "note.txt"
            document.write_text("private lighthouse", encoding="utf-8")
            index = root / "search.hsd"
            create_or_update(root, index, "correct")

            self.window.open_index(index, "correct")
            self.wait_idle()
            self.assertTrue(self.window.search_button.isEnabled())
            self.window.search("lighthouse")
            self.wait_idle()
            self.assertIn("private lighthouse", self.window.results.item(0).text())
            self.assertIn("score=", self.window.results.item(0).text())
            self.window.query_edit.setText("private lighthouse")
            self.window.lock_index()
            self.wait_idle()
            self.assertFalse(self.window.search_button.isEnabled())
            self.assertEqual(self.window.query_edit.text(), "")
            self.window.delete_index(index)
            self.wait_idle()
            self.assertFalse(index.exists())
            self.assertEqual(self.window.query_edit.text(), "")
            self.assertEqual(document.read_text(), "private lighthouse")

    def test_wrong_password_stays_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("private lighthouse")
            index = root / "search.hsd"
            create_or_update(root, index, "correct")
            self.window.open_index(index, "wrong")
            self.wait_idle()
            self.assertFalse(self.window.search_button.isEnabled())
            self.assertIn("password", self.window.status_label.text().lower())

    def test_creates_index_and_clears_results_when_opening_another(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "first.txt").write_text("lighthouse")
            first = root / "first.hsd"
            self.window.create_index(root, first, "correct")
            self.wait_idle()
            self.assertTrue(first.exists())
            self.window.search("lighthouse")
            self.wait_idle()
            self.assertEqual(self.window.results.count(), 1)
            self.window.query_edit.setText("lighthouse")

            (root / "first.txt").unlink()
            (root / "second.txt").write_text("harbor")
            second = root / "second.hsd"
            create_or_update(root, second, "correct")
            self.window.open_index(second, "correct")
            self.wait_idle()
            self.assertEqual(self.window.results.count(), 0)
            self.assertEqual(self.window.query_edit.text(), "")
            self.assertFalse(self.window.open_source_button.isEnabled())
            self.window.search("harbor")
            self.wait_idle()
            self.assertIn("harbor", self.window.results.item(0).text())

    def test_close_clears_visible_document_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("private lighthouse")
            index = root / "search.hsd"
            create_or_update(root, index, "correct")
            self.window.open_index(index, "correct")
            self.wait_idle()
            self.window.search("lighthouse")
            self.wait_idle()
            self.assertEqual(self.window.results.count(), 1)
            self.window.close()
            self.assertEqual(self.window.results.count(), 0)

    def test_legacy_index_without_root_explains_source_opening(self) -> None:
        self.window._on_done("open", (1, False, ""))
        self.window._hits = [object()]
        self.window._refresh_controls()
        self.assertFalse(self.window.open_source_button.isEnabled())
        self.assertIn("rebuild", self.window.status_label.text().lower())

    @unittest.skipUnless(os.environ.get("HSD_TEST_MODEL_DIR"), "set HSD_TEST_MODEL_DIR")
    def test_vector_and_hybrid_scores_are_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("A lighthouse on the shore.")
            index = root / "search.hsd"
            model = Path(os.environ["HSD_TEST_MODEL_DIR"])
            create_or_update(root, index, "correct", model)
            self.window.open_index(index, "correct", model)
            self.wait_idle()
            self.window.search("lighthouse", "vector")
            self.wait_idle()
            self.assertIn("score=", self.window.results.item(0).text())
            self.window.search("lighthouse", "hybrid")
            self.wait_idle()
            result = self.window.results.item(0).text()
            self.assertIn("bm25=", result)
            self.assertIn("cosine=", result)
            self.assertIn("rrf=", result)


if __name__ == "__main__":
    unittest.main()
