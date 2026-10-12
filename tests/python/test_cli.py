import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("HSD_TEST_MODEL_DIR"), "set HSD_TEST_MODEL_DIR")
    def test_hybrid_mode_prints_three_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.txt").write_text("Маяк стоит на берегу.", encoding="utf-8")
            database = root / "search.hsd"
            model_dir = os.environ["HSD_TEST_MODEL_DIR"]
            created = self.run_cli("index", str(root), "--output", str(database),
                                   "--model-dir", model_dir, "--password-stdin",
                                   password_input="correct\n")
            found = self.run_cli("search", str(database), "маяк у моря",
                                 "--mode", "hybrid", "--model-dir", model_dir,
                                 "--password-stdin", password_input="correct\n")
            self.assertEqual(created.returncode, 0, created.stderr)
            self.assertEqual(found.returncode, 0, found.stderr)
            self.assertIn("Маяк стоит на берегу.", found.stdout)
            self.assertIn("bm25=", found.stdout)
            self.assertIn("cosine=", found.stdout)
            self.assertIn("rrf=", found.stdout)

    def test_hybrid_mode_requires_index_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            no_index = self.run_cli("search", str(folder), "query", "--mode", "hybrid")
            no_model = self.run_cli("search", str(folder / "missing.hsd"), "query",
                                    "--mode", "hybrid")
            self.assertEqual(no_index.returncode, 2)
            self.assertIn("index", no_index.stderr.lower())
            self.assertEqual(no_model.returncode, 2)
            self.assertIn("--model-dir", no_model.stderr)

    def run_cli(self, *args: str, password_input: str | None = None,
                io_encoding: str | None = None) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "PYTHONIOENCODING": io_encoding} if io_encoding else None
        return subprocess.run(
            [sys.executable, "-m", "hsd", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            input=password_input,
            env=environment,
        )

    def test_help_identifies_hsd(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("HSD", result.stdout)

    def test_version(self) -> None:
        result = self.run_cli("--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn("HSD 0.1.0", result.stdout)

    def test_unknown_argument_fails(self) -> None:
        result = self.run_cli("--unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--unknown", result.stderr)

    def test_index_and_search_use_documents_without_changing_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            document = folder / "notes.md"
            original = "Привет, мир!\nДругой текст.".encode("utf-8")
            document.write_bytes(original)

            indexed = self.run_cli("index", str(folder))
            found = self.run_cli("search", str(folder), "ПРИВЕТ", io_encoding="cp1252")

            self.assertEqual(indexed.returncode, 0, indexed.stderr)
            self.assertIn("Indexed 1 fragment in memory", indexed.stdout)
            self.assertEqual(found.returncode, 0, found.stderr)
            self.assertIn(f"{document.resolve()}:0-{len(original)}", found.stdout)
            self.assertIn("Привет, мир!", found.stdout)
            self.assertIn("score=", found.stdout)
            self.assertEqual(document.read_bytes(), original)
            self.assertEqual(list(folder.iterdir()), [document])

    def test_empty_folder_and_no_match_are_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            empty = self.run_cli("search", temporary, "hello")
            self.assertEqual(empty.returncode, 0)
            self.assertIn("No supported text fragments", empty.stdout)

            (Path(temporary) / "a.txt").write_text("hello", encoding="utf-8")
            missing = self.run_cli("search", temporary, "pear")
            self.assertEqual(missing.returncode, 0)
            self.assertIn("No matching fragments", missing.stdout)

    def test_bad_folder_and_limit_have_clean_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bad_folder = self.run_cli("search", str(Path(temporary) / "missing"), "word")
            bad_limit = self.run_cli("search", temporary, "word", "--limit", "0")
            self.assertNotEqual(bad_folder.returncode, 0)
            self.assertIn("folder", bad_folder.stderr)
            self.assertNotIn("Traceback", bad_folder.stderr)
            self.assertNotEqual(bad_limit.returncode, 0)
            self.assertIn("positive", bad_limit.stderr)

    def test_invalid_document_encoding_exits_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "broken.txt").write_bytes(b"\xff private bytes")
            result = self.run_cli("search", temporary, "word")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("UTF-8", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertNotIn("private bytes", result.stderr)

    def test_control_characters_in_excerpts_are_escaped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "a.txt").write_text("\x1b[31m SECRET \u202e", encoding="utf-8")
            result = self.run_cli("search", temporary, "secret")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("\x1b", result.stdout)
            self.assertIn("\\u001b", result.stdout)
            self.assertNotIn("\u202e", result.stdout)
            self.assertIn("\\u202e", result.stdout)

    def test_persistent_index_reopens_and_can_be_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "note.txt"
            original = b"private lighthouse"
            document.write_bytes(original)
            database = root / "search.hsd"
            created = self.run_cli("index", str(root), "--output", str(database),
                                   "--password-stdin", password_input="long password\n")
            found = self.run_cli("search", str(database), "lighthouse",
                                 "--password-stdin", password_input="long password\n")
            journal = Path(str(database) + "-journal")
            journal.write_bytes(b"encrypted journal placeholder")
            removed = self.run_cli("delete-index", str(database))
            self.assertEqual(created.returncode, 0, created.stderr)
            self.assertEqual(found.returncode, 0, found.stderr)
            self.assertIn("private lighthouse", found.stdout)
            self.assertEqual(removed.returncode, 0, removed.stderr)
            self.assertFalse(database.exists())
            self.assertFalse(journal.exists())
            self.assertEqual(document.read_bytes(), original)

    def test_cli_reindexes_changes_and_keeps_previous_index_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "note.txt"
            document.write_text("old signal")
            database = root / "search.hsd"
            command = ("index", str(root), "--output", str(database), "--password-stdin")
            self.assertEqual(self.run_cli(*command, password_input="correct\n").returncode, 0)
            document.write_bytes(b"\xff")
            failed = self.run_cli(*command, password_input="correct\n")
            old = self.run_cli("search", str(database), "old", "--password-stdin",
                               password_input="correct\n")
            self.assertEqual(failed.returncode, 2)
            self.assertIn("old signal", old.stdout)
            document.write_text("new signal")
            updated = self.run_cli(*command, password_input="correct\n")
            new = self.run_cli("search", str(database), "new", "--password-stdin",
                               password_input="correct\n")
            self.assertEqual(updated.returncode, 0, updated.stderr)
            self.assertIn("new signal", new.stdout)

    def test_model_requires_encrypted_output_and_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("private text")
            no_output = self.run_cli("index", str(root), "--model-dir", str(root / "missing"))
            self.assertEqual(no_output.returncode, 2)
            self.assertIn("--output", no_output.stderr)
            database = root / "search.hsd"
            missing_model = self.run_cli("index", str(root), "--output", str(database),
                                         "--model-dir", str(root / "missing"),
                                         "--password-stdin", password_input="correct\n")
            self.assertEqual(missing_model.returncode, 2)
            self.assertNotIn("Traceback", missing_model.stderr)
            self.assertFalse(database.exists())

    @unittest.skipUnless(os.environ.get("HSD_TEST_MODEL_DIR"), "set HSD_TEST_MODEL_DIR")
    def test_vector_mode_searches_encrypted_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.txt").write_text("Маяк стоит на берегу.", encoding="utf-8")
            database = root / "search.hsd"
            model_dir = os.environ["HSD_TEST_MODEL_DIR"]
            created = self.run_cli("index", str(root), "--output", str(database),
                                   "--model-dir", model_dir, "--password-stdin",
                                   password_input="correct\n")
            found = self.run_cli("search", str(database), "маяк у моря",
                                 "--mode", "vector", "--model-dir", model_dir,
                                 "--password-stdin", password_input="correct\n")
            self.assertEqual(created.returncode, 0, created.stderr)
            self.assertEqual(found.returncode, 0, found.stderr)
            self.assertIn("Маяк стоит на берегу.", found.stdout)
            self.assertIn("score=", found.stdout)

    def test_vector_mode_requires_encrypted_index_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "a.txt").write_text("some text")
            no_index = self.run_cli("search", str(folder), "query", "--mode", "vector")
            no_model = self.run_cli("search", str(folder / "missing.hsd"), "query",
                                    "--mode", "vector")
            self.assertEqual(no_index.returncode, 2)
            self.assertIn("index", no_index.stderr.lower())
            self.assertEqual(no_model.returncode, 2)
            self.assertIn("--model-dir", no_model.stderr)


if __name__ == "__main__":
    unittest.main()
