import sqlite3
import os
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from hsd.search import open_index, save_index, search_encrypted_vectors, search_encrypted_hybrid
from hsd import _core


class StorageTests(unittest.TestCase):
    def test_incremental_reindex_updates_changed_and_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keep = root / "keep.txt"
            change = root / "change.md"
            removed = root / "removed.txt"
            keep.write_text("steady beacon")
            change.write_text("old harbor")
            removed.write_text("obsolete chart")
            database = root / "search.hsd"
            self.assertEqual(save_index(root, database, "correct"), 3)
            original_hashes = _core.EncryptedStore(str(database), "correct", False).read_file_hashes()[1]

            change.write_text("new harbor")
            removed.unlink()
            added = root / "added.txt"
            added.write_text("fresh chart")
            self.assertEqual(save_index(root, database, "correct"), 3)
            reopened = _core.EncryptedStore(str(database), "correct", False)
            indexed_root, hashes = reopened.read_file_hashes()
            self.assertEqual(indexed_root, str(root.resolve()))
            self.assertEqual(set(hashes), {str(keep.resolve()), str(change.resolve()), str(added.resolve())})
            self.assertEqual(hashes[str(keep.resolve())], original_hashes[str(keep.resolve())])
            self.assertNotEqual(hashes[str(change.resolve())], original_hashes[str(change.resolve())])
            self.assertEqual({chunk.text for chunk in reopened.read_chunks()},
                             {"steady beacon", "new harbor", "fresh chart"})
            del reopened

    def test_failed_incremental_update_preserves_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "note.txt"
            document.write_text("original")
            database = root / "search.hsd"
            save_index(root, database, "correct")
            original = _core.EncryptedStore(str(database), "correct", False).read_file_hashes()
            document.write_bytes(b"\xff")
            with self.assertRaises(Exception):
                save_index(root, database, "correct")
            reopened = _core.EncryptedStore(str(database), "correct", False)
            self.assertEqual(reopened.read_file_hashes(), original)
            self.assertEqual(reopened.read_chunks()[0].text, "original")
            del reopened
            document.write_text("recovered")
            save_index(root, database, "correct")
            self.assertEqual(open_index(database, "correct").search("recovered")[0].chunk.text,
                             "recovered")

    def test_untracked_embedded_index_requires_matching_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("replacement")
            database = root / "search.hsd"
            store = _core.EncryptedStore(str(database), "correct", True)
            store.replace_chunks_with_embeddings(
                [_core.Chunk(str(root / "note.txt"), "original", 0, 8)],
                [[1.0] + [0.0] * 383], "stored-model")
            del store
            with self.assertRaisesRegex(ValueError, "model"):
                save_index(root, database, "correct")
            reopened = _core.EncryptedStore(str(database), "correct", False)
            self.assertEqual(reopened.read_chunks()[0].text, "original")
            self.assertEqual(reopened.read_model_id(), "stored-model")
            del reopened

    @unittest.skipUnless(os.environ.get("HSD_TEST_MODEL_DIR"), "set HSD_TEST_MODEL_DIR")
    def test_incremental_update_reuses_unchanged_embeddings(self) -> None:
        model_dir = Path(os.environ["HSD_TEST_MODEL_DIR"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            steady = root / "steady.txt"
            changed = root / "changed.txt"
            steady.write_text("A lighthouse beside the sea")
            changed.write_text("An old bridge")
            database = root / "search.hsd"
            save_index(root, database, "correct", model_dir)
            before = _core.EncryptedStore(str(database), "correct", False).read_embeddings()[1]
            changed.write_text("A new bridge")
            save_index(root, database, "correct", model_dir)
            reopened = _core.EncryptedStore(str(database), "correct", False)
            by_path = dict(zip((chunk.source_path for chunk in reopened.read_chunks()),
                               reopened.read_embeddings()[1]))
            self.assertEqual(by_path[str(steady.resolve())], before[1])
            self.assertNotEqual(by_path[str(changed.resolve())], before[0])
            with self.assertRaisesRegex(ValueError, "model"):
                save_index(root, database, "correct")
            self.assertEqual(len(reopened.read_chunks()), 2)
            del reopened

    @unittest.skipUnless(os.environ.get("HSD_TEST_MODEL_DIR"), "set HSD_TEST_MODEL_DIR")
    def test_hybrid_search_reopens_index_and_combines_scores(self) -> None:
        from hsd.embeddings import LocalEmbedder

        model_dir = Path(os.environ["HSD_TEST_MODEL_DIR"])
        model = LocalEmbedder(model_dir)
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "search.hsd"
            chunks = [_core.Chunk("/lexical.txt", "маяк", 0, len("маяк".encode())),
                      _core.Chunk("/semantic.txt", "сигнальная башня", 0,
                                  len("сигнальная башня".encode()))]
            vectors = [model.embed_query("другая тема"), model.embed_query("маяк")]
            store = _core.EncryptedStore(str(database), "correct", True)
            store.replace_chunks_with_embeddings(chunks, vectors, model.model_id)
            del store

            count, results = search_encrypted_hybrid(database, "correct", "маяк", model_dir, 2)
            self.assertEqual(count, 2)
            self.assertEqual({result.chunk.source_path for result in results},
                             {"/lexical.txt", "/semantic.txt"})
            lexical = next(result for result in results if result.chunk.source_path == "/lexical.txt")
            semantic = next(result for result in results if result.chunk.source_path == "/semantic.txt")
            self.assertGreater(lexical.lexical_score, 0)
            self.assertEqual(semantic.lexical_score, 0)
            self.assertGreater(semantic.vector_score, lexical.vector_score)
            self.assertGreater(lexical.hybrid_score, 0)

    def test_hybrid_search_rejects_index_without_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("private lighthouse")
            database = root / "search.hsd"
            save_index(root, database, "correct")
            with self.assertRaisesRegex(ValueError, "embeddings"):
                search_encrypted_hybrid(database, "correct", "lighthouse", root)

    @unittest.skipUnless(os.environ.get("HSD_TEST_MODEL_DIR"), "set HSD_TEST_MODEL_DIR")
    def test_vector_search_reopens_index_with_matching_model(self) -> None:
        from hsd.embeddings import LocalEmbedder

        model_dir = Path(os.environ["HSD_TEST_MODEL_DIR"])
        model = LocalEmbedder(model_dir)
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "search.hsd"
            query = "маяк на берегу"
            query_vector = model.embed_query(query)
            other_vector = [0.0] * 384
            other_vector[0] = 1.0
            store = _core.EncryptedStore(str(database), "correct", True)
            store.replace_chunks_with_embeddings(
                [_core.Chunk("/near.txt", "near", 0, 4),
                 _core.Chunk("/other.txt", "other", 0, 5)],
                [query_vector, other_vector], model.model_id)
            del store

            count, results = search_encrypted_vectors(database, "correct", query, model_dir, 1)
            self.assertEqual(count, 2)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].chunk.source_path, "/near.txt")
            self.assertAlmostEqual(results[0].vector_score, 1.0, places=5)

    @unittest.skipUnless(os.environ.get("HSD_TEST_MODEL_DIR"), "set HSD_TEST_MODEL_DIR")
    def test_vector_search_rejects_model_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "search.hsd"
            store = _core.EncryptedStore(str(database), "correct", True)
            store.replace_chunks_with_embeddings(
                [_core.Chunk("/a.txt", "text", 0, 4)],
                [[1.0] + [0.0] * 383], "different-model")
            del store
            with self.assertRaisesRegex(ValueError, "model"):
                search_encrypted_vectors(database, "correct", "query",
                                         Path(os.environ["HSD_TEST_MODEL_DIR"]))

    def test_vector_search_rejects_index_without_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("private lighthouse")
            database = root / "search.hsd"
            save_index(root, database, "correct")
            with self.assertRaisesRegex(ValueError, "embeddings"):
                search_encrypted_vectors(database, "correct", "lighthouse", root)

    def test_reopen_search_and_plain_sqlite_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "private.md"
            content = "секретный маяк".encode("utf-8")
            document.write_bytes(content)
            database = root / "search.hsd"

            self.assertEqual(save_index(root, database, "correct horse"), 1)
            index = open_index(database, "correct horse")
            self.assertEqual(index.search("МАЯК")[0].chunk.text, content.decode())
            self.assertEqual(document.read_bytes(), content)
            self.assertNotIn(content, database.read_bytes())
            self.assertNotIn(str(document).encode(), database.read_bytes())
            with closing(sqlite3.connect(database)) as connection:
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute("SELECT * FROM chunks").fetchall()

    def test_wrong_password_does_not_reveal_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("private phrase")
            database = root / "search.hsd"
            save_index(root, database, "correct")
            with self.assertRaises(Exception) as caught:
                open_index(database, "incorrect")
            self.assertIn("password", str(caught.exception).lower())
            self.assertNotIn("private phrase", str(caught.exception))
            self.assertNotIn("note.txt", str(caught.exception))

    def test_failed_reindex_preserves_previous_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "note.txt"
            document.write_text("first private phrase")
            database = root / "search.hsd"
            save_index(root, database, "correct")
            document.write_bytes(b"\xff invalid")
            with self.assertRaises(Exception):
                save_index(root, database, "correct")
            self.assertEqual(open_index(database, "correct").search("first")[0].chunk.text,
                             "first private phrase")

    def test_wrong_password_reindex_preserves_previous_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "note.txt"
            document.write_text("first private phrase")
            database = root / "search.hsd"
            save_index(root, database, "correct")
            document.write_text("second private phrase")
            with self.assertRaises(Exception):
                save_index(root, database, "wrong")
            self.assertEqual(open_index(database, "correct").search("first")[0].chunk.text,
                             "first private phrase")

    def test_embedded_nul_in_fragment_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "search.hsd"
            store = _core.EncryptedStore(str(database), "correct", True)
            store.replace_chunks([_core.Chunk("/note.txt", "before\x00after", 0, 12)])
            reopened = _core.EncryptedStore(str(database), "correct", False)
            self.assertEqual(reopened.read_chunks()[0].text, "before\x00after")
            del reopened, store

    def test_embeddings_are_encrypted_and_reopen_with_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "search.hsd"
            store = _core.EncryptedStore(str(database), "correct", True)
            chunk = _core.Chunk("/note.txt", "private lighthouse", 0, 18)
            vector = [1.0] + [0.0] * 383
            store.replace_chunks_with_embeddings([chunk], [vector], "e5-test")
            del store
            reopened = _core.EncryptedStore(str(database), "correct", False)
            self.assertEqual(reopened.read_chunks()[0].text, "private lighthouse")
            model_id, vectors = reopened.read_embeddings()
            self.assertEqual(model_id, "e5-test")
            self.assertEqual(vectors, [vector])
            self.assertNotIn(b"e5-test", database.read_bytes())
            self.assertNotIn(b"private lighthouse", database.read_bytes())
            del reopened

    def test_invalid_embeddings_preserve_previous_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "search.hsd"
            store = _core.EncryptedStore(str(database), "correct", True)
            original = _core.Chunk("/note.txt", "original", 0, 8)
            store.replace_chunks_with_embeddings([original], [[1.0] + [0.0] * 383], "e5-test")
            with self.assertRaises(Exception):
                store.replace_chunks_with_embeddings(
                    [_core.Chunk("/note.txt", "replacement", 0, 11)], [[float("nan")] * 384], "e5-test"
                )
            self.assertEqual(store.read_chunks()[0].text, "original")
            self.assertEqual(store.read_embeddings()[0], "e5-test")
            del store

    @unittest.skipUnless(os.environ.get("HSD_TEST_MODEL_DIR"), "set HSD_TEST_MODEL_DIR for the real model")
    def test_indexing_with_model_persists_document_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "note.txt").write_text("Маяк стоит на берегу.", encoding="utf-8")
            database = root / "search.hsd"
            count = save_index(root, database, "correct", model_dir=Path(os.environ["HSD_TEST_MODEL_DIR"]))
            self.assertEqual(count, 1)
            store = _core.EncryptedStore(str(database), "correct", False)
            model_id, vectors = store.read_embeddings()
            self.assertTrue(model_id.startswith("intfloat/multilingual-e5-small@"))
            self.assertEqual(len(vectors), 1)
            self.assertEqual(len(vectors[0]), 384)
            self.assertEqual(open_index(database, "correct").search("МАЯК")[0].chunk.text,
                             "Маяк стоит на берегу.")
            del store


if __name__ == "__main__":
    unittest.main()
