import unittest

from hsd import _core


class BindingTests(unittest.TestCase):
    def test_hybrid_binding_returns_fused_and_source_scores(self) -> None:
        index = _core.HybridIndex()
        x = [1.0] + [0.0] * 383
        y = [0.0, 1.0] + [0.0] * 382
        index.add_chunks([_core.Chunk("/a.txt", "apple", 0, 5),
                          _core.Chunk("/b.txt", "banana", 2, 8)], [y, x])
        results = index.search("apple", x, 2)
        self.assertEqual(index.size, 2)
        self.assertEqual(results[0].chunk.source_path, "/a.txt")
        self.assertGreater(results[0].lexical_score, 0)
        self.assertEqual(results[0].vector_score, 0)
        self.assertGreater(results[0].hybrid_score, results[1].hybrid_score)
        self.assertEqual(results[1].lexical_score, 0)
        self.assertEqual(results[1].vector_score, 1)

    def test_vector_search_exposes_rank_and_source(self) -> None:
        index = _core.VectorIndex()
        x = [1.0] + [0.0] * 383
        y = [0.0, 1.0] + [0.0] * 382
        index.add_chunks([_core.Chunk("/docs/a.txt", "first", 2, 7),
                          _core.Chunk("/docs/b.txt", "second", 4, 10)], [x, y])
        results = index.search(x, 1)
        self.assertEqual(index.size, 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.source_path, "/docs/a.txt")
        self.assertEqual(results[0].chunk.start_byte, 2)
        self.assertAlmostEqual(results[0].vector_score, 1.0)

    def test_batch_round_trip_preserves_search_result(self) -> None:
        index = _core.LexicalIndex()
        index.add_chunks(
            [
                _core.Chunk("/docs/a.txt", "Привет, мир!", 7, 28),
                _core.Chunk("/docs/b.md", "Другой документ", 3, 34),
            ]
        )

        self.assertEqual(index.size, 2)
        results = index.search("ПРИВЕТ")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.source_path, "/docs/a.txt")
        self.assertEqual(results[0].chunk.text, "Привет, мир!")
        self.assertEqual((results[0].chunk.start_byte, results[0].chunk.end_byte), (7, 28))
        self.assertGreater(results[0].lexical_score, 0)


if __name__ == "__main__":
    unittest.main()
