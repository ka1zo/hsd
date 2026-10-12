import tempfile
import unittest
from pathlib import Path

from hsd.search import build_index


class SearchServiceTests(unittest.TestCase):
    def test_indexes_more_than_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number in range(3):
                (root / f"{number}.txt").write_text(f"word{number}", encoding="utf-8")

            index = build_index(root, batch_size=2)

            self.assertEqual(index.size, 3)
            results = index.search("word2")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].chunk.source_path, str((root / "2.txt").resolve()))

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                build_index(Path(temporary), batch_size=0)


if __name__ == "__main__":
    unittest.main()
