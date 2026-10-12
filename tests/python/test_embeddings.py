import math
import os
import socket
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from hsd.embeddings import LocalEmbedder, ModelError, mean_pool_normalize

ML_AVAILABLE = np is not None and find_spec("onnxruntime") is not None and find_spec("tokenizers") is not None


class EmbeddingTests(unittest.TestCase):
    @unittest.skipUnless(ML_AVAILABLE, "install the ml extra")
    def test_mean_pool_ignores_padding_and_returns_unit_vectors(self) -> None:
        tokens = np.array([[[3.0, 4.0], [0.0, 10.0], [100.0, 100.0]]], dtype=np.float32)
        mask = np.array([[1, 1, 0]], dtype=np.int64)
        result = mean_pool_normalize(tokens, mask)
        self.assertEqual(result.shape, (1, 2))
        self.assertAlmostEqual(float(result[0, 0]), 3 / math.sqrt(205), places=6)
        self.assertAlmostEqual(float(result[0, 1]), 14 / math.sqrt(205), places=6)

    def test_missing_model_does_not_attempt_download(self) -> None:
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network used")):
            with self.assertRaises(ModelError):
                LocalEmbedder(Path("/definitely/missing/hsd-model"))

    @unittest.skipUnless(ML_AVAILABLE, "install the ml extra")
    def test_modified_model_files_are_rejected_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "model.onnx").write_bytes(b"changed model")
            (directory / "tokenizer.json").write_bytes(b"changed tokenizer")
            with self.assertRaisesRegex(ModelError, "SHA-256"):
                LocalEmbedder(directory)

    @unittest.skipUnless(ML_AVAILABLE and os.environ.get("HSD_TEST_MODEL_DIR"),
                         "install ml extra and set HSD_TEST_MODEL_DIR")
    def test_real_model_embeds_query_and_passages_offline(self) -> None:
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network used")):
            model = LocalEmbedder(Path(os.environ["HSD_TEST_MODEL_DIR"]))
            query = model.embed_query("где находится маяк?")
            passages = model.embed_passages(["Маяк стоит на берегу.", "The lighthouse is on the shore."])
        self.assertEqual(len(query), 384)
        self.assertEqual(len(passages), 2)
        for vector in [query, *passages]:
            self.assertEqual(len(vector), 384)
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0, places=5)
        self.assertNotEqual(query, passages[0])


if __name__ == "__main__":
    unittest.main()
