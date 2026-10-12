"""Checks for the explicit CI model fetch and benchmark summary tools."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import fetch_model, report_metrics


class FetchModelTests(unittest.TestCase):
    def test_download_checks_hash_before_replacing_model(self) -> None:
        payload = b"verified model"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.onnx"
            target.write_bytes(b"old model")
            with patch.object(fetch_model, "urlopen", return_value=io.BytesIO(payload)):
                fetch_model.download_one("https://example.test/model", target,
                                         hashlib.sha256(payload).hexdigest())
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(list(target.parent.glob("*.partial")), [])

    def test_bad_hash_keeps_old_file_and_removes_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.onnx"
            target.write_bytes(b"old model")
            with patch.object(fetch_model, "urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    fetch_model.download_one("https://example.test/model", target,
                                             hashlib.sha256(b"expected").hexdigest())
            self.assertEqual(target.read_bytes(), b"old model")
            self.assertEqual(list(target.parent.glob("*.partial")), [])


class ReportMetricsTests(unittest.TestCase):
    def test_summary_contains_all_modes_and_measured_values(self) -> None:
        result = {"platform": "test-os", "documents": 12, "queries": 8, "k": 3,
                  "repeats_per_query": 10, "modes": {
                      mode: {"recall_at_k": 0.5, "mrr_at_k": 0.25,
                             "median_query_ms": 1.234, "p95_query_ms": 2.345}
                      for mode in ("lexical", "vector", "hybrid")}}
        summary = report_metrics.render(result)
        self.assertIn("test-os", summary)
        self.assertIn("12 documents", summary)
        self.assertIn("hybrid | 0.5000 | 0.2500 | 1.234 | 2.345", summary)


if __name__ == "__main__":
    unittest.main()
