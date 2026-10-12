import unittest

from benchmarks.stage8_eval import evaluate_rankings


class EvaluationMetricTests(unittest.TestCase):
    def test_recall_and_mrr_at_k_use_annotated_relevance(self) -> None:
        relevant = {"q1": {"a", "b"}, "q2": {"c"}}
        ranked = {"q1": ["a", "x", "b"], "q2": ["x", "c", "y"]}
        at_two = evaluate_rankings(relevant, ranked, 2)
        self.assertAlmostEqual(at_two["recall_at_k"], 0.75)
        self.assertAlmostEqual(at_two["mrr_at_k"], 0.75)
        at_one = evaluate_rankings(relevant, ranked, 1)
        self.assertAlmostEqual(at_one["recall_at_k"], 0.25)
        self.assertAlmostEqual(at_one["mrr_at_k"], 0.5)

    def test_metrics_reject_missing_or_empty_labels(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_rankings({"q": {"a"}}, {}, 1)
        with self.assertRaises(ValueError):
            evaluate_rankings({"q": set()}, {"q": ["a"]}, 1)
        with self.assertRaises(ValueError):
            evaluate_rankings({"q": {"a"}}, {"q": ["a"]}, 0)


if __name__ == "__main__":
    unittest.main()
