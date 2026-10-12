"""Offline stage 8 quality and warm-query latency on fixed synthetic examples."""

import argparse
import json
import math
import platform
import statistics
import tempfile
import time
from pathlib import Path

from hsd import _core
from hsd.embeddings import LocalEmbedder
from hsd.search import save_index


CASES = Path(__file__).with_name("stage8_cases.json")


def evaluate_rankings(relevant: dict[str, set[str]],
                      ranked: dict[str, list[str]], k: int) -> dict[str, float]:
    """Macro average Recall@k and reciprocal first relevant rank at k."""
    if k <= 0 or not relevant or relevant.keys() != ranked.keys() or any(
        not labels for labels in relevant.values()
    ):
        raise ValueError("queries and nonempty relevance labels are required")
    recalls = []
    reciprocals = []
    for query_id, labels in relevant.items():
        top = ranked[query_id][:k]
        recalls.append(len(set(top) & labels) / len(labels))
        reciprocals.append(next((1 / rank for rank, doc_id in enumerate(top, 1)
                                 if doc_id in labels), 0.0))
    return {"recall_at_k": statistics.mean(recalls),
            "mrr_at_k": statistics.mean(reciprocals)}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))]


def run(model_dir: Path, k: int = 3, repeats: int = 10) -> dict:
    if k <= 0 or repeats <= 0:
        raise ValueError("k and repeats must be positive")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    documents = cases["documents"]
    queries = cases["queries"]
    ids = {item["id"] for item in documents}
    relevant = {item["id"]: set(item["relevant"]) for item in queries}
    if len(ids) != len(documents) or len(relevant) != len(queries) or any(
        not labels or not labels <= ids for labels in relevant.values()
    ):
        raise ValueError("invalid evaluation examples")

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        source = directory / "documents"
        source.mkdir()
        for item in documents:
            (source / (item["id"] + ".txt")).write_text(item["text"], encoding="utf-8")
        database = directory / "evaluation.hsd"
        save_index(source, database, "synthetic-evaluation-only", model_dir=model_dir)
        store = _core.EncryptedStore(str(database), "synthetic-evaluation-only", False)
        model_id, embeddings = store.read_embeddings()
        chunks = store.read_chunks()
        del store
        model = LocalEmbedder(model_dir)
        if model_id != model.model_id or len(chunks) != len(documents):
            raise ValueError("evaluation index does not match the fixed examples")
        lexical = _core.LexicalIndex()
        vector = _core.VectorIndex()
        hybrid = _core.HybridIndex()
        lexical.add_chunks(chunks)
        vector.add_chunks(chunks, embeddings)
        hybrid.add_chunks(chunks, embeddings)

        modes = {
            "lexical": lambda text: lexical.search(text, k),
            "vector": lambda text: vector.search(model.embed_query(text), k),
            "hybrid": lambda text: hybrid.search(text, model.embed_query(text), k),
        }
        metrics = {}
        for name, search in modes.items():
            rankings = {}
            latencies = []
            for item in queries:
                search(item["text"])  # Warm the model and indexes before timing.
                for _ in range(repeats):
                    start = time.perf_counter_ns()
                    results = search(item["text"])
                    latencies.append((time.perf_counter_ns() - start) / 1_000_000)
                rankings[item["id"]] = [Path(row.chunk.source_path).stem for row in results]
            values = evaluate_rankings(relevant, rankings, k)
            metrics[name] = {
                **values,
                "median_query_ms": statistics.median(latencies),
                "p95_query_ms": _percentile(latencies, 0.95),
                "rankings": rankings,
            }
    return {"platform": platform.platform(), "documents": len(documents),
            "queries": len(queries), "k": k, "repeats_per_query": repeats,
            "modes": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        help="also write the complete results as UTF-8 JSON")
    options = parser.parse_args()
    serialized = json.dumps(run(options.model_dir, options.k, options.repeats),
                            ensure_ascii=False, indent=2)
    print(serialized)
    if options.output:
        options.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
