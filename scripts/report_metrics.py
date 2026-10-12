"""Render the fixed retrieval evaluation in a GitHub job summary."""

import argparse
import json
import os
from pathlib import Path


def render(result: dict) -> str:
    modes = result["modes"]
    lines = [
        f"## HSD retrieval evaluation — {result['platform']}",
        "",
        f"{result['documents']} documents, {result['queries']} queries, "
        f"k={result['k']}, {result['repeats_per_query']} repeats per query.",
        "",
        "| Mode | Recall@k | MRR@k | Median (ms) | p95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("lexical", "vector", "hybrid"):
        data = modes[name]
        lines.append(f"| {name} | {data['recall_at_k']:.4f} | {data['mrr_at_k']:.4f} "
                     f"| {data['median_query_ms']:.3f} | {data['p95_query_ms']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    options = parser.parse_args()
    summary = render(json.loads(options.result.read_text(encoding="utf-8")))
    print(summary, end="")
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as output:
            output.write(summary)


if __name__ == "__main__":
    main()
