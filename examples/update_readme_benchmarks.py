"""
Reads benchmark_results.json (produced by modal_smoke_benchmark.py) and
rewrites the 'Results at a Glance' section in README.md with verified numbers.

Usage:
    python examples/update_readme_benchmarks.py [--results benchmark_results.json]
"""

import json
import re
import sys
import argparse
from pathlib import Path


def _fmt(value, unit=""):
    if isinstance(value, float):
        return f"{value:.2f}{unit}"
    return f"{value}{unit}"


def build_table(results: dict) -> str:
    l4 = results.get("L4", {})
    t4 = results.get("T4", {})

    rows = [
        ("Inference Latency",    _fmt(l4.get("inference_avg_ms","n/a"), " ms"),   _fmt(t4.get("inference_avg_ms","n/a"), " ms")),
        ("Control Frequency",    _fmt(l4.get("control_hz","n/a"), " Hz"),          _fmt(t4.get("control_hz","n/a"), " Hz")),
        ("Peak Training VRAM",   _fmt(l4.get("peak_vram_alloc_gb","n/a"), " GB"),  _fmt(t4.get("peak_vram_alloc_gb","n/a"), " GB")),
        ("Training Throughput",  _fmt(l4.get("train_its","n/a"), " it/s"),          _fmt(t4.get("train_its","n/a"), " it/s")),
        ("Step Time",            _fmt(l4.get("train_step_avg_ms","n/a"), " ms"),   _fmt(t4.get("train_step_avg_ms","n/a"), " ms")),
    ]

    header = (
        "## Results at a Glance\n\n"
        "> Measured on Modal.com with `examples/modal_smoke_benchmark.py`.\n"
        "> OpenVLA-style model, batch=1 inference / batch=2 training.\n\n"
        "| Metric | NVIDIA L4 (Ada, 24 GB) | NVIDIA T4 (Turing, 16 GB) |\n"
        "| :--- | :--- | :--- |\n"
    )
    body = ""
    for name, l4v, t4v in rows:
        body += f"| {name} | **{l4v}** | **{t4v}** |\n"

    return header + body


def update_readme(results: dict, readme_path: Path):
    text = readme_path.read_text(encoding="utf-8")
    new_section = build_table(results)

    # Replace existing "Results at a Glance" section up to next "---" or "##"
    pattern = re.compile(r"## Results at a Glance.*?(?=\n---|\n## )", re.DOTALL)
    if pattern.search(text):
        updated = pattern.sub(new_section.rstrip(), text)
    else:
        updated = text.replace(
            "## Installation",
            new_section + "\n---\n\n## Installation",
        )

    readme_path.write_text(updated, encoding="utf-8")
    print(f"✅  README updated at {readme_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="benchmark_results.json")
    parser.add_argument("--readme",  default="README.md")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌  {results_path} not found. Run modal_smoke_benchmark.py first.")
        sys.exit(1)

    results = json.loads(results_path.read_text())
    update_readme(results, Path(args.readme))
    print("\nNew table:\n")
    print(build_table(results))


if __name__ == "__main__":
    main()
