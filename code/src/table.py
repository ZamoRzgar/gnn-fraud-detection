"""Build the results comparison table from results/results.csv.

Usage (from code/):
    python -m src.table            # markdown table, best run per (dataset, model)
    python -m src.table --all      # every run, newest last
"""

import argparse
import csv
from pathlib import Path

RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "results.csv"

METRICS = ["test_auroc", "test_auprc", "test_f1_macro", "test_recall@100"]


def load_rows():
    if not RESULTS_CSV.exists():
        raise SystemExit(f"no results yet: {RESULTS_CSV} not found (run src.train first)")
    with RESULTS_CSV.open() as f:
        return list(csv.DictReader(f))


def best_rows(rows):
    """Keep the run with the highest test AUROC per (dataset, model)."""
    best = {}
    for r in rows:
        key = (r["dataset"], r["model"])
        if key not in best or float(r["test_auroc"]) > float(best[key]["test_auroc"]):
            best[key] = r
    return sorted(best.values(), key=lambda r: (r["dataset"], r["model"]))


def print_table(rows):
    header = ["dataset", "model", "seed", "aux_w"] + METRICS
    cols = {"dataset": "dataset", "model": "model", "seed": "seed", "aux_w": "aux_weight",
            **{m: m for m in METRICS}}
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r[cols[h]] for h in header) + " |")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="show every run, not just best per (dataset, model)")
    args = p.parse_args()
    rows = load_rows()
    print_table(rows if args.all else best_rows(rows))


if __name__ == "__main__":
    main()
