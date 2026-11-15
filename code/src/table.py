"""Build the results comparison table from results/results.csv.

Usage (from code/):
    python -m src.table            # markdown table, best run per (dataset, model)
    python -m src.table --all      # every run, newest last
    python -m src.table --meanstd  # mean ± std across seeds per (dataset, model)
"""

import argparse
import csv
import statistics
from pathlib import Path

RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "results.csv"

METRICS = ["test_auroc", "test_auprc", "test_f1_macro", "test_recall@100"]


def load_rows():
    if not RESULTS_CSV.exists():
        raise SystemExit(f"no results yet: {RESULTS_CSV} not found (run src.train first)")
    with RESULTS_CSV.open() as f:
        return list(csv.DictReader(f))


def _group_key(r):
    """Grouping key; old rows without rho/loss columns get the defaults."""
    return (
        r["dataset"],
        r["model"],
        r.get("class_weight", "1"),
        r.get("rho", "0.5"),
        r.get("loss", "ce"),
    )


def best_rows(rows):
    """Keep the run with the highest test AUROC per group (dataset, model,
    class_weight, rho, loss)."""
    best = {}
    for r in rows:
        key = _group_key(r)
        if key not in best or float(r["test_auroc"]) > float(best[key]["test_auroc"]):
            best[key] = r
    return sorted(best.values(), key=_group_key)


def print_table(rows):
    header = ["dataset", "model", "seed", "aux_w", "cw", "rho", "loss"] + METRICS
    cols = {"dataset": "dataset", "model": "model", "seed": "seed", "aux_w": "aux_weight",
            "cw": "class_weight", "rho": "rho", "loss": "loss", **{m: m for m in METRICS}}
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r.get(cols[h], "") for h in header) + " |")
    print("\n".join(lines))


def meanstd_rows(rows):
    """Aggregate runs across seeds: mean ± std per group (dataset, model,
    class_weight, rho, loss)."""
    groups = {}
    for r in rows:
        groups.setdefault(_group_key(r), []).append(r)
    out = []
    for (dataset, model, cw, rho, loss), rs in sorted(groups.items()):
        row = {"dataset": dataset, "model": model, "cw": cw, "rho": rho,
               "loss": loss, "n": str(len(rs))}
        for m in METRICS:
            vals = [float(r[m]) for r in rs]
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row[m] = f"{statistics.mean(vals):.4f}±{std:.4f}"
        out.append(row)
    return out


def print_meanstd(rows):
    header = ["dataset", "model", "cw", "rho", "loss", "n"] + METRICS
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r[h] for h in header) + " |")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="show every run, not just best per (dataset, model)")
    p.add_argument("--meanstd", action="store_true", help="mean ± std across seeds per (dataset, model)")
    args = p.parse_args()
    rows = load_rows()
    if args.meanstd:
        print_meanstd(meanstd_rows(rows))
    else:
        print_table(rows if args.all else best_rows(rows))


if __name__ == "__main__":
    main()
