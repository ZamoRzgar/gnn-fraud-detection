"""Generate presentation figures from results/results.csv.

Usage (from code/):
    python make_figures.py

Writes PNGs (dark theme, matching the slide deck brief) to
../presentation/figures/.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_CSV = Path(__file__).parent / "results" / "results.csv"
OUT_DIR = Path(__file__).parent.parent / "presentation" / "figures"

BG = "#0f172a"        # dark navy
FG = "#e2e8f0"        # light text
GRID = "#334155"
C_GCN = "#64748b"     # slate
C_SEMI = "#38bdf8"    # sky blue
C_OURS = "#a78bfa"    # violet
C_ABL = "#f472b6"     # pink accent for ablations

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "axes.grid": True,
    "grid.color": GRID, "grid.alpha": 0.4, "font.size": 13,
    "font.family": "DejaVu Sans",
})


def load():
    with RESULTS_CSV.open() as f:
        return list(csv.DictReader(f))


def stats(rows, dataset, model, cw="1"):
    """Mean/std over seeds for a (dataset, model, class_weight) cell."""
    sel = [r for r in rows
           if r["dataset"] == dataset and r["model"] == model
           and r.get("class_weight", "1") == cw]
    out = {}
    for m in ["test_auroc", "test_auprc", "test_f1_macro"]:
        vals = [float(r[m]) for r in sel]
        out[m] = (np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0)
    return out


def fig_main(rows):
    models = [("gcn", "GCN", C_GCN), ("semignn", "SemiGNN", C_SEMI),
              ("ours", "Ours", C_OURS)]
    metrics = [("test_auroc", "AUROC"), ("test_auprc", "AUPRC"),
               ("test_f1_macro", "F1-macro")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, ds, title in zip(axes, ["yelp", "amazon"], ["YelpChi", "Amazon"]):
        x = np.arange(len(metrics))
        w = 0.26
        for i, (mkey, label, color) in enumerate(models):
            means = [stats(rows, ds, mkey)[mk][0] for mk, _ in metrics]
            stds = [stats(rows, ds, mkey)[mk][1] for mk, _ in metrics]
            bars = ax.bar(x + (i - 1) * w, means, w, label=label, color=color,
                          yerr=stds, capsize=4,
                          error_kw={"ecolor": FG, "lw": 1.2})
            for b, v in zip(bars, means):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                        ha="center", fontsize=9, color=FG)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics])
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
        ax.set_ylim(0, 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=12)
    fig.suptitle("Main results (mean ± std, 3 seeds)", fontsize=14, y=1.09)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "fig1_main_results.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def fig_ablation(rows):
    variants = [("ours", "Full model", C_OURS),
                ("ours_nofilter", "− similarity filter", C_ABL),
                ("ours_nosampler", "− balanced sampler", C_GCN)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, ds, title in zip(axes, ["yelp", "amazon"], ["YelpChi", "Amazon"]):
        x = np.arange(2)
        w = 0.26
        for i, (mkey, label, color) in enumerate(variants):
            s = stats(rows, ds, mkey)
            vals = [s["test_auroc"][0], s["test_f1_macro"][0]]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                        ha="center", fontsize=9, color=FG)
        ax.set_xticks(x)
        ax.set_xticklabels(["AUROC", "F1-macro"])
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
        ax.set_ylim(0, 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=12)
    fig.suptitle("Ablation study (seed 42)", fontsize=14, y=1.09)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "fig2_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_imbalance(rows):
    fraud = {"YelpChi": 14.5, "Amazon": 9.5}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, pct) in zip(axes, fraud.items()):
        ax.pie([pct, 100 - pct], labels=["fraud", "genuine"],
               colors=[C_OURS, GRID], autopct="%.1f%%", startangle=90,
               wedgeprops={"width": 0.42, "edgecolor": BG},
               textprops={"color": FG, "fontsize": 12})
        ax.set_title(name, fontsize=16, fontweight="bold")
    fig.suptitle("Class imbalance in the benchmarks", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_imbalance_donuts.png", dpi=200)
    plt.close(fig)


def fig_interaction(rows):
    cells = [("1", "ours", "weights ON\nsampler ON"),
             ("1", "ours_nosampler", "weights ON\nsampler OFF"),
             ("0", "ours", "weights OFF\nsampler ON"),
             ("0", "ours_nosampler", "weights OFF\nsampler OFF")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, ds, title in zip(axes, ["yelp", "amazon"], ["YelpChi", "Amazon"]):
        x = np.arange(len(cells))
        auroc = [stats(rows, ds, m, cw)["test_auroc"][0] for cw, m, _ in cells]
        f1 = [stats(rows, ds, m, cw)["test_f1_macro"][0] for cw, m, _ in cells]
        w = 0.38
        ax.bar(x - w / 2, auroc, w, label="AUROC", color=C_SEMI)
        ax.bar(x + w / 2, f1, w, label="F1-macro", color=C_OURS)
        for xi, v in zip(x - w / 2, auroc):
            ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color=FG)
        for xi, v in zip(x + w / 2, f1):
            ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color=FG)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, _, lbl in cells], fontsize=10)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
        ax.set_ylim(0, 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=12)
    fig.suptitle("Imbalance mechanisms: 2×2 interaction (seed 42)",
                 fontsize=14, y=1.09)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "fig4_interaction.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load()
    fig_main(rows)
    fig_ablation(rows)
    fig_imbalance(rows)
    fig_interaction(rows)
    print("wrote figures to", OUT_DIR)
    for p in sorted(OUT_DIR.glob("*.png")):
        print(" -", p.name)
