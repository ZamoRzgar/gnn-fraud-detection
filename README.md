# GNN-based Fraud Detection — Data Mining Course Project

**Topic:** Graph Neural Networks for Fraud/Anomaly Detection
**Task type:** Graph mining — semi-supervised node classification (imbalanced, camouflage-robust)
**Datasets:** YelpChi and Amazon fraud benchmarks (released with CARE-GNN)

## Course requirements

| Component | Weight | Due |
|---|---|---|
| Midterm project (survey) | 30% | Oct 29 |
| Final project | 60% | Jan 15 |
| — Presentation, 10 min | | Weeks 6–7 |
| — Report, 6 pages, KDD/ICDM format (LaTeX) | | Weeks 14–15 |
| — Code | | with report |
| Attendance | 10% | — |

Final project must: pick a topic, survey **≥3 papers** from top venues
(KDD, ICDM, SIGMOD, ICDE, VLDB; IEEE/ACM Transactions journals), identify a
problem under the topic, propose a solution, and implement it on **two datasets**.

## Problem

GNN fraud detectors struggle with (1) **extreme class imbalance** (fraudsters
are <5% of nodes) and (2) **camouflage** (fraudsters mimic benign behavior,
breaking homophily assumptions of standard GNNs).

## Proposed approach

Imbalance-aware GNN combining camouflage-resistant neighbor selection
(CARE-GNN) with balanced neighborhood sampling (PC-GNN), trained with a
semi-supervised objective (SemiGNN). Baselines: GCN, SemiGNN, CARE-GNN, PC-GNN.
Metrics: AUROC, AUPRC, F1-macro.

## Papers

See `papers/README.md` for the verified list and download links.

## Layout

```
papers/    paper PDFs + citations
report/    6-page LaTeX report (IEEEtran / ICDM format) + references.bib
code/      implementation (PyTorch + PyG), src/ with data/models/train/evaluate
```
