# Project Plan — GNN-based Fraud Detection

Timeline from mid-September to the final deadline (Jan 15), mapped to the
course deliverables in `README.md`.

## Phase 1 — Survey & midterm (mid-Sept → Oct 29)

Goal: midterm survey (30% of grade) due **Oct 29**.

- ~Sep 15–21: finalize topic; collect and skim core papers (GCN, SemiGNN,
  CARE-GNN, PC-GNN, GraphConsis); set up repo (`papers/`, `report/`, `code/`).
- ~Sep 22–Oct 5: deep-read ≥3 papers from top venues; write notes per paper
  (problem, method, datasets, metrics, limitations) in `papers/README.md`.
- ~Oct 6–19: draft survey section of the report (`report/main.tex`):
  problem definition, taxonomy (imbalance vs. camouflage), method comparison
  table.
- ~Oct 20–28: polish midterm survey; internal review pass; buffer.
- **Oct 29: submit midterm survey.**
- Weeks 6–7 of semester: 10-min presentation — prepare ~10 slides
  (problem, related work, proposed idea, plan) during the presentation week.

## Phase 2 — Implementation (Nov → early Jan)

### 2a. Data loader + baselines (Nov)

- Week 1 (Nov 1–7): obtain YelpChi/Amazon `.mat` files from the CARE-GNN
  release; implement `code/src/data.py` (loaders, masks, PyG `Data`).
- Week 2 (Nov 8–14): implement `code/src/evaluate.py` (AUROC, AUPRC,
  F1-macro, Recall@k) and `GCNBaseline` in `code/src/models.py`.
- Week 3 (Nov 15–21): implement `code/src/train.py` (class-weighted CE,
  best-val checkpointing, seeds); reproduce GCN baseline numbers on both
  datasets (sanity: YelpChi AUROC ≈ 0.60, Amazon ≈ 0.84).
- Week 4 (Nov 22–30): implement SemiGNN baseline; run baselines on both
  datasets; log all results.

### 2b. Proposed imbalance-aware model (Dec)

- Week 5 (Dec 1–7): CARE-GNN-style camouflage-resistant neighbor filtering
  (reinforcement-learned / similarity-based neighbor selection).
- Week 6 (Dec 8–14): PC-GNN-style balanced neighbor sampling for the
  minority class; integrate via `build_model()` (e.g. `--model ours`).
- Week 7 (Dec 15–21): combine with the semi-supervised objective; debug
  training stability; hyperparameter search on the validation split.
- Dec 22–31: buffer for implementation slip.

### 2c. Experiments + ablation (late Dec → early Jan)

- Jan 1–7: full experiment matrix: {GCN, SemiGNN, CARE-GNN, PC-GNN, ours} ×
  {YelpChi, Amazon}; ablations (no neighbor filtering, no balanced sampling,
  no class weighting); multiple seeds, report mean ± std.

## Phase 3 — Report + presentation + submission (Jan 1–15)

- Jan 1–7 (parallel with 2c): write methods/experiments sections of the
  6-page IEEEtran (ICDM-format) report; results tables and plots.
- Jan 8–12: full report draft; presentation slides for the final talk.
- Jan 13–14: proofread, reproducibility check (fresh clone → run), code
  cleanup.
- **Jan 15: submit final project (report + code).**
