# Slide deck brief — Data Mining course project presentation (10 minutes)

> Instructions for the AI designer: Create a ~12-slide academic-tech
> presentation with a MODERN, market-current design language (2025 style):
> dark navy/charcoal background with one vibrant accent gradient
> (electric blue → violet), glassmorphism cards for content blocks, large
> bold typography (e.g., Inter/Space Grotesk), generous whitespace, and
> data-viz-first layouts — every quantitative claim gets a chart, not a
> bullet. Use these dynamics where the tool supports them: animated bar/line
> charts that build on click, a node-link graph diagram with subtle motion,
> morph/fade transitions between slides, and step-by-step reveals for the
> model pipeline. Max 5 short bullets per slide; numbers big and highlighted.
> Audience: data mining professor and classmates.
> Presenter: Zamo Rzgar Ahmed, Software Engineering, Beihang University.
> Course instructor: Prof. Richong Zhang.

## Slide 1 — Title
- Title: "Graph Neural Networks for Fraud Detection: A Survey and an
  Imbalance-Aware Approach"
- Subtitle: Data Mining Course Project — Midterm Presentation
- Name, affiliation, email, date

## Slide 2 — Motivation: why fraud detection on graphs?
- Online platforms lose money and trust to fake reviews and fraudulent accounts
- Fraudsters act in groups and mimic normal users ("camouflage")
- Key idea: users, reviews, products form a GRAPH — relationships reveal fraud
- Visual: animated node-link diagram of a review graph (users–reviews–
  products); fraud ring nodes glow red and pulse; edges draw in on click

## Slide 3 — Problem definition
- Task: semi-supervised node classification — label each node fraud (1) or genuine (0)
- Only a small fraction of nodes are labeled
- Challenge 1: extreme class imbalance (fraud < 15% of nodes)
- Challenge 2: camouflage — fraudsters connect to honest entities to look normal
- Visual: donut chart animating to show the 15%/85% class split; small graph
  inset with one red (fraud) node among gray nodes; the two challenges appear
  as glass cards, one per click

## Slide 4 — Survey: structure-based methods
- FRAUDAR (KDD 2016): finds dense subgraphs = lockstep fraud groups
- Robust to camouflage, near-linear scalability, provable bounds
- Limitation: ignores node features and labels
- Visual: adjacency matrix heatmap with a highlighted dense block;
  camouflage edges animate in (fraud nodes sprouting links to honest nodes)
  to show how the block "dissolves"

## Slide 5 — Survey: GNN-based methods
- SemiGNN (ICDM 2019): multi-view attention + semi-supervised loss (financial fraud)
- CARE-GNN (CIKM 2020): label-aware similarity + RL neighbor selection against
  camouflage; released the YelpChi & Amazon benchmarks
- Limitation: RL selector is costly/unstable; imbalance handled only implicitly

## Slide 6 — Survey: imbalance-aware methods + gap
- PC-GNN (WWW 2021): "pick and choose" — label-balanced training samples +
  similarity-based neighbor sampling
- Comparison table (from the report): method × camouflage × imbalance
- GAP: no method handles both camouflage and imbalance in a simple, stable way

## Slide 7 — Proposed method (overview)
- Our model: imbalance-aware multi-relation GNN, three components:
  A) similarity filtering of camouflaged neighbors (replaces RL with a fixed
     label-supervised filter)
  B) balanced neighborhood sampling (fraud-favoring, cap 32 neighbors)
  C) relation-wise attention aggregation + fusion
- Visual: horizontal pipeline diagram that builds step-by-step on click:
  multi-relation graph → similarity filter (edges to dissimilar neighbors
  fade out) → balanced sampler (fraud neighbors highlighted) → attention
  aggregation (edges thicken by attention weight) → fraud score gauge

## Slide 8 — Experimental setup
- Datasets: YelpChi (45,954 nodes, 14.5% fraud) and Amazon (11,944 nodes, 9.5% fraud)
- 3 relations per dataset; 40/20/40 stratified train/val/test splits
- Metrics: AUROC, AUPRC, F1-macro, Recall@100
- Baselines: GCN, SemiGNN; 3 random seeds, mean ± std

## Slide 9 — Main results
- Results table (YelpChi): GCN 0.585, SemiGNN 0.801, Ours 0.854 AUROC —
  best on all metrics
- Results table (Amazon): SemiGNN best AUROC (0.959), Ours best F1 (0.904)
- Takeaway: relation-aware models beat vanilla GCN by a huge margin; ours wins
  YelpChi decisively
- Visual: two grouped bar charts (one per dataset), bars rise on click;
  our model's bar uses the accent gradient and a subtle glow; values labeled
  on top. Optionally a second view: radar/spider chart of AUROC/AUPRC/F1
  per model

## Slide 10 — Ablation & interaction study (what we learned)
- Removing similarity filter: −4.5 AUROC on YelpChi → filter is the key component
- Removing balanced sampling: no change (< 0.1 AUROC) → negative result
- 2×2 study (sampling × class-weighted loss): loss-level reweighting is what
  actually handles imbalance
- Message: honest ablations tell us which ideas actually work
- Visual: "component knockout" graphic — full model as 3 blocks; the ablated
  block greys out and the AUROC number visibly drops (animated count-down
  0.856 → 0.811); the sampler block greys out and the number stays still —
  the contrast is the punchline

## Slide 11 — Work done so far & next steps
- Done: survey of 5 papers (KDD/ICDM/CIKM/WWW/TKDE), full implementation
  (data loader, 3 models, metrics), 26 experiment runs, report draft in ICDM
  LaTeX format, code on GitHub
- Next: multi-seed ablations, hyperparameter tuning (sampling boost, class
  weights), possibly a third dataset (Elliptic), final report (6 pages)

## Slide 12 — Thank you / Questions
- Contact email
- Link to GitHub repo (ZamoRzgar/gnn-fraud-detection)

## Speaker notes (not on slides)
- Timing: ~45 s per slide; spend most time on slides 7, 9, 10
- Be ready to explain: why GCN fails (homophily assumption), what the
  similarity filter does, why the negative sampling result is still valuable
- Backup numbers: YelpChi fraud rate 14.5%, Amazon 9.5%; training ~15–60 min
  per run on CPU
