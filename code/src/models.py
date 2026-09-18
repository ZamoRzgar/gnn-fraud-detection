"""Model definitions.

Baselines and target models from the surveyed papers:
- GCN: standard baseline (Kipf & Welling 2017)
- SemiGNN: Wang et al., ICDM 2019 (arXiv:2003.01171)
- ImbalanceAwareGNN: proposed model combining CARE-GNN-style camouflage-
  resistant neighbor filtering (Dou et al., CIKM 2020) with PC-GNN-style
  balanced neighbor sampling (Liu et al., WWW 2021)

Model interface conventions (used by src/train.py):
- All models are called as model(x, **kwargs) and return [num_nodes, 2] logits.
- Models that consume per-relation graphs set the class attribute
  uses_relations = True and receive relation_edge_index (dict name -> [2, E]).
- Models whose aggregation is label-aware set needs_labels = True and
  additionally receive y and train_mask.
- Models may expose aux_loss(data) -> scalar, added to the CE loss in
  train.py with weight --aux_weight.
- Edge-wise computations are chunked (EDGE_CHUNK) so full-batch forward
  passes on the ~8M-edge relations stay within a few GB of RAM.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax as segment_softmax

EDGE_CHUNK = 1_000_000


def _edge_scores(edge_index, score_fn, chunk=EDGE_CHUNK):
    """Evaluate score_fn(src, dst) -> [B] over edges in chunks (bounded RAM)."""
    out = []
    for s in range(0, edge_index.shape[1], chunk):
        ei = edge_index[:, s : s + chunk]
        out.append(score_fn(ei[0], ei[1]))
    return torch.cat(out)


def _chunked_weighted_add(out, dst, src_emb, weight, chunk=EDGE_CHUNK):
    """out[u] += sum_i weight[i] * src_emb[i], grouped by dst, in chunks."""
    for s in range(0, dst.numel(), chunk):
        out.index_add_(0, dst[s : s + chunk], src_emb[s : s + chunk] * weight[s : s + chunk, None])
    return out


def _topk_mask_per_node(dst, key, k, num_nodes):
    """Boolean edge mask keeping, per center node dst, the k[dst] edges with
    the largest key. Vectorized via a single sort on (dst, key): keys are
    min-max scaled into [0, 1) so dst * 2 + key orders primarily by node.
    """
    E = dst.numel()
    if E == 0:
        return torch.zeros(0, dtype=torch.bool)
    key01 = (key - key.min()) / (key.max() - key.min() + 1e-12)
    order = torch.argsort(dst.double() * 2.0 + key01.double())
    dst_sorted = dst[order]
    deg = torch.bincount(dst, minlength=num_nodes)
    seg_end = torch.cumsum(deg, 0)
    from_end = seg_end[dst_sorted] - torch.arange(E, device=dst.device) - 1
    keep_sorted = from_end < k[dst_sorted]
    mask = torch.zeros(E, dtype=torch.bool, device=dst.device)
    mask[order[keep_sorted]] = True
    return mask


def _dot_attention_aggregate(h, edge_index, attn_mat, num_nodes):
    """Per-edge scaled dot-product scores (h_src @ attn_mat) . h_dst,
    softmax-normalized per center node, then weighted sum of neighbor
    embeddings. Shared by SemiGNN and ImbalanceAwareGNN."""
    agg = torch.zeros(num_nodes, h.shape[1], dtype=h.dtype, device=h.device)
    if edge_index.shape[1] == 0:
        return agg
    scale = 1.0 / math.sqrt(h.shape[1])

    def score_fn(src, dst):
        return ((h[src] @ attn_mat) * h[dst]).sum(-1) * scale

    scores = _edge_scores(edge_index, score_fn)
    alpha = segment_softmax(scores, edge_index[1], num_nodes=num_nodes)
    return _chunked_weighted_add(agg, edge_index[1], h[edge_index[0]], alpha)


class GCNBaseline(nn.Module):
    """2-layer GCN baseline for node classification."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


class SemiGNN(nn.Module):
    """Semi-supervised graph attentive network (Wang et al., ICDM 2019).

    Hierarchical attention over the multi-relation (multi-view) graph:

    1. Node-level attention (per view): neighbors of a node are weighted by
       a softmax over scaled dot-product scores (h_i H^v) . h_u, where H^v is
       a learnable per-view attention matrix (paper Eq. A / Eq. 1).
    2. Per-view MLP projection (paper Eq. 2, one layer) mapping view-specific
       embeddings into a common space.
    3. View-level attention (paper Eqs. 3-4): view importance from the dot
       product of the view embedding with a learnable preference vector phi^v;
       the joint embedding is the concatenation of the attention-weighted
       view embeddings.
    4. One-layer perceptron + 2-class linear head (paper Eq. 5).

    aux_loss(data) implements the paper's unsupervised graph-based loss
    (Eq. 6): embeddings of adjacent nodes are pulled together (positive
    edges) while Q=3 negative samples per edge, drawn proportionally to
    degree^0.75, are pushed apart (DeepWalk-style negative sampling).

    Simplifications vs. the paper:
    - The paper uses per-view embedding lookup tables as node input; we use a
      shared linear projection of the node features (the benchmark graphs
      provide features, and per-view N x d tables do not scale).
    - The paper's view preference vectors phi_u^v are per-user; we use global
      per-view preference vectors (per-user parameters are only meaningful in
      the transductive Alipay setting and add N*R*d parameters).
    - The paper samples positive contexts from random walks seeded at labeled
      users; we use 1-hop 'homo' edges incident to labeled (train) nodes,
      subsampled to at most max_aux_edges per call.
    """

    uses_relations = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        num_relations: int = 3,
        aux_negatives: int = 3,
        max_aux_edges: int = 200_000,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.attn_mats = nn.Parameter(
            torch.eye(hidden_dim).unsqueeze(0).repeat(num_relations, 1, 1)
        )
        self.view_mlps = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)]
        )
        self.view_pref = nn.Parameter(torch.randn(num_relations, hidden_dim) * 0.1)
        self.out_proj = nn.Linear(num_relations * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout
        self.aux_negatives = aux_negatives
        self.max_aux_edges = max_aux_edges
        self._embedding = None

    def forward(self, x, relation_edge_index):
        num_nodes = x.shape[0]
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        views = []
        for r, rel in enumerate(sorted(relation_edge_index)):
            agg = _dot_attention_aggregate(
                h, relation_edge_index[rel], self.attn_mats[r], num_nodes
            )
            views.append(F.relu(self.view_mlps[r](agg)))
        hv = torch.stack(views, dim=1)  # [N, R, H]

        beta = torch.softmax((hv * self.view_pref).sum(-1), dim=1)  # [N, R]
        joint = (beta.unsqueeze(-1) * hv).reshape(num_nodes, -1)  # [N, R*H]
        a = F.relu(self.out_proj(joint))
        a = F.dropout(a, p=self.dropout, training=self.training)
        self._embedding = a  # kept for aux_loss()
        return self.classifier(a)

    def aux_loss(self, data):
        """Semi-supervised social regularization (paper Eq. 6). Must be
        called after forward() in the same iteration; uses the cached final
        embeddings. Positive edges are 1-hop 'homo' edges touching at least
        one labeled (train) node; negatives are sampled ~ degree^0.75."""
        assert self._embedding is not None, "call forward() before aux_loss()"
        a = self._embedding
        ei = data.edge_index
        incident = data.train_mask[ei[0]] | data.train_mask[ei[1]]
        ei = ei[:, incident]
        if ei.shape[1] > self.max_aux_edges:
            sel = torch.randperm(ei.shape[1])[: self.max_aux_edges]
            ei = ei[:, sel]
        src, dst = ei[0], ei[1]

        pos = (a[src] * a[dst]).sum(-1)
        deg = torch.bincount(data.edge_index[0], minlength=a.shape[0]).float()
        neg_idx = torch.multinomial(
            deg.pow(0.75) + 1e-12, src.numel() * self.aux_negatives, replacement=True
        ).view(self.aux_negatives, -1)
        neg = (a[src].unsqueeze(0) * a[neg_idx]).sum(-1)  # [Q, E_pos]

        return -(
            F.logsigmoid(pos).mean()
            + self.aux_negatives * F.logsigmoid(-neg).mean()
        )


class ImbalanceAwareGNN(nn.Module):
    """Proposed model: imbalance- and camouflage-aware multi-relation GNN.

    Combines the two core ideas from the surveyed papers on top of a
    SemiGNN-style multi-relation backbone:

    (a) Camouflage-resistant neighbor selection (CARE-GNN, Dou et al., CIKM
        2020): per relation, a small MLP scores the similarity between each
        center node and each neighbor from their embeddings; only the top
        `keep_ratio` fraction of neighbors per node is kept, so camouflaged
        fraudsters' dissimilar (benign-looking) neighbors are dropped before
        aggregation. CARE-GNN learns per-relation filtering thresholds with
        an RL controller; we use a fixed keep-ratio per relation as a simpler,
        deterministic starting point (the RL controller can be added later).

    (b) Balanced neighbor sampling (PC-GNN, Liu et al., WWW 2021): the kept
        neighborhood is further capped at `max_degree` neighbors per node,
        sampled with a bias that favors neighbors known (from the training
        labels) to be fraud, countering the class imbalance each node sees in
        its aggregated neighborhood. Sampling uses weighted reservoir keys
        (Efraimidis-Spirakis: key = log(u) / w, u ~ Uniform) so it is
        differentiable-safe and vectorized; at eval time the sampling is
        deterministic (key = log w). This is a simplification of PC-GNN's
        two-step "pick & choose", which trains a separate neighbor predictor.

    (c) Relation attention: filtered/sampled neighborhoods are aggregated per
        relation with dot-product attention (shared helper with SemiGNN),
        combined with a residual self-connection, and fused across relations
        by a learnable attention query before the 2-class head.

    Note on training the similarity MLPs: hard top-k filtering is
    non-differentiable, so the similarity scorers cannot learn from the
    classification loss. Like CARE-GNN's label-aware similarity measure, they
    are therefore trained with label supervision instead — see aux_loss(),
    which teaches the scorers to predict whether a neighbor shares the center
    node's label (camouflaged fraud neighbors should score low).
    """

    uses_relations = True
    needs_labels = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        num_relations: int = 3,
        keep_ratio: float = 0.5,
        max_degree: int = 32,
        fraud_boost: float = 4.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        # (a) per-relation similarity scorers: [h_u | h_i] -> similarity logit
        self.sim_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_relations)
            ]
        )
        # per-relation attention matrices for aggregation
        self.attn_mats = nn.Parameter(
            torch.eye(hidden_dim).unsqueeze(0).repeat(num_relations, 1, 1)
        )
        # (c) cross-relation attention query
        self.rel_query = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout
        self.keep_ratio = keep_ratio
        self.max_degree = max_degree
        self.fraud_boost = fraud_boost
        self.max_aux_pairs = 50_000
        self._h = None

    def _filter_and_sample(self, h, edge_index, sim_mlp, num_nodes, y, train_mask):
        """Steps (a) and (b): similarity filtering, then fraud-biased
        degree-capped sampling. Returns the kept edge subset."""
        # (a) similarity filtering: keep top keep_ratio per center node
        def sim_fn(src, dst):
            return sim_mlp(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)

        sim = _edge_scores(edge_index, sim_fn)
        dst = edge_index[1]
        deg = torch.bincount(dst, minlength=num_nodes)
        keep_k = (deg.float() * self.keep_ratio).ceil().long()
        keep = _topk_mask_per_node(dst, sim, keep_k, num_nodes)
        edge_index = edge_index[:, keep]

        # (b) balanced sampling: cap degree, favor known-fraud neighbors
        if self.max_degree is not None and edge_index.shape[1] > 0:
            w = torch.ones(edge_index.shape[1], device=h.device)
            if y is not None and train_mask is not None:
                known_fraud = train_mask[edge_index[0]] & (y[edge_index[0]] == 1)
                w = w + self.fraud_boost * known_fraud.float()
            if self.training:
                key = torch.log(torch.rand(w.numel(), device=w.device) + 1e-12) / w
            else:
                key = torch.log(w)
            deg = torch.bincount(edge_index[1], minlength=num_nodes)
            cap = deg.clamp(max=self.max_degree)
            keep = _topk_mask_per_node(edge_index[1], key, cap, num_nodes)
            edge_index = edge_index[:, keep]
        return edge_index

    def forward(self, x, relation_edge_index, y=None, train_mask=None):
        num_nodes = x.shape[0]
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        self._h = h  # kept for aux_loss()

        views = []
        for r, rel in enumerate(sorted(relation_edge_index)):
            ei = self._filter_and_sample(
                h, relation_edge_index[rel], self.sim_mlps[r], num_nodes, y, train_mask
            )
            agg = _dot_attention_aggregate(h, ei, self.attn_mats[r], num_nodes)
            views.append(F.relu(agg + h))  # residual self-connection
        hv = torch.stack(views, dim=1)  # [N, R, H]

        alpha = torch.softmax(
            hv @ self.rel_query / math.sqrt(hv.shape[-1]), dim=1
        )  # [N, R]
        z = (alpha.unsqueeze(-1) * hv).sum(dim=1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.classifier(z)

    def aux_loss(self, data):
        """Label-aware similarity supervision (CARE-GNN): trains each
        relation's similarity MLP to predict whether a neighbor has the same
        label as the center node, over edges where both endpoints are labeled
        (train) nodes. Must be called after forward() in the same iteration."""
        assert self._h is not None, "call forward() before aux_loss()"
        h = self._h
        y, train_mask = data.y, data.train_mask
        losses = []
        for r, rel in enumerate(sorted(data.relation_edge_index)):
            ei = data.relation_edge_index[rel]
            both_labeled = train_mask[ei[0]] & train_mask[ei[1]]
            ei = ei[:, both_labeled]
            if ei.shape[1] == 0:
                continue
            if ei.shape[1] > self.max_aux_pairs:
                sel = torch.randperm(ei.shape[1])[: self.max_aux_pairs]
                ei = ei[:, sel]
            logit = self.sim_mlps[r](
                torch.cat([h[ei[1]], h[ei[0]]], dim=-1)
            ).squeeze(-1)
            same_label = (y[ei[1]] == y[ei[0]]).float()
            losses.append(F.binary_cross_entropy_with_logits(logit, same_label))
        return torch.stack(losses).mean()


def build_model(name: str, in_dim: int, **kwargs) -> nn.Module:
    models = {"gcn": GCNBaseline, "semignn": SemiGNN, "ours": ImbalanceAwareGNN}
    if name not in models:
        raise ValueError(f"unknown model: {name!r} (choices: {sorted(models)})")
    return models[name](in_dim, **kwargs)
