"""
SceneGuided_ConGNN 
================================================
Group Emotion Recognition with:
  - Hybrid spatial+semantic face graph (GATv2)
  - Scene-guided emotion contagion (cross-modal attention + per-node gate)
  - Adaptive gated fusion (learned softmax weights per sample)
  - Multi-task Focal Loss + auxiliary branch losses

Training protocol:
  - 10 independent runs, full reset per run
  - Reports Mean ± Std over runs
  - Supports GAF2/GAF3 (no Test split) and GroupEmoW (has Test split)

Datasets expected layout:
  face_dir/   {split}/{class}/*.npz   → keys: features (N,4096), boxes (N,4)
  scene_dir/  {split}/**/*.npy        → shape (1024,)
  object_dir/ {split}/{class}/*.npz   → keys: features (M,2048), boxes (M,4)
"""

from __future__ import annotations

import gc
import os
import glob
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Paths (override per dataset) ─────────────────────────────────────────
    face_dir:   str = ""
    scene_dir:  str = ""
    object_dir: str = ""
    output_dir: str = "/kaggle/working/outputs"

    # ── Feature dims ─────────────────────────────────────────────────────────
    face_dim:   int = 4096
    scene_dim:  int = 1024
    object_dim: int = 2048

    # ── Model ─────────────────────────────────────────────────────────────────
    gat_hidden:          int   = 256
    num_classes:         int   = 3
    dropout:             float = 0.6
    red_dropout:         float = 0.5
    clf_w_dropout:       float = 0.7
    attention_dropout:   float = 0.5
    knn_k:               int   = 3
    dense_fallback_k:    int   = 4   # use dense graph when n_nodes <= this
    max_faces:           Optional[int] = None

    # ── Hybrid graph flags ────────────────────────────────────────────────────
    use_semantic_graph: bool  = False
    k_semantic:         int   = 2
    sim_threshold:      float = 0.6
    drop_edge_rate:     float = 0.0

    # ── Architecture flags (ablation) ─────────────────────────────────────────
    use_attn_pool_context: bool = True
    use_gated_fusion:      bool = True

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:    int   = 32
    lr:            float = 3e-5
    weight_decay:  float = 0.05
    grad_clip:     float = 0.5
    epochs:        int   = 50
    patience:      int   = 999   # set low to enable early stopping

    # ── Branch loss schedule ─────────────────────────────────────────────────
    branch_w_max:  float = 0.30
    branch_w_min:  float = 0.05
    warmup_epochs: int   = 20
    decay_end:     int   = 80

    # ── Loss weights ─────────────────────────────────────────────────────────
    class_counts:  List[int] = field(default_factory=lambda: [1, 1, 1])
    neutral_w:     float = 1.8
    positive_w:    float = 1.3

    # ── Experiment ────────────────────────────────────────────────────────────
    n_runs:       int  = 10
    use_ram_cache: bool = True
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH UTILITIES  (fully vectorized, no Python node-loops at inference)
# ─────────────────────────────────────────────────────────────────────────────

def build_dense_edges(n: int) -> Tensor:
    """Return all (i→j, i≠j) edge pairs for a fully-connected graph of n nodes."""
    if n <= 1:
        return torch.zeros(2, 1, dtype=torch.long)
    idx = torch.arange(n)
    src, dst = torch.meshgrid(idx, idx, indexing="ij")
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)  # (2, n*(n-1))


def build_knn_edges(boxes: np.ndarray, k: int, dense_fallback_k: int) -> Tensor:
    """
    Spatial k-NN edge index from bounding-box centroids.

    Args:
        boxes: (N, 4) array [x1, y1, x2, y2]
        k: number of spatial neighbours
        dense_fallback_k: use dense graph when N <= this threshold

    Returns:
        edge_index: (2, E) LongTensor
    """
    n = len(boxes)
    if n <= 1:
        return torch.zeros(2, 1, dtype=torch.long)
    if n <= dense_fallback_k:
        return build_dense_edges(n)

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers = np.stack([cx, cy], axis=1)               # (N, 2)

    # Vectorized pairwise L2 — no Python loop over nodes
    diff = centers[:, None, :] - centers[None, :, :]   # (N, N, 2)
    dist = np.sqrt((diff ** 2).sum(-1))                 # (N, N)
    np.fill_diagonal(dist, np.inf)

    actual_k = min(k, n - 1)
    nn_idx = np.argsort(dist, axis=1)[:, :actual_k]    # (N, k)

    src = np.repeat(np.arange(n), actual_k)
    dst = nn_idx.ravel()
    pairs = set(zip(src.tolist(), dst.tolist()))
    pairs |= {(d, s) for s, d in pairs}                # undirected
    ei = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    return ei


def build_semantic_edges(
    features: np.ndarray,
    k_semantic: int,
    sim_threshold: float,
) -> Optional[Tensor]:
    """
    Semantic k-NN edges via cosine similarity of fine-tuned face features.

    Motivation (emotional contagion): individuals with similar affective
    representations tend to synchronise emotions regardless of spatial distance
    [Kelly & Barsade, 2001].

    Args:
        features:      (N, D) face feature matrix
        k_semantic:    max semantic neighbours per node
        sim_threshold: cosine similarity threshold τ

    Returns:
        edge_index (2, E) or None if no edge passes threshold
    """
    n = len(features)
    if n <= 1:
        return None

    norm   = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = features / norm                                  # (N, D)
    sim    = normed @ normed.T                                # (N, N) — vectorized
    np.fill_diagonal(sim, -1.0)

    actual_k = min(k_semantic, n - 1)
    top_k    = np.argsort(sim, axis=1)[:, -actual_k:]        # (N, k)

    src_list, dst_list = [], []
    for i in range(n):
        for j in top_k[i]:
            if sim[i, j] > sim_threshold:
                src_list.extend([i, j])
                dst_list.extend([j, i])

    if not src_list:
        return None
    edges = torch.tensor([src_list, dst_list], dtype=torch.long)
    return torch.unique(edges, dim=1)


def build_hybrid_edges(
    features: np.ndarray,
    boxes: np.ndarray,
    k_spatial: int,
    k_semantic: int,
    sim_threshold: float,
    dense_fallback_k: int,
) -> Tensor:
    """Union of spatial k-NN and semantic k-NN edge sets (deduplicated)."""
    n = len(features)
    if n <= 1:
        return torch.zeros(2, 1, dtype=torch.long)
    if n <= dense_fallback_k:
        return build_dense_edges(n)

    spatial_ei  = build_knn_edges(boxes, k_spatial, dense_fallback_k)
    semantic_ei = build_semantic_edges(features, k_semantic, sim_threshold)

    if semantic_ei is None:
        return spatial_ei

    combined = torch.cat([spatial_ei, semantic_ei], dim=1)
    return torch.unique(combined, dim=1)


def drop_edge(edge_index: Tensor, rate: float, training: bool) -> Tensor:
    """DropEdge regularization [Rong et al., ICLR 2020]."""
    if not training or rate <= 0.0 or edge_index.size(1) == 0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) >= rate
    if keep.sum() == 0:
        keep[0] = True
    return edge_index[:, keep]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "negative": 0, "neutral": 1, "positive": 2,
    "Negative": 0, "Neutral": 1, "Positive": 2,
}


class GERDataset(Dataset):
    """
    Unified dataset for GAF2 / GAF3 / GroupEmoW.

    Directory layout expected:
        face_dir/{split}/{class}/*.npz   (keys: features, boxes)
        scene_dir/{split}/**/*.npy
        object_dir/{split}/{class}/*.npz (keys: features, boxes)
    """

    def __init__(self, cfg: Config, split: str) -> None:
        self.cfg       = cfg
        self.max_faces = cfg.max_faces

        # ── Face files ───────────────────────────────────────────────────────
        pattern = os.path.join(cfg.face_dir, split, "**", "*.npz")
        self.face_files: List[str] = glob.glob(pattern, recursive=True)
        if not self.face_files:
            raise RuntimeError(f"No face .npz found under {cfg.face_dir}/{split}/")
        print(f"[{split}] face={len(self.face_files)}")

        # ── Scene index: stem → path ──────────────────────────────────────────
        self._scene_idx: Dict[str, str] = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(
                os.path.join(cfg.scene_dir, split, "**", "*.npy"), recursive=True
            )
        }

        # ── Object index: stem → path ─────────────────────────────────────────
        self._obj_idx: Dict[str, str] = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(
                os.path.join(cfg.object_dir, split, "**", "*.npz"), recursive=True
            )
        }
        print(f"[{split}] scene={len(self._scene_idx)} object={len(self._obj_idx)}")

        # ── Optional RAM cache ────────────────────────────────────────────────
        self._cache: Optional[List] = None
        if cfg.use_ram_cache:
            self._cache = [
                self._load(i)
                for i in tqdm(range(len(self.face_files)), desc=f"Cache {split}")
            ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_face(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        data     = np.load(path)
        feats_np = data["features"]
        boxes_np = data["boxes"]
        if self.max_faces is not None:
            feats_np = feats_np[: self.max_faces]
            boxes_np = boxes_np[: self.max_faces]
        return feats_np, boxes_np

    def _load_scene(self, stem: str) -> np.ndarray:
        p = self._scene_idx.get(stem)
        if p is None:
            return np.zeros(self.cfg.scene_dim, dtype=np.float32)
        feat = np.load(p)
        feat = feat.mean(axis=0).flatten() if feat.ndim >= 2 else feat.flatten()
        return feat[: self.cfg.scene_dim].astype(np.float32)

    def _load_object(self, stem: str) -> np.ndarray:
        p = self._obj_idx.get(stem)
        if p is None:
            return np.zeros((1, self.cfg.object_dim), dtype=np.float32)
        d = np.load(p)
        return (d["features"] if "features" in d else d[d.files[0]]).astype(np.float32)

    def _build_face_edges(self, feats_np: np.ndarray, boxes_np: np.ndarray) -> Tensor:
        cfg = self.cfg
        if cfg.use_semantic_graph:
            return build_hybrid_edges(
                feats_np, boxes_np,
                k_spatial=cfg.knn_k,
                k_semantic=cfg.k_semantic,
                sim_threshold=cfg.sim_threshold,
                dense_fallback_k=cfg.dense_fallback_k,
            )
        return build_knn_edges(boxes_np, cfg.knn_k, cfg.dense_fallback_k)

    def _load(self, idx: int) -> dict:
        path      = self.face_files[idx]
        stem      = os.path.splitext(os.path.basename(path))[0]
        label     = LABEL_MAP.get(os.path.basename(os.path.dirname(path)).lower(), 1)
        feats_np, boxes_np = self._load_face(path)
        o_feat    = self._load_object(stem)

        return {
            "face_x":  torch.from_numpy(feats_np).float(),
            "face_ei": self._build_face_edges(feats_np, boxes_np),
            "scene_x": torch.from_numpy(self._load_scene(stem)).float(),
            "obj_x":   torch.from_numpy(o_feat).float(),
            "obj_ei":  build_dense_edges(len(o_feat)),
            "y":       label,
        }

    def __len__(self) -> int:
        return len(self.face_files)

    def __getitem__(self, idx: int) -> dict:
        return self._cache[idx] if self._cache is not None else self._load(idx)


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE
# ─────────────────────────────────────────────────────────────────────────────

class Batch:
    """Mini-batch container with offset-adjusted edge indices."""

    __slots__ = (
        "face_x", "scene_x", "obj_x", "y",
        "face_ei", "obj_ei", "face_batch", "obj_batch",
    )

    def __init__(self, samples: List[dict]) -> None:
        self.face_x  = torch.cat([s["face_x"] for s in samples])
        self.scene_x = torch.stack([s["scene_x"] for s in samples])
        self.obj_x   = torch.cat([s["obj_x"] for s in samples])
        self.y       = torch.tensor([s["y"] for s in samples], dtype=torch.long)

        f_ei, o_ei, f_b, o_b, fp, op = [], [], [], [], 0, 0
        for i, s in enumerate(samples):
            nf = s["face_x"].size(0)
            no = s["obj_x"].size(0)
            f_ei.append(s["face_ei"] + fp)
            o_ei.append(s["obj_ei"] + op)
            f_b.append(torch.full((nf,), i, dtype=torch.long))
            o_b.append(torch.full((no,), i, dtype=torch.long))
            fp += nf
            op += no

        self.face_ei    = torch.cat(f_ei, dim=1)
        self.obj_ei     = torch.cat(o_ei, dim=1)
        self.face_batch = torch.cat(f_b)
        self.obj_batch  = torch.cat(o_b)

    def to(self, device: torch.device) -> "Batch":
        for k in self.__slots__:
            v = getattr(self, k)
            if isinstance(v, Tensor):
                setattr(self, k, v.to(device))
        return self


# ─────────────────────────────────────────────────────────────────────────────
# MODEL MODULES  (each with single responsibility)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureReducer(nn.Module):
    """
    Projects a raw modality feature into shared embedding space D.

    Math:
        h = ReLU( LN( W x + b ) ),   x ∈ R^{d_in},  h ∈ R^D

    Args:
        d_in:    input feature dimension
        d_out:   shared embedding dimension D
        dropout: dropout rate applied before linear projection
    """

    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, d_in) → (N, D)"""
        return self.net(x)


class AttentionPool(nn.Module):
    """
    Soft-attention graph pooling: learns per-node importance weights.

    Math:
        e_i  = MLP(h_i) ∈ R
        α_i  = softmax_batch(e_i)
        z    = Σ_i α_i · h_i   ∈ R^D

    Args:
        dim:     node feature dimension D
        dropout: dropout inside the attention MLP
    """

    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        Args:
            x:     (N_total, D) node features
            batch: (N_total,) graph assignment index

        Returns:
            z: (B, D) pooled representations
        """
        B      = batch.max().item() + 1
        logits = self.mlp(x)                                           # (N, 1)
        logits = logits - logits.max()                                 # numerical stability
        exp_w  = torch.exp(logits)                                     # (N, 1)
        denom  = torch.zeros(B, 1, device=x.device).scatter_add_(
            0, batch.unsqueeze(1), exp_w
        )
        alpha = exp_w / (denom[batch] + 1e-8)                         # (N, 1)
        return torch.zeros(B, x.size(1), device=x.device).scatter_add_(
            0, batch.unsqueeze(1).expand_as(x), alpha * x
        )


class MultiLayerGATv2(nn.Module):
    """
    Stacked GATv2 layers with residual connections and LayerNorm.

    Each layer:
        h^{(l+1)}_i = h^{(l)}_i + Dropout( ELU( LN( GATv2(h^{(l)}, E) ) ) )

    Supports DropEdge [Rong et al., 2020] at each layer independently.

    Args:
        dim:           node feature dimension (in = out = D)
        heads:         number of attention heads
        num_layers:    number of GATv2 layers
        dropout:       node feature dropout
        drop_edge_rate: fraction of edges to drop per layer during training
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        num_layers: int,
        dropout: float,
        drop_edge_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers     = nn.ModuleList([
            GATv2Conv(dim, dim // heads, heads=heads, dropout=0.3)
            for _ in range(num_layers)
        ])
        self.norms          = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.drop           = nn.Dropout(dropout)
        self.drop_edge_rate = drop_edge_rate

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x:          (N, D) node features
            edge_index: (2, E) edge index

        Returns:
            h: (N, D) updated node features
        """
        h = x
        for gat, norm in zip(self.layers, self.norms):
            ei = drop_edge(edge_index, self.drop_edge_rate, self.training)
            h  = h + self.drop(F.elu(norm(gat(h, ei))))
        return h


class SceneContagion(nn.Module):
    """
    Scene-Guided Emotion Contagion mechanism.

    Step 1 — Cross-modal grounding:
        f_s = LN( s + MHA(Q=s, K=V=[H_f; H_o]) )   ∈ R^D

    Step 2 — Per-node contagion gate:
        g_i   = σ( W_g h_i + b_g )                 ∈ R^D
        h_i  ← LN( h_i + g_i ⊙ (α ⊙ f_s) )

    The gate g_i is conditioned on each node's own representation,
    allowing selective scene absorption per individual.

    Args:
        dim:   shared embedding dimension D
        heads: number of attention heads in MHA
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.cross_attn       = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate_linear      = nn.Linear(dim, dim)
        self.contagion_alpha  = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.gate_linear.bias, -1.0)  # gate starts ~0.27

    def forward(
        self,
        hf: Tensor,           # (N_face, D) post-GATv2 face nodes
        ho: Tensor,           # (N_obj,  D) post-GATv2 object nodes
        ps: Tensor,           # (B, D) projected scene features
        face_batch: Tensor,   # (N_face,) batch assignment
        obj_batch:  Tensor,   # (N_obj,)  batch assignment
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            hf_updated: (N_face, D) scene-modulated face features
            fs:         (B, D)     scene embedding grounded in nodes
        """
        # ── Step 1: ground scene in all nodes ────────────────────────────────
        all_nodes  = torch.cat([hf, ho], dim=0)
        all_batch  = torch.cat([face_batch, obj_batch], dim=0)
        nodes_dense, key_mask = to_dense_batch(all_nodes, all_batch)  # (B, L, D)
        attn_out, _  = self.cross_attn(
            ps.unsqueeze(1),       # query: (B, 1, D)
            nodes_dense,           # key
            nodes_dense,           # value
            key_padding_mask=~key_mask,
        )
        fs = F.layer_norm(ps + attn_out.squeeze(1), (ps.size(-1),))   # (B, D)

        # ── Step 2: per-node gate ─────────────────────────────────────────────
        gate    = torch.sigmoid(self.gate_linear(hf))                  # (N_face, D)
        signal  = self.contagion_alpha * fs[face_batch]                # (N_face, D)
        hf_new  = F.layer_norm(hf + gate * signal, (hf.size(-1),))

        return hf_new, fs


class GatedFusion(nn.Module):
    """
    Adaptive per-sample modality gating.

    Math:
        g = softmax( MLP([z_f; z_c; z_s]) )   ∈ R^3
        z = LN( g_1·W_f·z_f + g_2·W_c·z_c + g_3·W_s·z_s )

    Args:
        dim:     embedding dimension D
        dropout: dropout in gate MLP
    """

    def __init__(self, dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.proj_f   = nn.Linear(dim, dim)
        self.proj_c   = nn.Linear(dim, dim)
        self.proj_s   = nn.Linear(dim, dim)
        self.gate_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim, 3),
        )
        self.classifier = nn.Linear(dim, 3)

    def forward(
        self,
        zf: Tensor,  # (B, D) face
        zc: Tensor,  # (B, D) context (object)
        zs: Tensor,  # (B, D) scene
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            logits: (B, num_classes)
            gates:  (B, 3) softmax gate weights for logging/ablation
        """
        gates  = torch.softmax(self.gate_mlp(torch.cat([zf, zc, zs], dim=1)), dim=-1)
        fused  = (
            gates[:, 0:1] * self.proj_f(zf)
            + gates[:, 1:2] * self.proj_c(zc)
            + gates[:, 2:3] * self.proj_s(zs)
        )
        fused  = F.layer_norm(fused, (fused.size(-1),))
        return self.classifier(fused), gates


class ConcatFusion(nn.Module):
    """
    Baseline concatenation fusion (no gating).

    Math:
        logits = MLP( LN([z_s; z_f; z_c]) )
    """

    def __init__(self, dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim, 3),
        )

    def forward(self, zf: Tensor, zc: Tensor, zs: Tensor) -> Tuple[Tensor, None]:
        return self.net(torch.cat([zs, zf, zc], dim=1)), None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SceneGuided_ConGNN(nn.Module):
    """
    SceneGuided-ConGNN: multimodal GNN for Group Emotion Recognition.

    Architecture (top-level data flow):
        face features  (N, 4096) ──► FeatureReducer ──► MultiLayerGATv2 ──┐
        scene features (B, 1024) ──► FeatureReducer ──────────────────────┼──► SceneContagion
        object features(M, 2048) ──► FeatureReducer ──► MultiLayerGATv2 ──┘
                                                                           │
                            ┌──────────────────────────────────────────────┘
                            ▼
                    AttentionPool × 3  (face, object, scene-as-is)
                            │
                    GatedFusion / ConcatFusion
                            │
                    logits (B, 3)  ← final prediction

    Branch classifiers (face / context / scene) are auxiliary outputs
    used only during training for multi-task supervision.

    Args:
        cfg: Config dataclass controlling all hyperparameters and flags.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        D   = cfg.gat_hidden
        drp = cfg.dropout
        rd  = cfg.red_dropout
        cwd = cfg.clf_w_dropout

        self.use_attn_ctx  = cfg.use_attn_pool_context
        self.use_gated     = cfg.use_gated_fusion

        # ── Reducers ──────────────────────────────────────────────────────────
        self.red_f = FeatureReducer(cfg.face_dim,   D, rd)
        self.red_o = FeatureReducer(cfg.object_dim, D, rd)
        self.red_s = FeatureReducer(cfg.scene_dim,  D, rd)

        # ── Graph encoders ────────────────────────────────────────────────────
        der = cfg.drop_edge_rate
        self.f_gat = MultiLayerGATv2(D, heads=4, num_layers=2, dropout=drp, drop_edge_rate=der)
        self.o_gat = MultiLayerGATv2(D, heads=4, num_layers=2, dropout=drp, drop_edge_rate=der)

        # ── Scene-guided contagion ────────────────────────────────────────────
        self.contagion = SceneContagion(D, heads=4)

        # ── Pooling ───────────────────────────────────────────────────────────
        self.pool_f    = AttentionPool(D, drp)
        self.pool_f_br = AttentionPool(D, drp)   # auxiliary face branch
        self.pool_o    = AttentionPool(D, drp) if self.use_attn_ctx else None

        # ── Branch classifiers (auxiliary, training only) ─────────────────────
        self.clf_f = nn.Linear(D, 3)
        self.clf_c = nn.Linear(D, 3)
        self.clf_s = nn.Linear(D, 3)

        # ── Final fusion classifier ───────────────────────────────────────────
        self.clf_w: nn.Module = (
            GatedFusion(D, dropout=cwd) if self.use_gated
            else ConcatFusion(D, dropout=cwd)
        )

    def forward(self, data: Batch) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            data: Batch object (face_x, scene_x, obj_x, *_ei, *_batch, y)

        Returns:
            out_f: (B, 3)  face branch logits
            out_c: (B, 3)  context branch logits
            out_s: (B, 3)  scene branch logits
            out_w: (B, 3)  final fused logits
        """
        # ── 1. Reduce dimensions ──────────────────────────────────────────────
        pf = self.red_f(data.face_x)    # (N_face, D)
        po = self.red_o(data.obj_x)     # (N_obj,  D)
        ps = self.red_s(data.scene_x)   # (B,      D)

        # ── 2. GATv2 message passing ──────────────────────────────────────────
        hf = self.f_gat(pf, data.face_ei)   # (N_face, D)
        ho = self.o_gat(po, data.obj_ei)    # (N_obj,  D)

        # ── 3. Auxiliary branch logits ────────────────────────────────────────
        out_f = self.clf_f(self.pool_f_br(hf, data.face_batch))

        ctx_pool = (
            self.pool_o(ho, data.obj_batch)
            if self.use_attn_ctx
            else global_mean_pool(ho, data.obj_batch)
        )
        out_c = self.clf_c(ctx_pool)
        out_s = self.clf_s(ps)

        # ── 4. Scene-guided contagion ─────────────────────────────────────────
        hf, fs = self.contagion(hf, ho, ps, data.face_batch, data.obj_batch)

        # ── 5. Final pooling ──────────────────────────────────────────────────
        zf = self.pool_f(hf + pf, data.face_batch)
        zc = (
            self.pool_o(ho + po, data.obj_batch)
            if self.use_attn_ctx
            else global_mean_pool(ho + po, data.obj_batch)
        )

        # ── 6. Fuse and classify ──────────────────────────────────────────────
        out_w, _ = self.clf_w(zf, zc, fs)

        return out_f, out_c, out_s, out_w


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

class GERCriterion(nn.Module):
    """
    Multi-task criterion for GER.

    L_total = L_focal(ŷ_w, y)  +  λ_t · [ L_CE(ŷ_f) + L_CE(ŷ_c) + L_CE(ŷ_s) ]

    where:
        L_focal  = class-weighted Focal Loss (γ=2) on the fused prediction
        L_CE     = cross-entropy with label smoothing (ε=0.1) on branch outputs
        λ_t      = branch weight annealed from λ_max to λ_min over training

    Args:
        class_counts: [n_neg, n_neu, n_pos] training sample counts
        neutral_w:    additional multiplier for neutral class weight
        positive_w:   additional multiplier for positive class weight
        gamma:        focal loss focusing parameter
        device:       torch device
    """

    def __init__(
        self,
        class_counts: List[int],
        neutral_w:  float,
        positive_w: float,
        gamma: float = 2.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.gamma = gamma

        counts = torch.tensor(class_counts, dtype=torch.float)
        w = counts.sum() / (3.0 * counts)   # inverse-frequency base weights
        w[1] *= neutral_w
        w[2] *= positive_w
        w /= w.mean()
        self.register_buffer("alpha", w.to(device))

        self.ce_branch = nn.CrossEntropyLoss(label_smoothing=0.1)

    def focal_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        ce   = F.cross_entropy(pred, target, reduction="none", label_smoothing=0.1)
        pt   = torch.exp(-ce)
        loss = self.alpha[target] * (1.0 - pt) ** self.gamma * ce
        return loss.mean()

    def forward(
        self,
        out_f: Tensor,
        out_c: Tensor,
        out_s: Tensor,
        out_w: Tensor,
        y:     Tensor,
        branch_w: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            total_loss:  scalar to backward
            whole_loss:  focal loss on ow (for logging)
            branch_loss: sum of CE branch losses (for logging)
        """
        whole_loss  = self.focal_loss(out_w, y)
        branch_loss = (
            self.ce_branch(out_f, y)
            + self.ce_branch(out_c, y)
            + self.ce_branch(out_s, y)
        )
        return whole_loss + branch_w * branch_loss, whole_loss, branch_loss


# ─────────────────────────────────────────────────────────────────────────────
# BRANCH WEIGHT SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def branch_weight(epoch: int, cfg: Config) -> float:
    """Linear annealing of branch loss weight from λ_max to λ_min."""
    if epoch < cfg.warmup_epochs:
        return cfg.branch_w_max
    if epoch >= cfg.decay_end:
        return cfg.branch_w_min
    t = (epoch - cfg.warmup_epochs) / (cfg.decay_end - cfg.warmup_epochs)
    return cfg.branch_w_max + t * (cfg.branch_w_min - cfg.branch_w_max)


# ─────────────────────────────────────────────────────────────────────────────
# ONE RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_once(
    cfg: Config,
    train_ds: GERDataset,
    val_ds:   GERDataset,
    run_id:   str,
    seed:     int,
) -> Tuple[float, dict, dict]:
    """
    Full training loop for a single run.

    Model, optimizer, and scheduler are freshly initialised from seed
    to prevent any state leakage between runs.

    Returns:
        best_val_acc:  best validation accuracy achieved
        best_state:    model state_dict at best epoch (on CPU)
        history:       dict of per-epoch metric lists
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=Batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=Batch
    )

    # ── Fresh model, optimizer, scheduler ────────────────────────────────────
    model     = SceneGuided_ConGNN(cfg).to(cfg.device)
    criterion = GERCriterion(
        cfg.class_counts, cfg.neutral_w, cfg.positive_w, device=cfg.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    history: dict = {k: [] for k in [
        "train_acc", "val_acc",
        "train_whole", "val_whole",
        "val_f", "val_c", "val_s", "bw",
    ]}

    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0

    for ep in range(cfg.epochs):
        bw = branch_weight(ep, cfg)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_whole = t_acc = t_total = 0
        for b in train_loader:
            b.to(cfg.device)
            optimizer.zero_grad()
            of, oc, os_, ow = model(b)
            loss, wl, _ = criterion(of, oc, os_, ow, b.y, bw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            t_whole += wl.item()
            t_acc   += (ow.argmax(1) == b.y).sum().item()
            t_total += b.y.size(0)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_whole = v_f = v_c = v_s = v_acc = v_total = 0
        with torch.no_grad():
            for b in val_loader:
                b.to(cfg.device)
                of, oc, os_, ow = model(b)
                n = b.y.size(0)
                _, wl, _ = criterion(of, oc, os_, ow, b.y, bw)
                v_whole += wl.item() * n
                v_f     += F.cross_entropy(of,  b.y).item() * n
                v_c     += F.cross_entropy(oc,  b.y).item() * n
                v_s     += F.cross_entropy(os_, b.y).item() * n
                v_acc   += (ow.argmax(1) == b.y).sum().item()
                v_total += n

        val_acc = v_acc / v_total
        scheduler.step()

        # ── Log ───────────────────────────────────────────────────────────────
        history["train_acc"].append(t_acc / t_total)
        history["val_acc"].append(val_acc)
        history["train_whole"].append(t_whole / len(train_loader))
        history["val_whole"].append(v_whole / v_total)
        history["val_f"].append(v_f / v_total)
        history["val_c"].append(v_c / v_total)
        history["val_s"].append(v_s / v_total)
        history["bw"].append(bw)

        print(
            f"  [{run_id}|s{seed}] Ep{ep+1:03d} "
            f"TrainAcc={t_acc/t_total:.4f} ValAcc={val_acc:.4f} "
            f"WholeLoss={v_whole/v_total:.4f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"  Early stop at ep{ep+1}")
                break

    return best_val_acc, best_state, history


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    cfg: Config,
    model: SceneGuided_ConGNN,
    dataset: GERDataset,
    split_name: str = "Val",
) -> Tuple[float, List[int], List[int]]:
    """Full evaluation with per-branch accuracy and classification report."""
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=Batch)
    model.eval()
    acc_f = acc_c = acc_s = acc_w = total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for b in loader:
            b.to(cfg.device)
            of, oc, os_, ow = model(b)
            n = b.y.size(0)
            acc_f += (of.argmax(1)  == b.y).sum().item()
            acc_c += (oc.argmax(1)  == b.y).sum().item()
            acc_s += (os_.argmax(1) == b.y).sum().item()
            acc_w += (ow.argmax(1)  == b.y).sum().item()
            total += n
            y_true.extend(b.y.cpu().tolist())
            y_pred.extend(ow.argmax(1).cpu().tolist())

    print(f"\n{'='*60}")
    print(f"[{split_name}] Face={acc_f/total:.4f} | "
          f"Ctx={acc_c/total:.4f} | Scene={acc_s/total:.4f} | "
          f"FINAL={acc_w/total:.4f}")
    print(f"{'='*60}")
    return acc_w / total, y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-RUN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    cfg: Config,
    train_ds: GERDataset,
    val_ds:   GERDataset,
    exp_id:   str,
) -> Tuple[float, float, float, List[int], List[int], dict]:
    """
    Run cfg.n_runs independent training runs, each with a fresh model.

    Returns:
        best_acc, mean_acc, std_acc, y_true, y_pred, best_history
    """
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_id}")
    print(f"  attn_ctx={'ON' if cfg.use_attn_pool_context else 'OFF'} | "
          f"gated={'ON' if cfg.use_gated_fusion else 'OFF'} | "
          f"sem={'ON' if cfg.use_semantic_graph else 'OFF'} | "
          f"de={cfg.drop_edge_rate}")
    print(f"{'='*70}")

    all_accs       = []
    best_acc_global = 0.0
    best_state_global = None
    best_history   = None

    for i in range(cfg.n_runs):
        seed = 42 + i * 7
        print(f"\n--- Run {i+1}/{cfg.n_runs}  (seed={seed}) ---")
        acc, state, hist = run_once(cfg, train_ds, val_ds, exp_id, seed)
        all_accs.append(acc)
        print(f"  → Val Acc = {acc:.4f}")

        if acc > best_acc_global:
            best_acc_global   = acc
            best_state_global = state
            best_history      = hist
            ckpt_path = os.path.join(cfg.output_dir, f"best_{exp_id}.pth")
            torch.save(state, ckpt_path)
            print(f"  🔥 New best → {ckpt_path}")

    mean_acc = float(np.mean(all_accs))
    std_acc  = float(np.std(all_accs))

    print(f"\n{'='*70}")
    print(f"  RESULTS: {exp_id}")
    print(f"  Runs   : {[f'{a:.4f}' for a in all_accs]}")
    print(f"  Mean   : {mean_acc:.4f}")
    print(f"  Std    : {std_acc:.4f}")
    print(f"  Best   : {best_acc_global:.4f}")
    print(f"{'='*70}")

    # Evaluate best model
    model = SceneGuided_ConGNN(cfg).to(cfg.device)
    model.load_state_dict({k: v.to(cfg.device) for k, v in best_state_global.items()})
    final_acc, y_true, y_pred = evaluate(cfg, model, val_ds)

    return final_acc, mean_acc, std_acc, y_true, y_pred, best_history


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    history: dict,
    y_true: List[int],
    y_pred: List[int],
    exp_id: str,
    output_dir: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(exp_id, fontsize=14, fontweight="bold")

    axes[0, 0].plot(history["val_f"],   label="Face")
    axes[0, 0].plot(history["val_c"],   label="Context")
    axes[0, 0].plot(history["val_s"],   label="Scene")
    axes[0, 0].legend(); axes[0, 0].set_title("Branch Losses (Val)")

    axes[0, 1].plot(history["train_acc"], label="Train", color="blue")
    axes[0, 1].plot(history["val_acc"],   label="Val",   color="red")
    axes[0, 1].legend(); axes[0, 1].set_title("Accuracy")

    axes[0, 2].plot(history["bw"], color="orange")
    axes[0, 2].set_title("Branch Weight Schedule")

    axes[1, 0].plot(history["train_whole"], label="Train", color="steelblue")
    axes[1, 0].plot(history["val_whole"],   label="Val",   color="crimson")
    axes[1, 0].legend(); axes[1, 0].set_title("Focal Loss — Whole")

    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1, 1],
                xticklabels=["Neg", "Neu", "Pos"],
                yticklabels=["Neg", "Neu", "Pos"])
    axes[1, 1].set_title("Confusion Matrix (%)")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.05, 0.5,
        classification_report(y_true, y_pred,
                              target_names=["Negative", "Neutral", "Positive"],
                              digits=4),
        fontsize=9, family="monospace", verticalalignment="center",
    )

    plt.tight_layout()
    path = os.path.join(output_dir, f"{exp_id}.png")
    plt.savefig(path, dpi=150)
    print(f"📊 Plot → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — GAF2 sweep (sw5–sw8)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Paths ─────────────────────────────────────────────────────────────────
    PATHS = dict(
        face_dir   = "/kaggle/input/datasets/trieung11/gaf2-face",
        scene_dir  = "/kaggle/input/datasets/trieung11/gaf2-fearture/scene_features_gaf2000_v2/scene_features_final/scenes",
        object_dir = "/kaggle/input/datasets/trieung11/gaf2-fearture/gaf2_object_features",
        output_dir = "/kaggle/working/outputs",
    )

    # ── Base config ───────────────────────────────────────────────────────────
    BASE = Config(
        **PATHS,
        gat_hidden    = 256,
        dropout       = 0.6,
        red_dropout   = 0.5,
        clf_w_dropout = 0.7,
        neutral_w     = 1.8,
        positive_w    = 1.3,
        lr            = 3e-5,
        weight_decay  = 0.05,
        epochs        = 50,
        patience      = 999,
        n_runs        = 10,
        # GAF2 Train counts: Neg=1159, Neu=1199, Pos=1272
        class_counts  = [1159, 1199, 1272],
    )

    # ── Cache datasets once — reuse across all sweeps ─────────────────────────
    print("📦 Loading datasets into RAM...")
    train_ds = GERDataset(BASE, split="Train")
    val_ds   = GERDataset(BASE, split="Val")

    # ── Sweep definitions ─────────────────────────────────────────────────────
    from dataclasses import replace

    SWEEPS = {
        "sw5_semantic": replace(
            BASE,
            use_semantic_graph    = True,
            use_attn_pool_context = False,
            use_gated_fusion      = False,
            positive_w            = 1.0,
        ),
        "sw6_full": replace(
            BASE,
            use_semantic_graph    = True,
            use_attn_pool_context = True,
            use_gated_fusion      = True,
        ),
        "sw7_dropedge": replace(
            BASE,
            use_semantic_graph    = False,
            use_attn_pool_context = False,
            use_gated_fusion      = False,
            drop_edge_rate        = 0.3,
            positive_w            = 1.0,
        ),
        "sw8_full_de": replace(
            BASE,
            use_semantic_graph    = True,
            use_attn_pool_context = True,
            use_gated_fusion      = True,
            drop_edge_rate        = 0.3,
        ),
    }

    # ── Run all sweeps ────────────────────────────────────────────────────────
    results = {}
    for exp_id, cfg in SWEEPS.items():
        final_acc, mean_acc, std_acc, y_true, y_pred, hist = run_experiment(
            cfg, train_ds, val_ds, exp_id
        )
        results[exp_id] = (final_acc, mean_acc, std_acc, y_true, y_pred, hist)
        gc.collect()
        torch.cuda.empty_cache()

    # ── Final report (Mean ± Std over 10 runs) ────────────────────────────────
    print(f"\n{'='*70}")
    print("📊 FINAL RESULTS — GAF2  (sorted by Best Val Acc)")
    print(f"{'='*70}")
    for exp_id, (facc, macc, sacc, *_) in sorted(
        results.items(), key=lambda x: -x[1][0]
    ):
        print(f"  Best={facc:.4f} | Mean={macc:.4f} ± {sacc:.4f} | {exp_id}")
    print(f"{'='*70}")

    # ── Plot best sweep ───────────────────────────────────────────────────────
    best_id, best_data = max(results.items(), key=lambda x: x[1][0])
    facc, _, _, y_true, y_pred, hist = best_data
    print(f"\n🏆 Best: {best_id}  FINAL={facc:.4f}")
    plot_results(hist, y_true, y_pred, best_id, BASE.output_dir)