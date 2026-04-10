import os
import re
import csv
import json
import time
import gc
import glob
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ============================================================
# BASE CONFIG  (giữ nguyên từ code gốc V10)
# ============================================================
CONFIG = {
    'face_dir':   '/kaggle/input/datasets/nguynnhtlam12/face-featuresv2',
    'scene_dir':  '/kaggle/input/datasets/drakhight/8726scene-features/scene_features_final/scenes',
    'object_dir': '/kaggle/input/datasets/trieung11/fearturecongnn/objects/objects',
    'output_dir': '/kaggle/working/ablation_outputs',

    'face_dim':   4096,
    'object_dim': 2048,
    'scene_dim':  1024,

    'gat_hidden':  512,
    'num_classes': 3,
    'gat_layers':  2,
    'num_heads':   4,
    'dropout':     0.5,
    'attention_dropout': 0.5,

    'knn_k': 3,
    'label_smoothing': 0.1,

    'batch_size':      32,
    'num_workers':     0,
    'prefetch_factor': None,

    'lr':           1e-5,
    'weight_decay': 1e-1,
    'grad_clip':    0.5,
    'epochs':       150,
    'patience':     40,

    'scheduler_patience': 10,
    'scheduler_factor':   0.5,
    'min_lr':             1e-6,

    'branch_w_max':  0.30,
    'branch_w_min':  0.30,
    'warmup_epochs': 20,
    'decay_end':     80,

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'debug_mode': False,
    'use_ram_cache': True,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# ============================================================
# VARIANTS REGISTRY
# ============================================================
# Mỗi entry là dict[str, bool/float] — AblationModel đọc flags này.
#
# Flags có thể dùng:
#   use_scene      : bool — có dùng nhánh scene không
#   use_obj        : bool — có dùng nhánh object không
#   use_sgf        : bool — có dùng SceneGuidedFusion không
#   use_ec         : bool — có dùng EmotionalContagion không
#   use_residual   : bool — có dùng node-level residual không
#   use_branch_loss: bool — có tính loss cho nhánh phụ không
#
# Ràng buộc tự động trong AblationModel:
#   - use_scene=False → use_sgf=False, use_ec=False (bắt buộc)
#   - use_obj=False   → use_sgf=False (vì SGF cần Key từ cả face lẫn obj)
# ============================================================
VARIANTS = {
    # ════════════════════════════════════════════════════════════════════════
    # GROUP A — MODALITY: tầm quan trọng từng luồng dữ liệu
    # ════════════════════════════════════════════════════════════════════════
    "A1_full_model": {
        "description": "Baseline đầy đủ — giữ nguyên 100% V10",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   True,  "sgf_mode": "original",
        "use_ec":    True,  "ec_mode":  "node_level",
        "use_residual": True, "use_branch_loss": True,
    },
    "A2_no_scene": {
        "description": "Bỏ nhánh Scene → SGF và EC tự động tắt",
        "use_scene": False, "use_obj": True,
        "use_sgf":   False, "sgf_mode": None,
        "use_ec":    False, "ec_mode":  None,
        "use_residual": True, "use_branch_loss": True,
    },
    "A3_no_obj": {
        "description": "Bỏ nhánh Object",
        "use_scene": True,  "use_obj": False,
        "use_sgf":   False, "sgf_mode": None,
        "use_ec":    False, "ec_mode":  None,
        "use_residual": True, "use_branch_loss": True,
    },
    "A4_face_only": {
        "description": "Chỉ Face + GATv2 — bỏ Scene và Object hoàn toàn",
        "use_scene": False, "use_obj": False,
        "use_sgf":   False, "sgf_mode": None,
        "use_ec":    False, "ec_mode":  None,
        "use_residual": True, "use_branch_loss": True,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP B — ARCHITECTURE COMPONENTS
    # ════════════════════════════════════════════════════════════════════════
    "B3_no_sgf_ec": {
        "description": "Giữ Scene nhưng tắt SGF+EC — scene thô concat cuối",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   False, "sgf_mode": None,
        "use_ec":    False, "ec_mode":  None,
        "use_residual": True, "use_branch_loss": True,
    },
    "B4_no_residual": {
        "description": "Tắt Node-level Residual Connections",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   True,  "sgf_mode": "original",
        "use_ec":    True,  "ec_mode":  "node_level",
        "use_residual": False, "use_branch_loss": True,
    },
    "B5_no_branch_loss": {
        "description": "Chỉ loss whole — bỏ branch losses",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   True,  "sgf_mode": "original",
        "use_ec":    True,  "ec_mode":  "node_level",
        "use_residual": True, "use_branch_loss": False,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP C — FUSION STRATEGY: so sánh 3 cách fuse scene với face/obj
    #
    # Lý do SGF gốc chưa hiệu quả (hypothesis):
    #   scene_proj là 1 vector duy nhất làm Query → attention weights trải đều,
    #   không đủ "góc nhìn" để attend khác nhau vào face nodes vs obj nodes.
    #
    # C1_concat: không dùng attention, chỉ concat thô → lower bound
    # C2_face_as_query: đảo vai trò, Face làm Query attend Scene → thử xem
    #                   scene nên là "context" thay vì "driver"
    # C3_sgf_expanded: fix vấn đề 1-vector-query bằng cách project scene
    #                  thành K=4 vectors (multi-query) trước khi cross-attend
    # ════════════════════════════════════════════════════════════════════════
    "C1_concat_only": {
        "description": "Fusion = simple concat (scene+face+obj) không attention",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   False, "sgf_mode": "concat",   # concat thô, không phải None
        "use_ec":    True,  "ec_mode":  "node_level",  # EC dùng scene_proj thô
        "use_residual": True, "use_branch_loss": True,
    },
    "C2_face_as_query": {
        "description": "Face làm Query, Scene làm Key-Value trong cross-attention",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   True,  "sgf_mode": "face_as_query",
        "use_ec":    True,  "ec_mode":  "node_level",
        "use_residual": True, "use_branch_loss": True,
    },
    "C3_sgf_expanded": {
        "description": "SGF cải tiến: scene → K=4 queries trước cross-attention",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   True,  "sgf_mode": "expanded",
        "use_ec":    True,  "ec_mode":  "node_level",
        "use_residual": True, "use_branch_loss": True,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP D — EC LEVEL: node-level vs graph-level (pool-level)
    #
    # D1_pool_ec: áp dụng EC SAU attention pool thay vì TRƯỚC
    #   → pool không còn "học ai phản ánh atmosphere tốt nhất"
    #   → nhưng đơn giản hơn, ít risk gradient vanishing hơn
    # ════════════════════════════════════════════════════════════════════════
    "D1_pool_level_ec": {
        "description": "EC áp dụng SAU attention pool (graph-level thay vì node-level)",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   True,  "sgf_mode": "original",
        "use_ec":    True,  "ec_mode":  "pool_level",
        "use_residual": True, "use_branch_loss": True,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP S — INCREMENTAL STACK (bottom-up contribution analysis)
    #
    # Mỗi variant thêm đúng 1 component so với variant trước.
    # Đây là cách trình bày chuẩn nhất trong paper/luận văn:
    #   "Each component contributes positively to the final performance."
    #
    #   S0: Face only                    → lower bound
    #   S1: S0 + Object branch           → +obj giúp bao nhiêu?
    #   S2: S1 + Scene (raw concat)      → +scene thô giúp bao nhiêu?
    #   S3: S2 + SGF (cross-attention)   → attention tốt hơn concat bao nhiêu?
    #   S4: S3 + EC (node-level)         → contagion thêm bao nhiêu?
    #   S5: S4 + Residual = Full model   → residual thêm bao nhiêu?
    # ════════════════════════════════════════════════════════════════════════
    "S0_face_only": {
        "description": "[Stack] Face + GATv2 only — lower bound",
        "use_scene": False, "use_obj": False,
        "use_sgf":   False, "sgf_mode": None,
        "use_ec":    False, "ec_mode":  None,
        "use_residual": False, "use_branch_loss": True,
    },
    "S1_face_obj": {
        "description": "[Stack] Face + Object (no Scene)",
        "use_scene": False, "use_obj": True,
        "use_sgf":   False, "sgf_mode": None,
        "use_ec":    False, "ec_mode":  None,
        "use_residual": False, "use_branch_loss": True,
    },
    "S2_face_obj_scene": {
        "description": "[Stack] Face + Object + Scene (raw concat, no SGF)",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   False, "sgf_mode": "concat",
        "use_ec":    False, "ec_mode":  None,
        "use_residual": False, "use_branch_loss": True,
    },
    "S3_plus_sgf": {
        "description": "[Stack] S2 + SceneGuidedFusion (cross-attention)",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   True,  "sgf_mode": "original",
        "use_ec":    False, "ec_mode":  None,
        "use_residual": False, "use_branch_loss": True,
    },
    "S4_plus_ec": {
        "description": "[Stack] S3 + EmotionalContagion (node-level)",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   True,  "sgf_mode": "original",
        "use_ec":    True,  "ec_mode":  "node_level",
        "use_residual": False, "use_branch_loss": True,
    },
    "S5_full": {
        "description": "[Stack] S4 + Residual = Full model (replicate A1)",
        "use_scene": True,  "use_obj": True,
        "use_sgf":   True,  "sgf_mode": "original",
        "use_ec":    True,  "ec_mode":  "node_level",
        "use_residual": True, "use_branch_loss": True,
    },
}


# ============================================================
# HELPER UTILITIES
# ============================================================
def build_knn_edges(boxes, k=3):
    n = len(boxes)
    if n <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers = np.stack([cx, cy], axis=1)
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    src_list, dst_list = [], []
    actual_k = min(k, n - 1)
    for i in range(n):
        d = dist[i].copy(); d[i] = np.inf
        nn_idx = np.argsort(d)[:actual_k]
        for j in nn_idx:
            src_list.extend([i, j]); dst_list.extend([j, i])
    edges = list(set(zip(src_list, dst_list)))
    if not edges:
        return torch.tensor([[0], [0]], dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_dense_edges(num_nodes):
    if num_nodes <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def get_lr(optimizer):
    return next(iter(optimizer.param_groups))['lr']


def get_branch_weight(epoch):
    w_max = CONFIG['branch_w_max']; w_min = CONFIG['branch_w_min']
    ep_warm = CONFIG['warmup_epochs']; ep_decay = CONFIG['decay_end']
    if epoch < ep_warm: return w_max
    if epoch >= ep_decay: return w_min
    progress = (epoch - ep_warm) / (ep_decay - ep_warm)
    return w_max + progress * (w_min - w_max)


# ============================================================
# DATASET  (giữ nguyên logic từ code gốc)
# ============================================================
class ConGNN_Dataset(TorchDataset):
    def __init__(self, split='train', max_faces=32, max_objects=10, use_cache=None):
        self.face_root   = CONFIG['face_dir']
        self.scene_root  = CONFIG['scene_dir']
        self.obj_root    = CONFIG['object_dir']
        self.max_faces   = max_faces
        self.max_objects = max_objects
        self.label_map   = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.use_cache   = CONFIG['use_ram_cache'] if use_cache is None else use_cache

        pattern = os.path.join(self.face_root, 'faces', split, '**', '*.npz')
        self.face_files = glob.glob(pattern, recursive=True)
        if not self.face_files:
            pattern = os.path.join(self.face_root, split, '**', '*.npz')
            self.face_files = glob.glob(pattern, recursive=True)

        print(f"  📊 {split.upper()}: {len(self.face_files)} samples")
        if not self.face_files:
            raise ValueError(f"No data found for split '{split}'")

        self._build_scene_index(); self._build_object_index()
        self._cache = None
        if self.use_cache:
            self._preload_all(split)

    def _build_scene_index(self):
        self._scene_index = {}
        for p in glob.glob(os.path.join(self.scene_root, '**', '*.npy'), recursive=True):
            self._scene_index[os.path.splitext(os.path.basename(p))[0]] = p

    def _build_object_index(self):
        self._obj_index = {}
        for p in glob.glob(os.path.join(self.obj_root, '**', '*.npz'), recursive=True):
            self._obj_index[os.path.splitext(os.path.basename(p))[0]] = p

    def _get_paired_path(self, face_path, target_type):
        stem = os.path.splitext(os.path.basename(face_path))[0]
        if target_type == 'scenes': return self._scene_index.get(stem)
        if target_type == 'objects': return self._obj_index.get(stem)
        return None

    def _preload_all(self, split):
        print(f"  💾 Preloading {split.upper()} into RAM...")
        self._cache = [self._load_sample(i) for i in
                       tqdm(range(len(self.face_files)), desc=f"  Cache {split}", leave=False)]

    def _load_sample(self, idx):
        face_file = self.face_files[idx]
        label = self.label_map.get(
            os.path.basename(os.path.dirname(face_file)).lower(), 1)

        # Face
        try:
            data = np.load(face_file)
            face_feat = data['features']; face_boxes = data['boxes']
            if len(face_boxes) > 0:
                si = np.argsort(face_boxes[:, 0])
                face_feat = face_feat[si]; face_boxes = face_boxes[si]
        except Exception:
            face_feat = np.zeros((1, CONFIG['face_dim']), dtype=np.float32)
            face_boxes = np.zeros((1, 4), dtype=np.float32)

        face_feat  = face_feat[:self.max_faces]  if len(face_feat)  > 0 else np.zeros((1, CONFIG['face_dim']),  dtype=np.float32)
        face_boxes = face_boxes[:self.max_faces] if len(face_boxes) > 0 else np.zeros((1, 4),                   dtype=np.float32)
        face_x          = torch.tensor(face_feat, dtype=torch.float32)
        face_edge_index = build_knn_edges(face_boxes, k=CONFIG['knn_k'])

        # Scene
        scene_path = self._get_paired_path(face_file, 'scenes')
        try:
            if scene_path and os.path.exists(scene_path):
                sf = np.load(scene_path)
                if sf.ndim == 4:   sf = sf.mean(axis=(0, 2, 3))
                elif sf.ndim == 3: sf = sf.mean(axis=(-2, -1))
                elif sf.ndim == 2:
                    sf = sf.squeeze(0) if sf.shape[0] == 1 else sf.mean(axis=0)
                sf = sf.flatten()[:CONFIG['scene_dim']]
                if len(sf) < CONFIG['scene_dim']:
                    sf = np.pad(sf, (0, CONFIG['scene_dim'] - len(sf)))
                scene_feat = sf.astype(np.float32)
            else:
                scene_feat = np.zeros(CONFIG['scene_dim'], dtype=np.float32)
        except Exception:
            scene_feat = np.zeros(CONFIG['scene_dim'], dtype=np.float32)
        scene_x = torch.tensor(scene_feat, dtype=torch.float32)

        # Object
        obj_path = self._get_paired_path(face_file, 'objects')
        try:
            if obj_path and os.path.exists(obj_path):
                od = np.load(obj_path)
                obj_feat = od['features'] if 'features' in od else od[od.files[0]]
            else:
                obj_feat = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)
        except Exception:
            obj_feat = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)

        obj_feat = obj_feat[:self.max_objects]
        context_x = (torch.tensor(obj_feat, dtype=torch.float32) if len(obj_feat) > 0
                     else torch.zeros((1, CONFIG['object_dim']), dtype=torch.float32))
        context_edge_index = build_dense_edges(len(context_x))

        return {
            'face_x': face_x, 'face_edge_index': face_edge_index,
            'context_x': context_x, 'context_edge_index': context_edge_index,
            'scene_x': scene_x, 'y': label
        }

    def __len__(self): return len(self.face_files)

    def __getitem__(self, idx):
        return self._cache[idx] if self._cache is not None else self._load_sample(idx)


# ============================================================
# COLLATE
# ============================================================
class SimpleBatch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

    def to(self, device):
        for attr in ['face_x','face_edge_index','face_batch',
                     'context_x','context_edge_index','context_batch',
                     'scene_x','y']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


def custom_collate(batch):
    fx, fei, fb, cx, cei, cb, sx, yl = [], [], [], [], [], [], [], []
    fn = cn = 0
    for gi, s in enumerate(batch):
        nf = s['face_x'].size(0)
        fx.append(s['face_x']); fei.append(s['face_edge_index'] + fn)
        fb.append(torch.full((nf,), gi, dtype=torch.long)); fn += nf
        nc = s['context_x'].size(0)
        cx.append(s['context_x']); cei.append(s['context_edge_index'] + cn)
        cb.append(torch.full((nc,), gi, dtype=torch.long)); cn += nc
        sx.append(s['scene_x']); yl.append(s['y'])

    return SimpleBatch(
        face_x=torch.cat(fx), face_edge_index=torch.cat(fei, dim=1),
        face_batch=torch.cat(fb), context_x=torch.cat(cx),
        context_edge_index=torch.cat(cei, dim=1), context_batch=torch.cat(cb),
        scene_x=torch.stack(sx), y=torch.tensor(yl, dtype=torch.long),
        num_graphs=len(batch)
    )


# ============================================================
# MODEL SUB-MODULES (giữ nguyên từ V10)
# ============================================================
class MultiLayerGATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, num_layers=2,
                 dropout=0.5, attention_dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                          dropout=attention_dropout, add_self_loops=True,
                          concat=True, bias=False))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.drop(F.relu(self.input_norm(self.input_proj(x))))
        for gat, norm in zip(self.gat_layers, self.norms):
            h_new = self.drop(F.elu(norm(gat(h, edge_index))))
            h = h + h_new if h.shape == h_new.shape else h_new
        return h


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 4, 1))

    def forward(self, x, batch):
        scores = self.score_mlp(x)
        scores = scores - scores.max()
        exp_s  = torch.exp(scores)
        B = batch.max().item() + 1
        denom = torch.zeros(B, 1, device=x.device)
        denom.scatter_add_(0, batch.unsqueeze(1), exp_s)
        weight = exp_s / (denom[batch] + 1e-8)
        out = torch.zeros(B, x.size(1), device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), weight * x)
        return out


class SceneGuidedFusion(nn.Module):
    """
    SGF gốc (sgf_mode='original'):
        Query = scene_proj (1 vector) attend over (face_nodes + obj_nodes)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, scene_feat, face_nodes, obj_nodes, face_batch, obj_batch):
        all_nodes = torch.cat([face_nodes, obj_nodes], dim=0)
        all_batch = torch.cat([face_batch, obj_batch],  dim=0)
        dense_nodes, mask = to_dense_batch(all_nodes, all_batch)
        query = scene_feat.unsqueeze(1)
        attn_out, attn_weights = self.cross_attn(
            query, dense_nodes, dense_nodes, key_padding_mask=~mask)
        fused = self.layer_norm(query + attn_out).squeeze(1)
        return fused, attn_weights


class SceneGuidedFusion_Expanded(nn.Module):
    """
    SGF cải tiến (sgf_mode='expanded') — fix vấn đề 1-vector-query:

    Vấn đề với SGF gốc:
        1 scene vector → 1 attention distribution trải đều → không đủ
        "góc nhìn" để attend khác nhau vào face vs object nodes.

    Giải pháp — Multi-Query Expansion:
        scene_proj (D) → expand_proj → (K, D)   K=4 queries độc lập
        Mỗi query học attend 1 khía cạnh khác nhau của group
        (vd: query1=emotional tone, query2=activity, query3=spatial, query4=object context)
        Sau attention: mean-pool K outputs → 1 vector D như cũ
        → Tương thích hoàn toàn với phần còn lại của model.

    Tại sao K=4?
        = num_heads (4) → mỗi query tương ứng 1 head trong multi-head attention
        → giảm risk redundancy, training ổn định hơn.
    """
    def __init__(self, hidden_dim, num_queries=4):
        super().__init__()
        self.K = num_queries
        # Project 1 scene vector → K query vectors
        self.expand_proj = nn.Linear(hidden_dim, hidden_dim * num_queries)
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm  = nn.LayerNorm(hidden_dim)
        self.gate        = nn.Linear(hidden_dim, hidden_dim)   # gating sau pool

    def forward(self, scene_feat, face_nodes, obj_nodes, face_batch, obj_batch):
        B, D = scene_feat.shape

        # Expand scene → K queries
        queries = self.expand_proj(scene_feat)           # [B, K*D]
        queries = queries.view(B, self.K, D)             # [B, K, D]

        # Build dense key-value matrix từ face + obj nodes
        all_nodes = torch.cat([face_nodes, obj_nodes], dim=0)
        all_batch = torch.cat([face_batch, obj_batch],  dim=0)
        dense_nodes, mask = to_dense_batch(all_nodes, all_batch)  # [B, N_max, D]

        # Cross-attention: K queries attend over N nodes
        attn_out, attn_weights = self.cross_attn(
            queries, dense_nodes, dense_nodes,
            key_padding_mask=~mask)                      # [B, K, D]

        # Residual + mean-pool K outputs → 1 vector
        attn_out = self.layer_norm(queries + attn_out)   # [B, K, D]
        fused    = attn_out.mean(dim=1)                  # [B, D]

        # Gating: scene_feat kiểm soát mức độ ảnh hưởng của attention output
        gate  = torch.sigmoid(self.gate(scene_feat))
        fused = gate * fused + (1 - gate) * scene_feat  # [B, D]

        return fused, attn_weights


class FaceQueryFusion(nn.Module):
    """
    SGF đảo vai trò (sgf_mode='face_as_query') — C2 variant:

    Thay vì Scene attend Face/Obj, ta để Face attend Scene.
    Trực giác: "Scene là context, Face chủ động tìm thông tin từ scene."
        Query = face_pooled (mean-pooled face nodes per graph)
        Key/Value = scene_proj (1 vector per image)

    Output: enriched_face (D) thay thế fused_scene
        → concat với feat_face và feat_obj để phân loại.

    Lưu ý: scene vẫn được dùng làm branch clf_scene riêng.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, scene_feat, face_nodes, obj_nodes, face_batch, obj_batch):
        # Query = mean-pooled face features per graph
        face_pooled = global_mean_pool(face_nodes, face_batch)  # [B, D]
        query = face_pooled.unsqueeze(1)                         # [B, 1, D]

        # Key/Value = scene (broadcast thành sequence length=1)
        scene_kv = scene_feat.unsqueeze(1)                       # [B, 1, D]

        attn_out, attn_weights = self.cross_attn(query, scene_kv, scene_kv)
        enriched = self.layer_norm(query + attn_out).squeeze(1)  # [B, D]

        # Trả về enriched_face_scene (đóng vai trò fused_scene trong pipeline)
        return enriched, attn_weights


class EmotionalContagion(nn.Module):
    """Node-level EC (ec_mode='node_level') — gốc từ V10."""
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, H_face, fused_scene, face_batch):
        atmosphere = fused_scene[face_batch]
        return self.norm(H_face + self.drop(self.alpha * atmosphere))


class EmotionalContagion_PoolLevel(nn.Module):
    """
    Pool-level EC (ec_mode='pool_level') — D1 variant:

    Áp dụng contagion SAU khi pool thay vì trước.
        feat_face_ec = feat_face + beta * fused_scene

    Ưu: đơn giản, không ảnh hưởng attention pool weights.
    Nhược: pool không học được "ai phản ánh atmosphere tốt nhất".
    So sánh với node-level để đánh giá tầm quan trọng của thứ tự này.
    """
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, feat_face_pooled, fused_scene):
        """
        feat_face_pooled: [B, D] — đã pool xong
        fused_scene:      [B, D]
        """
        return self.norm(feat_face_pooled + self.drop(self.beta * fused_scene))


# ============================================================
# ABLATION MODEL  ← trái tim của framework
# ============================================================
class AblationModel(nn.Module):
    """
    Model đa năng — bật/tắt VÀ thay đổi chiến lược từng thành phần
    dựa theo `flags` dict.

    Flags:
        use_scene      : bool
        use_obj        : bool
        use_sgf        : bool
        sgf_mode       : None | 'original' | 'expanded' | 'face_as_query' | 'concat'
        use_ec         : bool
        ec_mode        : None | 'node_level' | 'pool_level'
        use_residual   : bool
        use_branch_loss: bool

    Ràng buộc tự động (enforce trong __init__):
        use_scene=False  → use_sgf=False, use_ec=False
        use_obj=False    → use_sgf=False (SGF cần obj nodes làm KV)
        sgf_mode='concat'→ use_sgf=False (concat không phải attention)

    Combined dim trước clf_whole:
        face(D) + [obj(D)] + [scene/fused(D)]  →  1D, 2D, hoặc 3D
    """

    def __init__(self, flags: dict):
        super().__init__()
        D       = CONFIG['gat_hidden']
        drp     = CONFIG['dropout']
        att_drp = CONFIG['attention_dropout']

        # ── Enforce constraints ──────────────────────────────────────────────
        self.flags = dict(flags)
        f = self.flags

        if not f['use_scene']:
            f['use_sgf'] = False; f['sgf_mode'] = None
            f['use_ec']  = False; f['ec_mode']  = None
        if not f['use_obj']:
            # SGF cần obj nodes làm Key-Value → tắt khi không có obj
            f['use_sgf'] = False; f['sgf_mode'] = None
            # EC mất fused_scene → tắt luôn để tránh dùng scene_proj thô
            # (nếu muốn test "EC với scene thô" thì dùng sgf_mode='concat' + ec)
            f['use_ec']  = False; f['ec_mode']  = None
        # sgf_mode='concat' nghĩa là không attention, chỉ concat thô
        if f.get('sgf_mode') == 'concat':
            f['use_sgf'] = False

        # ── Input projections ────────────────────────────────────────────────
        self.reduce_face = nn.Sequential(
            nn.LayerNorm(CONFIG['face_dim']),
            nn.Linear(CONFIG['face_dim'], 1024),
            nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(drp),
            nn.Linear(1024, D), nn.LayerNorm(D), nn.ReLU()
        )
        if f['use_obj']:
            self.reduce_obj = nn.Sequential(
                nn.LayerNorm(CONFIG['object_dim']),
                nn.Linear(CONFIG['object_dim'], D),
                nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp)
            )
        if f['use_scene']:
            self.reduce_scene = nn.Sequential(
                nn.LayerNorm(CONFIG['scene_dim']),
                nn.Linear(CONFIG['scene_dim'], D),
                nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp)
            )

        # ── GAT ─────────────────────────────────────────────────────────────
        self.face_gat = MultiLayerGATv2(
            in_dim=D, hidden_dim=D,
            num_heads=CONFIG['num_heads'], num_layers=CONFIG['gat_layers'],
            dropout=drp, attention_dropout=att_drp)
        if f['use_obj']:
            self.context_gat = MultiLayerGATv2(
                in_dim=D, hidden_dim=D,
                num_heads=CONFIG['num_heads'], num_layers=CONFIG['gat_layers'],
                dropout=drp, attention_dropout=att_drp)

        # ── Branch classifiers ───────────────────────────────────────────────
        self.attn_pool_face_branch = AttentionPool(D, dropout=drp)
        self.clf_face = nn.Linear(D, CONFIG['num_classes'])
        if f['use_obj']:
            self.clf_context = nn.Linear(D, CONFIG['num_classes'])
        if f['use_scene']:
            self.clf_scene = nn.Linear(D, CONFIG['num_classes'])

        # ── Fusion module — khởi tạo theo sgf_mode ──────────────────────────
        sgf_mode = f.get('sgf_mode')
        if sgf_mode == 'original':
            self.fusion = SceneGuidedFusion(hidden_dim=D)
        elif sgf_mode == 'expanded':
            self.fusion = SceneGuidedFusion_Expanded(hidden_dim=D, num_queries=4)
        elif sgf_mode == 'face_as_query':
            self.fusion = FaceQueryFusion(hidden_dim=D)
        else:
            self.fusion = None     # 'concat' hoặc None → không dùng attention

        # ── EC module — khởi tạo theo ec_mode ───────────────────────────────
        ec_mode = f.get('ec_mode')
        if ec_mode == 'node_level':
            self.ec = EmotionalContagion(D, dropout=drp)
        elif ec_mode == 'pool_level':
            self.ec = EmotionalContagion_PoolLevel(D, dropout=drp)
        else:
            self.ec = None

        # ── Node-level Residual ──────────────────────────────────────────────
        if f['use_residual']:
            self.lambda_face   = nn.Parameter(torch.tensor(0.5))
            self.raw_face_proj = nn.Linear(D, D)
            if f['use_obj']:
                self.lambda_obj   = nn.Parameter(torch.tensor(0.5))
                self.raw_obj_proj = nn.Linear(D, D)

        # ── Attention pool cho whole head ─────────────────────────────────────
        self.attn_pool_face = AttentionPool(D, dropout=drp)

        # ── clf_whole: kích thước đầu vào phụ thuộc flags ───────────────────
        combined_dim = self._calc_combined_dim(D)
        self.clf_whole = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(combined_dim, D),
            nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp),
            nn.Linear(D, CONFIG['num_classes'])
        )

        active = {k: v for k, v in f.items()
                  if k not in ('description',) and v not in (False, None)}
        print(f"  [AblationModel] combined_dim={combined_dim}D | {active}")

    def _calc_combined_dim(self, D):
        f = self.flags
        dim = D                       # face luôn có
        if f['use_obj']:   dim += D   # object pooled
        if f['use_scene']: dim += D   # scene / fused_scene (1 slot bất kể mode)
        return dim

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, data):
        f     = self.flags
        sgf_m = f.get('sgf_mode')
        ec_m  = f.get('ec_mode')

        # ── Projections ──────────────────────────────────────────────────────
        face_x_proj = self.reduce_face(data.face_x)
        obj_x_proj  = self.reduce_obj(data.context_x)  if f['use_obj']   else None
        scene_proj  = self.reduce_scene(data.scene_x)  if f['use_scene'] else None

        # ── GAT ──────────────────────────────────────────────────────────────
        H_face = self.face_gat(face_x_proj, data.face_edge_index)
        H_obj  = (self.context_gat(obj_x_proj, data.context_edge_index)
                  if f['use_obj'] else None)

        # ── Branch classifiers ────────────────────────────────────────────────
        out_face    = self.clf_face(
            self.attn_pool_face_branch(H_face, data.face_batch))
        out_context = (self.clf_context(global_mean_pool(H_obj, data.context_batch))
                       if f['use_obj'] else None)
        out_scene   = self.clf_scene(scene_proj) if f['use_scene'] else None

        # ── Fusion ───────────────────────────────────────────────────────────
        if f['use_sgf'] and self.fusion is not None:
            # Tất cả attention variants cần (scene, H_face, H_obj, batches)
            fused_scene, _ = self.fusion(
                scene_proj, H_face, H_obj,
                data.face_batch, data.context_batch)
        else:
            # Không attention → fused_scene = scene_proj thô (hoặc None)
            fused_scene = scene_proj

        # ── EC node-level (TRƯỚC pool) ────────────────────────────────────────
        if ec_m == 'node_level' and self.ec is not None and fused_scene is not None:
            H_face = self.ec(H_face, fused_scene, data.face_batch)

        # ── Node-level Residual ───────────────────────────────────────────────
        if f['use_residual']:
            H_face = H_face + self.lambda_face * self.raw_face_proj(face_x_proj)
            if f['use_obj'] and H_obj is not None:
                H_obj = H_obj + self.lambda_obj * self.raw_obj_proj(obj_x_proj)

        # ── Pooling ───────────────────────────────────────────────────────────
        feat_face = self.attn_pool_face(H_face, data.face_batch)
        feat_obj  = (global_mean_pool(H_obj, data.context_batch)
                     if f['use_obj'] else None)

        # ── EC pool-level (SAU pool) ──────────────────────────────────────────
        if ec_m == 'pool_level' and self.ec is not None and fused_scene is not None:
            feat_face = self.ec(feat_face, fused_scene)

        # ── Build combined vector ─────────────────────────────────────────────
        parts = [feat_face]
        if f['use_obj']:   parts.append(feat_obj)
        if f['use_scene']: parts.append(fused_scene)

        combined  = torch.cat(parts, dim=1)
        out_whole = self.clf_whole(combined)

        return out_face, out_context, out_scene, out_whole


# ============================================================
# LOSS
# ============================================================
def compute_loss(out_f, out_c, out_s, out_w, labels,
                 ce_criterion, branch_w: float, use_branch_loss: bool):
    """
    Linh hoạt theo flags:
    - use_branch_loss=True:  L = CE(whole) + branch_w*(CE(face)+CE(ctx)+CE(scene))
    - use_branch_loss=False: L = CE(whole)  (B5 variant)
    - Nhánh bị tắt (None):  tự động bỏ qua
    """
    labels = labels.long()
    L_w = ce_criterion(out_w, labels)

    if not use_branch_loss:
        L_f = L_c = L_s = 0.0
        return L_w, 0.0, 0.0, 0.0, L_w.item()

    L_f = ce_criterion(out_f, labels) if out_f is not None else torch.tensor(0.0)
    L_c = ce_criterion(out_c, labels) if out_c is not None else torch.tensor(0.0)
    L_s = ce_criterion(out_s, labels) if out_s is not None else torch.tensor(0.0)

    L_total = L_w + branch_w * (L_f + L_c + L_s)
    return (L_total,
            L_f.item() if torch.is_tensor(L_f) else L_f,
            L_c.item() if torch.is_tensor(L_c) else L_c,
            L_s.item() if torch.is_tensor(L_s) else L_s,
            L_w.item())


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=40, min_delta=0.001, path='checkpoint.pt'):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best_acc = 0.0; self.best_loss = float('inf')
        self.early_stop = False; self.path = path

    def __call__(self, val_loss, val_acc, model):
        acc_improved  = val_acc  > (self.best_acc  + self.min_delta)
        loss_improved = (val_loss < self.best_loss - self.min_delta and
                         val_acc  >= self.best_acc - 0.002)
        if acc_improved or loss_improved:
            torch.save(model.state_dict(), self.path)
            self.counter = 0
            if val_acc  > self.best_acc:  self.best_acc  = val_acc
            if val_loss < self.best_loss: self.best_loss = val_loss
            print(f"  🔥 Saved  Loss={val_loss:.4f} Acc={val_acc:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ============================================================
# TRAIN ONE VARIANT
# ============================================================
def train_variant(variant_name: str, flags: dict,
                  train_loader, val_loader) -> dict:
    """
    Huấn luyện một variant và trả về dict kết quả trên val.
    """
    print(f"\n{'='*70}")
    print(f"▶  VARIANT: {variant_name}")
    print(f"   {flags['description']}")
    print(f"{'='*70}")

    var_dir = os.path.join(CONFIG['output_dir'], variant_name)
    os.makedirs(var_dir, exist_ok=True)
    ckpt_path = os.path.join(var_dir, 'best_model.pth')

    model = AblationModel(flags).to(CONFIG['device'])
    tp = sum(p.numel() for p in model.parameters())
    print(f"   Params: {tp:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    early_stop   = EarlyStopping(CONFIG['patience'], path=ckpt_path)

    use_branch_loss = flags.get('use_branch_loss', True)

    history = {k: [] for k in ['train_loss','val_loss',
                                'val_acc_whole','val_acc_face',
                                'val_acc_context','val_acc_scene']}
    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(CONFIG['epochs']):
        branch_w = get_branch_weight(epoch)

        # ── Train ──
        model.train(); t_loss = 0
        for batch in tqdm(train_loader,
                          desc=f"  [{variant_name}] Ep {epoch+1:03d} Train",
                          leave=False):
            try:
                batch = batch.to(CONFIG['device'])
                optimizer.zero_grad()
                out_f, out_c, out_s, out_w = model(batch)
                loss, *_ = compute_loss(out_f, out_c, out_s, out_w, batch.y,
                                        ce_criterion, branch_w, use_branch_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                optimizer.step()
                t_loss += loss.item()
            except Exception as e:
                if CONFIG['debug_mode']: raise
                continue

        # ── Validate ──
        model.eval()
        v_loss = v_af = v_ac = v_as = v_aw = total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader,
                              desc=f"  [{variant_name}] Ep {epoch+1:03d} Val",
                              leave=False):
                try:
                    batch = batch.to(CONFIG['device'])
                    out_f, out_c, out_s, out_w = model(batch)
                    loss, *_ = compute_loss(out_f, out_c, out_s, out_w, batch.y,
                                            ce_criterion, branch_w, use_branch_loss)
                    bs = len(batch.y)
                    v_loss += loss.item() * bs; total += bs
                    v_af += (out_f.argmax(1) == batch.y).sum().item() if out_f is not None else 0
                    v_ac += (out_c.argmax(1) == batch.y).sum().item() if out_c is not None else 0
                    v_as += (out_s.argmax(1) == batch.y).sum().item() if out_s is not None else 0
                    v_aw += (out_w.argmax(1) == batch.y).sum().item()
                except Exception as e:
                    if CONFIG['debug_mode']: raise
                    continue

        scheduler.step()

        vl  = v_loss / total
        vaw = v_aw / total
        vaf = v_af / total if flags.get('use_obj', True) or True else 0
        vac = v_ac / total if flags.get('use_obj', True) else 0
        vas = v_as / total if flags.get('use_scene', True) else 0

        for k, v in zip(['train_loss','val_loss','val_acc_whole',
                         'val_acc_face','val_acc_context','val_acc_scene'],
                        [t_loss/len(train_loader), vl, vaw, vaf, vac, vas]):
            history[k].append(v)

        if vaw > best_val_acc: best_val_acc = vaw

        print(f"  Ep {epoch+1:03d} | ValLoss={vl:.4f} | Whole={vaw:.4f} "
              f"| Face={vaf:.4f} | Ctx={vac:.4f} | Scene={vas:.4f}")

        early_stop(vl, vaw, model)
        if early_stop.early_stop:
            print(f"  🛑 Early stop ep {epoch+1}")
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); gc.collect()

    elapsed = time.time() - t_start
    result = {
        'variant':      variant_name,
        'description':  flags['description'],
        'best_val_acc': round(best_val_acc, 4),
        'epochs_run':   epoch + 1,
        'elapsed_min':  round(elapsed / 60, 1),
        'ckpt_path':    ckpt_path,
        'history':      history,
    }

    # Lưu history JSON
    with open(os.path.join(var_dir, 'history.json'), 'w') as fp:
        h_save = {k: v for k, v in history.items()}
        json.dump(h_save, fp, indent=2)

    return result


# ============================================================
# EVALUATE ON TEST SET
# ============================================================
def evaluate_on_test(variant_name: str, flags: dict, test_loader) -> dict:
    ckpt_path = os.path.join(CONFIG['output_dir'], variant_name, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        print(f"  ⚠ Checkpoint not found: {ckpt_path}")
        return {}

    model = AblationModel(flags).to(CONFIG['device'])
    model.load_state_dict(torch.load(ckpt_path, map_location=CONFIG['device']))
    model.eval()

    y_true, y_pw, y_pf, y_pc, y_ps = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  [{variant_name}] Test", leave=False):
            try:
                batch = batch.to(CONFIG['device'])
                out_f, out_c, out_s, out_w = model(batch)
                y_true.extend(batch.y.cpu().numpy())
                y_pw.extend(out_w.argmax(1).cpu().numpy())
                if out_f is not None: y_pf.extend(out_f.argmax(1).cpu().numpy())
                if out_c is not None: y_pc.extend(out_c.argmax(1).cpu().numpy())
                if out_s is not None: y_ps.extend(out_s.argmax(1).cpu().numpy())
            except Exception as e:
                if CONFIG['debug_mode']: raise
                continue

    acc_whole = accuracy_score(y_true, y_pw)
    acc_face  = accuracy_score(y_true, y_pf) if y_pf else 0.0
    acc_ctx   = accuracy_score(y_true, y_pc) if y_pc else 0.0
    acc_scene = accuracy_score(y_true, y_ps) if y_ps else 0.0

    print(f"\n  [{variant_name}] TEST → Whole={acc_whole:.4f} | "
          f"Face={acc_face:.4f} | Ctx={acc_ctx:.4f} | Scene={acc_scene:.4f}")
    print(classification_report(y_true, y_pw,
                                target_names=['Neg','Neu','Pos'], digits=4))

    # Vẽ confusion matrix cho variant này
    var_dir = os.path.join(CONFIG['output_dir'], variant_name)
    cm_pct  = confusion_matrix(y_true, y_pw, normalize='true') * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=['Neg','Neu','Pos'],
                yticklabels=['Neg','Neu','Pos'],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f'Confusion Matrix — {variant_name}\nAcc={acc_whole:.4f}', fontsize=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(var_dir, 'confusion_matrix.png'), dpi=120)
    plt.close()

    return {
        'test_acc_whole': round(acc_whole, 4),
        'test_acc_face':  round(acc_face, 4),
        'test_acc_ctx':   round(acc_ctx, 4),
        'test_acc_scene': round(acc_scene, 4),
        'y_true': y_true, 'y_pred': y_pw,
    }


# ============================================================
# SUMMARY PLOT
# ============================================================
def plot_ablation_summary(all_results: list):
    """
    Vẽ 2 biểu đồ:
    1. Bar chart tổng hợp tất cả variants, tô màu theo nhóm (A/B/C/D/S)
    2. Line chart incremental stack (S0 → S5) thể hiện contribution từng module
    """
    names   = [r['variant'] for r in all_results]
    accs    = [r.get('test_acc_whole', r.get('best_val_acc', 0)) for r in all_results]
    baseline = next((r.get('test_acc_whole', 0) for r in all_results
                     if r['variant'] == 'A1_full_model'), 0.8974)

    # ── Màu theo nhóm ──────────────────────────────────────────────────────
    GROUP_COLORS = {
        'A': '#e74c3c',   # đỏ — modality
        'B': '#3498db',   # xanh dương — architecture
        'C': '#2ecc71',   # xanh lá — fusion strategy
        'D': '#9b59b6',   # tím — EC level
        'S': '#f39c12',   # cam — incremental stack
    }
    colors = [GROUP_COLORS.get(n[0], '#95a5a6') for n in names]

    fig = plt.figure(figsize=(max(14, len(names) * 1.3), 12))

    # ── Plot 1: All variants bar chart ──────────────────────────────────────
    ax1 = fig.add_subplot(2, 1, 1)
    bars = ax1.bar(range(len(names)), accs, color=colors,
                   edgecolor='white', linewidth=1.5, width=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=35, ha='right', fontsize=10)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Ablation Study — Test Accuracy by Variant', fontsize=14, fontweight='bold')
    ax1.set_ylim(max(0, min(accs) - 0.05), 1.0)
    ax1.axhline(baseline, color='red', linestyle='--', linewidth=1.5,
                label=f'Baseline A1 = {baseline:.4f}')
    ax1.grid(axis='y', alpha=0.3)

    # Delta label trên mỗi bar
    for bar, acc, name in zip(bars, accs, names):
        delta = acc - baseline
        color_txt = '#c0392b' if delta < 0 else '#27ae60'
        sign = '+' if delta >= 0 else ''
        label = f'{acc:.4f}\n({sign}{delta:.4f})'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 label, ha='center', va='bottom', fontsize=8,
                 color=color_txt if name != 'A1_full_model' else '#2c3e50',
                 fontweight='bold')

    # Legend nhóm
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=f'Group {g}')
                    for g, c in GROUP_COLORS.items()]
    legend_elems.append(plt.Line2D([0], [0], color='red', linestyle='--', label='Baseline'))
    ax1.legend(handles=legend_elems, fontsize=9, loc='lower right')

    # ── Plot 2: Incremental Stack line chart ─────────────────────────────────
    stack_variants = ['S0_face_only','S1_face_obj','S2_face_obj_scene',
                      'S3_plus_sgf','S4_plus_ec','S5_full']
    stack_labels   = ['S0\nFace only','S1\n+Object','S2\n+Scene(raw)',
                      'S3\n+SGF','S4\n+EC','S5\n+Residual\n(=Full)']

    stack_accs = []
    for sv in stack_variants:
        matched = next((r.get('test_acc_whole', r.get('best_val_acc', None))
                        for r in all_results if r['variant'] == sv), None)
        stack_accs.append(matched)

    if any(v is not None for v in stack_accs):
        ax2 = fig.add_subplot(2, 1, 2)
        valid_x = [i for i, v in enumerate(stack_accs) if v is not None]
        valid_y = [stack_accs[i] for i in valid_x]

        ax2.plot(valid_x, valid_y, 'o-', color='#f39c12', linewidth=2.5,
                 markersize=10, markerfacecolor='white', markeredgewidth=2.5,
                 markeredgecolor='#f39c12', zorder=5)
        ax2.fill_between(valid_x, [min(valid_y)-0.02]*len(valid_x), valid_y,
                         alpha=0.15, color='#f39c12')

        for xi, yi in zip(valid_x, valid_y):
            delta = yi - baseline
            sign  = '+' if delta >= 0 else ''
            ax2.annotate(f'{yi:.4f}\n({sign}{delta:.4f})',
                         xy=(xi, yi), xytext=(0, 14),
                         textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold',
                         color='#d35400')

        # Mũi tên annotation cho từng bước nhảy
        for i in range(len(valid_x) - 1):
            x1, y1 = valid_x[i], valid_y[i]
            x2, y2 = valid_x[i+1], valid_y[i+1]
            delta_step = y2 - y1
            mid_x = (x1 + x2) / 2
            sign = '+' if delta_step >= 0 else ''
            color = '#27ae60' if delta_step >= 0 else '#c0392b'
            ax2.annotate(f'{sign}{delta_step:.4f}',
                         xy=(mid_x, (y1+y2)/2), fontsize=8,
                         ha='center', color=color, fontweight='bold')

        ax2.set_xticks(range(len(stack_variants)))
        ax2.set_xticklabels(stack_labels, fontsize=10)
        ax2.set_ylabel('Test Accuracy', fontsize=12)
        ax2.set_title('Incremental Component Stack — Contribution per Module',
                      fontsize=13, fontweight='bold')
        ax2.axhline(baseline, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
                    label=f'Baseline A1 = {baseline:.4f}')
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)
        y_min = min(v for v in stack_accs if v) - 0.03
        ax2.set_ylim(max(0, y_min), min(1.0, max(v for v in stack_accs if v) + 0.06))

    plt.tight_layout(pad=2.0)
    out_path = os.path.join(CONFIG['output_dir'], 'ablation_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Summary plot saved: {out_path}")


# ============================================================
# SAVE CSV
# ============================================================
def save_csv(all_results: list):
    csv_path = os.path.join(CONFIG['output_dir'], 'ablation_results.csv')
    fieldnames = ['variant','description','test_acc_whole','test_acc_face',
                  'test_acc_ctx','test_acc_scene','best_val_acc',
                  'epochs_run','elapsed_min']
    with open(csv_path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"✅ CSV saved: {csv_path}")


# ============================================================
# MAIN RUNNER
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Tên variants muốn chạy (vd: A1 B3). Mặc định: tất cả.')
    parser.add_argument('--list', action='store_true',
                        help='Liệt kê tất cả variants rồi thoát.')
    parser.add_argument('--skip_train', action='store_true',
                        help='Bỏ qua training, chỉ eval checkpoint đã có.')
    args, _ = parser.parse_known_args()

    if args.list:
        print("\n📋 AVAILABLE VARIANTS:")
        for k, v in VARIANTS.items():
            print(f"  {k:25s} — {v['description']}")
        return

    # Lọc variants muốn chạy
    if args.variants:
        selected = {}
        for req in args.variants:
            # Tìm theo prefix (A1, B3, ...) hoặc tên đầy đủ
            matched = [k for k in VARIANTS if k.startswith(req)]
            if not matched:
                print(f"⚠ Không tìm thấy variant '{req}'")
            for m in matched:
                selected[m] = VARIANTS[m]
        run_variants = selected
    else:
        run_variants = VARIANTS

    print(f"\n🧪 SẼ CHẠY {len(run_variants)} VARIANTS: {list(run_variants.keys())}\n")

    # Load datasets
    print("📂 Loading datasets...")
    train_ds = ConGNN_Dataset('train')
    val_ds   = ConGNN_Dataset('val')
    test_ds  = ConGNN_Dataset('test')

    kw = dict(batch_size=CONFIG['batch_size'], collate_fn=custom_collate,
              num_workers=CONFIG['num_workers'],
              pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)
    print(f"✅ Loaders ready | train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}\n")

    all_results = []

    for variant_name, flags in run_variants.items():
        try:
            # ── Training ──────────────────────────────────────────────────────
            if not args.skip_train:
                train_result = train_variant(variant_name, flags,
                                             train_loader, val_loader)
            else:
                train_result = {
                    'variant': variant_name,
                    'description': flags['description'],
                    'best_val_acc': 0.0,
                    'epochs_run': 0,
                    'elapsed_min': 0.0,
                }

            # ── Test Evaluation ───────────────────────────────────────────────
            test_result = evaluate_on_test(variant_name, flags, test_loader)
            combined = {**train_result, **test_result}
            combined.pop('history', None)   # không cần lưu vào CSV
            combined.pop('y_true',  None)
            combined.pop('y_pred',  None)
            combined.pop('ckpt_path', None)
            all_results.append(combined)

        except Exception as e:
            print(f"\n❌ Variant {variant_name} FAILED: {e}")
            if CONFIG['debug_mode']: raise
            all_results.append({
                'variant': variant_name,
                'description': flags['description'],
                'test_acc_whole': -1, 'best_val_acc': -1,
                'epochs_run': 0, 'elapsed_min': 0
            })
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache(); gc.collect()

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 ABLATION STUDY — KẾT QUẢ TỔNG HỢP")
    print("="*70)
    header = f"{'Variant':<25} {'Description':<45} {'Test Acc':>10}"
    print(header); print("-"*len(header))
    baseline_acc = None
    for r in all_results:
        acc = r.get('test_acc_whole', -1)
        if r['variant'] == 'A1_full_model': baseline_acc = acc
        delta = f"({acc - baseline_acc:+.4f})" if baseline_acc and r['variant'] != 'A1_full_model' else ""
        print(f"  {r['variant']:<23} {r['description']:<45} {acc:>8.4f}  {delta}")

    save_csv(all_results)
    try:
        plot_ablation_summary(all_results)
    except Exception as e:
        print(f"⚠ Plot error: {e}")

    print("\n✅ ABLATION STUDY HOÀN THÀNH!")
    print(f"   Kết quả: {CONFIG['output_dir']}/ablation_results.csv")
    print(f"   Biểu đồ: {CONFIG['output_dir']}/ablation_summary.png")


if __name__ == "__main__":
    main()