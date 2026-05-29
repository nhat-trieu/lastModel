import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch
import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import gc

warnings.filterwarnings('ignore')

# ==========================================
# KAGGLE PATHS — GAF3
# Dataset trên Kaggle thường mount tại /kaggle/input/<dataset-slug>/
# ==========================================
KAGGLE_PATHS = {
    # face: {Train|Validation}/{positive|neutral|negative}/*.npz
    'face_dir':   '/kaggle/input/datasets/trieung11/facegaf3',

    # scene: {Train|Validation}/{Label}/{Label}/*.npy  (subfolder trùng tên label)
    'scene_dir':  '/kaggle/input/datasets/trieung11/70accscenegaf3/scene_features_final/scenes',

    # object: {Train|Validation}/{Positive|Neutral|Negative}/*.npz
    'object_dir': '/kaggle/input/datasets/trieung11/objectgaf3',
}

# Thư mục lưu model checkpoint và biểu đồ
OUTPUT_DIR = '/kaggle/working/outputs'

def setup_data():
    """Trả về paths đã cấu hình — trên Kaggle không cần unzip."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for key, path in KAGGLE_PATHS.items():
        status = '✅' if os.path.exists(path) else '❌ NOT FOUND'
        print(f'  {status}  {key}: {path}')
    return KAGGLE_PATHS


# ==========================================
# CONFIG — GAF3
# class_counts: Train Pos=3977, Neu=3080, Neg=2758
# ==========================================
CONFIG = {
    'face_dim':   4096,
    'object_dim': 2048,
    'scene_dim':  1024,

    'gat_hidden':  512,
    'num_classes': 3,
    'dropout':     0.5,
    'red_dropout': 0.4,
    'clf_w_dropout': 0.6,
    'attention_dropout': 0.5,

    'knn_k': 3,
    'dense_fallback_threshold': 4,
    'max_faces': 32,

    # Semantic-aware graph
    'use_semantic_graph': False,  # True → hybrid spatial+semantic edges
    'k_semantic':         2,      # số semantic neighbor mỗi node
    'sim_threshold':      0.6,    # ngưỡng cosine similarity để nối edge

    # DropEdge
    'drop_edge_rate': 0.0,        # 0.0 = tắt, 0.3 = drop 30% edges khi train

    'batch_size':    32,
    'lr':            3e-5,
    'weight_decay':  0.05,
    'grad_clip':     0.5,
    'epochs':        100,         # GAF3: dataset lớn hơn cần train lâu hơn
    'patience':      20,
    'neutral_w':     1.8,
    'positive_w':    1.0,

    'branch_w_max':  0.30,
    'branch_w_min':  0.05,
    'warmup_epochs': 20,
    'decay_end':     80,

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'use_ram_cache': True,

    # NEW: flags kiến trúc — bật/tắt từng upgrade
    'use_attn_pool_context': False,
    'use_gated_fusion':      False,

    # GAF3 class counts (Train): Neg=2758, Neu=3080, Pos=3977
    'class_counts': [2758, 3080, 3977],

    'output_dir': OUTPUT_DIR,
    'n_runs': 10,
}


# ==========================================
# UTILS & GRAPH BUILDER
# ==========================================
def build_dense_edges(num_nodes):
    if num_nodes <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def build_knn_edges(boxes, k=3):
    """Spatial KNN: nối các face gần nhau về vị trí pixel."""
    n = len(boxes)
    if n <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    if n <= CONFIG['dense_fallback_threshold']:
        return build_dense_edges(n)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers = np.stack([cx, cy], axis=1)
    dist = np.sqrt(((centers[:, None, :] - centers[None, :, :])**2).sum(axis=-1))
    src, dst = [], []
    actual_k = min(k, n - 1)
    for i in range(n):
        d = dist[i].copy(); d[i] = np.inf
        nn_idx = np.argsort(d)[:actual_k]
        for j in nn_idx:
            src.extend([i, j]); dst.extend([j, i])
    return torch.tensor(list(set(zip(src, dst))), dtype=torch.long).t().contiguous()


def build_semantic_edges(features, k_semantic=2, sim_threshold=0.6):
    """
    Semantic KNN: nối các face có cam xuc tuong dong nhau
    dua tren cosine similarity cua fine-tuned face features.
    """
    n = len(features)
    if n <= 1:
        return None

    norm   = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = features / norm
    sim    = normed @ normed.T
    np.fill_diagonal(sim, -1.0)

    edges    = set()
    actual_k = min(k_semantic, n - 1)
    for i in range(n):
        top_k = np.argsort(sim[i])[-actual_k:]
        for j in top_k:
            if sim[i, j] > sim_threshold:
                edges.add((i, j))
                edges.add((j, i))

    if not edges:
        return None
    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()


def build_hybrid_edges(features, boxes, k_spatial=3, k_semantic=2, sim_threshold=0.6):
    """Hybrid graph = Spatial KNN union Semantic KNN."""
    n = len(features)
    if n <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    if n <= CONFIG['dense_fallback_threshold']:
        return build_dense_edges(n)

    spatial_ei  = build_knn_edges(boxes, k=k_spatial)
    semantic_ei = build_semantic_edges(features, k_semantic, sim_threshold)

    if semantic_ei is None:
        return spatial_ei

    combined = torch.cat([spatial_ei, semantic_ei], dim=1)
    combined = torch.unique(combined, dim=1)
    return combined


def drop_edge(edge_index, drop_rate, training=True):
    """DropEdge: random drop edges khi training de chong over-smoothing."""
    if not training or drop_rate <= 0.0:
        return edge_index
    E = edge_index.size(1)
    if E == 0:
        return edge_index
    keep_mask = torch.rand(E, device=edge_index.device) >= drop_rate
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    return edge_index[:, keep_mask]


# ==========================================
# DATASET
# ==========================================
class GAF3_Dataset(TorchDataset):
    def __init__(self, cfg, split='Train'):
        self.label_map = {
            'negative': 0, 'neutral': 1, 'positive': 2,
            'Negative': 0, 'Neutral': 1, 'Positive': 2,
        }
        self.cfg = cfg
        self.max_faces = cfg.get('max_faces', 32)

        pattern = os.path.join(cfg['face_dir'], split, '**', '*.npz')
        self.face_files = glob.glob(pattern, recursive=True)
        print(f"[{split}] Found {len(self.face_files)} face files")

        self._scene_idx = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(cfg['scene_dir'], split, '**', '*.npy'), recursive=True)
        }

        self._obj_idx = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(cfg['object_dir'], split, '**', '*.npz'), recursive=True)
        }

        print(f"[{split}] scene={len(self._scene_idx)} object={len(self._obj_idx)}")

        self._cache = [
            self._load_sample(i)
            for i in tqdm(range(len(self.face_files)), desc=f"Caching {split}")
        ] if cfg.get('use_ram_cache', True) else None

    def _load_sample(self, idx):
        f_path = self.face_files[idx]
        stem   = os.path.splitext(os.path.basename(f_path))[0]
        label  = self.label_map.get(os.path.basename(os.path.dirname(f_path)).lower(), 1)

        data     = np.load(f_path)
        feats_np = data['features'][:self.max_faces]
        boxes_np = data['boxes'][:self.max_faces]
        face_x   = torch.from_numpy(feats_np).float()

        if self.cfg.get('use_semantic_graph', False):
            face_ei = build_hybrid_edges(
                feats_np, boxes_np,
                k_spatial    = self.cfg.get('knn_k', 3),
                k_semantic   = self.cfg.get('k_semantic', 2),
                sim_threshold= self.cfg.get('sim_threshold', 0.6),
            )
        else:
            face_ei = build_knn_edges(boxes_np, k=self.cfg.get('knn_k', 3))

        s_path = self._scene_idx.get(stem)
        if s_path:
            s_feat = np.load(s_path)
            if s_feat.ndim >= 2:
                s_feat = s_feat.mean(axis=0).flatten()[:CONFIG['scene_dim']]
            else:
                s_feat = s_feat[:CONFIG['scene_dim']]
        else:
            s_feat = np.zeros(CONFIG['scene_dim'])

        o_path = self._obj_idx.get(stem)
        if o_path:
            o_data = np.load(o_path)
            o_feat = o_data['features'][:10] if 'features' in o_data else o_data[o_data.files[0]][:10]
        else:
            o_feat = np.zeros((1, CONFIG['object_dim']))

        return {
            'face_x':  face_x,
            'face_ei': face_ei,
            'scene_x': torch.from_numpy(s_feat).float(),
            'obj_x':   torch.from_numpy(o_feat).float(),
            'obj_ei':  build_dense_edges(len(o_feat)),
            'y':       label,
        }

    def __len__(self): return len(self.face_files)
    def __getitem__(self, idx):
        return self._cache[idx] if self._cache else self._load_sample(idx)


# ==========================================
# BATCHING
# ==========================================
class SimpleBatch:
    def __init__(self, batch):
        self.face_x  = torch.cat([s['face_x'] for s in batch])
        self.scene_x = torch.stack([s['scene_x'] for s in batch])
        self.obj_x   = torch.cat([s['obj_x'] for s in batch])
        self.y       = torch.tensor([s['y'] for s in batch], dtype=torch.long)
        f_ei, o_ei, f_b, o_b, f_ptr, o_ptr = [], [], [], [], 0, 0
        for i, s in enumerate(batch):
            nf, no = s['face_x'].size(0), s['obj_x'].size(0)
            f_ei.append(s['face_ei'] + f_ptr); o_ei.append(s['obj_ei'] + o_ptr)
            f_b.append(torch.full((nf,), i, dtype=torch.long))
            o_b.append(torch.full((no,), i, dtype=torch.long))
            f_ptr += nf; o_ptr += no
        self.face_ei    = torch.cat(f_ei, dim=1)
        self.obj_ei     = torch.cat(o_ei, dim=1)
        self.face_batch = torch.cat(f_b)
        self.obj_batch  = torch.cat(o_b)

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor): setattr(self, k, v.to(device))
        return self


# ==========================================
# MODEL BUILDING BLOCKS
# ==========================================
class AttentionPool(nn.Module):
    """Soft attention pooling: học trọng số tầm quan trọng của từng node."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(dim // 4, 1)
        )

    def forward(self, x, batch):
        s      = self.mlp(x)
        weights = s - s.max()
        exp_w  = torch.exp(weights)
        denom  = torch.zeros(batch.max() + 1, 1, device=x.device).scatter_add_(
            0, batch.unsqueeze(1), exp_w)
        attn = exp_w / (denom[batch] + 1e-8)
        return torch.zeros(batch.max() + 1, x.size(1), device=x.device).scatter_add_(
            0, batch.unsqueeze(1).expand_as(x), attn * x)


class MultiLayerGATv2(nn.Module):
    def __init__(self, in_d, hid_d, heads, layers, dropout, drop_edge_rate=0.0):
        super().__init__()
        self.gats           = nn.ModuleList([GATv2Conv(hid_d, hid_d // heads, heads=heads, dropout=0.3) for _ in range(layers)])
        self.norms          = nn.ModuleList([nn.LayerNorm(hid_d) for _ in range(layers)])
        self.drop           = nn.Dropout(dropout)
        self.drop_edge_rate = drop_edge_rate

    def forward(self, x, ei):
        h = x
        for gat, norm in zip(self.gats, self.norms):
            ei_drop = drop_edge(ei, self.drop_edge_rate, training=self.training)
            h = h + self.drop(F.elu(norm(gat(h, ei_drop))))
        return h


class GatedFusion(nn.Module):
    """
    Gated Fusion cho 3 nhánh: Face (f), Context (c), Scene (s).
    Gate input: concat(f, c, s) → Linear → Sigmoid → 3 scalar weights
    Output:     α_f * W_f(f) + α_c * W_c(c) + α_s * W_s(s)
    """
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.proj_f = nn.Linear(dim, dim)
        self.proj_c = nn.Linear(dim, dim)
        self.proj_s = nn.Linear(dim, dim)

        self.gate_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim, 3),
        )
        self.classifier = nn.Linear(dim, 3)

    def forward(self, feat_f, feat_c, feat_s):
        concat = torch.cat([feat_f, feat_c, feat_s], dim=1)
        gates  = torch.softmax(self.gate_net(concat), dim=-1)

        fused = (gates[:, 0:1] * self.proj_f(feat_f) +
                 gates[:, 1:2] * self.proj_c(feat_c) +
                 gates[:, 2:3] * self.proj_s(feat_s))
        fused = F.layer_norm(fused, (fused.size(-1),))
        return self.classifier(fused), gates


# ==========================================
# MAIN MODEL
# ==========================================
class SceneGuided_ConGNN(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None: cfg = CONFIG
        D   = cfg['gat_hidden']
        drp = cfg['dropout']
        rd  = cfg['red_dropout']
        cwd = cfg['clf_w_dropout']

        self.use_attn_pool_context = cfg.get('use_attn_pool_context', False)
        self.use_gated_fusion      = cfg.get('use_gated_fusion', False)

        # ── Dimension reducers ──────────────────────────────────────────────
        self.red_f = nn.Sequential(nn.LayerNorm(cfg['face_dim']),   nn.Dropout(rd), nn.Linear(cfg['face_dim'],   D), nn.LayerNorm(D), nn.ReLU())
        self.red_o = nn.Sequential(nn.LayerNorm(cfg['object_dim']), nn.Dropout(rd), nn.Linear(cfg['object_dim'], D), nn.LayerNorm(D), nn.ReLU())
        self.red_s = nn.Sequential(nn.LayerNorm(cfg['scene_dim']),  nn.Dropout(rd), nn.Linear(cfg['scene_dim'],  D), nn.LayerNorm(D), nn.ReLU())

        # ── GATv2 layers ────────────────────────────────────────────────────
        der        = cfg.get('drop_edge_rate', 0.0)
        self.f_gat = MultiLayerGATv2(D, D, heads=4, layers=2, dropout=drp, drop_edge_rate=der)
        self.o_gat = MultiLayerGATv2(D, D, heads=4, layers=2, dropout=drp, drop_edge_rate=der)

        # ── Contagion mechanism ─────────────────────────────────────────────
        self.contagion_alpha = nn.Parameter(torch.zeros(D))
        self.contagion_gate  = nn.Linear(D, D)
        nn.init.constant_(self.contagion_gate.bias, -2.0)

        # ── Cross-modal fusion (scene-guided attention) ─────────────────────
        self.fusion = nn.MultiheadAttention(D, 4, batch_first=True)

        # ── Pooling layers ──────────────────────────────────────────────────
        self.pool_f    = AttentionPool(D, drp)
        self.pool_f_br = AttentionPool(D, drp)

        if self.use_attn_pool_context:
            self.pool_o = AttentionPool(D, drp)
        else:
            self.pool_o = None

        # ── Branch classifiers ──────────────────────────────────────────────
        self.clf_f = nn.Linear(D, 3)
        self.clf_c = nn.Linear(D, 3)
        self.clf_s = nn.Linear(D, 3)

        # ── Final prediction classifier ──────────────────────────────────────
        if self.use_gated_fusion:
            self.clf_w = GatedFusion(D, dropout=cwd)
        else:
            self.clf_w = nn.Sequential(
                nn.Dropout(cwd), nn.Linear(D * 3, D), nn.LayerNorm(D),
                nn.ReLU(), nn.Dropout(drp), nn.Linear(D, 3)
            )

    def forward(self, data):
        pf = self.red_f(data.face_x)
        po = self.red_o(data.obj_x)
        ps = self.red_s(data.scene_x)

        hf = self.f_gat(pf, data.face_ei)
        ho = self.o_gat(po, data.obj_ei)

        out_f = self.clf_f(self.pool_f_br(hf, data.face_batch))

        if self.use_attn_pool_context:
            ctx_pooled = self.pool_o(ho, data.obj_batch)
        else:
            ctx_pooled = global_mean_pool(ho, data.obj_batch)

        out_c = self.clf_c(ctx_pooled)
        out_s = self.clf_s(ps)

        nodes, mask = to_dense_batch(
            torch.cat([hf, ho], 0),
            torch.cat([data.face_batch, data.obj_batch], 0)
        )
        attn_s, _ = self.fusion(ps.unsqueeze(1), nodes, nodes, key_padding_mask=~mask)
        fs = F.layer_norm(ps + attn_s.squeeze(1), (ps.size(-1),))

        gate = torch.sigmoid(self.contagion_gate(hf))
        hf   = F.layer_norm(
            hf + gate * (self.contagion_alpha * fs[data.face_batch]),
            (hf.size(-1),)
        )

        feat_f = self.pool_f(hf + pf, data.face_batch)

        if self.use_attn_pool_context:
            feat_c = self.pool_o(ho + po, data.obj_batch)
        else:
            feat_c = global_mean_pool(ho + po, data.obj_batch)

        if self.use_gated_fusion:
            out_w, gates = self.clf_w(feat_f, feat_c, fs)
        else:
            out_w = self.clf_w(torch.cat([fs, feat_f, feat_c], dim=1))

        return out_f, out_c, out_s, out_w


# ==========================================
# LOSS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', alpha)

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none', label_smoothing=0.1)
        return (self.alpha[target] * (1 - torch.exp(-ce))**self.gamma * ce).mean()


def build_focal_loss(cfg):
    counts = torch.tensor(cfg['class_counts']).float()
    w = counts.sum() / (3 * counts)
    w[1] *= cfg.get('neutral_w', 1.8)
    w[2] *= cfg.get('positive_w', 1.0)
    w /= w.mean()
    return FocalLoss(alpha=w.to(cfg['device']))


# ==========================================
# PLOT
# ==========================================
def plot_all(history, y_true, y_pred, run_id, cfg, split_name='Val'):
    """
    Layout 2×3:
      [0,0] Branch Losses (val_f, val_c, val_s)
      [0,1] Train vs Val Accuracy
      [0,2] Branch Loss Weight Schedule
      [1,0] Prediction Loss – Whole (train_whole vs val_whole)
      [1,1] Confusion Matrix (%)
      [1,2] Classification Report
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f"Best Run: {run_id}", fontsize=14, fontweight='bold')

    axes[0, 0].plot(history['val_f'], label='Face')
    axes[0, 0].plot(history['val_c'], label='Context')
    axes[0, 0].plot(history['val_s'], label='Scene')
    axes[0, 0].legend()
    axes[0, 0].set_title(f"Branch Losses ({split_name})")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("CE Loss")

    axes[0, 1].plot(history['acc_t'], label='Train Acc', color='blue', lw=2)
    axes[0, 1].plot(history['acc_w'], label=f'{split_name} Acc', color='red', lw=2)
    axes[0, 1].legend()
    axes[0, 1].set_title(f"Train vs {split_name} Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")

    axes[0, 2].plot(history['bw'], label='Branch Weight', color='orange')
    axes[0, 2].set_title("Branch Loss Weight Schedule")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("bw")

    axes[1, 0].plot(history['train_whole'], label='Train Whole',  color='steelblue', lw=2)
    axes[1, 0].plot(history['val_whole'],   label=f'{split_name} Whole', color='crimson', lw=2)
    axes[1, 0].legend()
    axes[1, 0].set_title("Prediction Loss – Whole (Focal Loss of ow only)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Focal Loss")

    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Neg', 'Neu', 'Pos'],
                yticklabels=['Neg', 'Neu', 'Pos'])
    axes[1, 1].set_title(f"Confusion Matrix (%) — {split_name}")

    axes[1, 2].axis('off')
    axes[1, 2].text(
        0.05, 0.5,
        classification_report(y_true, y_pred,
                              target_names=['Negative', 'Neutral', 'Positive'],
                              digits=4),
        fontsize=9, family='monospace', verticalalignment='center'
    )

    plt.tight_layout()
    save_path = os.path.join(cfg['output_dir'], f'{run_id}.png')
    plt.savefig(save_path)
    print(f"📊 Plot saved: {save_path}")
    plt.show()


# ==========================================
# TRAIN 1 LẦN
# ==========================================
def train_once(cfg, train_ds, val_ds, run_id, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  collate_fn=SimpleBatch)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, collate_fn=SimpleBatch)

    model = SceneGuided_ConGNN(cfg).to(cfg['device'])
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])

    crit_w  = build_focal_loss(cfg)
    crit_br = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {k: [] for k in [
        'train_whole', 'val_whole',
        'val_f', 'val_c', 'val_s',
        'acc_w', 'acc_t', 'bw',
    ]}

    best_acc, patience_cnt = 0, 0
    best_state = None

    for ep in range(cfg['epochs']):
        # ── Branch weight schedule ────────────────────────────────────────────
        if ep < cfg['warmup_epochs']:
            bw = cfg['branch_w_max']
        elif ep >= cfg['decay_end']:
            bw = cfg['branch_w_min']
        else:
            bw = cfg['branch_w_max'] + (ep - cfg['warmup_epochs']) / \
                 (cfg['decay_end'] - cfg['warmup_epochs']) * \
                 (cfg['branch_w_min'] - cfg['branch_w_max'])

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        t_whole_sum = t_acc = t_total = 0

        for b in train_loader:
            b.to(cfg['device'])
            opt.zero_grad()
            of, oc, os_, ow = model(b)

            whole_loss  = crit_w(ow, b.y)
            branch_reg  = crit_br(of, b.y) + crit_br(oc, b.y) + crit_br(os_, b.y)
            total_loss  = whole_loss + bw * branch_reg

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()

            t_whole_sum += whole_loss.item()
            t_acc       += (ow.argmax(1) == b.y).sum().item()
            t_total     += b.y.size(0)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        v_whole_sum = v_f = v_c = v_s = v_acc = total = 0

        with torch.no_grad():
            for b in val_loader:
                b.to(cfg['device'])
                of, oc, os_, ow = model(b)
                n = b.y.size(0)

                v_whole_sum += crit_w(ow, b.y).item() * n
                v_f  += crit_br(of,  b.y).item() * n
                v_c  += crit_br(oc,  b.y).item() * n
                v_s  += crit_br(os_, b.y).item() * n
                v_acc  += (ow.argmax(1) == b.y).sum().item()
                total  += n

        history['train_whole'].append(t_whole_sum / len(train_loader))
        history['val_whole'].append(v_whole_sum / total)
        history['val_f'].append(v_f / total)
        history['val_c'].append(v_c / total)
        history['val_s'].append(v_s / total)
        history['acc_w'].append(v_acc / total)
        history['acc_t'].append(t_acc / t_total)
        history['bw'].append(bw)
        sch.step()

        print(
            f"  [{run_id}|seed{seed}] Ep{ep+1:03d} | "
            f"TrainAcc={t_acc/t_total:.4f} | ValAcc={v_acc/total:.4f} | "
            f"TrainWhole={t_whole_sum/len(train_loader):.4f} | "
            f"ValWhole={v_whole_sum/total:.4f}"
        )

        if (v_acc / total) > best_acc:
            best_acc     = v_acc / total
            patience_cnt = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg['patience']:
                print(f"  Early stop at ep{ep+1}")
                break

    return best_acc, best_state, history


# ==========================================
# EVALUATE
# ==========================================
def evaluate(cfg, model, val_ds, split_name='Val'):
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=SimpleBatch)
    model.eval()
    acc_f = acc_c = acc_s = acc_w = total = 0
    y_t, y_p = [], []
    with torch.no_grad():
        for b in val_loader:
            b.to(cfg['device'])
            of, oc, os_, ow = model(b)
            n = b.y.size(0)
            acc_f += (of.argmax(1)  == b.y).sum().item()
            acc_c += (oc.argmax(1)  == b.y).sum().item()
            acc_s += (os_.argmax(1) == b.y).sum().item()
            acc_w += (ow.argmax(1)  == b.y).sum().item()
            total += n
            y_t.extend(b.y.cpu().numpy())
            y_p.extend(ow.argmax(1).cpu().numpy())
    print(f"\n{'='*60}")
    print(f"[{split_name}] Face={acc_f/total:.4f} | Ctx={acc_c/total:.4f} | Scene={acc_s/total:.4f} | FINAL={acc_w/total:.4f}")
    print(f"{'='*60}")
    return acc_w / total, y_t, y_p


# ==========================================
# TRAIN N LẦN LẤY BEST
# ==========================================
def train_multi_run(cfg, train_ds, val_ds, sweep_id):
    n_runs  = cfg.get('n_runs', 10)
    out_dir = cfg['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    arch_flags = (
        f"attn_ctx={'ON' if cfg.get('use_attn_pool_context') else 'OFF'} | "
        f"gated_fusion={'ON' if cfg.get('use_gated_fusion') else 'OFF'}"
    )
    print(f"\n{'='*70}")
    print(f"🚀 SWEEP: {sweep_id}")
    print(f"   neutral_w={cfg['neutral_w']} | positive_w={cfg.get('positive_w',1.0)} | "
          f"lr={cfg['lr']} | wd={cfg['weight_decay']} | max_faces={cfg.get('max_faces',32)}")
    print(f"   Architecture: {arch_flags}")
    print(f"{'='*70}")

    all_accs = []
    best_overall_acc     = 0
    best_overall_state   = None
    best_overall_history = None

    for run_i in range(n_runs):
        seed = 42 + run_i * 7
        print(f"\n--- Run {run_i+1}/{n_runs} (seed={seed}) ---")
        acc, state, history = train_once(cfg, train_ds, val_ds, sweep_id, seed=seed)
        all_accs.append(acc)
        print(f"  → Val Acc = {acc:.4f}")

        if acc > best_overall_acc:
            best_overall_acc     = acc
            best_overall_state   = state
            best_overall_history = history
            torch.save(state, os.path.join(out_dir, f'best_{sweep_id}.pth'))
            print(f"  🔥 New best saved!")

    print(f"\n📊 {sweep_id} | {n_runs} runs:")
    print(f"   Accs: {[f'{a:.4f}' for a in all_accs]}")
    print(f"   Mean={np.mean(all_accs):.4f} | Std={np.std(all_accs):.4f} | Best={best_overall_acc:.4f}")

    # Evaluate best model on Val
    model = SceneGuided_ConGNN(cfg).to(cfg['device'])
    model.load_state_dict({k: v.to(cfg['device']) for k, v in best_overall_state.items()})
    final_acc, y_t, y_p = evaluate(cfg, model, val_ds, split_name='Val')

    return final_acc, y_t, y_p, best_overall_history, np.mean(all_accs), np.std(all_accs)


# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    print("📂 Checking dataset paths...")
    paths = setup_data()
    CONFIG.update(paths)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("\n📦 Caching datasets vào RAM...")
    train_ds = GAF3_Dataset(CONFIG, split='Train')
    val_ds   = GAF3_Dataset(CONFIG, split='Validation')

    # ==========================================
    # SWEEP — 4 kịch bản thực nghiệm (sw5–sw8) — GAF3
    # ==========================================
    BASE = {
        **CONFIG,
        'epochs':   100,
        'patience': 20,
        'use_attn_pool_context': False,
        'use_gated_fusion':      False,
        'positive_w':            1.0,
    }

    SWEEP = [
        # ── sw5: SEMANTIC GRAPH ──────────────────────────────────────────────
        {
            **BASE,
            'gat_hidden': 256, 'dropout': 0.6, 'red_dropout': 0.5, 'clf_w_dropout': 0.7,
            'neutral_w':  1.8, 'positive_w': 1.0,
            'lr': 3e-5, 'weight_decay': 0.05,
            'use_attn_pool_context': False,
            'use_gated_fusion':      False,
            'use_semantic_graph':    True,
            'k_semantic':            2,
            'sim_threshold':         0.6,
        },

        # ── sw6: FULL SYSTEM ─────────────────────────────────────────────────
        {
            **BASE,
            'gat_hidden': 256, 'dropout': 0.6, 'red_dropout': 0.5, 'clf_w_dropout': 0.7,
            'neutral_w':  1.8, 'positive_w': 1.3,
            'lr': 3e-5, 'weight_decay': 0.05,
            'use_attn_pool_context': True,
            'use_gated_fusion':      True,
            'use_semantic_graph':    True,
            'k_semantic':            2,
            'sim_threshold':         0.6,
            'drop_edge_rate':        0.0,
        },

        # ── sw7: BASELINE + DROPEDGE ─────────────────────────────────────────
        {
            **BASE,
            'gat_hidden': 256, 'dropout': 0.6, 'red_dropout': 0.5, 'clf_w_dropout': 0.7,
            'neutral_w':  1.8, 'positive_w': 1.0,
            'lr': 3e-5, 'weight_decay': 0.05,
            'use_attn_pool_context': False,
            'use_gated_fusion':      False,
            'use_semantic_graph':    False,
            'drop_edge_rate':        0.3,
        },

        # ── sw8: FULL SYSTEM + DROPEDGE ──────────────────────────────────────
        {
            **BASE,
            'gat_hidden': 256, 'dropout': 0.6, 'red_dropout': 0.5, 'clf_w_dropout': 0.7,
            'neutral_w':  1.8, 'positive_w': 1.3,
            'lr': 3e-5, 'weight_decay': 0.05,
            'use_attn_pool_context': True,
            'use_gated_fusion':      True,
            'use_semantic_graph':    True,
            'k_semantic':            2,
            'sim_threshold':         0.6,
            'drop_edge_rate':        0.3,
        },
    ]

    # ==========================================
    # CHẠY SWEEP
    # ==========================================
    sweep_results = []

    for i, cfg in enumerate(SWEEP):
        sweep_id = (
            f"gaf3_sw{i+5}"
            f"_nw{cfg['neutral_w']}"
            f"_pw{cfg.get('positive_w', 1.0)}"
            f"_lr{cfg['lr']}"
            f"{'_attn' if cfg.get('use_attn_pool_context') else ''}"
            f"{'_gate' if cfg.get('use_gated_fusion') else ''}"
            f"{'_sem' if cfg.get('use_semantic_graph') else ''}"
            f"{'_de' + str(cfg.get('drop_edge_rate', 0.0)) if cfg.get('drop_edge_rate', 0.0) > 0 else ''}"
        )

        final_acc, y_t, y_p, best_hist, mean_acc, std_acc = \
            train_multi_run(cfg, train_ds, val_ds, sweep_id)

        sweep_results.append((sweep_id, final_acc, mean_acc, std_acc, y_t, y_p, best_hist))
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================
    # TỔNG KẾT SWEEP
    # ==========================================
    print(f"\n{'='*70}")
    print("📊 SWEEP RESULTS (sorted by best val acc) — GAF3")
    print(f"{'='*70}")
    for sid, facc, macc, sacc, *_ in sorted(sweep_results, key=lambda x: -x[1]):
        print(f"  Best={facc:.4f} | Mean={macc:.4f}±{sacc:.4f} | {sid}")
    print(f"{'='*70}")

    best = max(sweep_results, key=lambda x: x[1])
    best_sid, best_facc, _, _, best_yt, best_yp, best_hist = best
    print(f"\n🏆 Best overall: {best_sid} | FINAL={best_facc:.4f}")
    plot_all(best_hist, best_yt, best_yp, best_sid, CONFIG, split_name='Val')