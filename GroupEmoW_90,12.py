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
# CONFIG
# ============================================================
CONFIG = {
    'face_dir':   '/kaggle/input/datasets/nguynnhtlam12/face-featuresv2',
    'scene_dir':  '/kaggle/input/datasets/drakhight/8726scene-features/scene_features_final/scenes',
    'object_dir': '/kaggle/input/datasets/trieung11/fearturecongnn/objects/objects',
    'output_dir': '/kaggle/working/output',

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
# DATASET
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
# MODEL SUB-MODULES
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
    Scene-Guided Fusion (SGF):
        Query = scene_proj (1 vector per image) attend over (face_nodes + obj_nodes).
        Output: fused_scene — scene vector enriched với context từ face và object.
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


class EmotionalContagion(nn.Module):
    """
    Node-level Emotional Contagion (EC):
        Mỗi face node được điều chỉnh bởi atmosphere của scene (fused_scene).
        Áp dụng TRƯỚC attention pool để pool học được ai phản ánh scene tốt nhất.
    """
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, H_face, fused_scene, face_batch):
        atmosphere = fused_scene[face_batch]
        return self.norm(H_face + self.drop(self.alpha * atmosphere))


# ============================================================
# FULL MODEL (A1)
# ============================================================
class ConGNN(nn.Module):
    """
    ConGNN — Full Model:
        - Face branch  : MultiLayerGATv2 + AttentionPool
        - Object branch: MultiLayerGATv2 + global_mean_pool
        - Scene branch : Linear projection
        - SceneGuidedFusion (SGF): scene cross-attends face+object nodes
        - EmotionalContagion (EC): node-level, applied before pooling
        - Node-level Residual Connections (face + object)
        - Branch classifiers (face, object, scene) + whole classifier
    """

    def __init__(self):
        super().__init__()
        D       = CONFIG['gat_hidden']
        drp     = CONFIG['dropout']
        att_drp = CONFIG['attention_dropout']

        # ── Input projections ────────────────────────────────────────────────
        self.reduce_face = nn.Sequential(
            nn.LayerNorm(CONFIG['face_dim']),
            nn.Linear(CONFIG['face_dim'], 1024),
            nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(drp),
            nn.Linear(1024, D), nn.LayerNorm(D), nn.ReLU()
        )
        self.reduce_obj = nn.Sequential(
            nn.LayerNorm(CONFIG['object_dim']),
            nn.Linear(CONFIG['object_dim'], D),
            nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp)
        )
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
        self.context_gat = MultiLayerGATv2(
            in_dim=D, hidden_dim=D,
            num_heads=CONFIG['num_heads'], num_layers=CONFIG['gat_layers'],
            dropout=drp, attention_dropout=att_drp)

        # ── Branch classifiers ───────────────────────────────────────────────
        self.attn_pool_face_branch = AttentionPool(D, dropout=drp)
        self.clf_face    = nn.Linear(D, CONFIG['num_classes'])
        self.clf_context = nn.Linear(D, CONFIG['num_classes'])
        self.clf_scene   = nn.Linear(D, CONFIG['num_classes'])

        # ── Scene-Guided Fusion ──────────────────────────────────────────────
        self.fusion = SceneGuidedFusion(hidden_dim=D)

        # ── Emotional Contagion (node-level) ─────────────────────────────────
        self.ec = EmotionalContagion(D, dropout=drp)

        # ── Node-level Residual ──────────────────────────────────────────────
        self.lambda_face   = nn.Parameter(torch.tensor(0.5))
        self.raw_face_proj = nn.Linear(D, D)
        self.lambda_obj    = nn.Parameter(torch.tensor(0.5))
        self.raw_obj_proj  = nn.Linear(D, D)

        # ── Attention pool for whole head ─────────────────────────────────────
        self.attn_pool_face = AttentionPool(D, dropout=drp)

        # ── Whole classifier: face(D) + obj(D) + scene(D) → 3D ──────────────
        combined_dim = D * 3
        self.clf_whole = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(combined_dim, D),
            nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp),
            nn.Linear(D, CONFIG['num_classes'])
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  [ConGNN] Params: {total_params:,} | combined_dim={combined_dim}")

    def forward(self, data):
        # ── Projections ──────────────────────────────────────────────────────
        face_x_proj = self.reduce_face(data.face_x)
        obj_x_proj  = self.reduce_obj(data.context_x)
        scene_proj  = self.reduce_scene(data.scene_x)

        # ── GAT ──────────────────────────────────────────────────────────────
        H_face = self.face_gat(face_x_proj, data.face_edge_index)
        H_obj  = self.context_gat(obj_x_proj, data.context_edge_index)

        # ── Branch classifiers ────────────────────────────────────────────────
        out_face    = self.clf_face(self.attn_pool_face_branch(H_face, data.face_batch))
        out_context = self.clf_context(global_mean_pool(H_obj, data.context_batch))
        out_scene   = self.clf_scene(scene_proj)

        # ── Scene-Guided Fusion ───────────────────────────────────────────────
        fused_scene, _ = self.fusion(
            scene_proj, H_face, H_obj,
            data.face_batch, data.context_batch)

        # ── Emotional Contagion (node-level, BEFORE pool) ─────────────────────
        H_face = self.ec(H_face, fused_scene, data.face_batch)

        # ── Node-level Residual ───────────────────────────────────────────────
        H_face = H_face + self.lambda_face * self.raw_face_proj(face_x_proj)
        H_obj  = H_obj  + self.lambda_obj  * self.raw_obj_proj(obj_x_proj)

        # ── Pooling ───────────────────────────────────────────────────────────
        feat_face = self.attn_pool_face(H_face, data.face_batch)
        feat_obj  = global_mean_pool(H_obj, data.context_batch)

        # ── Whole classifier ──────────────────────────────────────────────────
        combined  = torch.cat([feat_face, feat_obj, fused_scene], dim=1)
        out_whole = self.clf_whole(combined)

        return out_face, out_context, out_scene, out_whole


# ============================================================
# LOSS
# ============================================================
def compute_loss(out_f, out_c, out_s, out_w, labels,
                 ce_criterion, branch_w: float):
    """
    L = CE(whole) + branch_w * (CE(face) + CE(obj) + CE(scene))
    """
    labels = labels.long()
    L_w = ce_criterion(out_w, labels)
    L_f = ce_criterion(out_f, labels)
    L_c = ce_criterion(out_c, labels)
    L_s = ce_criterion(out_s, labels)
    L_total = L_w + branch_w * (L_f + L_c + L_s)
    return L_total, L_f.item(), L_c.item(), L_s.item(), L_w.item()


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
# TRAINING
# ============================================================
def train(train_loader, val_loader) -> dict:
    out_dir = CONFIG['output_dir']
    ckpt_path = os.path.join(out_dir, 'best_model.pth')

    model = ConGNN().to(CONFIG['device'])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    early_stop   = EarlyStopping(CONFIG['patience'], path=ckpt_path)

    history = {k: [] for k in ['train_loss', 'val_loss',
                                'val_acc_whole', 'val_acc_face',
                                'val_acc_context', 'val_acc_scene']}
    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(CONFIG['epochs']):
        branch_w = get_branch_weight(epoch)

        # ── Train ──
        model.train(); t_loss = 0
        for batch in tqdm(train_loader, desc=f"  Ep {epoch+1:03d} Train", leave=False):
            try:
                batch = batch.to(CONFIG['device'])
                optimizer.zero_grad()
                out_f, out_c, out_s, out_w = model(batch)
                loss, *_ = compute_loss(out_f, out_c, out_s, out_w, batch.y,
                                        ce_criterion, branch_w)
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
            for batch in tqdm(val_loader, desc=f"  Ep {epoch+1:03d} Val", leave=False):
                try:
                    batch = batch.to(CONFIG['device'])
                    out_f, out_c, out_s, out_w = model(batch)
                    loss, *_ = compute_loss(out_f, out_c, out_s, out_w, batch.y,
                                            ce_criterion, branch_w)
                    bs = len(batch.y)
                    v_loss += loss.item() * bs; total += bs
                    v_af += (out_f.argmax(1) == batch.y).sum().item()
                    v_ac += (out_c.argmax(1) == batch.y).sum().item()
                    v_as += (out_s.argmax(1) == batch.y).sum().item()
                    v_aw += (out_w.argmax(1) == batch.y).sum().item()
                except Exception as e:
                    if CONFIG['debug_mode']: raise
                    continue

        scheduler.step()

        vl  = v_loss / total
        vaw = v_aw / total
        vaf = v_af / total
        vac = v_ac / total
        vas = v_as / total

        for k, v in zip(['train_loss', 'val_loss', 'val_acc_whole',
                         'val_acc_face', 'val_acc_context', 'val_acc_scene'],
                        [t_loss / len(train_loader), vl, vaw, vaf, vac, vas]):
            history[k].append(v)

        if vaw > best_val_acc: best_val_acc = vaw

        print(f"  Ep {epoch+1:03d} | ValLoss={vl:.4f} | Whole={vaw:.4f} "
              f"| Face={vaf:.4f} | Ctx={vac:.4f} | Scene={vas:.4f}")

        early_stop(vl, vaw, model)
        if early_stop.early_stop:
            print(f"  🛑 Early stop at epoch {epoch+1}")
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); gc.collect()

    elapsed = time.time() - t_start
    print(f"\n✅ Training done | Best val acc: {best_val_acc:.4f} | "
          f"Time: {elapsed/60:.1f} min")

    # Save history
    with open(os.path.join(out_dir, 'history.json'), 'w') as fp:
        json.dump(history, fp, indent=2)

    return {'best_val_acc': best_val_acc, 'history': history,
            'ckpt_path': ckpt_path, 'elapsed_min': round(elapsed / 60, 1)}


# ============================================================
# EVALUATE ON TEST SET
# ============================================================
def evaluate_on_test(test_loader) -> dict:
    ckpt_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
    if not os.path.exists(ckpt_path):
        print(f"  ⚠ Checkpoint not found: {ckpt_path}")
        return {}

    model = ConGNN().to(CONFIG['device'])
    model.load_state_dict(torch.load(ckpt_path, map_location=CONFIG['device']))
    model.eval()

    y_true, y_pw, y_pf, y_pc, y_ps = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Test", leave=False):
            try:
                batch = batch.to(CONFIG['device'])
                out_f, out_c, out_s, out_w = model(batch)
                y_true.extend(batch.y.cpu().numpy())
                y_pw.extend(out_w.argmax(1).cpu().numpy())
                y_pf.extend(out_f.argmax(1).cpu().numpy())
                y_pc.extend(out_c.argmax(1).cpu().numpy())
                y_ps.extend(out_s.argmax(1).cpu().numpy())
            except Exception as e:
                if CONFIG['debug_mode']: raise
                continue

    acc_whole = accuracy_score(y_true, y_pw)
    acc_face  = accuracy_score(y_true, y_pf) if y_pf else 0.0
    acc_ctx   = accuracy_score(y_true, y_pc) if y_pc else 0.0
    acc_scene = accuracy_score(y_true, y_ps) if y_ps else 0.0

    print(f"\n  TEST RESULTS:")
    print(f"  Whole={acc_whole:.4f} | Face={acc_face:.4f} | "
          f"Ctx={acc_ctx:.4f} | Scene={acc_scene:.4f}")
    print(classification_report(y_true, y_pw,
                                target_names=['Neg', 'Neu', 'Pos'], digits=4))

    # Confusion matrix
    cm_pct = confusion_matrix(y_true, y_pw, normalize='true') * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=['Neg', 'Neu', 'Pos'],
                yticklabels=['Neg', 'Neu', 'Pos'],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f'Confusion Matrix — ConGNN\nAcc={acc_whole:.4f}', fontsize=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(CONFIG['output_dir'], 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")

    return {
        'test_acc_whole': round(acc_whole, 4),
        'test_acc_face':  round(acc_face, 4),
        'test_acc_ctx':   round(acc_ctx, 4),
        'test_acc_scene': round(acc_scene, 4),
    }


# ============================================================
# PLOT TRAINING HISTORY
# ============================================================
def plot_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['val_acc_whole'],   label='Whole')
    axes[1].plot(history['val_acc_face'],    label='Face')
    axes[1].plot(history['val_acc_context'], label='Context')
    axes[1].plot(history['val_acc_scene'],   label='Scene')
    axes[1].set_title('Val Accuracy'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(CONFIG['output_dir'], 'training_history.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ History plot saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ConGNN — Full Model Training")
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training, only evaluate checkpoint.')
    args, _ = parser.parse_known_args()

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
    print(f"✅ Loaders ready | train={len(train_loader)} "
          f"val={len(val_loader)} test={len(test_loader)}\n")

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.skip_train:
        print("🚀 Starting training...\n")
        train_result = train(train_loader, val_loader)
        plot_history(train_result['history'])
    else:
        print("⏩ Skipping training — evaluating existing checkpoint.\n")

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print("\n🔍 Evaluating on test set...")
    test_result = evaluate_on_test(test_loader)

    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    for k, v in test_result.items():
        print(f"  {k:<20s}: {v:.4f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n✅ Done! Outputs saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()