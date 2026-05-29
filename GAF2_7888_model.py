import os
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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CÁCH DÙNG
# ============================================================
# Sau khi loss_sweep_gaf2.py xong, xem kết quả rồi chạy:
#
#   python branchw_sweep_gaf2.py --best_loss L5
#
# Script sẽ tự động lấy config loss của L5 (hoặc bất kỳ Lx nào)
# rồi sweep branch_w = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
#
# Nếu chưa biết best loss, mặc định dùng L5 (focal γ=2.0, neg=2.0)
# ============================================================

# ============================================================
# BASE CONFIG
# ============================================================
CONFIG = {
    'face_dir':   '/kaggle/input/datasets/trieung11/gaf2-face',
    'scene_dir':  '/kaggle/input/datasets/trieung11/gaf2-fearture/scene_features_gaf2000_v2/scene_features_final/scenes',
    'object_dir': '/kaggle/input/datasets/trieung11/gaf2-fearture/gaf2_object_features',
    'output_dir': '/kaggle/working/branchw_sweep_outputs',

    'face_dim':   4096,
    'object_dim': 2048,
    'scene_dim':  1024,

    'gat_hidden':  512,
    'num_classes': 3,
    'gat_layers':  2,
    'num_heads':   4,
    'dropout':     0.5,
    'attention_dropout': 0.5,

    'knn_k':           3,
    'label_smoothing': 0,

    'batch_size':  32,
    'num_workers': 0,

    'lr':           1e-5,
    'weight_decay': 1e-1,
    'grad_clip':    0.5,
    'epochs':       150,
    'patience':     40,
    'min_lr':       1e-6,

    # GAF2 train counts: Neg=1159, Neu=1199, Pos=1272
    'class_counts': [1159, 1199, 1272],

    'device':        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'use_ram_cache': True,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================
# BRANCH_W SWEEP VALUES
# ============================================================
BRANCH_W_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# ============================================================
# LOSS CONFIGS  (copy từ loss_sweep_gaf2.py để tham chiếu)
# Sau khi loss sweep xong → truyền --best_loss Lx
# ============================================================
LOSS_CONFIGS = {
    'L1_focal20_asym_branch': {
        'loss_type': 'focal', 'neg_w': 1.2, 'neu_w': 1.5, 'pos_w': 1.0,
        'gamma': 2.0, 'branch_loss_type': 'ce_plain',
    },
    'L8_focal20_asym_branch': {
        'loss_type': 'focal', 'neg_w': 2.0, 'neu_w': 1.5, 'pos_w': 1.0,
        'gamma': 2.0, 'branch_loss_type': 'ce_plain',
    },
    'L9_focal20_neu25': {
        'loss_type': 'focal', 'neg_w': 1.5, 'neu_w': 2.5, 'pos_w': 1.0,
        'gamma': 2.0, 'branch_loss_type': 'focal',
    },
}


# ============================================================
# LOSS BUILDERS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('alpha', alpha)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (self.alpha[targets] * (1 - pt) ** self.gamma * ce).mean()


def build_class_weights(loss_cfg):
    counts = torch.tensor(CONFIG['class_counts'], dtype=torch.float)
    if loss_cfg['neg_w'] is None:
        w = counts.sum() / (3 * counts)
    else:
        w = torch.tensor([loss_cfg['neg_w'], loss_cfg['neu_w'], loss_cfg['pos_w']],
                         dtype=torch.float)
    return (w / w.mean()).to(CONFIG['device'])


def build_criterion(loss_cfg):
    alpha = build_class_weights(loss_cfg)

    if loss_cfg['loss_type'] == 'focal':
        whole_crit = FocalLoss(alpha, gamma=loss_cfg['gamma'],
                               label_smoothing=CONFIG['label_smoothing'])
    else:
        whole_crit = nn.CrossEntropyLoss(weight=alpha,
                                         label_smoothing=CONFIG['label_smoothing'])

    bt = loss_cfg.get('branch_loss_type', 'ce_weighted')
    if bt == 'focal':
        branch_crit = FocalLoss(alpha, gamma=loss_cfg['gamma'],
                                label_smoothing=CONFIG['label_smoothing'])
    elif bt == 'ce_weighted':
        branch_crit = nn.CrossEntropyLoss(weight=alpha,
                                          label_smoothing=CONFIG['label_smoothing'])
    else:
        branch_crit = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])

    return whole_crit, branch_crit


# ============================================================
# UTILITIES
# ============================================================
def build_knn_edges(boxes, k=3):
    n = len(boxes)
    if n <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers = np.stack([cx, cy], axis=1)
    dist = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(-1))
    src_list, dst_list = [], []
    actual_k = min(k, n - 1)
    for i in range(n):
        d = dist[i].copy(); d[i] = np.inf
        for j in np.argsort(d)[:actual_k]:
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


# ============================================================
# DATASET
# ============================================================
class ConGNN_Dataset(TorchDataset):
    def __init__(self, split='Train', max_faces=32, max_objects=10):
        self.face_root   = CONFIG['face_dir']
        self.scene_root  = CONFIG['scene_dir']
        self.obj_root    = CONFIG['object_dir']
        self.max_faces   = max_faces
        self.max_objects = max_objects
        self.label_map   = {
            'negative': 0, 'neutral': 1, 'positive': 2,
            'Negative': 0, 'Neutral': 1, 'Positive': 2,
        }

        pattern = os.path.join(self.face_root, 'faces', split, '**', '*.npz')
        self.face_files = glob.glob(pattern, recursive=True)
        if not self.face_files:
            pattern = os.path.join(self.face_root, split, '**', '*.npz')
            self.face_files = glob.glob(pattern, recursive=True)

        print(f"  📊 {split}: {len(self.face_files)} samples")
        if not self.face_files:
            raise ValueError(f"No data for split='{split}'")

        self._scene_idx = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(self.scene_root, '**', '*.npy'), recursive=True)
        }
        self._obj_idx = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(self.obj_root, '**', '*.npz'), recursive=True)
        }

        print(f"  💾 Caching {split}...")
        self._cache = [self._load(i) for i in
                       tqdm(range(len(self.face_files)), leave=False)]

    def _load(self, idx):
        fp    = self.face_files[idx]
        label = self.label_map.get(os.path.basename(os.path.dirname(fp)).strip(), 1)
        stem  = os.path.splitext(os.path.basename(fp))[0]

        try:
            d = np.load(fp)
            ff, fb = d['features'], d['boxes']
            if len(fb) > 0:
                si = np.argsort(fb[:, 0]); ff = ff[si]; fb = fb[si]
        except Exception:
            ff = np.zeros((1, CONFIG['face_dim']),  dtype=np.float32)
            fb = np.zeros((1, 4), dtype=np.float32)

        ff = ff[:self.max_faces]  if len(ff) > 0 else np.zeros((1, CONFIG['face_dim']),  dtype=np.float32)
        fb = fb[:self.max_faces]  if len(fb) > 0 else np.zeros((1, 4), dtype=np.float32)

        try:
            sp = self._scene_idx.get(stem)
            if sp:
                sf = np.load(sp)
                if sf.ndim == 4:   sf = sf.mean((0, 2, 3))
                elif sf.ndim == 3: sf = sf.mean((-2, -1))
                elif sf.ndim == 2: sf = sf.squeeze(0) if sf.shape[0] == 1 else sf.mean(0)
                sf = sf.flatten()[:CONFIG['scene_dim']]
                if len(sf) < CONFIG['scene_dim']:
                    sf = np.pad(sf, (0, CONFIG['scene_dim'] - len(sf)))
                scene = sf.astype(np.float32)
            else:
                scene = np.zeros(CONFIG['scene_dim'], dtype=np.float32)
        except Exception:
            scene = np.zeros(CONFIG['scene_dim'], dtype=np.float32)

        try:
            op = self._obj_idx.get(stem)
            if op:
                od = np.load(op)
                obj = od['features'] if 'features' in od else od[od.files[0]]
            else:
                obj = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)
        except Exception:
            obj = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)

        obj = obj[:self.max_objects]
        cx  = (torch.tensor(obj, dtype=torch.float32) if len(obj) > 0
               else torch.zeros((1, CONFIG['object_dim']), dtype=torch.float32))

        return {
            'face_x':              torch.tensor(ff, dtype=torch.float32),
            'face_edge_index':     build_knn_edges(fb, CONFIG['knn_k']),
            'context_x':           cx,
            'context_edge_index':  build_dense_edges(len(cx)),
            'scene_x':             torch.tensor(scene, dtype=torch.float32),
            'y':                   label,
        }

    def __len__(self):  return len(self.face_files)
    def __getitem__(self, i): return self._cache[i]


# ============================================================
# COLLATE
# ============================================================
class SimpleBatch:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)

    def to(self, device):
        for a in ['face_x','face_edge_index','face_batch',
                  'context_x','context_edge_index','context_batch',
                  'scene_x','y']:
            setattr(self, a, getattr(self, a).to(device))
        return self


def custom_collate(batch):
    fx, fei, fb_l, cx, cei, cb_l, sx, yl = [], [], [], [], [], [], [], []
    fn = cn = 0
    for gi, s in enumerate(batch):
        nf = s['face_x'].size(0)
        fx.append(s['face_x']); fei.append(s['face_edge_index'] + fn)
        fb_l.append(torch.full((nf,), gi, dtype=torch.long)); fn += nf
        nc = s['context_x'].size(0)
        cx.append(s['context_x']); cei.append(s['context_edge_index'] + cn)
        cb_l.append(torch.full((nc,), gi, dtype=torch.long)); cn += nc
        sx.append(s['scene_x']); yl.append(s['y'])
    return SimpleBatch(
        face_x=torch.cat(fx), face_edge_index=torch.cat(fei, 1),
        face_batch=torch.cat(fb_l), context_x=torch.cat(cx),
        context_edge_index=torch.cat(cei, 1), context_batch=torch.cat(cb_l),
        scene_x=torch.stack(sx), y=torch.tensor(yl, dtype=torch.long),
        num_graphs=len(batch),
    )


# ============================================================
# MODEL  (A1_full_model — giữ nguyên 100%)
# ============================================================
class MultiLayerGATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, num_layers=2,
                 dropout=0.5, attention_dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                      dropout=attention_dropout, add_self_loops=True,
                      concat=True, bias=False)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, ei):
        h = self.drop(F.relu(self.input_norm(self.input_proj(x))))
        for gat, norm in zip(self.gat_layers, self.norms):
            h_new = self.drop(F.elu(norm(gat(h, ei))))
            h = h + h_new if h.shape == h_new.shape else h_new
        return h


class AttentionPool(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(dim // 4, 1))

    def forward(self, x, batch):
        s = self.mlp(x); s = s - s.max()
        e = torch.exp(s)
        B = batch.max().item() + 1
        d = torch.zeros(B, 1, device=x.device).scatter_add_(0, batch.unsqueeze(1), e)
        w = e / (d[batch] + 1e-8)
        return torch.zeros(B, x.size(1), device=x.device).scatter_add_(
            0, batch.unsqueeze(1).expand_as(x), w * x)


class SceneGuidedFusion(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, 4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(D)

    def forward(self, scene, H_face, H_obj, fb, ob):
        nodes, mask = to_dense_batch(torch.cat([H_face, H_obj]),
                                     torch.cat([fb, ob]))
        out, w = self.attn(scene.unsqueeze(1), nodes, nodes, key_padding_mask=~mask)
        return self.norm(scene + out.squeeze(1)), w


class EmotionalContagion(nn.Module):
    def __init__(self, D, dropout=0.5):
        super().__init__()
        self.gate  = nn.Linear(D, D)
        self.alpha = nn.Parameter(torch.zeros(D))
        self.norm  = nn.LayerNorm(D)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, H, scene, batch):
        g = torch.sigmoid(self.gate(H))
        return self.norm(H + g * (self.alpha * scene[batch]))


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        D = CONFIG['gat_hidden']
        drp = CONFIG['dropout']
        att = CONFIG['attention_dropout']

        self.reduce_face = nn.Sequential(
            nn.LayerNorm(CONFIG['face_dim']),
            nn.Linear(CONFIG['face_dim'], 1024), nn.LayerNorm(1024),
            nn.ReLU(), nn.Dropout(drp),
            nn.Linear(1024, D), nn.LayerNorm(D), nn.ReLU())
        self.reduce_obj = nn.Sequential(
            nn.LayerNorm(CONFIG['object_dim']),
            nn.Linear(CONFIG['object_dim'], D), nn.LayerNorm(D),
            nn.ReLU(), nn.Dropout(drp))
        self.reduce_scene = nn.Sequential(
            nn.LayerNorm(CONFIG['scene_dim']),
            nn.Linear(CONFIG['scene_dim'], D), nn.LayerNorm(D),
            nn.ReLU(), nn.Dropout(drp))

        self.face_gat    = MultiLayerGATv2(D, D, CONFIG['num_heads'],
                                           CONFIG['gat_layers'], drp, att)
        self.context_gat = MultiLayerGATv2(D, D, CONFIG['num_heads'],
                                           CONFIG['gat_layers'], drp, att)

        self.pool_face_br = AttentionPool(D, drp)
        self.pool_face    = AttentionPool(D, drp)

        self.clf_face    = nn.Linear(D, 3)
        self.clf_context = nn.Linear(D, 3)
        self.clf_scene   = nn.Linear(D, 3)

        self.fusion = SceneGuidedFusion(D)
        self.ec     = EmotionalContagion(D, drp)

        self.lambda_face   = nn.Parameter(torch.tensor(0.5))
        self.raw_face_proj = nn.Linear(D, D)
        self.lambda_obj    = nn.Parameter(torch.tensor(0.5))
        self.raw_obj_proj  = nn.Linear(D, D)

        self.clf_whole = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(D * 3, D),
            nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp),
            nn.Linear(D, 3))

        print(f"  [FullModel] Params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, data):
        fp = self.reduce_face(data.face_x)
        op = self.reduce_obj(data.context_x)
        sp = self.reduce_scene(data.scene_x)

        Hf = self.face_gat(fp, data.face_edge_index)
        Ho = self.context_gat(op, data.context_edge_index)

        out_f = self.clf_face(self.pool_face_br(Hf, data.face_batch))
        out_c = self.clf_context(global_mean_pool(Ho, data.context_batch))
        out_s = self.clf_scene(sp)

        fs, _ = self.fusion(sp, Hf, Ho, data.face_batch, data.context_batch)
        Hf    = self.ec(Hf, fs, data.face_batch)

        Hf = Hf + self.lambda_face * self.raw_face_proj(fp)
        Ho = Ho + self.lambda_obj  * self.raw_obj_proj(op)

        feat_f = self.pool_face(Hf, data.face_batch)
        feat_c = global_mean_pool(Ho, data.context_batch)

        out_w = self.clf_whole(torch.cat([feat_f, feat_c, fs], dim=1))
        return out_f, out_c, out_s, out_w


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=40, min_delta=0.001, path='ckpt.pt'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_acc   = 0.0
        self.best_loss  = float('inf')
        self.early_stop = False
        self.path       = path

    def __call__(self, val_loss, val_acc, model):
        improved = (val_acc  > self.best_acc  + self.min_delta or
                   (val_loss < self.best_loss - self.min_delta and
                    val_acc  >= self.best_acc  - 0.002))
        if improved:
            torch.save(model.state_dict(), self.path)
            self.counter = 0
            if val_acc  > self.best_acc:  self.best_acc  = val_acc
            if val_loss < self.best_loss: self.best_loss = val_loss
            print(f"  🔥 Saved  Loss={val_loss:.4f}  Acc={val_acc:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ============================================================
# TRAIN ONE (branch_w, loss_cfg) PAIR
# ============================================================
def train_one(variant_name, branch_w, loss_cfg, train_loader, val_loader):
    print(f"\n{'='*70}")
    print(f"▶  {variant_name}  |  branch_w={branch_w}")
    print(f"   loss={loss_cfg['loss_type']}  neg_w={loss_cfg['neg_w']}  "
          f"neu_w={loss_cfg['neu_w']}  γ={loss_cfg.get('gamma')}")
    print(f"{'='*70}")

    var_dir   = os.path.join(CONFIG['output_dir'], variant_name)
    os.makedirs(var_dir, exist_ok=True)
    ckpt_path = os.path.join(var_dir, 'best_model.pth')

    torch.manual_seed(42); np.random.seed(42)

    model     = FullModel().to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    whole_crit, branch_crit = build_criterion(loss_cfg)
    stopper = EarlyStopping(CONFIG['patience'], path=ckpt_path)

    history = {k: [] for k in [
        'train_loss', 'val_loss',
        'val_acc_whole', 'val_acc_face', 'val_acc_ctx', 'val_acc_scene',
        'val_neg_acc', 'val_neu_acc', 'val_pos_acc',
    ]}

    print(f"\n  {'Ep':>4} | {'TrLoss':>8} | {'VlLoss':>8} | "
          f"{'Whole':>7} | {'Neg':>7} | {'Neu':>7} | {'Pos':>7}")
    print(f"  {'-'*65}")

    t0 = time.time()
    for ep in range(CONFIG['epochs']):
        # ── Train ──────────────────────────────────────────────────────
        model.train(); tl = tn = 0
        for b in tqdm(train_loader, desc=f"  Tr ep{ep+1:03d}", leave=False):
            try:
                b = b.to(CONFIG['device']); optimizer.zero_grad()
                of, oc, os_, ow = model(b)
                loss = (whole_crit(ow, b.y) +
                        branch_w * (branch_crit(of, b.y) +
                                    branch_crit(oc, b.y) +
                                    branch_crit(os_, b.y)))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                optimizer.step()
                tl += loss.item() * len(b.y); tn += len(b.y)
            except Exception: continue

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        vl = vaf = vac = vas = vaw = vn = 0
        preds, labels = [], []
        with torch.no_grad():
            for b in tqdm(val_loader, desc=f"  Val ep{ep+1:03d}", leave=False):
                try:
                    b = b.to(CONFIG['device'])
                    of, oc, os_, ow = model(b)
                    loss = (whole_crit(ow, b.y) +
                            branch_w * (branch_crit(of, b.y) +
                                        branch_crit(oc, b.y) +
                                        branch_crit(os_, b.y)))
                    bs = len(b.y)
                    vl  += loss.item() * bs; vn += bs
                    vaf += (of.argmax(1) == b.y).sum().item()
                    vac += (oc.argmax(1) == b.y).sum().item()
                    vas += (os_.argmax(1) == b.y).sum().item()
                    vaw += (ow.argmax(1) == b.y).sum().item()
                    preds.extend(ow.argmax(1).cpu().numpy())
                    labels.extend(b.y.cpu().numpy())
                except Exception: continue

        scheduler.step()

        vl_ = vl / vn; vaw_ = vaw / vn
        p = np.array(preds); l = np.array(labels)
        neg_a = (p[l==0]==0).mean() if (l==0).sum()>0 else 0.
        neu_a = (p[l==1]==1).mean() if (l==1).sum()>0 else 0.
        pos_a = (p[l==2]==2).mean() if (l==2).sum()>0 else 0.

        for k, v in [('train_loss', tl/tn), ('val_loss', vl_),
                     ('val_acc_whole', vaw_), ('val_acc_face', vaf/vn),
                     ('val_acc_ctx', vac/vn), ('val_acc_scene', vas/vn),
                     ('val_neg_acc', neg_a), ('val_neu_acc', neu_a),
                     ('val_pos_acc', pos_a)]:
            history[k].append(v)

        stopper(vl_, vaw_, model)
        print(f"  {ep+1:>4} | {tl/tn:>8.4f} | {vl_:>8.4f} | "
              f"{vaw_:>7.4f} | {neg_a:>7.4f} | {neu_a:>7.4f} | {pos_a:>7.4f}"
              + (" ★" if stopper.counter == 0 else ""))

        if stopper.early_stop:
            print(f"  ⏹ Early stop ep {ep+1}"); break

    elapsed = (time.time() - t0) / 60
    with open(os.path.join(var_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return {
        'variant': variant_name, 'branch_w': branch_w,
        'best_val_acc': stopper.best_acc,
        'best_val_loss': stopper.best_loss,
        'epochs_run': ep + 1, 'elapsed_min': round(elapsed, 1),
        'history': history, 'ckpt_path': ckpt_path,
    }


# ============================================================
# EVALUATE BEST CHECKPOINT
# ============================================================
def evaluate_best(variant_name, val_loader):
    ckpt = os.path.join(CONFIG['output_dir'], variant_name, 'best_model.pth')
    if not os.path.exists(ckpt):
        print(f"  ⚠ Not found: {ckpt}"); return {}

    model = FullModel().to(CONFIG['device'])
    model.load_state_dict(torch.load(ckpt, map_location=CONFIG['device']))
    model.eval()

    preds, labels = [], []
    af = ac = as_ = aw = n = 0
    with torch.no_grad():
        for b in val_loader:
            b = b.to(CONFIG['device'])
            of, oc, os_, ow = model(b)
            bs = len(b.y)
            af  += (of.argmax(1) == b.y).sum().item()
            ac  += (oc.argmax(1) == b.y).sum().item()
            as_ += (os_.argmax(1) == b.y).sum().item()
            aw  += (ow.argmax(1) == b.y).sum().item()
            n   += bs
            preds.extend(ow.argmax(1).cpu().numpy())
            labels.extend(b.y.cpu().numpy())

    p = np.array(preds); l = np.array(labels)
    print(f"\n  ── EVAL [{variant_name}] ──")
    print(f"  Face={af/n:.4f} Obj={ac/n:.4f} Scene={as_/n:.4f} Whole={aw/n:.4f}")
    print(classification_report(l, p, target_names=['Neg','Neu','Pos'], digits=4))

    cm = confusion_matrix(l, p)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
    plt.title(f'{variant_name}'); plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], variant_name, 'cm.png'), dpi=120)
    plt.close()

    return {
        'val_acc_whole': aw/n, 'val_neg_acc': cm[0,0]/cm[0].sum(),
        'val_neu_acc': cm[1,1]/cm[1].sum(), 'val_pos_acc': cm[2,2]/cm[2].sum(),
    }


# ============================================================
# SUMMARY PLOT
# ============================================================
def plot_summary(all_results, best_loss_name):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f'Branch_w Sweep — GAF2 | Loss={best_loss_name}',
                 fontsize=13, fontweight='bold')

    bws  = [r['branch_w'] for r in all_results]
    cols = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # 1. Val acc over epochs
    ax = axes[0]
    for r, c in zip(all_results, cols):
        h = r.get('history', {})
        if 'val_acc_whole' in h:
            ax.plot(range(1, len(h['val_acc_whole'])+1),
                    h['val_acc_whole'], label=f"bw={r['branch_w']}", lw=1.5, color=c)
    ax.axhline(0.7888, color='red', ls='--', lw=1.2, label='Baseline 78.88%')
    ax.axhline(0.8197, color='gold', ls='--', lw=1.2, label='SOTA 81.97%')
    ax.set_title('Val Whole Acc — All branch_w'); ax.legend(fontsize=8)
    ax.set_xlabel('Epoch')

    # 2. Per-class acc vs branch_w
    ax = axes[1]
    neg_a = [r.get('eval', {}).get('val_neg_acc', 0) for r in all_results]
    neu_a = [r.get('eval', {}).get('val_neu_acc', 0) for r in all_results]
    pos_a = [r.get('eval', {}).get('val_pos_acc', 0) for r in all_results]
    ax.plot(bws, neg_a, 'o-', color='#e74c3c', lw=2, label='Negative')
    ax.plot(bws, neu_a, 's-', color='#3498db', lw=2, label='Neutral')
    ax.plot(bws, pos_a, '^-', color='#2ecc71', lw=2, label='Positive')
    ax.axhline(0.7251, color='#e74c3c', ls=':', lw=1.2, alpha=0.7)
    ax.axhline(0.8096, color='#3498db', ls=':', lw=1.2, alpha=0.7)
    ax.axhline(0.8952, color='#2ecc71', ls=':', lw=1.2, alpha=0.7)
    ax.set_xticks(bws); ax.set_xlabel('branch_w')
    ax.set_title('Per-Class Acc vs branch_w\n(dotted = SOTA)')
    ax.legend()

    # 3. Overall acc vs branch_w
    ax = axes[2]
    whole = [r.get('eval', {}).get('val_acc_whole', r['best_val_acc'])
             for r in all_results]
    bars = ax.bar([str(b) for b in bws], whole, color=cols, alpha=0.85)
    ax.axhline(0.7888, color='red',  ls='--', lw=1.5, label='Baseline 78.88%')
    ax.axhline(0.8197, color='gold', ls='--', lw=1.5, label='SOTA 81.97%')
    for bar, v in zip(bars, whole):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('branch_w'); ax.set_ylim(0.70, 0.86)
    ax.set_title('Val Whole Acc vs branch_w'); ax.legend()

    plt.tight_layout()
    out = os.path.join(CONFIG['output_dir'], 'branchw_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Plot saved: {out}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    # SỬA LỖI Ở ĐÂY: Đổi default sang L9_focal20_neu25
    parser.add_argument('--best_loss', default='L9_focal20_neu25',
                        help='Key của loss config tốt nhất từ loss_sweep. '
                             'Ví dụ: L9_focal20_neu25 hoặc chỉ cần prefix: L9')
    parser.add_argument('--branch_w', nargs='+', type=float, default=None,
                        help='Chỉ chạy một số branch_w cụ thể. Vd: --branch_w 0.3 0.4 0.5')
    parser.add_argument('--list_loss', action='store_true',
                        help='Liệt kê các loss config có thể dùng rồi thoát.')
    args, _ = parser.parse_known_args()

    if args.list_loss:
        print("\n📋 AVAILABLE LOSS CONFIGS:")
        for k, v in LOSS_CONFIGS.items():
            print(f"  {k}")
        return

    # Tìm loss config theo prefix hoặc tên đầy đủ
    loss_key = next((k for k in LOSS_CONFIGS if k.startswith(args.best_loss)), None)
    if loss_key is None:
        print(f"❌ Không tìm thấy loss config '{args.best_loss}'")
        print(f"   Có thể dùng: {list(LOSS_CONFIGS.keys())}")
        return

    loss_cfg = LOSS_CONFIGS[loss_key]
    bw_list  = args.branch_w if args.branch_w else BRANCH_W_VALUES

    print(f"\n{'='*70}")
    print(f"🔬 BRANCH_W SWEEP — GAF2")
    print(f"   Best loss config : {loss_key}")
    print(f"   loss_type={loss_cfg['loss_type']} | neg_w={loss_cfg['neg_w']} | "
          f"neu_w={loss_cfg['neu_w']} | γ={loss_cfg.get('gamma')}")
    print(f"   branch_w values  : {bw_list}")
    print(f"   Baseline: 78.88% | SOTA: 81.97%")
    print(f"{'='*70}\n")

    # Datasets
    print("📂 Loading datasets...")
    train_ds = ConGNN_Dataset('Train')
    val_ds   = ConGNN_Dataset('Val')
    kw = dict(batch_size=CONFIG['batch_size'], collate_fn=custom_collate,
              num_workers=CONFIG['num_workers'], pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    print(f"✅ train={len(train_loader)} batches | val={len(val_loader)} batches\n")

    all_results = []
    summary     = []

    for bw in bw_list:
        vname = f"BW{str(bw).replace('.','')}__{loss_key}"
        try:
            result = train_one(vname, bw, loss_cfg, train_loader, val_loader)
            eval_r = evaluate_best(vname, val_loader)
            result['eval'] = eval_r
            all_results.append(result)

            summary.append({
                'variant':       vname,
                'branch_w':      bw,
                'loss_config':   loss_key,
                'best_val_acc':  round(result['best_val_acc'], 4),
                'eval_whole':    round(eval_r.get('val_acc_whole', 0), 4),
                'eval_neg':      round(eval_r.get('val_neg_acc',   0), 4),
                'eval_neu':      round(eval_r.get('val_neu_acc',   0), 4),
                'eval_pos':      round(eval_r.get('val_pos_acc',   0), 4),
                'epochs_run':    result['epochs_run'],
                'elapsed_min':   result['elapsed_min'],
            })

        except Exception as e:
            print(f"\n❌ branch_w={bw} FAILED: {e}")
            import traceback; traceback.print_exc()
            continue

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*75}")
    print(f"📊 BRANCH_W SWEEP — KẾT QUẢ | loss={loss_key}")
    print(f"   Baseline: 78.88% | SOTA: 81.97%")
    print(f"{'='*75}")
    print(f"  {'branch_w':>9} | {'BestAcc':>8} | {'Whole':>7} | "
          f"{'Neg':>7} | {'Neu':>7} | {'Pos':>7}")
    print(f"  {'-'*60}")
    for r in sorted(summary, key=lambda x: x['eval_whole'], reverse=True):
        marker = ' ← BEST' if r == sorted(summary, key=lambda x: x['eval_whole'], reverse=True)[0] else ''
        print(f"  {r['branch_w']:>9} | {r['best_val_acc']:>8.4f} | "
              f"{r['eval_whole']:>7.4f} | {r['eval_neg']:>7.4f} | "
              f"{r['eval_neu']:>7.4f} | {r['eval_pos']:>7.4f}{marker}")

    # ── Save ──────────────────────────────────────────────────────────────
    with open(os.path.join(CONFIG['output_dir'], 'branchw_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(CONFIG['output_dir'], 'branchw_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        keys = ['variant','branch_w','loss_config','best_val_acc',
                'eval_whole','eval_neg','eval_neu','eval_pos','epochs_run','elapsed_min']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(summary)
    print(f"\n✅ CSV: {csv_path}")

    try:
        plot_summary(all_results, loss_key)
    except Exception as e:
        print(f"⚠ Plot error: {e}")

    print(f"\n✅ DONE! Outputs: {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()