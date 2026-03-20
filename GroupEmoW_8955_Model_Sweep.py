# ==========================================
# 
# gat_hidden    = 512
# dropout       = 0.5
# red_dropout   = 0.4      # reduction layer dropout
# clf_w_dropout = 0.6
# max_faces     = 32

# # Training
# lr            = 3e-5
# weight_decay  = 0.05
# batch_size    = 32
# epochs        = 150
# patience      = 25
# grad_clip     = 0.5

# # Loss
# neutral_w     = 1.8      # boost Neutral class weight
# branch_w_max  = 0.30     # auxiliary loss weight max
# branch_w_min  = 0.05     # auxiliary loss weight min
# warmup_epochs = 20
# decay_end     = 80
# ==========================================

import os
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import time
import gc

warnings.filterwarnings('ignore')

# ==========================================
# CONFIG V12-FINAL: DỨT ĐIỂM THESIS
# ==========================================
CONFIG = {
    'face_dir':   '/kaggle/input/datasets/nguynnhtlam12/face-featuresv2',
    'scene_dir':  '/kaggle/input/datasets/drakhight/8726scene-features/scene_features_final/scenes',
    'object_dir': '/kaggle/input/datasets/trieung11/fearturecongnn/objects/objects',
    'output_dir': '/kaggle/working/outputs_v12_final',

    'face_dim':   4096,
    'object_dim': 2048,
    'scene_dim':  1024,

    'gat_hidden':  512,
    'num_classes': 3,
    'dropout':     0.6,
    'attention_dropout': 0.6,

    'knn_k': 3,
    'dense_fallback_threshold': 4,

    'batch_size':      32,
    'lr':           5e-6,
    'weight_decay': 1e-1,
    'grad_clip':    0.5,
    'epochs':       150,
    'patience':     20,

    'branch_w_max':   0.30,
    'branch_w_min':   0.05,
    'warmup_epochs':  20,
    'decay_end':      80,

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'use_ram_cache': True,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==========================================
# UTILS & GRAPH BUILDER
# ==========================================
def build_dense_edges(num_nodes):
    if num_nodes <= 1: return torch.tensor([[0], [0]], dtype=torch.long)
    edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def build_knn_edges(boxes, k=3):
    n = len(boxes)
    if n <= 1: return torch.tensor([[0], [0]], dtype=torch.long)
    if n <= CONFIG['dense_fallback_threshold']: return build_dense_edges(n)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0
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

# ==========================================
# DATASET
# ==========================================
class ConGNN_Dataset(TorchDataset):
    def __init__(self, split='train'):
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        pattern = os.path.join(CONFIG['face_dir'], split, '**', '*.npz')
        self.face_files = glob.glob(pattern, recursive=True)
        if not self.face_files:
            self.face_files = glob.glob(os.path.join(CONFIG['face_dir'], 'faces', split, '**', '*.npz'), recursive=True)

        self._scene_idx = {os.path.splitext(os.path.basename(p))[0]: p
                           for p in glob.glob(os.path.join(CONFIG['scene_dir'], '**', '*.npy'), recursive=True)}
        self._obj_idx = {os.path.splitext(os.path.basename(p))[0]: p
                         for p in glob.glob(os.path.join(CONFIG['object_dir'], '**', '*.npz'), recursive=True)}

        self.max_faces = CONFIG.get('max_faces', 32)
        self._cache = [self._load_sample(i) for i in tqdm(range(len(self.face_files)), desc=f"Caching {split}")] if CONFIG['use_ram_cache'] else None

    def _load_sample(self, idx):
        f_path = self.face_files[idx]
        stem = os.path.splitext(os.path.basename(f_path))[0]
        label = self.label_map.get(os.path.basename(os.path.dirname(f_path)).lower(), 1)

        data = np.load(f_path)
        face_x = torch.from_numpy(data['features'][:self.max_faces]).float()
        face_ei = build_knn_edges(data['boxes'][:self.max_faces])

        s_path = self._scene_idx.get(stem)
        if s_path:
            s_feat = np.load(s_path)
            if s_feat.ndim >= 2:
                s_feat = s_feat.mean(axis=0).flatten()[:CONFIG['scene_dim']]
            else:
                s_feat = s_feat[:CONFIG['scene_dim']]  # (1024,) → lấy thẳng
        else: s_feat = np.zeros(CONFIG['scene_dim'])

        o_path = self._obj_idx.get(stem)
        if o_path:
            o_data = np.load(o_path)
            o_feat = o_data['features'][:10] if 'features' in o_data else o_data[o_data.files[0]][:10]
        else: o_feat = np.zeros((1, CONFIG['object_dim']))

        return {'face_x': face_x, 'face_ei': face_ei, 'scene_x': torch.from_numpy(s_feat).float(),
                'obj_x': torch.from_numpy(o_feat).float(), 'obj_ei': build_dense_edges(len(o_feat)), 'y': label}

    def __len__(self): return len(self.face_files)
    def __getitem__(self, idx): return self._cache[idx] if self._cache else self._load_sample(idx)

# ==========================================
# BATCHING & MODEL
# ==========================================
class SimpleBatch:
    def __init__(self, batch):
        self.face_x = torch.cat([s['face_x'] for s in batch])
        self.scene_x = torch.stack([s['scene_x'] for s in batch])
        self.obj_x = torch.cat([s['obj_x'] for s in batch])
        self.y = torch.tensor([s['y'] for s in batch], dtype=torch.long)
        f_ei, o_ei, f_b, o_b, f_ptr, o_ptr = [], [], [], [], 0, 0
        for i, s in enumerate(batch):
            nf, no = s['face_x'].size(0), s['obj_x'].size(0)
            f_ei.append(s['face_ei'] + f_ptr); o_ei.append(s['obj_ei'] + o_ptr)
            f_b.append(torch.full((nf,), i, dtype=torch.long)); o_b.append(torch.full((no,), i, dtype=torch.long))
            f_ptr += nf; o_ptr += no
        self.face_ei, self.obj_ei = torch.cat(f_ei, dim=1), torch.cat(o_ei, dim=1)
        self.face_batch, self.obj_batch = torch.cat(f_b), torch.cat(o_b)
    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor): setattr(self, k, v.to(device))
        return self

class AttentionPool(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim//4), nn.Tanh(), nn.Dropout(dropout), nn.Linear(dim//4, 1))
    def forward(self, x, batch):
        s = self.mlp(x); weights = s - s.max()
        exp_w = torch.exp(weights)
        denom = torch.zeros(batch.max()+1, 1, device=x.device).scatter_add_(0, batch.unsqueeze(1), exp_w)
        attn = exp_w / (denom[batch] + 1e-8)
        return torch.zeros(batch.max()+1, x.size(1), device=x.device).scatter_add_(0, batch.unsqueeze(1).expand_as(x), attn * x)

class SceneGuided_ConGNN(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None: cfg = CONFIG
        D   = cfg['gat_hidden']
        drp = cfg['dropout']
        rd  = cfg['red_dropout']      # dropout ở reduction layers (chỗ nguy hiểm nhất)
        cwd = cfg['clf_w_dropout']    # dropout trước clf_w

        self.red_f = nn.Sequential(nn.LayerNorm(cfg['face_dim']),   nn.Dropout(rd), nn.Linear(cfg['face_dim'],   D), nn.LayerNorm(D), nn.ReLU())
        self.red_o = nn.Sequential(nn.LayerNorm(cfg['object_dim']), nn.Dropout(rd), nn.Linear(cfg['object_dim'], D), nn.LayerNorm(D), nn.ReLU())
        self.red_s = nn.Sequential(nn.LayerNorm(cfg['scene_dim']),  nn.Dropout(rd), nn.Linear(cfg['scene_dim'],  D), nn.LayerNorm(D), nn.ReLU())

        self.f_gat = MultiLayerGATv2(D, D, heads=4, layers=2, dropout=drp)
        self.o_gat = MultiLayerGATv2(D, D, heads=4, layers=2, dropout=drp)
        self.contagion_alpha = nn.Parameter(torch.zeros(D))
        self.contagion_gate  = nn.Linear(D, D)
        nn.init.constant_(self.contagion_gate.bias, -2.0)
        self.fusion    = nn.MultiheadAttention(D, 4, batch_first=True)
        self.pool_f    = AttentionPool(D, drp)
        self.pool_o    = AttentionPool(D, drp)
        self.pool_f_br = AttentionPool(D, drp)
        self.clf_f = nn.Linear(D, 3)
        self.clf_c = nn.Linear(D, 3)
        self.clf_s = nn.Linear(D, 3)
        self.clf_w = nn.Sequential(nn.Dropout(cwd), nn.Linear(D*3, D), nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp), nn.Linear(D, 3))

    def forward(self, data):
        pf, po, ps = self.red_f(data.face_x), self.red_o(data.obj_x), self.red_s(data.scene_x)
        hf, ho = self.f_gat(pf, data.face_ei), self.o_gat(po, data.obj_ei)
        out_f = self.clf_f(self.pool_f_br(hf, data.face_batch))
        out_c = self.clf_c(global_mean_pool(ho, data.obj_batch))
        out_s = self.clf_s(ps)
        nodes, mask = to_dense_batch(torch.cat([hf, ho], 0), torch.cat([data.face_batch, data.obj_batch], 0))
        attn_s, _ = self.fusion(ps.unsqueeze(1), nodes, nodes, key_padding_mask=~mask)
        fs = F.layer_norm(ps + attn_s.squeeze(1), (ps.size(-1),))
        gate = torch.sigmoid(self.contagion_gate(hf))
        hf = F.layer_norm(hf + gate * (self.contagion_alpha * fs[data.face_batch]), (hf.size(-1),))
        feat_f, feat_o = self.pool_f(hf + pf, data.face_batch), self.pool_o(ho + po, data.obj_batch)
        out_w = self.clf_w(torch.cat([fs, feat_f, feat_o], 1))
        return out_f, out_c, out_s, out_w

class MultiLayerGATv2(nn.Module):
    def __init__(self, in_d, hid_d, heads, layers, dropout):
        super().__init__()
        self.gats = nn.ModuleList([GATv2Conv(hid_d if i>0 else hid_d, hid_d//heads, heads=heads, dropout=0.3) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid_d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
    def forward(self, x, ei):
        h = x
        for gat, norm in zip(self.gats, self.norms):
            h = h + self.drop(F.elu(norm(gat(h, ei))))
        return h

# ==========================================
# TRAINING LOGIC
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.gamma = gamma; self.register_buffer('alpha', alpha)
    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none', label_smoothing=0.1)
        return (self.alpha[target] * (1-torch.exp(-ce))**self.gamma * ce).mean()

def plot_all(history, y_true, y_pred, run_id='best'):
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f"Best Run: {run_id}", fontsize=14, fontweight='bold')
    # Branch losses
    axes[0,0].plot(history['val_f'], label='Face')
    axes[0,0].plot(history['val_c'], label='Ctx')
    axes[0,0].plot(history['val_w'], label='Whole', lw=2, color='red')
    axes[0,0].legend(); axes[0,0].set_title("Branch Losses")
    # Train vs Val Accuracy
    axes[0,1].plot(history['acc_t'], label='Train Acc', color='blue', lw=2)
    axes[0,1].plot(history['acc_w'], label='Val Acc',   color='red',  lw=2)
    axes[0,1].legend(); axes[0,1].set_title("Train vs Val Accuracy")
    # Branch weight schedule
    axes[0,2].plot(history['bw'], label='Weight', color='orange')
    axes[0,2].set_title("Branch Loss Weight Schedule")
    # Total loss
    axes[1,0].plot(history['train_l'], label='Train')
    axes[1,0].plot(history['val_l'],   label='Val')
    axes[1,0].legend(); axes[1,0].set_title("Total Loss (fixed bw=0.30)")
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1,1],
                xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
    axes[1,1].set_title("Confusion Matrix (%)")
    # Classification report
    axes[1,2].axis('off')
    axes[1,2].text(0.1, 0.5, classification_report(y_true, y_pred, digits=4),
                   fontsize=10, family='monospace', verticalalignment='center')
    plt.tight_layout()
    save_path = f"{CONFIG['output_dir']}/best_{run_id}.png"
    plt.savefig(save_path)
    print(f"📊 Plot saved: {save_path}")
    plt.show()



def train_one(cfg, train_ds, val_ds, test_ds, run_id):
    """Train một config, dùng lại cached dataset — không leak data."""
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  collate_fn=SimpleBatch)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], collate_fn=SimpleBatch)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'], collate_fn=SimpleBatch)

    model = SceneGuided_ConGNN(cfg).to(cfg['device'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])

    counts = torch.tensor([4833, 5635, 5082]).float()
    w = (counts.sum()/(3*counts))
    w[1] *= cfg.get('neutral_w', 1.2)  # ✅ neutral weight từ cfg
    w /= w.mean()
    crit_w  = FocalLoss(alpha=w.to(cfg['device']))
    crit_br = nn.CrossEntropyLoss(label_smoothing=0.1)
    FIXED_BW = cfg['branch_w_max']

    history = {k: [] for k in ['train_l', 'val_l', 'val_f', 'val_c', 'val_w', 'acc_w', 'acc_t', 'bw']}
    best_acc, patience_cnt = 0, 0
    out_dir = cfg['output_dir']

    for ep in range(cfg['epochs']):
        bw = cfg['branch_w_min'] if ep >= cfg['decay_end'] else cfg['branch_w_max'] if ep < cfg['warmup_epochs'] else \
             cfg['branch_w_max'] + (ep-cfg['warmup_epochs'])/(cfg['decay_end']-cfg['warmup_epochs'])*(cfg['branch_w_min']-cfg['branch_w_max'])

        model.train(); t_l = t_acc = t_total = 0
        for b in train_loader:
            b.to(cfg['device']); opt.zero_grad()
            of, oc, os_, ow = model(b)
            loss = crit_w(ow, b.y) + bw*(crit_br(of, b.y) + crit_br(oc, b.y) + crit_br(os_, b.y))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step(); t_l += loss.item()
            t_acc += (ow.argmax(1) == b.y).sum().item(); t_total += b.y.size(0)

        model.eval(); v_l = v_f = v_c = v_w = v_acc = total = 0
        with torch.no_grad():
            for b in val_loader:
                b.to(cfg['device']); of, oc, os_, ow = model(b)
                v_l  += (crit_w(ow, b.y) + FIXED_BW*(crit_br(of, b.y) + crit_br(oc, b.y) + crit_br(os_, b.y))).item()*b.y.size(0)
                v_f  += crit_br(of,  b.y).item()*b.y.size(0)
                v_c  += crit_br(oc,  b.y).item()*b.y.size(0)
                v_w  += crit_w(ow,   b.y).item()*b.y.size(0)
                v_acc += (ow.argmax(1) == b.y).sum().item(); total += b.y.size(0)

        history['train_l'].append(t_l/len(train_loader)); history['val_l'].append(v_l/total)
        history['val_f'].append(v_f/total); history['val_c'].append(v_c/total)
        history['val_w'].append(v_w/total); history['acc_w'].append(v_acc/total)
        history['acc_t'].append(t_acc/t_total); history['bw'].append(bw)
        sch.step()

        print(f"[{run_id}] Ep{ep+1} | TrainAcc={t_acc/t_total:.4f} | ValAcc={v_acc/total:.4f} | ValLoss={v_l/total:.4f}")
        if (v_acc/total) > best_acc:
            best_acc = v_acc/total; patience_cnt = 0
            torch.save(model.state_dict(), f"{out_dir}/best_{run_id}.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg['patience']: break

    # Test
    model.load_state_dict(torch.load(f"{out_dir}/best_{run_id}.pth")); model.eval()
    acc_f = acc_c = acc_s = acc_w = total_test = 0; y_t, y_p = [], []
    with torch.no_grad():
        for b in test_loader:
            b.to(cfg['device']); of, oc, os_, ow = model(b); n = b.y.size(0)
            acc_f += (of.argmax(1)==b.y).sum().item(); acc_c += (oc.argmax(1)==b.y).sum().item()
            acc_s += (os_.argmax(1)==b.y).sum().item(); acc_w += (ow.argmax(1)==b.y).sum().item()
            total_test += n; y_t.extend(b.y.cpu().numpy()); y_p.extend(ow.argmax(1).cpu().numpy())

    print(f"\n{'='*60}")
    print(f"[{run_id}] Face={acc_f/total_test:.4f} | Ctx={acc_c/total_test:.4f} | Scene={acc_s/total_test:.4f} | FINAL={acc_w/total_test:.4f}")
    print(f"{'='*60}\n")
    return acc_w / total_test, y_t, y_p, history


if __name__ == "__main__":
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Config tốt nhất từ sweep trước (run1)
    BEST_CFG = {**CONFIG,
        'gat_hidden': 512, 'lr': 3e-5, 'weight_decay': 0.05,
        'dropout': 0.5, 'red_dropout': 0.4, 'clf_w_dropout': 0.6,
        'epochs': 150, 'patience': 25,
        'branch_w_max': 0.30, 'branch_w_min': 0.05,
        'warmup_epochs': 20, 'decay_end': 80,
        'grad_clip': 0.5, 'batch_size': 32,
    }

    # ==========================================
    # SWEEP 2 — test neutral weight + face count
    # run1: baseline   w*1.2 + 32 face
    # run2: w*1.5      + 32 face
    # run3: w*1.8      + 32 face
    # run4: w*1.2      + 16 face
    # run5: w*1.8      + 16 face  ← kết hợp tốt nhất
    # ==========================================
    SWEEP = [
        {**BEST_CFG, 'neutral_w': 1.2, 'max_faces': 32},  # run1 baseline
        {**BEST_CFG, 'neutral_w': 1.5, 'max_faces': 32},  # run2
        {**BEST_CFG, 'neutral_w': 1.8, 'max_faces': 32},  # run3
        {**BEST_CFG, 'neutral_w': 1.2, 'max_faces': 16},  # run4
        {**BEST_CFG, 'neutral_w': 1.8, 'max_faces': 16},  # run5
    ]

    results = []
    cached_datasets = {}  # cache theo max_faces để không load lại thừa
    best_history = None
    best_yt, best_yp = None, None

    for i, cfg in enumerate(SWEEP):
        mf = cfg['max_faces']
        run_id = f"run{i+1}_nw{cfg['neutral_w']}_f{mf}"

        # Cache dataset nếu chưa có cho max_faces này
        if mf not in cached_datasets:
            print(f"\n📦 Caching datasets với max_faces={mf}...")
            CONFIG['max_faces'] = mf  # patch để ConGNN_Dataset dùng đúng
            cached_datasets[mf] = (
                ConGNN_Dataset('train'),
                ConGNN_Dataset('val'),
                ConGNN_Dataset('test'),
            )
        train_ds, val_ds, test_ds = cached_datasets[mf]

        print(f"\n{'='*70}")
        print(f"🚀 {run_id}")
        print(f"   neutral_w={cfg['neutral_w']} | max_faces={mf} | lr={cfg['lr']} | wd={cfg['weight_decay']}")
        print(f"{'='*70}")

        acc, y_t, y_p, history = train_one(cfg, train_ds, val_ds, test_ds, run_id)
        results.append((run_id, acc, y_t, y_p, history))
        gc.collect(); torch.cuda.empty_cache()

    # Tổng kết
    print(f"\n{'='*70}")
    print("📊 SWEEP RESULTS (sorted)")
    print(f"{'='*70}")
    for run_id, acc, *_ in sorted(results, key=lambda x: -x[1]):
        print(f"  {acc:.4f}  {run_id}")
    print(f"{'='*70}")

    # ✅ Vẽ biểu đồ cho run tốt nhất
    best = max(results, key=lambda x: x[1])
    best_run_id, best_acc, best_yt, best_yp, best_history = best
    print(f"\n🏆 Best run: {best_run_id} | FINAL={best_acc:.4f}")
    plot_all(best_history, best_yt, best_yp, best_run_id)