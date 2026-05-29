"""
Paste thẳng vào 1 cell Kaggle, sửa các path ở phần CONFIG bên dưới rồi Run.
Đã fix lỗi trích xuất embedding, thêm PCA và tăng perplexity.
"""

import os, gc, glob, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # [FIX 2] Import thêm PCA

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# ✏️  SỬA PATH Ở ĐÂY
# ═══════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Path tới file .pth
    'a1_ckpt': '/kaggle/input/datasets/drakhight/tsnegroupemow/best_model.pth',
    'z0_ckpt': '/kaggle/input/datasets/drakhight/tsnegroupemow/concatbest_model.pth',

    # Accuracy để hiện trên title (lấy từ kết quả training)
    'acc_a1': 0.9012,
    'acc_z0': 0.8785,

    # Path data
    'face_dir':   '/kaggle/input/datasets/nguynnhtlam12/face-featuresv2',
    'scene_dir':  '/kaggle/input/datasets/drakhight/8726scene-features/scene_features_final/scenes',
    'object_dir': '/kaggle/input/datasets/trieung11/fearturecongnn/objects/objects',

    # Output
    'output_dir': '/kaggle/working/tsne_replot',

    # Thông số model (không cần sửa nếu dùng config gốc)
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

    # t-SNE [FIX 3] Tăng perplexity và n_iter
    'tsne_max_samples': 2000,
    'tsne_perplexity':  50,   
    'tsne_n_iter':      2000, 

    'batch_size':  32,
    'num_workers': 0,
    'split':       'test',   # hoặc 'val' nếu muốn

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
COLOR_MAP   = {0: '#e05252', 1: '#5b9bd5', 2: '#6abf69'}
DRAW_ORDER  = [2, 0, 1]

print(f"Device: {CONFIG['device']}")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════
class ConGNN_Dataset(TorchDataset):
    def __init__(self, split='test', max_faces=32, max_objects=10):
        self.face_root   = CONFIG['face_dir']
        self.scene_root  = CONFIG['scene_dir']
        self.obj_root    = CONFIG['object_dir']
        self.max_faces   = max_faces
        self.max_objects = max_objects
        self.label_map   = {'negative': 0, 'neutral': 1, 'positive': 2}

        pattern = os.path.join(self.face_root, 'faces', split, '**', '*.npz')
        self.face_files = glob.glob(pattern, recursive=True)
        if not self.face_files:
            pattern = os.path.join(self.face_root, split, '**', '*.npz')
            self.face_files = glob.glob(pattern, recursive=True)

        print(f"📊 {split.upper()}: {len(self.face_files)} samples")
        self._build_scene_index()
        self._build_object_index()

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
        if target_type == 'scenes':  return self._scene_index.get(stem)
        if target_type == 'objects': return self._obj_index.get(stem)

    def _load_sample(self, idx):
        face_file = self.face_files[idx]
        label = self.label_map.get(
            os.path.basename(os.path.dirname(face_file)).lower(), 1)

        try:
            data = np.load(face_file)
            face_feat = data['features']; face_boxes = data['boxes']
            if len(face_boxes) > 0:
                si = np.argsort(face_boxes[:, 0])
                face_feat = face_feat[si]; face_boxes = face_boxes[si]
        except Exception:
            face_feat  = np.zeros((1, CONFIG['face_dim']),  dtype=np.float32)
            face_boxes = np.zeros((1, 4), dtype=np.float32)

        face_feat  = face_feat[:self.max_faces]  if len(face_feat)  > 0 else np.zeros((1, CONFIG['face_dim']),  dtype=np.float32)
        face_boxes = face_boxes[:self.max_faces] if len(face_boxes) > 0 else np.zeros((1, 4), dtype=np.float32)
        face_x          = torch.tensor(face_feat, dtype=torch.float32)
        face_edge_index = build_knn_edges(face_boxes, k=CONFIG['knn_k'])

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

        obj_path = self._get_paired_path(face_file, 'objects')
        try:
            if obj_path and os.path.exists(obj_path):
                od = np.load(obj_path)
                obj_feat = od['features'] if 'features' in od else od[od.files[0]]
            else:
                obj_feat = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)
        except Exception:
            obj_feat = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)

        obj_feat  = obj_feat[:self.max_objects]
        context_x = (torch.tensor(obj_feat, dtype=torch.float32) if len(obj_feat) > 0
                     else torch.zeros((1, CONFIG['object_dim']), dtype=torch.float32))
        context_edge_index = build_dense_edges(len(context_x))

        raw_face = face_feat.mean(axis=0).astype(np.float32)
        raw_obj  = obj_feat.mean(axis=0).astype(np.float32) if len(obj_feat) > 0 \
                   else np.zeros(CONFIG['object_dim'], dtype=np.float32)

        return {
            'face_x': face_x, 'face_edge_index': face_edge_index,
            'context_x': context_x, 'context_edge_index': context_edge_index,
            'scene_x': scene_x,
            'raw_face': torch.tensor(raw_face),
            'raw_obj':  torch.tensor(raw_obj),
            'y': label
        }

    def __len__(self):       return len(self.face_files)
    def __getitem__(self, i): return self._load_sample(i)

# ═══════════════════════════════════════════════════════════════════════════
# COLLATE
# ═══════════════════════════════════════════════════════════════════════════
class SimpleBatch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
    def to(self, device):
        for attr in ['face_x','face_edge_index','face_batch',
                     'context_x','context_edge_index','context_batch',
                     'scene_x','raw_face','raw_obj','y']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

def custom_collate(batch):
    fx, fei, fb, cx, cei, cb, sx, rf, ro, yl = [], [], [], [], [], [], [], [], [], []
    fn = cn = 0
    for gi, s in enumerate(batch):
        nf = s['face_x'].size(0)
        fx.append(s['face_x']); fei.append(s['face_edge_index'] + fn)
        fb.append(torch.full((nf,), gi, dtype=torch.long)); fn += nf
        nc = s['context_x'].size(0)
        cx.append(s['context_x']); cei.append(s['context_edge_index'] + cn)
        cb.append(torch.full((nc,), gi, dtype=torch.long)); cn += nc
        sx.append(s['scene_x']); rf.append(s['raw_face']); ro.append(s['raw_obj'])
        yl.append(s['y'])
    return SimpleBatch(
        face_x=torch.cat(fx), face_edge_index=torch.cat(fei, dim=1),
        face_batch=torch.cat(fb), context_x=torch.cat(cx),
        context_edge_index=torch.cat(cei, dim=1), context_batch=torch.cat(cb),
        scene_x=torch.stack(sx), raw_face=torch.stack(rf), raw_obj=torch.stack(ro),
        y=torch.tensor(yl, dtype=torch.long), num_graphs=len(batch)
    )

# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════
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
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, H_face, fused_scene, face_batch):
        return self.norm(H_face + self.drop(self.alpha * fused_scene[face_batch]))

class A1FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        D   = CONFIG['gat_hidden']
        drp = CONFIG['dropout']
        att = CONFIG['attention_dropout']
        self.reduce_face = nn.Sequential(
            nn.LayerNorm(CONFIG['face_dim']),
            nn.Linear(CONFIG['face_dim'], 1024), nn.LayerNorm(1024),
            nn.ReLU(), nn.Dropout(drp),
            nn.Linear(1024, D), nn.LayerNorm(D), nn.ReLU()
        )
        self.reduce_obj = nn.Sequential(
            nn.LayerNorm(CONFIG['object_dim']),
            nn.Linear(CONFIG['object_dim'], D), nn.LayerNorm(D),
            nn.ReLU(), nn.Dropout(drp)
        )
        self.reduce_scene = nn.Sequential(
            nn.LayerNorm(CONFIG['scene_dim']),
            nn.Linear(CONFIG['scene_dim'], D), nn.LayerNorm(D),
            nn.ReLU(), nn.Dropout(drp)
        )
        self.face_gat    = MultiLayerGATv2(D, D, CONFIG['num_heads'], CONFIG['gat_layers'], drp, att)
        self.context_gat = MultiLayerGATv2(D, D, CONFIG['num_heads'], CONFIG['gat_layers'], drp, att)
        self.attn_pool_face_branch = AttentionPool(D, drp)
        self.clf_face    = nn.Linear(D, CONFIG['num_classes'])
        self.clf_context = nn.Linear(D, CONFIG['num_classes'])
        self.clf_scene   = nn.Linear(D, CONFIG['num_classes'])
        self.fusion      = SceneGuidedFusion(D)
        self.ec          = EmotionalContagion(D, drp)
        self.lambda_face   = nn.Parameter(torch.tensor(0.5))
        self.raw_face_proj = nn.Linear(D, D)
        self.lambda_obj    = nn.Parameter(torch.tensor(0.5))
        self.raw_obj_proj  = nn.Linear(D, D)
        self.attn_pool_face = AttentionPool(D, drp)
        self.clf_whole = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(3*D, D), nn.LayerNorm(D),
            nn.ReLU(), nn.Dropout(drp), nn.Linear(D, CONFIG['num_classes'])
        )

    def forward(self, data, return_embedding=False):
        fp = self.reduce_face(data.face_x)
        op = self.reduce_obj(data.context_x)
        sp = self.reduce_scene(data.scene_x)
        H_face = self.face_gat(fp, data.face_edge_index)
        H_obj  = self.context_gat(op, data.context_edge_index)
        out_face    = self.clf_face(self.attn_pool_face_branch(H_face, data.face_batch))
        out_context = self.clf_context(global_mean_pool(H_obj, data.context_batch))
        out_scene   = self.clf_scene(sp)
        fused_scene, _ = self.fusion(sp, H_face, H_obj, data.face_batch, data.context_batch)
        H_face = self.ec(H_face, fused_scene, data.face_batch)
        H_face = H_face + self.lambda_face * self.raw_face_proj(fp)
        H_obj  = H_obj  + self.lambda_obj  * self.raw_obj_proj(op)
        feat_face = self.attn_pool_face(H_face, data.face_batch)
        feat_obj  = global_mean_pool(H_obj, data.context_batch)
        
        combined  = torch.cat([feat_face, feat_obj, fused_scene], dim=1)
        
        if return_embedding:
            # [FIX 1] Trích xuất vector 512-dim ở giữa classifier thay vì 1536-dim combined
            h = self.clf_whole[0](combined) # Dropout
            h = self.clf_whole[1](h)        # Linear (1536 -> 512)
            h = self.clf_whole[2](h)        # LayerNorm
            h = self.clf_whole[3](h)        # ReLU -> Đây chính là vector thực sự khác biệt
            out_whole = self.clf_whole(combined)
            return out_face, out_context, out_scene, out_whole, h
            
        out_whole = self.clf_whole(combined)
        return out_face, out_context, out_scene, out_whole


class Z0ConcatBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        D   = CONFIG['gat_hidden']
        drp = CONFIG['dropout']
        self.enc_face = nn.Sequential(
            nn.LayerNorm(CONFIG['face_dim']),
            nn.Linear(CONFIG['face_dim'], 1024), nn.ReLU(), nn.Dropout(drp),
            nn.Linear(1024, D), nn.LayerNorm(D), nn.ReLU()
        )
        self.enc_obj = nn.Sequential(
            nn.LayerNorm(CONFIG['object_dim']),
            nn.Linear(CONFIG['object_dim'], D), nn.LayerNorm(D),
            nn.ReLU(), nn.Dropout(drp)
        )
        self.enc_scene = nn.Sequential(
            nn.LayerNorm(CONFIG['scene_dim']),
            nn.Linear(CONFIG['scene_dim'], D), nn.LayerNorm(D),
            nn.ReLU(), nn.Dropout(drp)
        )
        self.clf = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(3*D, D), nn.LayerNorm(D),
            nn.ReLU(), nn.Dropout(drp), nn.Linear(D, CONFIG['num_classes'])
        )

    def forward(self, data, return_embedding=False):
        combined  = torch.cat([self.enc_face(data.raw_face),
                                self.enc_obj(data.raw_obj),
                                self.enc_scene(data.scene_x)], dim=1)
                                
        if return_embedding:
            # [FIX 1] Trích xuất vector 512-dim tương tự như A1 Full Model
            h = self.clf[0](combined) # Dropout
            h = self.clf[1](h)        # Linear (1536 -> 512)
            h = self.clf[2](h)        # LayerNorm
            h = self.clf[3](h)        # ReLU
            out_whole = self.clf(combined)
            return None, None, None, out_whole, h
            
        out_whole = self.clf(combined)
        return None, None, None, out_whole

# ═══════════════════════════════════════════════════════════════════════════
# EXTRACT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════
def extract_embeddings(model, ckpt_path, loader, max_samples=2000):
    print(f"  📦 Loading: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=CONFIG['device']))
    model = model.to(CONFIG['device']).eval()
    all_emb, all_y = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Extracting", leave=False):
            if len(all_y) >= max_samples:
                break
            try:
                batch = batch.to(CONFIG['device'])
                # Lấy phần tử [4] là biến h (512-dim) đã được trả về ở hàm forward
                emb = model(batch, return_embedding=True)[4]
                all_emb.append(emb.cpu().numpy())
                all_y.extend(batch.y.cpu().numpy())
            except Exception as e:
                print(f"  ⚠ skip: {e}"); continue
    emb_arr = np.concatenate(all_emb, axis=0)[:max_samples]
    y_arr   = np.array(all_y)[:max_samples]
    print(f"  ✅ Shape: {emb_arr.shape}")
    return emb_arr, y_arr


# ═══════════════════════════════════════════════════════════════════════════
# t-SNE + PLOT
# ═══════════════════════════════════════════════════════════════════════════
def run_tsne(emb):
    print("  📐 Running PCA(50) + t-SNE...")
    # [FIX 2] Chạy PCA(50) trước khi đưa vào t-SNE để giảm nhiễu
    emb_s = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=min(50, emb_s.shape[1]), random_state=42)
    emb_pca = pca.fit_transform(emb_s)
    
    return TSNE(n_components=2, perplexity=CONFIG['tsne_perplexity'],
                n_iter=CONFIG['tsne_n_iter'], random_state=42).fit_transform(emb_pca)


def draw_panel(ax, coords, labels, title):
    ax.set_facecolor('white')
    for cls_idx in DRAW_ORDER:
        mask = np.array(labels) == cls_idx
        pts  = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=COLOR_MAP[cls_idx], label=CLASS_NAMES[cls_idx],
                   alpha=0.75, s=20, edgecolors='none', rasterized=True)
    ax.set_title(title, fontsize=13, fontweight='normal', pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#bbbbbb')
        ax.spines[sp].set_linewidth(0.8)
    leg = ax.legend(title='Sentiment', title_fontsize=10, fontsize=10,
                    loc='upper left', frameon=True, framealpha=0.9,
                    edgecolor='#cccccc', markerscale=1.6)
    leg.get_title().set_fontweight('bold')


def save_plots(coords_z0, y_z0, coords_a1, y_a1):
    out = CONFIG['output_dir']

    # Plot riêng: Concat Baseline
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('white')
    draw_panel(ax, coords_z0, y_z0, 't-SNE Visualization of Concat Baseline')
    plt.tight_layout(pad=1.5)
    fig.savefig(f'{out}/tsne_concat_baseline.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig)
    print(f"✅ Saved: {out}/tsne_concat_baseline.png")

    # Plot riêng: Full Model
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('white')
    draw_panel(ax, coords_a1, y_a1, 't-SNE Visualization of Ours Method')
    plt.tight_layout(pad=1.5)
    fig.savefig(f'{out}/tsne_full_model.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig)
    print(f"✅ Saved: {out}/tsne_full_model.png")

    # Side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    draw_panel(axes[0], coords_z0, y_z0,
               f't-SNE Visualization of Concat Baseline\n(Test Acc = {CONFIG["acc_z0"]:.4f})')
    draw_panel(axes[1], coords_a1, y_a1,
               f't-SNE Visualization of Ours Method\n(Test Acc = {CONFIG["acc_a1"]:.4f})')
    plt.tight_layout(pad=2.5)
    fig.savefig(f'{out}/tsne_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig)
    print(f"✅ Saved: {out}/tsne_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════
print("📂 Loading dataset...")
ds     = ConGNN_Dataset(split=CONFIG['split'])
loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False,
                    collate_fn=custom_collate, num_workers=CONFIG['num_workers'],
                    pin_memory=torch.cuda.is_available())
print(f"✅ {len(loader)} batches\n")

print("🔵 Z0 Concat Baseline...")
emb_z0, y_z0 = extract_embeddings(Z0ConcatBaseline(), CONFIG['z0_ckpt'],
                                   loader, CONFIG['tsne_max_samples'])
gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n🟢 A1 Full Model...")
emb_a1, y_a1 = extract_embeddings(A1FullModel(), CONFIG['a1_ckpt'],
                                   loader, CONFIG['tsne_max_samples'])
gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n📐 t-SNE Z0...")
coords_z0 = run_tsne(emb_z0)

print("\n📐 t-SNE A1...")
coords_a1 = run_tsne(emb_a1)

print("\n🎨 Plotting & saving...")
save_plots(coords_z0, y_z0, coords_a1, y_a1)

print(f"\n🏁 Done! Files tại: {CONFIG['output_dir']}")