import os
import re
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
# CONFIG
# ==========================================
CONFIG = {
    'face_dir':   '/kaggle/input/datasets/nguynnhtlam12/face-featuresv2',
    'scene_dir':  '/kaggle/input/datasets/drakhight/8726scene-features/scene_features_final/scenes',
    'object_dir': '/kaggle/input/datasets/trieung11/fearturecongnn/objects/objects',
    'output_dir': '/kaggle/working/outputs_v10_emotional_contagion',

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

    # ── V6: Branch loss warm-up config ──────────────────────────────────────
    # Giai đoạn 1 (warm-up):   epoch [0, warmup_epochs)       → branch_w = branch_w_max
    # Giai đoạn 2 (decay):     epoch [warmup_epochs, decay_end] → tuyến tính từ max → min
    # Giai đoạn 3 (late):      epoch [decay_end, ∞)            → branch_w = branch_w_min
    #
    # Lý do: Giai đoạn đầu branch loss giúp các head học đặc trưng riêng (không conflict).
    # Sau đó giảm dần để out_whole được ưu tiên — tránh gradient conflict cuối training.
    'branch_w_max':   0.30,   # weight branch loss lúc đầu (giữ nguyên gốc V4)
    'branch_w_min':   0.30,   # weight branch loss cuối training
    'warmup_epochs':  20,     # giữ cao trong 20 epoch đầu
    'decay_end':      80,     # decay tuyến tính từ epoch 20 → 80
    # ────────────────────────────────────────────────────────────────────────

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'debug_mode': False,
    'use_ram_cache': True,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# ==========================================
# SYSTEM INFO
# ==========================================
def print_system_info():
    print("="*80)
    print("🖥️  SYSTEM INFORMATION")
    print("="*80)
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print(f"Device: {CONFIG['device']}")
    print("="*80 + "\n")


# ==========================================
# V6: Branch weight scheduler
# ==========================================
def get_branch_weight(epoch: int) -> float:
    """
    Trả về branch loss weight tại epoch hiện tại.

    Chiến lược 3 giai đoạn:
      [0,         warmup_epochs):  constant = branch_w_max
      [warmup_epochs, decay_end]:  linear decay  max → min
      (decay_end, ∞):              constant = branch_w_min

    Tại sao linear thay vì cosine?
    → Linear rõ ràng, dễ debug, không có "bump" ở đầu như cosine decay.
    → Tốc độ giảm đều mỗi epoch → gradient conflict giảm dần từ từ,
      không gây shock đột ngột cho các branch head.
    """
    w_max      = CONFIG['branch_w_max']
    w_min      = CONFIG['branch_w_min']
    ep_warm    = CONFIG['warmup_epochs']
    ep_decay   = CONFIG['decay_end']

    if epoch < ep_warm:
        return w_max

    if epoch >= ep_decay:
        return w_min

    # Linear interpolation trong [ep_warm, ep_decay]
    progress = (epoch - ep_warm) / (ep_decay - ep_warm)   # 0.0 → 1.0
    return w_max + progress * (w_min - w_max)              # w_max → w_min


# ==========================================
# KNN Graph Builder
# ==========================================
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
        d = dist[i].copy()
        d[i] = np.inf
        nn_idx = np.argsort(d)[:actual_k]
        for j in nn_idx:
            src_list.append(i)
            dst_list.append(j)
            src_list.append(j)
            dst_list.append(i)

    edges = list(set(zip(src_list, dst_list)))
    if len(edges) == 0:
        return torch.tensor([[0], [0]], dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_dense_edges(num_nodes):
    if num_nodes <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ==========================================
# DATASET
# ==========================================
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
        if len(self.face_files) == 0:
            pattern = os.path.join(self.face_root, split, '**', '*.npz')
            self.face_files = glob.glob(pattern, recursive=True)

        print(f"📊 {split.upper()}: Found {len(self.face_files)} samples")
        if len(self.face_files) == 0:
            raise ValueError(f"❌ No data found in {self.face_root} for split '{split}'")

        self._build_scene_index()
        self._build_object_index()

        self._cache = None
        if self.use_cache:
            self._preload_all(split)

    def _build_scene_index(self):
        print("  🗂️  Building scene file index...")
        self._scene_index = {}
        for npy_path in glob.glob(os.path.join(self.scene_root, '**', '*.npy'), recursive=True):
            stem = os.path.splitext(os.path.basename(npy_path))[0]
            self._scene_index[stem] = npy_path
        print(f"  ✅ Scene index: {len(self._scene_index)} files found")

    def _build_object_index(self):
        print("  🗂️  Building object file index...")
        self._obj_index = {}
        for npz_path in glob.glob(os.path.join(self.obj_root, '**', '*.npz'), recursive=True):
            stem = os.path.splitext(os.path.basename(npz_path))[0]
            self._obj_index[stem] = npz_path
        print(f"  ✅ Object index: {len(self._obj_index)} files found")

    def _get_paired_path(self, face_path, target_type):
        stem = os.path.splitext(os.path.basename(face_path))[0]
        if target_type == 'scenes':
            path = self._scene_index.get(stem, None)
            if path is None and CONFIG['debug_mode']:
                print(f"⚠️ Scene NOT FOUND for: {stem}")
            return path
        elif target_type == 'objects':
            path = self._obj_index.get(stem, None)
            if path is None and CONFIG['debug_mode']:
                print(f"⚠️ Object NOT FOUND for: {stem}")
            return path
        return None

    def _preload_all(self, split):
        print(f"  💾 Preloading {split.upper()} into RAM...")
        t0 = time.time()
        self._cache = [self._load_sample(i) for i in
                       tqdm(range(len(self.face_files)), desc=f"  Caching {split}", leave=False)]
        ram_mb = sum(s['face_x'].numel()*4 + s['context_x'].numel()*4
                     for s in self._cache) / 1e6
        print(f"  ✅ Cached {len(self._cache)} samples in {time.time()-t0:.1f}s | ~{ram_mb:.0f} MB\n")

    def _load_sample(self, idx):
        face_file = self.face_files[idx]
        label = self.label_map.get(os.path.basename(os.path.dirname(face_file)).lower(), 1)

        try:
            data = np.load(face_file)
            face_feat  = data['features']
            face_boxes = data['boxes']
            if len(face_boxes) > 0:
                sort_idx   = np.argsort(face_boxes[:, 0])
                face_feat  = face_feat[sort_idx]
                face_boxes = face_boxes[sort_idx]
        except Exception as e:
            if CONFIG['debug_mode']: print(f"⚠️ face load: {e}")
            face_feat  = np.zeros((1, CONFIG['face_dim']), dtype=np.float32)
            face_boxes = np.zeros((1, 4), dtype=np.float32)

        face_feat  = face_feat[:self.max_faces]  if len(face_feat)  > 0 else np.zeros((1, CONFIG['face_dim']), dtype=np.float32)
        face_boxes = face_boxes[:self.max_faces] if len(face_boxes) > 0 else np.zeros((1, 4), dtype=np.float32)

        face_x          = torch.tensor(face_feat, dtype=torch.float32)
        face_edge_index = build_knn_edges(face_boxes, k=CONFIG['knn_k'])

        scene_path = self._get_paired_path(face_file, 'scenes')
        try:
            if scene_path and os.path.exists(scene_path):
                scene_feat = np.load(scene_path)
                if scene_feat.ndim == 4:
                    scene_feat = scene_feat.mean(axis=(0, 2, 3))
                elif scene_feat.ndim == 3:
                    scene_feat = scene_feat.mean(axis=(-2, -1))
                elif scene_feat.ndim == 2:
                    if scene_feat.shape[0] == 1:
                        scene_feat = scene_feat.squeeze(0)
                    elif scene_feat.shape[-1] == CONFIG['scene_dim']:
                        scene_feat = scene_feat.mean(axis=0)
                    else:
                        scene_feat = scene_feat.mean(axis=-1)
                scene_feat = scene_feat.flatten()[:CONFIG['scene_dim']]
            else:
                scene_feat = np.zeros(CONFIG['scene_dim'], dtype=np.float32)
        except Exception as e:
            if CONFIG['debug_mode']: print(f"⚠️ scene load: {e}")
            scene_feat = np.zeros(CONFIG['scene_dim'], dtype=np.float32)

        if len(scene_feat) < CONFIG['scene_dim']:
            scene_feat = np.pad(scene_feat, (0, CONFIG['scene_dim'] - len(scene_feat)))
        scene_x = torch.tensor(scene_feat.astype(np.float32), dtype=torch.float32)

        obj_path = self._get_paired_path(face_file, 'objects')
        try:
            if obj_path and os.path.exists(obj_path):
                obj_data = np.load(obj_path)
                obj_feat = obj_data['features'] if 'features' in obj_data else obj_data[obj_data.files[0]]
            else:
                obj_feat = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)
        except Exception as e:
            if CONFIG['debug_mode']: print(f"⚠️ obj load: {e}")
            obj_feat = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)

        obj_feat = obj_feat[:self.max_objects]
        if len(obj_feat) > 0:
            context_x = torch.tensor(obj_feat, dtype=torch.float32)
        else:
            context_x = torch.zeros((1, CONFIG['object_dim']), dtype=torch.float32)

        context_edge_index = build_dense_edges(len(context_x))

        return {
            'face_x':             face_x,
            'face_edge_index':    face_edge_index,
            'context_x':          context_x,
            'context_edge_index': context_edge_index,
            'scene_x':            scene_x,
            'y':                  label
        }

    def __len__(self): return len(self.face_files)
    def __getitem__(self, idx):
        return self._cache[idx] if self._cache is not None else self._load_sample(idx)


# ==========================================
# CUSTOM COLLATE
# ==========================================
class SimpleBatch:
    def __init__(self, face_x, face_edge_index, face_batch,
                 context_x, context_edge_index, context_batch,
                 scene_x, y, num_graphs):
        self.face_x             = face_x
        self.face_edge_index    = face_edge_index
        self.face_batch         = face_batch
        self.context_x          = context_x
        self.context_edge_index = context_edge_index
        self.context_batch      = context_batch
        self.scene_x            = scene_x
        self.y                  = y
        self.num_graphs         = num_graphs

    def to(self, device):
        for attr in ['face_x','face_edge_index','face_batch',
                     'context_x','context_edge_index','context_batch',
                     'scene_x','y']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def __getstate__(self):    return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)


def custom_collate(batch):
    fx, fei, fb = [], [], []
    cx, cei, cb = [], [], []
    sx, yl = [], []
    fn = cn = 0

    for gi, s in enumerate(batch):
        nf = s['face_x'].size(0)
        fx.append(s['face_x'])
        fei.append(s['face_edge_index'] + fn)
        fb.append(torch.full((nf,), gi, dtype=torch.long))
        fn += nf

        nc = s['context_x'].size(0)
        cx.append(s['context_x'])
        cei.append(s['context_edge_index'] + cn)
        cb.append(torch.full((nc,), gi, dtype=torch.long))
        cn += nc

        sx.append(s['scene_x'])
        yl.append(s['y'])

    return SimpleBatch(
        face_x=torch.cat(fx),
        face_edge_index=torch.cat(fei, dim=1),
        face_batch=torch.cat(fb),
        context_x=torch.cat(cx),
        context_edge_index=torch.cat(cei, dim=1),
        context_batch=torch.cat(cb),
        scene_x=torch.stack(sx),
        y=torch.tensor(yl, dtype=torch.long),
        num_graphs=len(batch)
    )


# ==========================================
# GAT
# ==========================================
class MultiLayerGATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, num_layers=2,
                 dropout=0.5, attention_dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.norms      = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                          dropout=attention_dropout, add_self_loops=True,
                          concat=True, bias=False)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.drop(F.relu(self.input_norm(self.input_proj(x))))
        for gat, norm in zip(self.gat_layers, self.norms):
            h_new = self.drop(F.elu(norm(gat(h, edge_index))))
            h = h + h_new if h.shape == h_new.shape else h_new
        return h


# ==========================================
# Attention Pool
# ==========================================
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x, batch):
        scores = self.score_mlp(x)          # [N, 1]
        scores = scores - scores.max()      # numerical stability
        exp_s  = torch.exp(scores)          # [N, 1]

        B = batch.max().item() + 1
        denom = torch.zeros(B, 1, device=x.device)
        denom.scatter_add_(0, batch.unsqueeze(1), exp_s)
        weight = exp_s / (denom[batch] + 1e-8)

        out = torch.zeros(B, x.size(1), device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x),
                         weight * x)
        return out


class SceneGuidedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, scene_feat, face_nodes, obj_nodes, face_batch, obj_batch):
        all_nodes = torch.cat([face_nodes, obj_nodes], dim=0)
        all_batch = torch.cat([face_batch,  obj_batch],  dim=0)
        dense_nodes, mask = to_dense_batch(all_nodes, all_batch)
        query = scene_feat.unsqueeze(1)
        attn_out, attn_weights = self.cross_attn(
            query, dense_nodes, dense_nodes, key_padding_mask=~mask
        )
        fused_scene = self.layer_norm(query + attn_out).squeeze(1)
        return fused_scene, attn_weights



class EmotionalContagion(nn.Module):
    """
    V10: Emotional Contagion — bầu không khí nhóm uốn nắn biểu cảm từng người.

    Ý tưởng:
        H_face_final_i = H_face_i + alpha * fused_scene[graph_i]

    Tức là mỗi face node nhận thêm "group atmosphere" từ fused_scene của ảnh đó.
    - fused_scene đã encode cả scene + face + object qua SceneGuidedFusion
    - Broadcast theo face_batch: mỗi node lấy fused_scene của ảnh mình thuộc về
    - alpha: learnable scalar — model tự học mức độ ảnh hưởng của bầu không khí
      lên từng node (có thể = 0 nếu không có ích → safe to add)

    Tại sao làm ở node-level TRƯỚC khi pool?
    - Pool-level: cộng sau khi pool → mất thông tin "node nào bị ảnh hưởng nhiều hơn"
    - Node-level: attention pool SAU contagion → học được "node nào phản ánh
      group atmosphere tốt nhất" → focal person detection tự nhiên hơn

    Tại sao fused_scene thay vì scene_proj thuần?
    - fused_scene = scene attend over (face_nodes + obj_nodes) qua cross-attention
    - Tức là nó đã "nhìn thấy" group → encode group emotion tốt hơn scene thuần
    """
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        # alpha: learnable per-dim scale (không phải scalar để linh hoạt hơn)
        # Khởi tạo nhỏ để không làm loạn H_face ban đầu
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, H_face, fused_scene, face_batch):
        """
        H_face:      [N_total_faces, D]
        fused_scene: [B, D]            — group atmosphere per image
        face_batch:  [N_total_faces]   — graph index của từng node
        return:      [N_total_faces, D]
        """
        # Broadcast fused_scene → mỗi face node lấy atmosphere của ảnh mình
        atmosphere = fused_scene[face_batch]           # [N, D]
        # Contagion: cộng atmosphere được scale bởi alpha (per-dim)
        H_out = H_face + self.drop(self.alpha * atmosphere)
        return self.norm(H_out)

# ==========================================
# MAIN MODEL — V10 (base: V4 + branch weight warm-up)
# ==========================================
class SceneGuided_ConGNN(nn.Module):
    def __init__(self):
        super().__init__()
        D       = CONFIG['gat_hidden']
        drp     = CONFIG['dropout']
        att_drp = CONFIG['attention_dropout']

# Thêm LayerNorm ở ngay đầu tiên để chuẩn hóa raw features (chống sốc dữ liệu)
        self.reduce_face = nn.Sequential(
            nn.LayerNorm(CONFIG['face_dim']),   # <--- THÊM MỚI Ở ĐÂY
            nn.Linear(CONFIG['face_dim'], 1024),
            nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(drp),
            nn.Linear(1024, D),
            nn.LayerNorm(D), nn.ReLU()
        )
        
        self.reduce_obj = nn.Sequential(
            nn.LayerNorm(CONFIG['object_dim']), # <--- THÊM MỚI Ở ĐÂY
            nn.Linear(CONFIG['object_dim'], D),
            nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp)
        )
        
        self.reduce_scene = nn.Sequential(
            nn.LayerNorm(CONFIG['scene_dim']),  # <--- THÊM MỚI Ở ĐÂY
            nn.Linear(CONFIG['scene_dim'], D),
            nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp)
        )

        self.face_gat = MultiLayerGATv2(
            in_dim=D, hidden_dim=D,
            num_heads=CONFIG['num_heads'], num_layers=CONFIG['gat_layers'],
            dropout=drp, attention_dropout=att_drp
        )
        self.context_gat = MultiLayerGATv2(
            in_dim=D, hidden_dim=D,
            num_heads=CONFIG['num_heads'], num_layers=CONFIG['gat_layers'],
            dropout=drp, attention_dropout=att_drp
        )

        self.clf_face    = nn.Linear(D, CONFIG['num_classes'])
        self.clf_context = nn.Linear(D, CONFIG['num_classes'])
        self.clf_scene   = nn.Linear(D, CONFIG['num_classes'])

        self.scene_guided_fusion = SceneGuidedFusion(hidden_dim=D)

        # V3: node-level residual
        self.lambda_face   = nn.Parameter(torch.tensor(0.5))
        self.lambda_obj    = nn.Parameter(torch.tensor(0.5))
        self.raw_face_proj = nn.Linear(D, D)
        self.raw_obj_proj  = nn.Linear(D, D)

        # V10: Emotional Contagion
        self.emotional_contagion = EmotionalContagion(D, dropout=drp)

        # V4: attention pool cho face (branch + whole)
        self.attn_pool_face        = AttentionPool(D, dropout=drp)
        self.attn_pool_face_branch = AttentionPool(D, dropout=drp)

        self.clf_whole = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(D * 3, D),
            nn.LayerNorm(D), nn.ReLU(), nn.Dropout(drp),
            nn.Linear(D, CONFIG['num_classes'])
        )

    def forward(self, data):
        face_x_proj = self.reduce_face(data.face_x)
        obj_x_proj  = self.reduce_obj(data.context_x)
        scene_proj  = self.reduce_scene(data.scene_x)

        H_face = self.face_gat(face_x_proj, data.face_edge_index)
        H_obj  = self.context_gat(obj_x_proj, data.context_edge_index)

        # Branch classifiers
        out_face    = self.clf_face(self.attn_pool_face_branch(H_face, data.face_batch))
        out_context = self.clf_context(global_mean_pool(H_obj, data.context_batch))
        out_scene   = self.clf_scene(scene_proj)

        fused_scene, _ = self.scene_guided_fusion(
            scene_proj, H_face, H_obj,
            data.face_batch, data.context_batch
        )

        # V10: Emotional Contagion — group atmosphere uốn nắn từng face node
        # Áp dụng TRƯỚC node-level residual để residual học trên top của contagion
        H_face = self.emotional_contagion(H_face, fused_scene, data.face_batch)

        # Node-level residual trước khi pool
        H_face_res = H_face + self.lambda_face * self.raw_face_proj(face_x_proj)
        H_obj_res  = H_obj  + self.lambda_obj  * self.raw_obj_proj(obj_x_proj)

        feat_face = self.attn_pool_face(H_face_res, data.face_batch)
        feat_obj  = global_mean_pool(H_obj_res, data.context_batch)

        combined  = torch.cat([fused_scene, feat_face, feat_obj], dim=1)
        out_whole = self.clf_whole(combined)

        return out_face, out_context, out_scene, out_whole


# ==========================================
# Loss Function
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, num_classes=3, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        if alpha is None:
            self.alpha = torch.tensor([1.0, 2.0, 1.0])
        else:
            self.alpha = torch.tensor(alpha)

    def forward(self, pred, target):
        target     = target.long()
        self.alpha = self.alpha.to(pred.device)
        log_probs  = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        ce_loss = torch.sum(-true_dist * log_probs, dim=-1)
        pt      = torch.exp(log_probs).gather(1, target.unsqueeze(1)).squeeze(1)
        focal   = self.alpha[target] * ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


# ==========================================
# V6: compute_loss nhận thêm branch_w
# ==========================================
def compute_loss(out_f, out_c, out_s, out_w, labels,
                 focal_criterion, ce_criterion,
                 branch_w: float = 0.30):
    """
    L = FocalLoss(whole) + branch_w * [CE(face) + CE(ctx) + CE(scene)]

    branch_w giảm dần theo epoch (truyền vào từ training loop).
    Khi branch_w → 0, gradient chỉ flow qua clf_whole — không còn conflict.
    """
    labels  = labels.long()
    L_w     = focal_criterion(out_w, labels)
    L_f     = ce_criterion(out_f, labels)
    L_c     = ce_criterion(out_c, labels)
    L_s     = ce_criterion(out_s, labels)
    L_total = L_w + branch_w * (L_f + L_c + L_s) 0,7/0.3, 0.2/0,4
    return L_total, L_f.item(), L_c.item(), L_s.item(), L_w.item()


# ==========================================
# EARLY STOPPING
# ==========================================
class ImprovedEarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, path='checkpoint.pt'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = np.inf
        self.best_acc   = 0
        self.early_stop = False
        self.path       = path

    def __call__(self, val_loss, val_acc, model):
        improved = (val_acc > self.best_acc + self.min_delta or
                    (val_loss < self.best_loss - self.min_delta and
                     val_acc >= self.best_acc - 0.002))
        if improved:
            self.best_loss = val_loss
            self.best_acc  = val_acc
            torch.save(model.state_dict(), self.path)
            self.counter = 0
            print(f"  🔥 Best! Loss={val_loss:.4f} | Acc={val_acc:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_lr(optimizer):
    return next(iter(optimizer.param_groups))['lr']


# ==========================================
# PLOTTING
# ==========================================
def plot_results(history, y_true, y_pred, suffix=''):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(28, 22))
    gs  = GridSpec(
        2, 4, figure=fig,
        height_ratios=[0.9, 2.2],
        hspace=0.38, wspace=0.38
    )

    ax_branch  = fig.add_subplot(gs[0, 0:2])
    ax_acc     = fig.add_subplot(gs[0, 2])
    ax_branchw = fig.add_subplot(gs[0, 3])   # V6: thêm plot branch_w schedule
    ax_loss    = fig.add_subplot(gs[1, 0:2])
    ax_cm      = fig.add_subplot(gs[1, 2:4])

    # --- Branch Losses ---
    for name, key in [('Face','val_loss_f'), ('Context','val_loss_c'), ('Whole','val_loss_w')]:
        ax_branch.plot(history[key], label=name, lw=2)
    ax_branch.set_title('Branch Losses', fontsize=13)
    ax_branch.legend(); ax_branch.grid(alpha=0.3)

    # --- Val Accuracy ---
    ax_acc.plot(history['val_acc_whole'],   label='Whole',   lw=2.5, color='red')
    ax_acc.plot(history['val_acc_face'],    label='Face',    lw=2,   alpha=0.7)
    ax_acc.plot(history['val_acc_context'], label='Context', lw=2,   alpha=0.7)
    ax_acc.plot(history['val_acc_scene'],   label='Scene',   lw=2,   alpha=0.7, color='green')
    ax_acc.axhline(y=0.90, color='g', linestyle='--', label='Target 90%')
    ax_acc.set_title('Val Accuracy', fontsize=13)
    ax_acc.legend(fontsize=9); ax_acc.grid(alpha=0.3)

    # --- V6: Branch Weight Schedule ---
    ax_branchw.plot(history['branch_w'], lw=2, color='orange')
    ax_branchw.set_title('Branch Loss Weight', fontsize=13)
    ax_branchw.set_xlabel('Epoch')
    ax_branchw.set_ylim(0, CONFIG['branch_w_max'] * 1.1)
    ax_branchw.axhline(y=CONFIG['branch_w_min'], color='red', linestyle='--',
                       label=f"min={CONFIG['branch_w_min']}")
    ax_branchw.axvline(x=CONFIG['warmup_epochs'], color='gray', linestyle=':',
                       label=f"warm={CONFIG['warmup_epochs']}")
    ax_branchw.axvline(x=CONFIG['decay_end'],     color='gray', linestyle='--',
                       label=f"end={CONFIG['decay_end']}")
    ax_branchw.legend(fontsize=8); ax_branchw.grid(alpha=0.3)

    # --- Total Loss ---
    ax_loss.plot(history['train_loss'], label='Train', lw=2.5)
    ax_loss.plot(history['val_loss'],   label='Val',   lw=2.5)
    ax_loss.set_title('Total Loss', fontsize=17, fontweight='bold')
    ax_loss.legend(fontsize=13); ax_loss.grid(alpha=0.3)
    ax_loss.tick_params(labelsize=13)

    # --- Confusion Matrix ---
    cm_pct = confusion_matrix(y_true, y_pred, normalize='true') * 100
    sns.heatmap(
        cm_pct, annot=True, fmt='.2f', cmap='Blues', ax=ax_cm,
        xticklabels=['Neg', 'Neu', 'Pos'],
        yticklabels=['Neg', 'Neu', 'Pos'],
        annot_kws={"size": 24, "weight": "bold"},
        linewidths=2, linecolor='white',
        square=True
    )
    ax_cm.set_title('Confusion Matrix (%)', fontsize=17, fontweight='bold')
    ax_cm.tick_params(labelsize=15)
    ax_cm.set_xlabel('Predicted', fontsize=14)
    ax_cm.set_ylabel('True', fontsize=14)

    fname = f"{CONFIG['output_dir']}/results{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot saved: {fname}")


# ==========================================
# DEBUG
# ==========================================
def debug_data_loading(dataset, num_samples=5):
    print("\n🔍 DEBUG: Checking scene/object feature shapes...")
    scene_zeros = obj_zeros = 0
    for i in range(min(num_samples, len(dataset.face_files))):
        face_file  = dataset.face_files[i]
        scene_path = dataset._get_paired_path(face_file, 'scenes')
        obj_path   = dataset._get_paired_path(face_file, 'objects')
        scene_exists = os.path.exists(scene_path) if scene_path else False
        obj_exists   = os.path.exists(obj_path)   if obj_path   else False
        if scene_exists:
            s = np.load(scene_path)
            print(f"  Scene [{i}]: shape={s.shape}, dtype={s.dtype}")
        else:
            print(f"  Scene [{i}]: ❌ NOT FOUND — {scene_path}"); scene_zeros += 1
        if obj_exists:
            try:
                o    = np.load(obj_path)
                feat = o['features'] if 'features' in o else o[o.files[0]]
                print(f"  Obj   [{i}]: shape={feat.shape}, dtype={feat.dtype}")
            except Exception as e:
                print(f"  Obj   [{i}]: ❌ {e}")
        else:
            print(f"  Obj   [{i}]: ❌ NOT FOUND — {obj_path}"); obj_zeros += 1
    print(f"\n  Scene missing: {scene_zeros}/{num_samples}")
    print(f"  Obj   missing: {obj_zeros}/{num_samples}\n")


# ==========================================
# MAIN
# ==========================================
def main():
    print_system_info()

    if torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect()

    print("📂 Loading datasets...")
    t0 = time.time()
    try:
        train_ds = ConGNN_Dataset('train')
        val_ds   = ConGNN_Dataset('val')
        test_ds  = ConGNN_Dataset('test')
        print(f"✅ Loaded in {time.time()-t0:.1f}s\n")
    except Exception as e:
        print(f"❌ {e}"); return

    debug_data_loading(train_ds, num_samples=10)

    kw = {
        'batch_size': CONFIG['batch_size'],
        'collate_fn': custom_collate,
        'num_workers': CONFIG['num_workers'],
        'pin_memory': torch.cuda.is_available()
    }
    if CONFIG['num_workers'] > 0 and CONFIG['prefetch_factor']:
        kw['prefetch_factor'] = CONFIG['prefetch_factor']

    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)
    print(f"✅ Loaders: train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}\n")

    print("🧪 Sanity check first batch...")
    try:
        tb = next(iter(train_loader)).to(CONFIG['device'])
        print(f"✅ Face: {tb.face_x.shape} | Context: {tb.context_x.shape} | Scene: {tb.scene_x.shape}")
        scene_nonzero = (tb.scene_x.abs().sum(dim=1) > 0).float().mean()
        ctx_nonzero   = (tb.context_x.abs().sum(dim=1) > 0).float().mean()
        print(f"   Scene non-zero ratio: {scene_nonzero:.2%} | Context non-zero: {ctx_nonzero:.2%}")
        if scene_nonzero < 0.5: print("   ⚠️  WARNING: >50% scene features are zero!")
        if ctx_nonzero   < 0.5: print("   ⚠️  WARNING: >50% context features are zero!")
        del tb
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ {e}"); return

    print("\n🏗️  Building model...")
    model = SceneGuided_ConGNN().to(CONFIG['device'])
    tp = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {tp:,} | Trainable: {tr:,}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['epochs'],
        eta_min=CONFIG['min_lr']
    )

    #focal_criterion = FocalLoss(CONFIG['num_classes'], gamma=2.0,
    #                            label_smoothing=CONFIG['label_smoothing'])
    #ce_criterion    = nn.CrossEntropyLoss()###

    # Tạm cất FocalLoss, dùng đồng loạt CrossEntropy có Label Smoothing
    focal_criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    ce_criterion    = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    
    ckpt_path  = f"{CONFIG['output_dir']}/best_model.pth"
    early_stop = ImprovedEarlyStopping(CONFIG['patience'], path=ckpt_path)

    history = {k: [] for k in ['train_loss','val_loss','val_loss_f','val_loss_c',
                                'val_loss_s','val_loss_w',
                                'val_acc_whole','val_acc_face','val_acc_context',
                                'val_acc_scene','lr',
                                'branch_w']}   # V6: log branch_w per epoch

    print("="*80)
    print("🎯 TRAINING START  |  V10: V6 + Emotional Contagion (group atmosphere → face nodes)")
    print(f"   branch_w: {CONFIG['branch_w_max']} → {CONFIG['branch_w_min']}  "
          f"(warm {CONFIG['warmup_epochs']}ep, decay until {CONFIG['decay_end']}ep)")
    print("="*80 + "\n")

    for epoch in range(CONFIG['epochs']):
        t_ep = time.time()

        # ── V6: Tính branch_w cho epoch này ──────────────────────────────────
        branch_w = get_branch_weight(epoch)
        history['branch_w'].append(branch_w)
        # ─────────────────────────────────────────────────────────────────────

        # ---------- TRAIN ----------
        model.train(); t_loss = 0
        bar = tqdm(train_loader, desc=f"Ep {epoch+1:03d}/{CONFIG['epochs']} [Train] bw={branch_w:.3f}")
        for bi, batch in enumerate(bar):
            try:
                batch = batch.to(CONFIG['device'])
                optimizer.zero_grad()
                out_f, out_c, out_s, out_w = model(batch)
                loss, *_ = compute_loss(out_f, out_c, out_s, out_w, batch.y,
                                        focal_criterion, ce_criterion,
                                        branch_w=branch_w)   # ← V6
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                optimizer.step()
                t_loss += loss.item()
                bar.set_postfix(loss=f"{loss.item():.4f}")
                if bi % 50 == 0 and torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception as e:
                print(f"\n❌ train batch {bi}: {e}")
                if CONFIG['debug_mode']: raise
                continue

        # ---------- VALIDATE ----------
        model.eval()
        v_loss = v_lf = v_lc = v_ls = v_lw = 0
        v_af = v_ac = v_as = v_aw = total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1:03d}/{CONFIG['epochs']} [Val]  "):
                try:
                    batch = batch.to(CONFIG['device'])
                    out_f, out_c, out_s, out_w = model(batch)
                    loss, lf, lc, ls, lw = compute_loss(out_f, out_c, out_s, out_w, batch.y,
                                                         focal_criterion, ce_criterion,
                                                         branch_w=branch_w)   # ← V6
                    bs      = len(batch.y)
                    v_loss += loss.item() * bs
                    v_lf   += lf * bs; v_lc += lc * bs
                    v_ls   += ls * bs; v_lw += lw * bs
                    v_af   += (out_f.argmax(1) == batch.y).sum().item()
                    v_ac   += (out_c.argmax(1) == batch.y).sum().item()
                    v_as   += (out_s.argmax(1) == batch.y).sum().item()
                    v_aw   += (out_w.argmax(1) == batch.y).sum().item()
                    total  += bs
                except Exception as e:
                    print(f"\n❌ val batch: {e}")
                    if CONFIG['debug_mode']: raise
                    continue

        scheduler.step()

        t_loss /= len(train_loader)
        for k, v in zip(
            ['val_loss','val_loss_f','val_loss_c','val_loss_s','val_loss_w',
             'val_acc_face','val_acc_context','val_acc_scene','val_acc_whole'],
            [v_loss, v_lf, v_lc, v_ls, v_lw, v_af, v_ac, v_as, v_aw]
        ):
            history[k].append(v / total)
        history['train_loss'].append(t_loss)
        history['lr'].append(get_lr(optimizer))

        vl  = v_loss / total
        vaw = v_aw / total
        vaf = v_af / total
        vac = v_ac / total
        vas = v_as / total

        print(f"\nEp {epoch+1:03d} [{time.time()-t_ep:.0f}s] "
              f"bw={branch_w:.3f} | TrL={t_loss:.4f} | ValL={vl:.4f} | "
              f"Whole={vaw:.4f} | Face={vaf:.4f} | Ctx={vac:.4f} | "
              f"Scene={vas:.4f} | LR={get_lr(optimizer):.1e}")

        early_stop(vl, vaw, model)
        if early_stop.early_stop:
            print(f"\n🛑 Early stop at epoch {epoch+1}")
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); gc.collect()

    # ---------- TEST ----------
    print("\n" + "="*80 + "\n🏆 TEST\n" + "="*80 + "\n")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    y_true, y_pf, y_pc, y_ps, y_pw = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            try:
                batch = batch.to(CONFIG['device'])
                of, oc, os_, ow = model(batch)
                y_true.extend(batch.y.cpu().numpy())
                y_pf.extend(of.argmax(1).cpu().numpy())
                y_pc.extend(oc.argmax(1).cpu().numpy())
                y_ps.extend(os_.argmax(1).cpu().numpy())
                y_pw.extend(ow.argmax(1).cpu().numpy())
            except Exception as e:
                print(f"\n❌ test: {e}"); continue

    for name, preds in [('FACE', y_pf), ('CONTEXT', y_pc), ('SCENE', y_ps), ('WHOLE', y_pw)]:
        acc = accuracy_score(y_true, preds)
        print(f"\n🔹 {name} — Acc: {acc:.4f}")
        print(classification_report(y_true, preds, target_names=['Neg','Neu','Pos'], digits=4))

    final_acc = accuracy_score(y_true, y_pw)
    face_acc  = accuracy_score(y_true, y_pf)
    ctx_acc   = accuracy_score(y_true, y_pc)
    scene_acc = accuracy_score(y_true, y_ps)
    print("="*50)
    print(f"Face={face_acc:.4f} | Ctx={ctx_acc:.4f} | Scene={scene_acc:.4f} | FINAL={final_acc:.4f}")
    print("🎉 TARGET!" if final_acc >= 0.95 else f"📈 Gap: {(0.95-final_acc)*100:.2f}%")

    try:
        plot_results(history, y_true, y_pw)
    except Exception as e:
        print(f"⚠️ Plot: {e}")

    print("\n✅ DONE!")


if __name__ == "__main__":
    main()