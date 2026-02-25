# ==========================================
# CÀI ĐẶT THƯ VIỆN CẦN THIẾT
# ==========================================
print("⏳ Đang cài đặt PyTorch Geometric...")
# Uncomment these lines when running on Kaggle
# !pip install torch_geometric
# !pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
print("✅ Cài đặt hoàn tất!")

# ==========================================
# IMPORT LIBRARIES
# ==========================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
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
# CONFIG - ✅ OPTIMIZED FOR KAGGLE
# ==========================================
CONFIG = {
    'face_dir':   '/kaggle/input/datasets/trieung11/vggface-face-features-groupemow/face_features/groupemow',
    'scene_dir':  '/kaggle/input/fearturecongnn/scene_features_final',
    'object_dir': '/kaggle/input/fearturecongnn/objects',
    'output_dir': '/kaggle/working/outputs_hybrid_ipr',
    
    'face_dim': 4096,
    'object_dim': 2048,
    'scene_dim': 2048,
    
    'gat_hidden': 256,
    'whole_hidden': 512,
    'num_classes': 3,
    
    'gat_layers': 2,
    'num_heads': 4,
    'dropout': 0.5,             # ✅ Reduced from 0.4 (was too high)
    'attention_dropout': 0.5,   # ✅ Reduced from 0.4
    
    'num_vn': 4,
    'num_k': 2,
    'ipr_temperature': 1.0,
    
    # ✅ ECL settings (following ConGNN paper)
    'lambda_penalty': 0.2,       # Penalty coefficient (optimal value from paper Fig. 10)
    'label_smoothing': 0.1,      # Label smoothing for regularization
    
    # ✅ FIXED: Reduced batch_size and num_workers for Kaggle
    'batch_size': 16,  # Reduced from 32
    'num_workers': 0,  # Changed from 2 to 0 to avoid deadlock
    'prefetch_factor': None,  # Only used when num_workers > 0
    
    # ✅ FIXED: Better optimization settings
    'lr': 1e-5,                  # Reduced from 1e-4 to 5e-5
    'weight_decay': 5e-2,        # Reduced from 1e-4 to 5e-5
    'grad_clip': 0.5,            # Reduced from 1.0 to 0.5
    'epochs': 200,
    'patience': 20,              # Increased from 12 to 15
    'scheduler_patience': 6,     # Reduced from 8 to 6
    'scheduler_factor': 0.5,     # Reduced from 0.7 to 0.5
    'min_lr': 1e-6,              # Reduced from 5e-6 to 1e-6
    
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    # ✅ Debug settings
    'debug_mode': False,  # Set to True to see detailed logs
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==========================================
# SYSTEM INFO
# ==========================================
def print_system_info():
    print("="*80)
    print("🖥️  SYSTEM INFORMATION")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {CONFIG['device']}")
    print("="*80 + "\n")

# ==========================================
# DATASET
# ==========================================
class ConGNN_Dataset(TorchDataset):
    def __init__(self, split='train', max_faces=32, max_objects=10):
        self.face_root = CONFIG['face_dir']
        self.scene_root = CONFIG['scene_dir']
        self.obj_root = CONFIG['object_dir']
        self.max_faces = max_faces
        self.max_objects = max_objects
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        pattern = os.path.join(self.face_root, 'faces', split, '**', '*.npz')
        self.face_files = glob.glob(pattern, recursive=True)
        if len(self.face_files) == 0:
            pattern = os.path.join(self.face_root, split, '**', '*.npz')
            self.face_files = glob.glob(pattern, recursive=True)
        
        print(f"📊 {split.upper()}: Found {len(self.face_files)} samples")
        
        if len(self.face_files) == 0:
            raise ValueError(f"❌ No data found in {self.face_root} for split '{split}'")

    def __len__(self): 
        return len(self.face_files)

    def _get_paired_path(self, face_path, target_type):
        if '/faces/' in face_path: 
            rel_path = face_path.split('/faces/')[-1]
        else: 
            rel_path = face_path.split(self.face_root)[-1].lstrip('/')

        if target_type == 'scenes':
            new_path = os.path.join(self.scene_root, 'scenes', rel_path).replace('.npz', '.npy')
            if not os.path.exists(new_path): 
                new_path = os.path.join(self.scene_root, rel_path.replace('.npz', '.npy'))
            return new_path
        elif target_type == 'objects':
            new_path = os.path.join(self.obj_root, 'objects', rel_path)
            if not os.path.exists(new_path): 
                new_path = os.path.join(self.obj_root, rel_path)
            return new_path

    def _build_dense_edges(self, num_nodes):
        if num_nodes <= 1:
            return torch.tensor([[0], [0]], dtype=torch.long)
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def __getitem__(self, idx):
        face_file = self.face_files[idx]
        label_str = os.path.basename(os.path.dirname(face_file)).lower()
        label = self.label_map.get(label_str, 1)
        
        # Load Face
        try:
            data = np.load(face_file)
            face_feat = data['features']
            face_boxes = data['boxes']
            if len(face_boxes) > 0:
                sort_idx = np.argsort(face_boxes[:, 0])
                face_feat = face_feat[sort_idx]
        except Exception as e:
            if CONFIG['debug_mode']:
                print(f"⚠️  Error loading face: {e}")
            face_feat = np.zeros((1, CONFIG['face_dim']), dtype=np.float32)

        face_feat = face_feat[:self.max_faces] if len(face_feat) > 0 else np.zeros((1, CONFIG['face_dim']), dtype=np.float32)
        face_x = torch.tensor(face_feat, dtype=torch.float32)
        face_edge_index = self._build_dense_edges(len(face_x))

        # Load Scene
        scene_path = self._get_paired_path(face_file, 'scenes')
        try:
            scene_feat = np.load(scene_path) if os.path.exists(scene_path) else np.zeros(CONFIG['scene_dim'], dtype=np.float32)
        except:
            scene_feat = np.zeros(CONFIG['scene_dim'], dtype=np.float32)
        scene_x = torch.tensor(scene_feat, dtype=torch.float32).unsqueeze(0)
        
        # Load Objects
        obj_path = self._get_paired_path(face_file, 'objects')
        try:
            obj_feat = np.load(obj_path)['features'] if os.path.exists(obj_path) else np.zeros((0, CONFIG['object_dim']), dtype=np.float32)
        except:
            obj_feat = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)
        obj_feat = obj_feat[:self.max_objects]
        
        # Context = Objects + Scene
        if len(obj_feat) > 0:
            context_x = torch.cat([
                torch.tensor(obj_feat, dtype=torch.float32),
                scene_x
            ], dim=0)
        else:
            context_x = scene_x
        
        context_edge_index = self._build_dense_edges(len(context_x))

        return {
            'face_x': face_x,
            'face_edge_index': face_edge_index,
            'context_x': context_x,
            'context_edge_index': context_edge_index,
            'y': label
        }

# ==========================================
# CUSTOM COLLATE - ✅ FIXED FOR MULTIPROCESSING
# ==========================================
class SimpleBatch:
    """✅ Picklable batch object for DataLoader"""
    def __init__(self, face_x, face_edge_index, face_batch,
                 context_x, context_edge_index, context_batch,
                 y, num_graphs):
        self.face_x = face_x
        self.face_edge_index = face_edge_index
        self.face_batch = face_batch
        
        self.context_x = context_x
        self.context_edge_index = context_edge_index
        self.context_batch = context_batch
        
        self.y = y
        self.num_graphs = num_graphs
    
    def to(self, device):
        self.face_x = self.face_x.to(device)
        self.face_edge_index = self.face_edge_index.to(device)
        self.face_batch = self.face_batch.to(device)
        self.context_x = self.context_x.to(device)
        self.context_edge_index = self.context_edge_index.to(device)
        self.context_batch = self.context_batch.to(device)
        self.y = self.y.to(device)
        return self
    
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)

def custom_collate(batch):
    """
    ✅ Fixed collate function with proper error handling
    """
    face_x_list = []
    face_edge_index_list = []
    face_batch_list = []
    
    context_x_list = []
    context_edge_index_list = []
    context_batch_list = []
    
    y_list = []
    
    face_node_offset = 0
    context_node_offset = 0
    
    for graph_idx, sample in enumerate(batch):
        # Face
        num_face_nodes = sample['face_x'].size(0)
        face_x_list.append(sample['face_x'])
        face_edge_index_list.append(sample['face_edge_index'] + face_node_offset)
        face_batch_list.append(torch.full((num_face_nodes,), graph_idx, dtype=torch.long))
        face_node_offset += num_face_nodes
        
        # Context
        num_context_nodes = sample['context_x'].size(0)
        context_x_list.append(sample['context_x'])
        context_edge_index_list.append(sample['context_edge_index'] + context_node_offset)
        context_batch_list.append(torch.full((num_context_nodes,), graph_idx, dtype=torch.long))
        context_node_offset += num_context_nodes
        
        y_list.append(sample['y'])
    
    return SimpleBatch(
        face_x=torch.cat(face_x_list, dim=0),
        face_edge_index=torch.cat(face_edge_index_list, dim=1),
        face_batch=torch.cat(face_batch_list, dim=0),
        context_x=torch.cat(context_x_list, dim=0),
        context_edge_index=torch.cat(context_edge_index_list, dim=1),
        context_batch=torch.cat(context_batch_list, dim=0),
        y=torch.tensor(y_list, dtype=torch.long),
        num_graphs=len(batch)
    )

# ==========================================
# GAT MODEL
# ==========================================
class MultiLayerGATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=2, num_layers=2, dropout=0.5, attention_dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                          dropout=attention_dropout, add_self_loops=True, concat=True, bias=False)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h = self.dropout_layer(F.relu(self.input_norm(self.input_proj(x))))
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h_new = gat(h, edge_index)
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout_layer(h_new)
            if h.shape == h_new.shape: 
                h = h + h_new
            else: 
                h = h_new
        return h

# ==========================================
# CÔNG CỤ LẤY MẪU CHÍNH XÁC k PHẦN TỬ (Tương đương SIMPLE)
# ==========================================
def gumbel_top_k_sample(logits, k, temperature=1.0, training=True):
    """
    Thuật toán lấy mẫu k phần tử không trùng lặp (k-subset sampling).
    Thay thế cho Gumbel-Softmax (vốn chỉ lấy k=1).
    """
    if not training:
        # Khi test: Lấy cứng k giá trị có xác suất cao nhất
        _, indices = torch.topk(logits, k, dim=-1)
        hard_samples = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
        return hard_samples

    # Khi train: Thêm nhiễu Gumbel để tạo tính ngẫu nhiên và tính đạo hàm
    gumbels = -torch.empty_like(logits).exponential_().log()  # Tạo nhiễu Gumbel
    gumbel_logits = (logits + gumbels) / temperature
    
    # Kỹ thuật Straight-Through Estimator (STE) cho Top-K
    # 1. Forward pass: Lấy cứng k phần tử lớn nhất (0 hoặc 1)
    _, indices = torch.topk(gumbel_logits, k, dim=-1)
    hard_samples = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
    
    # 2. Backward pass: Mượn đạo hàm từ Softmax mềm
    soft_samples = torch.softmax(gumbel_logits, dim=-1)
    
    # Kết hợp: Forward giữ nguyên 'hard', Backward mượn đường 'soft'
    samples = (hard_samples - soft_samples).detach() + soft_samples
    return samples

# ==========================================
# IPR-MPNN LAYER (ĐÃ NÂNG CẤP LÊN K > 1)
# ==========================================
class IPR_MPNN_Layer(nn.Module):
    def __init__(self, hidden_dim, num_vn=4, num_k=2, dropout=0.3, temperature=1.0):
        super().__init__()
        self.num_vn = num_vn
        self.num_k = num_k  # ✅ THÊM MỚI: Số lượng node ảo mỗi node thật sẽ nối tới
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Kiểm tra tính hợp lệ
        assert self.num_k <= self.num_vn, "Lỗi: Số k (num_k) không được lớn hơn tổng số node ảo (num_vn)!"

        # Khởi tạo đặc trưng ban đầu cho các node ảo (Learnable)
        self.vn_init = nn.Parameter(torch.randn(1, num_vn, hidden_dim))
        
        # Mạng dự đoán xác suất nối (Router)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_vn) # Output ra xác suất cho 'num_vn' node
        )
        
        # Mạng biến đổi thông tin từ Node thật -> Node ảo
        self.msg_r2v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Mạng truyền tin giữa các Node Ảo với nhau (Fully Connected)
        self.v2v_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.v2v_norm = nn.LayerNorm(hidden_dim)
        
        # Mạng biến đổi thông tin từ Node ảo -> Node thật
        self.msg_v2r = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bộ nhớ GRU để cập nhật thông tin cho node thật
        self.gru_update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, real_nodes, batch, training=True):
        num_graphs = batch.max().item() + 1
        
        # 1. Khởi tạo Node ảo cho mỗi đồ thị trong batch
        vn_features = self.vn_init.expand(num_graphs, -1, -1).clone()
        
        # 2. TÍNH TOÁN LIÊN KẾT (ROUTING)
        routing_logits = self.router(real_nodes)
        
        # ✅ ĐIỂM KHÁC BIỆT: Thay thế Gumbel-Softmax bằng Gumbel-Top-K
        # Bây giờ 'weights' sẽ có đúng 'num_k' số 1 trên mỗi hàng, còn lại là 0.
        weights = gumbel_top_k_sample(
            routing_logits, 
            k=self.num_k, 
            temperature=self.temperature, 
            training=training
        )
        
        # 3. TRUYỀN TIN TỪ THẬT -> ẢO (Real to Virtual)
        messages_r2v = self.msg_r2v(real_nodes)
        vn_updates = []
        for v in range(self.num_vn):
            # Lấy ra cột trọng số của node ảo thứ v
            mask_v = weights[:, v].unsqueeze(-1) 
            # Chỉ những node thật được nối với node ảo v mới gửi thông tin đi
            masked_msg = messages_r2v * mask_v
            # Gom thông tin lại cho từng đồ thị
            agg_v = global_add_pool(masked_msg, batch)
            vn_updates.append(agg_v.unsqueeze(1))
        
        vn_agg = torch.cat(vn_updates, dim=1)
        vn_features = vn_features + vn_agg # Cộng dồn thông tin mới vào node ảo
        
        # 4. TRUYỀN TIN GIỮA CÁC NODE ẢO (Virtual to Virtual)
        # Các node ảo chia sẻ thông tin để dung hòa mâu thuẫn (như bài báo mô tả)
        vn_attn, _ = self.v2v_attn(vn_features, vn_features, vn_features)
        vn_features = self.v2v_norm(vn_features + vn_attn)
        
        # 5. TRUYỀN TIN TỪ ẢO -> THẬT (Virtual to Real)
        messages_v2r = self.msg_v2r(vn_features)
        vn_expanded = messages_v2r[batch] # Trải node ảo ra cho khớp với kích thước node thật
        
        # Gom thông tin từ k node ảo mà node thật này đã nối
        real_msg_received = (vn_expanded * weights.unsqueeze(-1)).sum(dim=1)
        
        # 6. CẬP NHẬT NODE THẬT BẰNG GRU
        updated_real = self.gru_update(real_msg_received, real_nodes)
        
        return updated_real

# ==========================================
# MODEL
# ==========================================
class VirtualNode_ConGNN_ECL(nn.Module):
    def __init__(self):
        super().__init__()
        gat_dim = CONFIG['gat_hidden']
        whole_dim = CONFIG['whole_hidden']
        drp = CONFIG['dropout']
        att_drp = CONFIG['attention_dropout']
        
        # 1. Input Normalization cho nhánh Face 4096
        self.face_input_norm = nn.LayerNorm(CONFIG['face_dim'])
        
        self.face_gat = MultiLayerGATv2(
            CONFIG['face_dim'], gat_dim, 
            CONFIG['num_heads'], CONFIG['gat_layers'], 
            drp, att_drp
        )
        self.clf_face = nn.Linear(gat_dim, CONFIG['num_classes'])
        
        self.context_gat = MultiLayerGATv2(
            CONFIG['object_dim'], gat_dim, 
            CONFIG['num_heads'], CONFIG['gat_layers'], 
            drp, att_drp
        )
        self.clf_context = nn.Linear(gat_dim, CONFIG['num_classes'])
        
        self.proj_to_whole = nn.Sequential(
            nn.Linear(gat_dim, whole_dim),
            nn.LayerNorm(whole_dim),
            nn.ReLU(),
            nn.Dropout(drp)
        )
        
        self.ipr_mpnn = IPR_MPNN_Layer(
            hidden_dim=whole_dim,
            num_vn=CONFIG['num_vn'],
            num_k=CONFIG['num_k'],
            dropout=drp,
            temperature=CONFIG['ipr_temperature']
        )

        # 2. Bottleneck nén Face (4096 -> 1024 -> whole_dim)
        # Thay BatchNorm bằng LayerNorm để ổn định hơn cho feature vector
        self.reduce_face = nn.Sequential(
            nn.Linear(CONFIG['face_dim'], 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(drp),
            nn.Linear(1024, whole_dim),
            nn.LayerNorm(whole_dim),
            nn.ReLU()
        )

        # 3. Bottleneck nén Context (2048 -> whole_dim)
        self.reduce_context = nn.Sequential(
            nn.Linear(CONFIG['object_dim'], whole_dim),
            nn.LayerNorm(whole_dim),
            nn.ReLU(),
            nn.Dropout(drp)
        )
        
        total_concat_dim = whole_dim + whole_dim + whole_dim
        
        self.clf_whole = nn.Sequential(
            nn.Linear(total_concat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(drp),
            nn.Linear(512, CONFIG['num_classes'])
        )

    def forward(self, data):
        # ✅ Fix 1: Sử dụng dữ liệu đã qua LayerNorm
        face_x_norm = self.face_input_norm(data.face_x)
        
        # 1. Nhánh GAT
        H_face = self.face_gat(face_x_norm, data.face_edge_index) # Dùng face_x_norm ở đây
        out_face = self.clf_face(global_mean_pool(H_face, data.face_batch))
        
        H_context = self.context_gat(data.context_x, data.context_edge_index)
        out_context = self.clf_context(global_mean_pool(H_context, data.context_batch))
        
        # 2. IPR-MPNN (Graph Branch)
        real_nodes_mix = torch.cat([H_face, H_context], dim=0)
        real_nodes_proj = self.proj_to_whole(real_nodes_mix)
        
        real_batch = torch.cat([data.face_batch, data.context_batch], dim=0)
        updated_real = self.ipr_mpnn(real_nodes_proj, batch=real_batch, training=self.training)
        
        whole_pooled = global_mean_pool(updated_real, real_batch)

        # 3. Skip Connections (Raw features pooling + Bottleneck)
        raw_face_pooled = global_mean_pool(face_x_norm, data.face_batch) # Pooling từ face_x_norm
        raw_context_pooled = global_mean_pool(data.context_x, data.context_batch)
        
        reduced_f = self.reduce_face(raw_face_pooled)
        reduced_c = self.reduce_context(raw_context_pooled)
        
        # 4. Final Concat
        combined = torch.cat([whole_pooled, reduced_f, reduced_c], dim=1)
        out_whole = self.clf_whole(combined)
        
        return out_face, out_context, out_whole
# ==========================================
# CẬP NHẬT LOSS: FOCAL LOSS + LABEL SMOOTHING
# ==========================================
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, num_classes=3, smoothing=0.1, gamma=2.0):
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        # Trọng số ưu tiên: [Negative, Neutral, Positive]
        # Ép mô hình tập trung x2 vào class Neutral (index 1)
        self.alpha = torch.tensor([1.0, 2.0, 1.0]) 

    def forward(self, pred, target):
        # Ép kiểu target về torch.long (int64) để tránh lỗi index/gather
        target = target.long() 
        self.alpha = self.alpha.to(pred.device)
        
        # 1. Label Smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # 2. Cross Entropy với Nhãn mềm
        ce_loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        # 3. Kích hoạt Focal Loss
        probs = torch.exp(log_probs)
        # Lấy xác suất mô hình dự đoán cho class đúng
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1) 
        
        # ✅ FIX LỖI: Lấy trọng số alpha bằng indexing thay vì gather
        at = self.alpha[target]
        
        # Công thức Focal: Phạt thật nặng những ca có xác suất (pt) thấp
        focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss
        
        # ✅ FIX LỖI 2: Hàm loss chỉ được trả về 1 giá trị vô hướng (scalar)
        # Trong hàm compute_ecl_loss_with_bpf, bạn dùng L_f = criterion(out_f, label)
        return focal_loss.mean()
def compute_ecl_loss_with_bpf(out_f, out_c, out_w, label, criterion):
    """
    ✅ NÂNG CẤP SOTA: Thay thế BPF bằng Self-Distillation Loss.
    Nhánh Whole (Siêu nhánh) làm Thầy, dạy cho Face và Context (Học trò).
    """
    # 1. Tính Loss cơ bản bằng criterion
    # (Lưu ý: criterion truyền vào từ main() đã có sẵn Label Smoothing và Class Weights rồi)
    L_f = criterion(out_f, label)
    L_c = criterion(out_c, label)
    L_w = criterion(out_w, label)
    
    # 2. SELF-DISTILLATION LOSS (Thầy dạy Trò)
    # Nhiệt độ (Temperature) làm "mềm" phân phối xác suất để học trò dễ học hơn
    T = 2.0 
    
    # Thầy (Whole) tạo ra đáp án mềm (Soft Target), phải dùng detach() để không lỗi đạo hàm
    soft_target_w = F.softmax(out_w.detach() / T, dim=1) 
    
    # Trò (Face, Context) đưa ra dự đoán mềm
    log_prob_f = F.log_softmax(out_f / T, dim=1)
    log_prob_c = F.log_softmax(out_c / T, dim=1)
    
    # Tính độ lệch giữa Thầy và Trò (KL Divergence)
    distill_loss_f = F.kl_div(log_prob_f, soft_target_w, reduction='batchmean') * (T ** 2)
    distill_loss_c = F.kl_div(log_prob_c, soft_target_w, reduction='batchmean') * (T ** 2)
    
    # 3. TỔNG HỢP LOSS THÔNG MINH
    # L_w (nhánh chính) giữ 100% sức mạnh.
    # Nhánh con học 50% từ Ground Truth (Nhãn thật), 50% học từ Thầy Whole.
    alpha = 0.5
    beta = 0.5
    
    total_loss = L_w + alpha * (L_f + L_c) + beta * (distill_loss_f + distill_loss_c)
    
    # Trả về penalty_ratio = 0.0 để biểu đồ không bị lỗi (ta không dùng BPF penalty nữa)
    return total_loss, L_f.item(), L_c.item(), L_w.item(), 0.0

# ==========================================
# EARLY STOPPING
# ==========================================
class ImprovedEarlyStopping:
    def __init__(self, patience=12, min_delta=0.001, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.best_acc = 0
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, val_acc, model):
        improved = False
        
        if val_loss < (self.best_loss - self.min_delta):
            improved = True
        elif val_acc > self.best_acc and val_loss < (self.best_loss + 0.05):
            improved = True
            
        if improved:
            self.best_loss = val_loss
            self.best_acc = val_acc
            torch.save(model.state_dict(), self.path)
            self.counter = 0
            print(f"  🔥 Best Model! Loss={val_loss:.4f} | Acc={val_acc:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ==========================================
# PLOTTING
# ==========================================
def plot_results(history, y_true, y_pred):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['val_loss_f'], label='Face', linewidth=2)
    axes[0, 1].plot(history['val_loss_c'], label='Context', linewidth=2)
    axes[0, 1].plot(history['val_loss_w'], label='Whole', linewidth=2)
    axes[0, 1].set_title('Branch Losses', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(history['penalty_ratio'], linewidth=2, color='orange')
    axes[0, 2].axhline(y=0.15, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].set_title('Emotion Bias Ratio', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Bias Ratio')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 0.3])
    
    axes[1, 0].plot(history['val_acc_whole'], label='Whole', linewidth=2.5, color='red')
    axes[1, 0].plot(history['val_acc_face'], label='Face', linewidth=2, alpha=0.7)
    axes[1, 0].plot(history['val_acc_context'], label='Context', linewidth=2, alpha=0.7)
    axes[1, 0].axhline(y=0.90, color='g', linestyle='--', label='Target 90%')
    axes[1, 0].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Thêm normalize='true' để tính tỷ lệ, nhân 100 để ra phần trăm
    cm_pct = confusion_matrix(y_true, y_pred, normalize='true') * 100
    # Đổi fmt='d' (số nguyên) thành fmt='.2f' (số thực 2 chữ số thập phân)
    sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
    # Nên thêm ký hiệu (%) 
    axes[1, 1].set_title('Confusion Matrix (%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('True')
    axes[1, 1].set_xlabel('Predicted')
    
    axes[1, 2].plot(history['lr'], linewidth=2, color='purple')
    axes[1, 2].set_title('Learning Rate', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('LR')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/hybrid_gat128_gnn512_results.png", dpi=300)
    plt.show()

# ==========================================
# ✅ MAIN FUNCTION WITH FIXES
# ==========================================
def main():
    print("="*80)
    print("🚀 HYBRID GAT + IPR-MPNN (GAT 128 + GNN 512)")
    print("="*80)
    print(f"📌 Architecture:")
    print(f"   - Face Branch: MultiLayerGATv2 (Hidden={CONFIG['gat_hidden']})")
    print(f"   - Context Branch: MultiLayerGATv2 (Hidden={CONFIG['gat_hidden']})")
    print(f"   - Projection: {CONFIG['gat_hidden']} → {CONFIG['whole_hidden']}")
    print(f"   - Whole Branch: IPR-MPNN (Hidden={CONFIG['whole_hidden']}, VN={CONFIG['num_vn']})")
    print("="*80 + "\n")
    
    # Print system info
    print_system_info()
    
    # ✅ Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("📂 Loading datasets...")
    start_time = time.time()
    
    try:
        train_dataset = ConGNN_Dataset('train')
        val_dataset = ConGNN_Dataset('val')
        test_dataset = ConGNN_Dataset('test')
        print(f"✅ Datasets loaded in {time.time() - start_time:.2f}s\n")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # ✅ Create DataLoaders with fixed settings
    print("🔄 Creating DataLoaders...")
    dataloader_kwargs = {
        'batch_size': CONFIG['batch_size'],
        'collate_fn': custom_collate,
        'num_workers': CONFIG['num_workers'],
        'pin_memory': True if torch.cuda.is_available() else False,
    }
    
    # Add prefetch_factor only if num_workers > 0
    if CONFIG['num_workers'] > 0 and CONFIG['prefetch_factor'] is not None:
        dataloader_kwargs['prefetch_factor'] = CONFIG['prefetch_factor']
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    
    print(f"✅ DataLoaders created")
    print(f"   - Batch size: {CONFIG['batch_size']}")
    print(f"   - Num workers: {CONFIG['num_workers']}")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Val batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}\n")
    
    # ✅ Test loading one batch
    print("🧪 Testing DataLoader (loading first batch)...")
    try:
        test_start = time.time()
        test_batch = next(iter(train_loader))
        test_batch = test_batch.to(CONFIG['device'])
        print(f"✅ First batch loaded successfully in {time.time() - test_start:.2f}s")
        print(f"   - Face nodes: {test_batch.face_x.shape}")
        print(f"   - Context nodes: {test_batch.context_x.shape}")
        print(f"   - Labels: {test_batch.y.shape}\n")
        del test_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error loading first batch: {e}")
        print("   Try reducing batch_size in CONFIG")
        return
    
    # Create model
    print("🏗️  Building model...")
    model = VirtualNode_ConGNN_ECL().to(CONFIG['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Statistics:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=CONFIG['lr'], 
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=CONFIG['scheduler_patience'],
        factor=CONFIG['scheduler_factor'],
        min_lr=CONFIG['min_lr']
    )
    
    criterion = FocalLossWithSmoothing(
        num_classes=CONFIG['num_classes'], 
        smoothing=CONFIG['label_smoothing'], 
        gamma=2.0
    )
    
    early_stop = ImprovedEarlyStopping(
        patience=CONFIG['patience'], 
        path=f"{CONFIG['output_dir']}/best_model_hybrid_gat128_gnn512.pth"
    )
    
    history = {
        'train_loss': [], 'val_loss': [],
        'val_loss_f': [], 'val_loss_c': [], 'val_loss_w': [],
        'val_acc_whole': [], 'val_acc_face': [], 'val_acc_context': [],
        'penalty_ratio': [], 'lr': []
    }
    
    print("="*80)
    print("🎯 STARTING TRAINING")
    print("="*80 + "\n")
    
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        
        # ==========================================
        # TRAINING
        # ==========================================
        model.train()
        t_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{CONFIG['epochs']} [Train]")
        
        for batch_idx, batch in enumerate(train_bar):
            try:
                batch = batch.to(CONFIG['device'])
                optimizer.zero_grad()
                
                out_f, out_c, out_w = model(batch)
                loss, _, _, _, _ = compute_ecl_loss_with_bpf(out_f, out_c, out_w, batch.y, criterion)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                optimizer.step()
                
                t_loss += loss.item()
                train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # ✅ Clear cache periodically
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\n❌ Error in training batch {batch_idx}: {e}")
                if CONFIG['debug_mode']:
                    raise e
                continue
        
        # ==========================================
        # VALIDATION
        # ==========================================
        model.eval()
        v_loss = 0
        v_loss_f, v_loss_c, v_loss_w = 0, 0, 0
        v_acc_f, v_acc_c, v_acc_w = 0, 0, 0
        penalty_total = 0
        total_samples = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{CONFIG['epochs']} [Val]  ")
            for batch in val_bar:
                try:
                    batch = batch.to(CONFIG['device'])
                    out_f, out_c, out_w = model(batch)
                    
                    loss, lf, lc, lw, penalty = compute_ecl_loss_with_bpf(out_f, out_c, out_w, batch.y, criterion)
                    
                    batch_size = len(batch.y)
                    v_loss += loss.item() * batch_size
                    v_loss_f += lf * batch_size
                    v_loss_c += lc * batch_size
                    v_loss_w += lw * batch_size
                    penalty_total += penalty * batch_size
                    
                    v_acc_f += (out_f.argmax(1) == batch.y).sum().item()
                    v_acc_c += (out_c.argmax(1) == batch.y).sum().item()
                    v_acc_w += (out_w.argmax(1) == batch.y).sum().item()
                    
                    total_samples += batch_size
                except Exception as e:
                    print(f"\n❌ Error in validation: {e}")
                    if CONFIG['debug_mode']:
                        raise e
                    continue
        
        # Calculate metrics
        t_loss /= len(train_loader)
        v_loss /= total_samples
        v_loss_f /= total_samples
        v_loss_c /= total_samples
        v_loss_w /= total_samples
        v_acc_f /= total_samples
        v_acc_c /= total_samples
        v_acc_w /= total_samples
        penalty_ratio = penalty_total / total_samples
        
        current_lr = get_lr(optimizer)
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_loss_f'].append(v_loss_f)
        history['val_loss_c'].append(v_loss_c)
        history['val_loss_w'].append(v_loss_w)
        history['val_acc_face'].append(v_acc_f)
        history['val_acc_context'].append(v_acc_c)
        history['val_acc_whole'].append(v_acc_w)
        history['penalty_ratio'].append(penalty_ratio)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch+1:02d} [{epoch_time:.1f}s]: TrLoss={t_loss:.4f} | ValLoss={v_loss:.4f} | "
              f"Whole={v_acc_w:.4f} | Face={v_acc_f:.4f} | Ctx={v_acc_c:.4f} | "
              f"Bias={penalty_ratio:.3f} | LR={current_lr:.2e}")
        
        # Scheduler step
        scheduler.step(v_loss)
        
        # Early stopping check
        early_stop(v_loss, v_acc_w, model)
        
        if early_stop.early_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
            break
        
        # ✅ Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print("\n" + "="*80)
    print("🏆 FINAL TEST EVALUATION")
    print("="*80 + "\n")
    
    model.load_state_dict(torch.load(f"{CONFIG['output_dir']}/best_model_hybrid_gat128_gnn512.pth"))
    model.eval()
    
    y_true, y_pred_f, y_pred_c, y_pred_w = [], [], [], []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing")
        for batch in test_bar:
            try:
                batch = batch.to(CONFIG['device'])
                out_f, out_c, out_w = model(batch)
                
                y_true.extend(batch.y.cpu().numpy())
                y_pred_f.extend(out_f.argmax(1).cpu().numpy())
                y_pred_c.extend(out_c.argmax(1).cpu().numpy())
                y_pred_w.extend(out_w.argmax(1).cpu().numpy())
            except Exception as e:
                print(f"\n❌ Error in test batch: {e}")
                continue
    
    print("\n🔹 FACE BRANCH:")
    print(f"   Accuracy: {accuracy_score(y_true, y_pred_f):.4f}")
    print(classification_report(y_true, y_pred_f, target_names=['Neg', 'Neu', 'Pos'], digits=4))
    
    print("\n🔹 CONTEXT BRANCH:")
    print(f"   Accuracy: {accuracy_score(y_true, y_pred_c):.4f}")
    print(classification_report(y_true, y_pred_c, target_names=['Neg', 'Neu', 'Pos'], digits=4))
    
    print("\n🔹 WHOLE BRANCH:")
    final_acc = accuracy_score(y_true, y_pred_w)
    print(f"   Accuracy: {final_acc:.4f}")
    print(classification_report(y_true, y_pred_w, target_names=['Neg', 'Neu', 'Pos'], digits=4))
    
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    print(f"Face: {accuracy_score(y_true, y_pred_f):.4f} | "
          f"Context: {accuracy_score(y_true, y_pred_c):.4f} | "
          f"FINAL: {final_acc:.4f}")
    
    if final_acc >= 0.90:
        print("\n🎉 TARGET ACHIEVED! 🎉")
    else:
        print(f"\n📈 Gap to 90%: {(0.90 - final_acc)*100:.2f}%")
    
    # Plot results
    print("\n📊 Generating plots...")
    try:
        plot_results(history, y_true, y_pred_w)
        print("✅ Plots saved!")
    except Exception as e:
        print(f"⚠️  Error generating plots: {e}")
    
    print("\n✅ TRAINING COMPLETE!")

if __name__ == "__main__":
    main()