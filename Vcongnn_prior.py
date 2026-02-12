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
    'face_dir':   '/kaggle/input/datasets/trieung11/v-congnn-addpriors/faces_priors_final',
    'scene_dir':  '/kaggle/input/datasets/trieung11/v-congnn-addpriors/scene_priors_final',
    'object_dir': '/kaggle/input/datasets/trieung11/v-congnn-addpriors/objects_features_priors',
    'output_dir': '/kaggle/working/outputs_prior_aware',
    
    'face_dim': 2048,
    'object_dim': 2048,
    'scene_dim': 2048,
    
    'gat_hidden': 128,
    'whole_hidden': 256,
    'num_classes': 3,
    
    'gat_layers': 2,
    'num_heads': 4,
    'dropout': 0.3,
    'attention_dropout': 0.3,
    
    'num_vn': 4,
    'num_k': 3,
    'ipr_temperature': 0.5,
    
    'lambda_penalty_max': 0.2,   # Giá trị lambda tối đa sau warm-up
    'label_smoothing': 0.1,
    
    'batch_size': 16,
    'num_workers': 0,
    
    'lr': 5e-5,
    'weight_decay': 5e-5,
    'grad_clip': 0.5,
    'epochs': 100,
    'patience': 15,
    'scheduler_patience': 6,
    'scheduler_factor': 0.5,
    'min_lr': 1e-6,
    
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'debug_mode': False,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==========================================
# DATASET WITH PRIORS - ✅ FIXED INDENTATION
# ==========================================
class ConGNN_Dataset_WithPriors(TorchDataset):
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
            raise ValueError(f"❌ No data found in {self.face_root}")

    def __len__(self): 
        return len(self.face_files)

    def _get_paired_path(self, face_path, target_type):
        rel_path = face_path.split('/faces/')[-1] if '/faces/' in face_path else face_path.split(self.face_root)[-1].lstrip('/')
        if target_type == 'scenes':
            new_path = os.path.join(self.scene_root, 'scenes', rel_path)
            return new_path if os.path.exists(new_path) else os.path.join(self.scene_root, rel_path)
        elif target_type == 'objects':
            new_path = os.path.join(self.obj_root, 'objects', rel_path)
            return new_path if os.path.exists(new_path) else os.path.join(self.obj_root, rel_path)

    def _build_dense_edges(self, num_nodes):
        if num_nodes <= 1: return torch.tensor([[0], [0]], dtype=torch.long)
        adj = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
        return adj.nonzero().t().contiguous()

    def __getitem__(self, idx):
        face_file = self.face_files[idx]
        label = self.label_map.get(os.path.basename(os.path.dirname(face_file)).lower(), 1)
        
        data = np.load(face_file)
        face_x = torch.tensor(data['features'][:self.max_faces], dtype=torch.float32)
        f_priors = data['priors'][:self.max_faces] if 'priors' in data else np.ones(len(face_x))
        face_priors = torch.tensor(f_priors, dtype=torch.float32).unsqueeze(-1)
        
        scene_path = self._get_paired_path(face_file, 'scenes')
        try:
            s_data = np.load(scene_path)
            scene_feat = s_data['features'] if 'features' in s_data else s_data
            s_prior = s_data['priors'] if 'priors' in s_data else [1.0]
        except:
            scene_feat = np.zeros((1, CONFIG['scene_dim']), dtype=np.float32)
            s_prior = [1.0]
        scene_x = torch.tensor(scene_feat, dtype=torch.float32).view(1, -1)
        scene_priors = torch.tensor(s_prior, dtype=torch.float32).view(1, 1)

        obj_path = self._get_paired_path(face_file, 'objects')
        try:
            o_data = np.load(obj_path)
            obj_feat = o_data['features'][:self.max_objects]
            o_prior = o_data['priors'][:self.max_objects] if 'priors' in o_data else np.ones(len(obj_feat))
        except:
            obj_feat = np.zeros((0, CONFIG['object_dim']), dtype=np.float32)
            o_prior = np.zeros(0)

        if len(obj_feat) > 0:
            context_x = torch.cat([torch.tensor(obj_feat, dtype=torch.float32), scene_x], dim=0)
            context_priors = torch.cat([torch.tensor(o_prior, dtype=torch.float32), scene_priors.view(-1)], dim=0)
        else:
            context_x, context_priors = scene_x, scene_priors.view(-1)
        
        return {
            'face_x': face_x, 'face_edge_index': self._build_dense_edges(len(face_x)), 'face_priors': face_priors,
            'context_x': context_x, 'context_edge_index': self._build_dense_edges(len(context_x)), 'context_priors': context_priors.unsqueeze(-1),
            'y': label
        }

# ==========================================
# MODELS & UTILS
# ==========================================
class SimpleBatch:
    def __init__(self, face_x, face_edge_index, face_batch, face_priors, context_x, context_edge_index, context_batch, context_priors, y, num_graphs):
        self.face_x, self.face_edge_index, self.face_batch, self.face_priors = face_x, face_edge_index, face_batch, face_priors
        self.context_x, self.context_edge_index, self.context_batch, self.context_priors = context_x, context_edge_index, context_batch, context_priors
        self.y, self.num_graphs = y, num_graphs
    def to(self, device):
        for attr in ['face_x', 'face_edge_index', 'face_batch', 'face_priors', 'context_x', 'context_edge_index', 'context_batch', 'context_priors', 'y']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

def custom_collate(batch):
    f_x, f_ei, f_b, f_p = [], [], [], []
    c_x, c_ei, c_b, c_p = [], [], [], []
    y, f_off, c_off = [], 0, 0
    for i, s in enumerate(batch):
        f_x.append(s['face_x']); f_ei.append(s['face_edge_index'] + f_off); f_b.append(torch.full((s['face_x'].size(0),), i, dtype=torch.long)); f_p.append(s['face_priors'])
        c_x.append(s['context_x']); c_ei.append(s['context_edge_index'] + c_off); c_b.append(torch.full((s['context_x'].size(0),), i, dtype=torch.long)); c_p.append(s['context_priors'])
        y.append(s['y']); f_off += s['face_x'].size(0); c_off += s['context_x'].size(0)
    return SimpleBatch(torch.cat(f_x), torch.cat(f_ei, 1), torch.cat(f_b), torch.cat(f_p), torch.cat(c_x), torch.cat(c_ei, 1), torch.cat(c_b), torch.cat(c_p), torch.tensor(y), len(batch))

class MultiLayerGATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=2, num_layers=2, dropout=0.3, attention_dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim//num_heads, heads=num_heads, dropout=attention_dropout) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, ei):
        h = self.dropout(F.relu(self.input_proj(x)))
        for gat, norm in zip(self.gat_layers, self.norms):
            h = norm(h + F.elu(gat(h, ei)))
        return h

class PriorAwareIPR_MPNN(nn.Module):
    def __init__(self, hidden_dim, num_vn=4, num_k=3, dropout=0.3, temperature=0.5):
        super().__init__()
        self.num_vn, self.num_k, self.temp = num_vn, num_k, temperature
        self.vn_init = nn.Parameter(torch.randn(1, num_vn, hidden_dim))
        self.router = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_vn))
        self.msg_r2v = nn.Linear(hidden_dim, hidden_dim)
        self.v2v_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, real_nodes, batch, priors, training=True):
        vn = self.vn_init.expand(batch.max()+1, -1, -1).clone()
        logits = self.router(torch.cat([real_nodes, priors], -1))
        if training:
            g = -torch.empty_like(logits).exponential_().log()
            soft = F.softmax((logits + g)/self.temp, -1)
            _, idx = torch.topk(logits + g, self.num_k, -1)
            hard = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
            w = (hard - soft).detach() + soft
        else:
            _, idx = torch.topk(logits, self.num_k, -1)
            w = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        
        r2v = self.msg_r2v(real_nodes)
        for v in range(self.num_vn):
            vn[:, v] += global_add_pool(r2v * w[:, v:v+1], batch)
        
        attn, _ = self.v2v_attn(vn, vn, vn)
        vn = vn + attn
        real_msg = (vn[batch] * w.unsqueeze(-1)).sum(1)
        return self.gru(real_msg, real_nodes)

class VirtualNode_ConGNN_PriorAware(nn.Module):
    def __init__(self):
        super().__init__()
        dim, w_dim = CONFIG['gat_hidden'], CONFIG['whole_hidden']
        self.face_gat = MultiLayerGATv2(CONFIG['face_dim'], dim)
        self.ctx_gat = MultiLayerGATv2(CONFIG['object_dim'], dim)
        self.proj = nn.Linear(dim, w_dim)
        self.ipr = PriorAwareIPR_MPNN(w_dim, CONFIG['num_vn'], CONFIG['num_k'])
        self.clf_f, self.clf_c, self.clf_w = nn.Linear(dim, 3), nn.Linear(dim, 3), nn.Linear(w_dim, 3)

    def forward(self, data):
            # Lấy số lượng đồ thị thực tế trong batch để đảm bảo đầu ra pooling luôn đủ size
            num_graphs = data.num_graphs 
            
            # Nhánh Face
            hf = self.face_gat(data.face_x, data.face_edge_index)
            # Ép size của pooling luôn bằng num_graphs
            out_face = self.clf_f(global_mean_pool(hf, data.face_batch, size=num_graphs))
            
            # Nhánh Context
            hc = self.ctx_gat(data.context_x, data.context_edge_index)
            out_context = self.clf_c(global_mean_pool(hc, data.context_batch, size=num_graphs))
            
            # Nhánh Whole (IPR-MPNN)
            real_nodes = self.proj(torch.cat([hf, hc], 0))
            real_batch = torch.cat([data.face_batch, data.context_batch], 0)
            real_priors = torch.cat([data.face_priors, data.context_priors], 0)
            
            updated = self.ipr(real_nodes, real_batch, real_priors, self.training)
            # Ép size của pooling luôn bằng num_graphs
            out_whole = self.clf_w(global_mean_pool(updated, real_batch, size=num_graphs))
            
            return out_face, out_context, out_whole

# ==========================================
# TRAINING & PLOTTING UTILS
# ==========================================
def compute_loss(out_f, out_c, out_w, label, lam):
    lf, lc, lw = [F.cross_entropy(o, label) for o in [out_f, out_c, out_w]]
    bias = (out_f.argmax(1) != out_c.argmax(1)).float().mean()
    total = (1 + lam * bias) * (lf + lc + lw)
    return total, lf.item(), lc.item(), lw.item(), bias.item()

def plot_dashboard(hist, y_true, y_pred, metrics):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes[0,0].plot(hist['train_loss'], label='Train'); axes[0,0].plot(hist['val_loss'], label='Val'); axes[0,0].set_title('Loss'); axes[0,0].legend()
    axes[0,1].plot(hist['val_loss_f'], label='Face'); axes[0,1].plot(hist['val_loss_c'], label='Ctx'); axes[0,1].plot(hist['val_loss_w'], label='Whole'); axes[0,1].set_title('Branch Losses'); axes[0,1].legend()
    axes[0,2].plot(hist['bias'], color='orange'); axes[0,2].set_title('Emotion Bias Ratio'); axes[0,2].set_ylim(0, 1)
    axes[1,0].plot(hist['val_acc_w'], label='Acc Whole', color='red'); axes[1,0].axhline(0.9, ls='--', color='g'); axes[1,0].set_title('Accuracy'); axes[1,0].legend()
    
    cm = confusion_matrix(y_true, y_pred)
    cm_p = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_p, annot=True, fmt='.1f', cmap='Blues', ax=axes[1,1], xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
    for t in axes[1,1].texts: t.set_text(t.get_text() + "%")
    axes[1,1].set_title('Confusion Matrix (%)')
    
    axes[1,2].bar(['Face', 'Ctx', 'Whole'], [metrics['f'], metrics['c'], metrics['w']], color=['blue', 'orange', 'red'])
    axes[1,2].set_title('Final Branch Acc Comparison'); axes[1,2].set_ylim(0.7, 1.0)
    plt.tight_layout(); plt.savefig(f"{CONFIG['output_dir']}/dashboard.png"); plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    train_loader = DataLoader(ConGNN_Dataset_WithPriors('train'), CONFIG['batch_size'], True, collate_fn=custom_collate)
    val_loader = DataLoader(ConGNN_Dataset_WithPriors('val'), CONFIG['batch_size'], collate_fn=custom_collate)
    test_loader = DataLoader(ConGNN_Dataset_WithPriors('test'), CONFIG['batch_size'], collate_fn=custom_collate)

    model = VirtualNode_ConGNN_PriorAware().to(CONFIG['device'])
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    hist = {k: [] for k in ['train_loss', 'val_loss', 'val_loss_f', 'val_loss_c', 'val_loss_w', 'val_acc_w', 'bias']}
    
    best_acc = 0
    for ep in range(CONFIG['epochs']):
        # Dynamic Lambda Warm-up (0 -> 0.2 over 20% epochs) 
        lam = (ep / (0.2 * CONFIG['epochs'])) * 0.2 if ep < (0.2 * CONFIG['epochs']) else 0.2
        
        model.train(); t_l = 0
        for b in tqdm(train_loader, desc=f"Ep {ep+1} [Train]"):
            b = b.to(CONFIG['device']); opt.zero_grad()
            of, oc, ow = model(b)
            loss, _, _, _, _ = compute_loss(of, oc, ow, b.y, lam)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            t_l += loss.item()
        
        model.eval(); v_l, v_lf, v_lc, v_lw, v_b, v_acc = 0,0,0,0,0,0
        y_t, y_p = [], []
        with torch.no_grad():
            for b in val_loader:
                b = b.to(CONFIG['device']); of, oc, ow = model(b)
                loss, lf, lc, lw, bias = compute_loss(of, oc, ow, b.y, lam)
                v_l += loss.item(); v_lf += lf; v_lc += lc; v_lw += lw; v_b += bias
                v_acc += (ow.argmax(1) == b.y).sum().item(); y_t.extend(b.y.cpu()); y_p.extend(ow.argmax(1).cpu())
        
        v_acc /= len(val_loader.dataset); v_l /= len(val_loader)
        hist['train_loss'].append(t_l/len(train_loader)); hist['val_loss'].append(v_l)
        hist['val_loss_f'].append(v_lf/len(val_loader)); hist['val_loss_c'].append(v_lc/len(val_loader))
        hist['val_loss_w'].append(v_lw/len(val_loader)); hist['val_acc_w'].append(v_acc); hist['bias'].append(v_b/len(val_loader))
        
        print(f"Val Acc: {v_acc:.4f} | Bias: {v_b/len(val_loader):.3f} | Lam: {lam:.2f}")
        if v_acc > best_acc: best_acc = v_acc; torch.save(model.state_dict(), f"{CONFIG['output_dir']}/best.pth")

    # Final Evaluation & Plotting
    model.load_state_dict(torch.load(f"{CONFIG['output_dir']}/best.pth"))
    model.eval(); y_t, y_pf, y_pc, y_pw = [], [], [], []
    with torch.no_grad():
        for b in test_loader:
            b = b.to(CONFIG['device']); of, oc, ow = model(b)
            y_t.extend(b.y.cpu()); y_pf.extend(of.argmax(1).cpu()); y_pc.extend(oc.argmax(1).cpu()); y_pw.extend(ow.argmax(1).cpu())
    
    metrics = {'f': accuracy_score(y_t, y_pf), 'c': accuracy_score(y_t, y_pc), 'w': accuracy_score(y_t, y_pw)}
    plot_dashboard(hist, y_t, y_pw, metrics)

if __name__ == '__main__': main()