import os
import copy
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
    'face_dir':   '/kaggle/input/datasets/trieung11/gaf2-fearture/face_features_bbox_gaf2',
    'scene_dir':  '/kaggle/input/datasets/trieung11/gaf2-fearture/scene_features_gaf2000_v2/scene_features_final/scenes',
    'object_dir': '/kaggle/input/datasets/trieung11/gaf2-fearture/gaf2_object_features',
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

    'min_lr':       1e-6,

    'branch_w_max':   0.30,
    'branch_w_min':   0.30,
    'warmup_epochs':  20,
    'decay_end':      80,

    'num_runs': 10,

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
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print(f"Device: {CONFIG['device']}")
    print("="*80 + "\n")


def get_branch_weight(epoch):
    w_max = CONFIG['branch_w_max']; w_min = CONFIG['branch_w_min']
    ep_warm = CONFIG['warmup_epochs']; ep_decay = CONFIG['decay_end']
    if epoch < ep_warm: return w_max
    if epoch >= ep_decay: return w_min
    return w_max + (epoch - ep_warm) / (ep_decay - ep_warm) * (w_min - w_max)


# ==========================================
# GRAPH BUILDERS
# ==========================================
def build_knn_edges(boxes, k=3):
    n = len(boxes)
    if n <= 1:
        return torch.tensor([[0],[0]], dtype=torch.long)
    cx = (boxes[:,0]+boxes[:,2])/2.0; cy = (boxes[:,1]+boxes[:,3])/2.0
    centers = np.stack([cx,cy],axis=1)
    diff = centers[:,None,:] - centers[None,:,:]
    dist = np.sqrt((diff**2).sum(axis=-1))
    src_list, dst_list = [], []
    actual_k = min(k, n-1)
    for i in range(n):
        d = dist[i].copy(); d[i] = np.inf
        for j in np.argsort(d)[:actual_k]:
            src_list+=[i,j]; dst_list+=[j,i]
    edges = list(set(zip(src_list,dst_list)))
    if not edges: return torch.tensor([[0],[0]], dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_dense_edges(num_nodes):
    if num_nodes <= 1:
        return torch.tensor([[0],[0]], dtype=torch.long)
    edges = [[i,j] for i in range(num_nodes) for j in range(num_nodes) if i!=j]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ==========================================
# DATASET
# ==========================================
class ConGNN_Dataset(TorchDataset):
    def __init__(self, split='train', max_faces=32, max_objects=10, use_cache=None):
        # Map tên split — GAF2: face=train/val | scene+object=Train/Val
        face_split  = split                                    # train / val
        scene_split = 'Train' if split == 'train' else 'Val'  # Train / Val
        obj_split   = 'Train' if split == 'train' else 'Val'  # Train / Val

        self.face_root  = CONFIG['face_dir']
        self.scene_root = os.path.join(CONFIG['scene_dir'],  scene_split)  # fix: tách đúng split
        self.obj_root   = os.path.join(CONFIG['object_dir'], obj_split)    # fix: tách đúng split
        self.max_faces  = max_faces
        self.max_objects = max_objects
        self.label_map   = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.use_cache   = CONFIG['use_ram_cache'] if use_cache is None else use_cache

        pattern = os.path.join(self.face_root, 'faces', face_split, '**', '*.npz')
        self.face_files = glob.glob(pattern, recursive=True)
        if not self.face_files:
            pattern = os.path.join(self.face_root, face_split, '**', '*.npz')
            self.face_files = glob.glob(pattern, recursive=True)

        print(f"📊 {split.upper()}: {len(self.face_files)} samples")
        if not self.face_files:
            raise ValueError(f"No data found for split '{split}'")

        self._build_scene_index(); self._build_object_index()
        self._cache = None
        if self.use_cache: self._preload_all(split)

    def _build_scene_index(self):
        self._scene_index = {}
        for p in glob.glob(os.path.join(self.scene_root,'**','*.npy'),recursive=True):
            self._scene_index[os.path.splitext(os.path.basename(p))[0]] = p

    def _build_object_index(self):
        self._obj_index = {}
        for p in glob.glob(os.path.join(self.obj_root,'**','*.npz'),recursive=True):
            self._obj_index[os.path.splitext(os.path.basename(p))[0]] = p

    def _get_paired_path(self, face_path, target_type):
        stem = os.path.splitext(os.path.basename(face_path))[0]
        return self._scene_index.get(stem) if target_type=='scenes' else self._obj_index.get(stem)

    def _preload_all(self, split):
        print(f"  💾 Preloading {split.upper()}...")
        t0 = time.time()
        self._cache = [self._load_sample(i) for i in
                       tqdm(range(len(self.face_files)), desc=f"  Cache {split}", leave=False)]
        print(f"  ✅ {len(self._cache)} samples in {time.time()-t0:.1f}s\n")

    def _load_sample(self, idx):
        face_file = self.face_files[idx]
        label = self.label_map.get(os.path.basename(os.path.dirname(face_file)).lower(), 1)
        try:
            data = np.load(face_file)
            face_feat, face_boxes = data['features'], data['boxes']
            if len(face_boxes) > 0:
                s = np.argsort(face_boxes[:,0])
                face_feat, face_boxes = face_feat[s], face_boxes[s]
        except:
            face_feat  = np.zeros((1,CONFIG['face_dim']),  dtype=np.float32)
            face_boxes = np.zeros((1,4), dtype=np.float32)

        face_feat  = face_feat[:self.max_faces]  if len(face_feat)  > 0 else np.zeros((1,CONFIG['face_dim']),  dtype=np.float32)
        face_boxes = face_boxes[:self.max_faces] if len(face_boxes) > 0 else np.zeros((1,4), dtype=np.float32)
        face_x          = torch.tensor(face_feat,  dtype=torch.float32)
        face_edge_index = build_knn_edges(face_boxes, k=CONFIG['knn_k'])

        scene_path = self._get_paired_path(face_file,'scenes')
        try:
            if scene_path and os.path.exists(scene_path):
                sf = np.load(scene_path)
                if sf.ndim==4: sf=sf.mean(axis=(0,2,3))
                elif sf.ndim==3: sf=sf.mean(axis=(-2,-1))
                elif sf.ndim==2:
                    sf = sf.squeeze(0) if sf.shape[0]==1 else (sf.mean(0) if sf.shape[-1]==CONFIG['scene_dim'] else sf.mean(-1))
                sf = sf.flatten()[:CONFIG['scene_dim']]
            else: sf = np.zeros(CONFIG['scene_dim'],dtype=np.float32)
        except: sf = np.zeros(CONFIG['scene_dim'],dtype=np.float32)
        if len(sf) < CONFIG['scene_dim']: sf = np.pad(sf,(0,CONFIG['scene_dim']-len(sf)))
        scene_x = torch.tensor(sf.astype(np.float32),dtype=torch.float32)

        obj_path = self._get_paired_path(face_file,'objects')
        try:
            if obj_path and os.path.exists(obj_path):
                od = np.load(obj_path)
                obj_feat = od['features'] if 'features' in od else od[od.files[0]]
            else: obj_feat = np.zeros((0,CONFIG['object_dim']),dtype=np.float32)
        except: obj_feat = np.zeros((0,CONFIG['object_dim']),dtype=np.float32)

        obj_feat  = obj_feat[:self.max_objects]
        context_x = torch.tensor(obj_feat,dtype=torch.float32) if len(obj_feat)>0 \
                    else torch.zeros((1,CONFIG['object_dim']),dtype=torch.float32)
        context_edge_index = build_dense_edges(len(context_x))

        return {'face_x':face_x,'face_edge_index':face_edge_index,
                'context_x':context_x,'context_edge_index':context_edge_index,
                'scene_x':scene_x,'y':label}

    def __len__(self): return len(self.face_files)
    def __getitem__(self, idx):
        return self._cache[idx] if self._cache is not None else self._load_sample(idx)


# ==========================================
# COLLATE
# ==========================================
class SimpleBatch:
    def __init__(self,face_x,face_edge_index,face_batch,
                 context_x,context_edge_index,context_batch,scene_x,y,num_graphs):
        self.face_x=face_x; self.face_edge_index=face_edge_index; self.face_batch=face_batch
        self.context_x=context_x; self.context_edge_index=context_edge_index; self.context_batch=context_batch
        self.scene_x=scene_x; self.y=y; self.num_graphs=num_graphs

    def to(self, device):
        for a in ['face_x','face_edge_index','face_batch','context_x',
                  'context_edge_index','context_batch','scene_x','y']:
            setattr(self,a,getattr(self,a).to(device))
        return self

    def __getstate__(self): return self.__dict__
    def __setstate__(self,d): self.__dict__.update(d)


def custom_collate(batch):
    fx,fei,fb,cx,cei,cb,sx,yl=[],[],[],[],[],[],[],[]
    fn=cn=0
    for gi,s in enumerate(batch):
        nf=s['face_x'].size(0); fx.append(s['face_x']); fei.append(s['face_edge_index']+fn)
        fb.append(torch.full((nf,),gi,dtype=torch.long)); fn+=nf
        nc=s['context_x'].size(0); cx.append(s['context_x']); cei.append(s['context_edge_index']+cn)
        cb.append(torch.full((nc,),gi,dtype=torch.long)); cn+=nc
        sx.append(s['scene_x']); yl.append(s['y'])
    return SimpleBatch(torch.cat(fx),torch.cat(fei,dim=1),torch.cat(fb),
                       torch.cat(cx),torch.cat(cei,dim=1),torch.cat(cb),
                       torch.stack(sx),torch.tensor(yl,dtype=torch.long),len(batch))


# ==========================================
# MODEL
# ==========================================
class MultiLayerGATv2(nn.Module):
    def __init__(self,in_dim,hidden_dim,num_heads=4,num_layers=2,dropout=0.5,attention_dropout=0.3):
        super().__init__()
        self.input_proj=nn.Linear(in_dim,hidden_dim); self.input_norm=nn.LayerNorm(hidden_dim)
        self.gat_layers=nn.ModuleList(); self.norms=nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(GATv2Conv(hidden_dim,hidden_dim//num_heads,heads=num_heads,
                                             dropout=attention_dropout,add_self_loops=True,concat=True,bias=False))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop=nn.Dropout(dropout)

    def forward(self,x,edge_index):
        h=self.drop(F.relu(self.input_norm(self.input_proj(x))))
        for gat,norm in zip(self.gat_layers,self.norms):
            h_new=self.drop(F.elu(norm(gat(h,edge_index))))
            h=h+h_new if h.shape==h_new.shape else h_new
        return h


class AttentionPool(nn.Module):
    def __init__(self,hidden_dim,dropout=0.3):
        super().__init__()
        self.score_mlp=nn.Sequential(nn.Linear(hidden_dim,hidden_dim//4),nn.Tanh(),
                                     nn.Dropout(dropout),nn.Linear(hidden_dim//4,1))

    def forward(self,x,batch):
        scores=self.score_mlp(x); scores=scores-scores.max(); exp_s=torch.exp(scores)
        B=batch.max().item()+1
        denom=torch.zeros(B,1,device=x.device); denom.scatter_add_(0,batch.unsqueeze(1),exp_s)
        weight=exp_s/(denom[batch]+1e-8)
        out=torch.zeros(B,x.size(1),device=x.device)
        out.scatter_add_(0,batch.unsqueeze(1).expand_as(x),weight*x)
        return out


class SceneGuidedFusion(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.cross_attn=nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=4,batch_first=True)
        self.layer_norm=nn.LayerNorm(hidden_dim)

    def forward(self,scene_feat,face_nodes,obj_nodes,face_batch,obj_batch):
        all_nodes=torch.cat([face_nodes,obj_nodes],dim=0)
        all_batch=torch.cat([face_batch,obj_batch],dim=0)
        dense_nodes,mask=to_dense_batch(all_nodes,all_batch)
        query=scene_feat.unsqueeze(1)
        attn_out,attn_w=self.cross_attn(query,dense_nodes,dense_nodes,key_padding_mask=~mask)
        return self.layer_norm(query+attn_out).squeeze(1),attn_w


class EmotionalContagion(nn.Module):
    def __init__(self,hidden_dim,dropout=0.3):
        super().__init__()
        self.alpha=nn.Parameter(torch.zeros(hidden_dim))
        self.norm=nn.LayerNorm(hidden_dim); self.drop=nn.Dropout(dropout)

    def forward(self,H_face,fused_scene,face_batch):
        return self.norm(H_face+self.drop(self.alpha*fused_scene[face_batch]))


class SceneGuided_ConGNN(nn.Module):
    def __init__(self):
        super().__init__()
        D=CONFIG['gat_hidden']; drp=CONFIG['dropout']; att_drp=CONFIG['attention_dropout']

        self.reduce_face=nn.Sequential(
            nn.LayerNorm(CONFIG['face_dim']),nn.Linear(CONFIG['face_dim'],1024),
            nn.LayerNorm(1024),nn.ReLU(),nn.Dropout(drp),nn.Linear(1024,D),nn.LayerNorm(D),nn.ReLU())
        self.reduce_obj=nn.Sequential(
            nn.LayerNorm(CONFIG['object_dim']),nn.Linear(CONFIG['object_dim'],D),
            nn.LayerNorm(D),nn.ReLU(),nn.Dropout(drp))
        self.reduce_scene=nn.Sequential(
            nn.LayerNorm(CONFIG['scene_dim']),nn.Linear(CONFIG['scene_dim'],D),
            nn.LayerNorm(D),nn.ReLU(),nn.Dropout(drp))

        self.face_gat=MultiLayerGATv2(D,D,CONFIG['num_heads'],CONFIG['gat_layers'],drp,att_drp)
        self.context_gat=MultiLayerGATv2(D,D,CONFIG['num_heads'],CONFIG['gat_layers'],drp,att_drp)

        self.clf_face=nn.Linear(D,CONFIG['num_classes'])
        self.clf_context=nn.Linear(D,CONFIG['num_classes'])
        self.clf_scene=nn.Linear(D,CONFIG['num_classes'])

        self.scene_guided_fusion=SceneGuidedFusion(D)
        self.emotional_contagion=EmotionalContagion(D,drp)

        self.lambda_face=nn.Parameter(torch.tensor(0.5))
        self.lambda_obj=nn.Parameter(torch.tensor(0.5))
        self.raw_face_proj=nn.Linear(D,D); self.raw_obj_proj=nn.Linear(D,D)

        self.attn_pool_face=AttentionPool(D,drp)
        self.attn_pool_face_branch=AttentionPool(D,drp)

        self.clf_whole=nn.Sequential(
            nn.Dropout(0.5),nn.Linear(D*3,D),nn.LayerNorm(D),nn.ReLU(),nn.Dropout(drp),
            nn.Linear(D,CONFIG['num_classes']))

    def forward(self,data):
        face_x_proj=self.reduce_face(data.face_x)
        obj_x_proj=self.reduce_obj(data.context_x)
        scene_proj=self.reduce_scene(data.scene_x)

        H_face=self.face_gat(face_x_proj,data.face_edge_index)
        H_obj=self.context_gat(obj_x_proj,data.context_edge_index)

        out_face=self.clf_face(self.attn_pool_face_branch(H_face,data.face_batch))
        out_context=self.clf_context(global_mean_pool(H_obj,data.context_batch))
        out_scene=self.clf_scene(scene_proj)

        fused_scene,_=self.scene_guided_fusion(scene_proj,H_face,H_obj,data.face_batch,data.context_batch)
        H_face=self.emotional_contagion(H_face,fused_scene,data.face_batch)

        H_face_res=H_face+self.lambda_face*self.raw_face_proj(face_x_proj)
        H_obj_res=H_obj+self.lambda_obj*self.raw_obj_proj(obj_x_proj)

        feat_face=self.attn_pool_face(H_face_res,data.face_batch)
        feat_obj=global_mean_pool(H_obj_res,data.context_batch)

        out_whole=self.clf_whole(torch.cat([fused_scene,feat_face,feat_obj],dim=1))
        return out_face,out_context,out_scene,out_whole


# ==========================================
# LOSS / EARLY STOPPING
# ==========================================
def compute_loss(out_f,out_c,out_s,out_w,labels,focal_crit,ce_crit,branch_w=0.30):
    labels=labels.long()
    L_w=focal_crit(out_w,labels); L_f=ce_crit(out_f,labels)
    L_c=ce_crit(out_c,labels);    L_s=ce_crit(out_s,labels)
    return L_w+branch_w*(L_f+L_c+L_s), L_f.item(),L_c.item(),L_s.item(),L_w.item()


class EarlyStopping:
    def __init__(self,patience=40,min_delta=0.001,path='best.pth'):
        self.patience=patience; self.min_delta=min_delta; self.path=path
        self.counter=0; self.best_loss=np.inf; self.best_acc=0; self.early_stop=False

    def __call__(self,val_loss,val_acc,model):
        improved=(val_acc > self.best_acc+self.min_delta or
                  (val_loss < self.best_loss-self.min_delta and val_acc >= self.best_acc-0.002))
        if improved:
            self.best_loss=val_loss; self.best_acc=val_acc
            torch.save(model.state_dict(),self.path)
            self.counter=0
            print(f"  🔥 Best! Loss={val_loss:.4f} | Acc={val_acc:.4f}")
        else:
            self.counter+=1
            if self.counter>=self.patience: self.early_stop=True


def get_lr(opt): return next(iter(opt.param_groups))['lr']


# ==========================================
# MAIN — 10 runs, global best → eval on val
# ==========================================
def main():
    print_system_info()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # ── Load data ────────────────────────────────────────────────────────────
    print("📂 Loading datasets...")
    train_ds = ConGNN_Dataset('train')
    val_ds   = ConGNN_Dataset('val')

    kw = dict(batch_size=CONFIG['batch_size'], collate_fn=custom_collate,
              num_workers=CONFIG['num_workers'], pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    print(f"✅ train={len(train_loader)} | val={len(val_loader)} batches\n")

    # ── Tracking globals ─────────────────────────────────────────────────────
    all_run_accs        = []   # best val acc mỗi run
    global_best_acc     = 0.0
    global_best_state   = None
    global_best_run     = -1
    global_best_history = None  # history của run tốt nhất để vẽ biểu đồ

    print("="*80)
    print(f"🎯 10-RUN PROTOCOL  |  V10 Emotional Contagion")
    print("="*80 + "\n")

    for run in range(CONFIG['num_runs']):
        print(f"\n{'▓'*70}")
        print(f"  RUN {run+1}/{CONFIG['num_runs']}")
        print(f"{'▓'*70}\n")

        # ── Reset hoàn toàn ───────────────────────────────────────────────
        model     = SceneGuided_ConGNN().to(CONFIG['device'])
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
        focal_crit = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        ce_crit    = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        ckpt_path  = f"{CONFIG['output_dir']}/run{run+1:02d}_best.pth"
        es         = EarlyStopping(CONFIG['patience'], path=ckpt_path)

        # history của run này
        history = {k: [] for k in ['train_loss', 'val_loss',
                                    'val_acc_whole', 'val_acc_face',
                                    'val_acc_context', 'val_acc_scene',
                                    'branch_w']}

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(CONFIG['epochs']):
            branch_w = get_branch_weight(epoch)
            history['branch_w'].append(branch_w)

            # Train
            model.train(); t_loss = 0
            bar = tqdm(train_loader,
                       desc=f"Run{run+1} Ep{epoch+1:03d} [Train] bw={branch_w:.3f}")
            for bi, batch in enumerate(bar):
                try:
                    batch = batch.to(CONFIG['device'])
                    optimizer.zero_grad()
                    out_f,out_c,out_s,out_w = model(batch)
                    loss,*_ = compute_loss(out_f,out_c,out_s,out_w,batch.y,
                                           focal_crit,ce_crit,branch_w)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                    optimizer.step()
                    t_loss += loss.item()
                    bar.set_postfix(loss=f"{loss.item():.4f}")
                    if bi%50==0 and torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception as e:
                    print(f"❌ train batch {bi}: {e}")
                    if CONFIG['debug_mode']: raise
                    continue

            # Validate
            model.eval()
            v_loss = v_aw = v_af = v_ac = v_as = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        batch = batch.to(CONFIG['device'])
                        out_f,out_c,out_s,out_w = model(batch)
                        loss,*_ = compute_loss(out_f,out_c,out_s,out_w,batch.y,
                                               focal_crit,ce_crit,branch_w)
                        bs = len(batch.y)
                        v_loss += loss.item()*bs
                        v_aw   += (out_w.argmax(1)==batch.y).sum().item()
                        v_af   += (out_f.argmax(1)==batch.y).sum().item()
                        v_ac   += (out_c.argmax(1)==batch.y).sum().item()
                        v_as   += (out_s.argmax(1)==batch.y).sum().item()
                        total  += bs
                    except Exception as e:
                        print(f"❌ val batch: {e}")
                        if CONFIG['debug_mode']: raise
                        continue

            scheduler.step()
            t_loss /= len(train_loader)
            vl=v_loss/total; vaw=v_aw/total; vaf=v_af/total; vac=v_ac/total; vas=v_as/total

            history['train_loss'].append(t_loss)
            history['val_loss'].append(vl)
            history['val_acc_whole'].append(vaw)
            history['val_acc_face'].append(vaf)
            history['val_acc_context'].append(vac)
            history['val_acc_scene'].append(vas)

            print(f"Run{run+1} Ep{epoch+1:03d} | TrL={t_loss:.4f} ValL={vl:.4f} | "
                  f"Whole={vaw:.4f} Face={vaf:.4f} Ctx={vac:.4f} Scene={vas:.4f} | "
                  f"LR={get_lr(optimizer):.1e}")

            es(vl, vaw, model)
            if es.early_stop:
                print(f"🛑 Early stop at epoch {epoch+1}")
                break
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

        # ── Kết thúc run ─────────────────────────────────────────────────
        best_acc_this_run = es.best_acc
        all_run_accs.append(best_acc_this_run)
        print(f"\n✅ Run {run+1} → Best Val Acc = {best_acc_this_run:.4f}")

        # Cập nhật global best
        if best_acc_this_run > global_best_acc:
            global_best_acc     = best_acc_this_run
            global_best_run     = run + 1
            global_best_history = {k: list(v) for k, v in history.items()}
            # Load ckpt tốt nhất của run này rồi deepcopy
            model.load_state_dict(torch.load(ckpt_path, map_location=CONFIG['device']))
            global_best_state = copy.deepcopy(model.state_dict())
            print(f"  🌟 New Global Best: {global_best_acc:.4f} (Run {run+1})")

        del model, optimizer, scheduler, es
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # ============================================================
    # FINAL EVALUATION — Global Best model trên val_loader
    # ============================================================
    print("\n" + "="*80)
    print(f"🏆  FINAL EVALUATION — Best Run = {global_best_run}  |  Val Acc = {global_best_acc:.4f}")
    print("="*80 + "\n")

    # Lưu global best
    global_ckpt = f"{CONFIG['output_dir']}/global_best_model.pth"
    torch.save(global_best_state, global_ckpt)
    print(f"💾 Global best saved: {global_ckpt}\n")

    # Load vào model mới
    final_model = SceneGuided_ConGNN().to(CONFIG['device'])
    final_model.load_state_dict(global_best_state)
    final_model.eval()

    focal_crit = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    ce_crit    = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final eval"):
            try:
                batch = batch.to(CONFIG['device'])
                _,_,_,out_w = final_model(batch)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(out_w.argmax(1).cpu().numpy())
            except Exception as e:
                print(f"❌ {e}"); continue

    final_acc = accuracy_score(y_true, y_pred)

    # ── In kết quả ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Best Run     : {global_best_run}")
    print(f"  Val Acc      : {final_acc*100:.2f}%")
    print("="*60)
    print(classification_report(y_true, y_pred,
                                target_names=['Negative','Neutral','Positive'],
                                digits=4))
    print("🎉 TARGET!" if final_acc >= 0.95 else f"📈 Gap to 95%: {(0.95-final_acc)*100:.2f}%")

    # ── 10-run summary ────────────────────────────────────────────────────────
    print("\n── 10-Run Summary ──────────────────────────────────────")
    for i, acc in enumerate(all_run_accs):
        marker = " ← BEST" if (i+1) == global_best_run else ""
        print(f"  Run {i+1:02d}: {acc*100:.2f}%{marker}")
    print(f"  Mean ± Std : {np.mean(all_run_accs)*100:.2f}% ± {np.std(all_run_accs)*100:.2f}%")
    print("─"*56)

    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    h      = global_best_history   # shorthand

    # =========================================================
    # FIGURE 1 — Training curves (正方形 2×2)
    # =========================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 14))
    fig1.suptitle(f'Training Curves — Best Run {global_best_run}  |  Val Acc = {final_acc*100:.2f}%',
                  fontsize=15, fontweight='bold', y=1.01)

    epochs_x = range(1, len(h['train_loss']) + 1)

    # (0,0) Total Loss
    ax = axes1[0, 0]
    ax.plot(epochs_x, h['train_loss'], label='Train Loss', lw=2.5, color='#e74c3c')
    ax.plot(epochs_x, h['val_loss'],   label='Val Loss',   lw=2.5, color='#3498db')
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    ax.set_aspect('auto')

    # (0,1) Val Accuracy — all branches
    ax = axes1[0, 1]
    ax.plot(epochs_x, h['val_acc_whole'],   label='Whole',   lw=2.5, color='#e74c3c')
    ax.plot(epochs_x, h['val_acc_face'],    label='Face',    lw=2,   color='#9b59b6', alpha=0.85)
    ax.plot(epochs_x, h['val_acc_context'], label='Object',  lw=2,   color='#f39c12', alpha=0.85)
    ax.plot(epochs_x, h['val_acc_scene'],   label='Scene',   lw=2,   color='#2ecc71', alpha=0.85)
    ax.axhline(y=0.90, color='gray', linestyle='--', lw=1.2, label='90% target')
    ax.set_title('Val Accuracy — All Branches', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # (1,0) Branch Weight Schedule
    ax = axes1[1, 0]
    ax.plot(epochs_x, h['branch_w'], lw=2.5, color='#e67e22')
    ax.axvline(x=CONFIG['warmup_epochs'], color='gray', linestyle=':', lw=1.5,
               label=f"warmup end ({CONFIG['warmup_epochs']}ep)")
    ax.axvline(x=CONFIG['decay_end'],     color='gray', linestyle='--', lw=1.5,
               label=f"decay end ({CONFIG['decay_end']}ep)")
    ax.axhline(y=CONFIG['branch_w_min'], color='#e74c3c', linestyle='--', lw=1.2,
               label=f"min={CONFIG['branch_w_min']}")
    ax.set_title('Branch Loss Weight Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('branch_w', fontsize=12)
    ax.set_ylim(0, CONFIG['branch_w_max'] * 1.2)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # (1,1) 10-Run Summary bar chart
    ax = axes1[1, 1]
    colors_bar = ['#2ecc71' if (i+1)==global_best_run else '#3498db'
                  for i in range(len(all_run_accs))]
    bars = ax.bar(range(1, len(all_run_accs)+1),
                  [a*100 for a in all_run_accs],
                  color=colors_bar, edgecolor='white', linewidth=1.5)
    mean_a = np.mean(all_run_accs)*100; std_a = np.std(all_run_accs)*100
    ax.axhline(mean_a, color='#e74c3c', linestyle='--', lw=2,
               label=f'Mean={mean_a:.2f}%')
    ax.fill_between(range(0, len(all_run_accs)+2),
                    mean_a-std_a, mean_a+std_a,
                    alpha=0.15, color='#e74c3c', label=f'±Std={std_a:.2f}%')
    for bar, v in zip(bars, all_run_accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{v*100:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_title('10-Run Val Accuracy Summary', fontsize=14, fontweight='bold')
    ax.set_xlabel('Run', fontsize=12); ax.set_ylabel('Val Acc (%)', fontsize=12)
    ax.set_xticks(range(1, len(all_run_accs)+1))
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    fig1.tight_layout()
    path1 = f"{CONFIG['output_dir']}/training_curves_best_run{global_best_run}.png"
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Training curves saved: {path1}")

    # =========================================================
    # FIGURE 2 — Confusion Matrices (正方形 1×2)
    # =========================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
    fig2.suptitle(f'Confusion Matrix — Best Run {global_best_run}  |  Val Acc = {final_acc*100:.2f}%',
                  fontsize=15, fontweight='bold')

    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[0],
                xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'],
                annot_kws={"size":20, "weight":"bold"}, linewidths=2,
                linecolor='white', square=True)
    axes2[0].set_title('Counts', fontsize=14, fontweight='bold')
    axes2[0].set_xlabel('Predicted', fontsize=13); axes2[0].set_ylabel('True', fontsize=13)

    # Normalized %
    sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Blues', ax=axes2[1],
                xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'],
                annot_kws={"size":20, "weight":"bold"}, linewidths=2,
                linecolor='white', square=True)
    axes2[1].set_title(f'Per-class % — Neg={cm_pct[0,0]:.2f}  Neu={cm_pct[1,1]:.2f}  Pos={cm_pct[2,2]:.2f}',
                       fontsize=13, fontweight='bold')
    axes2[1].set_xlabel('Predicted', fontsize=13); axes2[1].set_ylabel('True', fontsize=13)

    fig2.tight_layout()
    path2 = f"{CONFIG['output_dir']}/confusion_matrix_best_run{global_best_run}.png"
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Confusion matrix saved: {path2}")
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()