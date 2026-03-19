import os, glob, time, zipfile, re
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIG
# ==========================================
CONFIG = {
    # ── Dataset ───────────────────────────────────────────────────────────
    # Chọn dataset: 'groupemow' hoặc 'gaf2'
    'dataset': 'gaf2',

    # GroupEmoW (face crops)
    'groupemow_root': '/kaggle/input/datasets/trieung11/full-groupemow-face-cropped/GroupEmoW_Full_Face_Cropped',

    # GAF2 (face crops — cấu trúc Train/Val × Positive/Negative/Neutral)
    'gaf2_root': '/kaggle/input/datasets/wawuwaa/tr-facecropgaf2/GAF2_Full_Face_Cropped',

    # Pretrained VGGFace (Oxford weights hoặc checkpoint cũ)
    'pretrained_path': '/kaggle/input/datasets/trieung11/1821153vgg-face/VGG_FACE.pth',

    'output_dir':  '/kaggle/working/vggface_gaf2',
    'ckpt_path':   '/kaggle/working/vggface_gaf2/best.pth',

    # Thư mục lưu file .npy trích xuất đặc trưng
    'feat_dir':    '/kaggle/working/vggface_gaf2/features',

    'img_size':    224,
    'batch_size':  64,
    'num_workers': 2,

    'lr':           1e-5,   # Adam lr=1e-5 theo paper
    'weight_decay': 1e-4,
    'epochs':       50,
    'patience':     10,
    'scheduler_patience': 4,

    # Kaggle time limit: dừng trước 11.5h để kịp save
    'time_limit_sec': 11.5 * 3600,

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['feat_dir'],   exist_ok=True)

# ==========================================
# MODEL
# ==========================================
class VGG_16(nn.Module):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc6  = nn.Linear(512 * 7 * 7, 4096)
        self.fc7  = nn.Linear(4096, 4096)
        self.fc8  = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(dropout)

    def _features(self, x):
        x = self.relu(self.conv1_1(x)); x = self.relu(self.conv1_2(x)); x = self.pool(x)
        x = self.relu(self.conv2_1(x)); x = self.relu(self.conv2_2(x)); x = self.pool(x)
        x = self.relu(self.conv3_1(x)); x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x)); x = self.pool(x)
        x = self.relu(self.conv4_1(x)); x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x)); x = self.pool(x)
        x = self.relu(self.conv5_1(x)); x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x)); x = self.pool(x)
        x = x.view(x.size(0), -1)
        # FIX 4a: self.drop chỉ áp dụng trong training (model.train()),
        #         tự tắt khi model.eval() — hành vi đúng cho forward()
        x = self.relu(self.fc6(x)); x = self.drop(x)
        x = self.relu(self.fc7(x))   # 4096-dim
        return x

    def forward(self, x):
        return self.fc8(self.drop(self._features(x)))

    # FIX 4b: extract() đảm bảo model ở eval mode → dropout = 0
    #         Vector 4096-dim nhất quán, không bị nhiễu ngẫu nhiên
    def extract(self, x):
        """Inference-time: model.eval() đảm bảo dropout tắt, trả về fc7 4096-dim"""
        self.eval()
        with torch.no_grad():
            x = self.relu(self.conv1_1(x)); x = self.relu(self.conv1_2(x)); x = self.pool(x)
            x = self.relu(self.conv2_1(x)); x = self.relu(self.conv2_2(x)); x = self.pool(x)
            x = self.relu(self.conv3_1(x)); x = self.relu(self.conv3_2(x))
            x = self.relu(self.conv3_3(x)); x = self.pool(x)
            x = self.relu(self.conv4_1(x)); x = self.relu(self.conv4_2(x))
            x = self.relu(self.conv4_3(x)); x = self.pool(x)
            x = self.relu(self.conv5_1(x)); x = self.relu(self.conv5_2(x))
            x = self.relu(self.conv5_3(x)); x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc6(x))   # KHÔNG self.drop — đúng cho inference
            x = self.relu(self.fc7(x))   # 4096-dim output
        return x



# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features(model, dataset, split_name, device, feat_dir, batch_size=64, num_workers=2):
    """
    Trích xuất fc7 4096-dim cho toàn bộ một split, lưu thành:
      features/  <split_name>_features.npy   — shape (N, 4096)
      features/  <split_name>_labels.npy     — shape (N,)
      features/  <split_name>_paths.npy      — shape (N,)  (đường dẫn ảnh gốc)
    """
    # Dataset dùng val_tf (không augment) — quan trọng khi extract
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)

    model.eval()
    all_feats  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"  Extract [{split_name}]"):
            imgs = imgs.to(device)
            feats = model.extract(imgs)          # (B, 4096), dropout tắt
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    all_feats  = np.concatenate(all_feats,  axis=0)   # (N, 4096)
    all_labels = np.concatenate(all_labels, axis=0)   # (N,)
    all_paths  = np.array([p for p, _ in dataset.samples])  # (N,)

    feat_path  = os.path.join(feat_dir, f'{split_name}_features.npy')
    label_path = os.path.join(feat_dir, f'{split_name}_labels.npy')
    path_path  = os.path.join(feat_dir, f'{split_name}_paths.npy')

    np.save(feat_path,  all_feats)
    np.save(label_path, all_labels)
    np.save(path_path,  all_paths)

    print(f"  ✅ {split_name}: {all_feats.shape} → {feat_path}")
    return feat_path, label_path, path_path



class GroupEmoWFaceDataset(Dataset):
    LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __init__(self, root, split, transform):
        self.transform = transform
        self.samples   = []

        split_dir = os.path.join(root, split)
        for cls_name in os.listdir(split_dir):
            label = self.LABEL_MAP.get(cls_name.lower())
            if label is None:
                continue
            cls_dir = os.path.join(split_dir, cls_name)
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for p in glob.glob(os.path.join(cls_dir, ext)):
                    self.samples.append((p, label))

        print(f"  {split}: {len(self.samples)} face crops")
        for name, idx in self.LABEL_MAP.items():
            n = sum(1 for _, l in self.samples if l == idx)
            print(f"    {name}: {n} ({100*n/len(self.samples):.1f}%)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        return self.transform(img), label


# ==========================================
# DATASET — GAF2
# Cấu trúc: <root>/Train|Val / Positive|Negative|Neutral / *.jpg
# ==========================================
class GAF2Dataset(Dataset):
    # FIX: tên thư mục GAF2 viết hoa (Positive/Negative/Neutral)
    LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __init__(self, root, split, transform):
        self.transform = transform
        self.samples   = []

        split_dir = os.path.join(root, split)
        for cls_name in os.listdir(split_dir):
            label = self.LABEL_MAP.get(cls_name.lower())   # lower() để match cả Positive lẫn positive
            if label is None:
                continue
            cls_dir = os.path.join(split_dir, cls_name)
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for p in glob.glob(os.path.join(cls_dir, ext)):
                    self.samples.append((p, label))

        print(f"  {split}: {len(self.samples)} images")
        for name, idx in self.LABEL_MAP.items():
            n = sum(1 for _, l in self.samples if l == idx)
            if n > 0:
                print(f"    {name}: {n} ({100*n/len(self.samples):.1f}%)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        return self.transform(img), label


# ==========================================
# TRANSFORMS
# ==========================================
# FIX 1: RandomRotation(10) → RandomRotation(20) theo bài báo (±20°)
# FIX 2: ColorJitter bỏ — không có trong bài báo, gây nhiễu feature màu da
# FIX 3: Normalize mean đổi sang RGB order (Oxford VGGFace mean gốc là BGR)
#         BGR mean: [93.5940, 104.7624, 129.1863] / 255
#         → RGB order: [129.1863, 104.7624, 93.5940] / 255 ≈ [0.507, 0.411, 0.367]
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),              # FIX 1: 10 → 20
    # ColorJitter đã bỏ                         # FIX 2
    transforms.ToTensor(),
    transforms.Normalize([0.507, 0.411, 0.367], [1, 1, 1]),  # FIX 3: RGB order
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.507, 0.411, 0.367], [1, 1, 1]),  # FIX 3: RGB order
])


# ==========================================
# MAIN
# ==========================================
def main():
    print("="*70)
    print("  FINE-TUNE VGGFace")
    print(f"  Dataset : {CONFIG['dataset'].upper()}")
    print(f"  Device  : {CONFIG['device']}")
    print("="*70 + "\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    print("📂 Loading datasets...")

    if CONFIG['dataset'] == 'gaf2':
        root = CONFIG['gaf2_root']
        # GAF2 dùng Train/Val (viết hoa)
        train_ds = GAF2Dataset(root, 'Train', train_tf)
        val_ds   = GAF2Dataset(root, 'Val',   val_tf)
    else:
        root = CONFIG['groupemow_root']
        train_ds = GroupEmoWFaceDataset(root, 'train', train_tf)
        val_ds   = GroupEmoWFaceDataset(root, 'val',   val_tf)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n📦 Loading model...")
    model = VGG_16(num_classes=3, dropout=0.5).to(CONFIG['device'])

    start_epoch = 0
    best_acc    = 0.0
    no_improve  = 0

    resume_path = CONFIG['ckpt_path']
    load_path   = resume_path if os.path.exists(resume_path) else CONFIG['pretrained_path']

    if os.path.exists(load_path):
        print(f"  Loading: {load_path}")
        ckpt = torch.load(load_path, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        model_sd = model.state_dict()
        sd = {k: v for k, v in sd.items()
              if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  Loaded {len(sd)}/{len(model_sd)} layers")
        if missing:    print(f"  Missing keys   : {missing}")
        if unexpected: print(f"  Unexpected keys: {unexpected}")
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch']
            best_acc    = ckpt.get('best_acc', 0.0)
            print(f"  ✅ Resuming from epoch {start_epoch} | best_acc={best_acc:.4f}")
        else:
            print(f"  ✅ Pretrained weights loaded")
    else:
        print("  ⚠️ No pretrained weights found, training from scratch")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=CONFIG['scheduler_patience'], min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  ✅ Optimizer state resumed")

    # ── Training loop ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("🎯 TRAINING START")
    print("="*70 + "\n")

    global_start = time.time()

    for epoch in range(start_epoch, CONFIG['epochs']):
        t0 = time.time()

        # Train
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch+1:02d} [Train]", leave=False):
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss    += loss.item() * len(labels)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total   += len(labels)

        # Val
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Ep {epoch+1:02d} [Val]  ", leave=False):
                imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += len(labels)

        tr_acc  = tr_correct  / tr_total
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        lr_now = next(iter(optimizer.param_groups))['lr']
        print(f"Ep {epoch+1:02d} [{time.time()-t0:.0f}s] "
              f"TrLoss={tr_loss/tr_total:.4f} | TrAcc={tr_acc:.4f} | "
              f"ValAcc={val_acc:.4f} | LR={lr_now:.1e}")

        # Save best
        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            torch.save({
                'epoch':                epoch + 1,
                'state_dict':           model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc':             best_acc,
            }, CONFIG['ckpt_path'])
            print(f"  🔥 Best! ValAcc={val_acc:.4f} → saved")
        else:
            no_improve += 1
            print(f"  ⏳ No improve {no_improve}/{CONFIG['patience']}")
            if no_improve >= CONFIG['patience']:
                print("\n🛑 Early stopping")
                break

        # Kaggle time guard
        elapsed = time.time() - global_start
        if elapsed > CONFIG['time_limit_sec']:
            print(f"\n⏳ {elapsed/3600:.1f}h elapsed — stopping to save outputs")
            break

    # ── Zip checkpoint ────────────────────────────────────────────────────
    print(f"\n✅ Training done | Best ValAcc = {best_acc:.4f}")
    print("📦 Zipping checkpoint...")
    ckpt_zip = '/kaggle/working/vggface_gaf2_best.zip'
    with zipfile.ZipFile(ckpt_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(CONFIG['ckpt_path'], 'vggface_gaf2_best.pth')
    print(f"✅ Saved: {ckpt_zip}  ({os.path.getsize(ckpt_zip)/1e6:.1f} MB)")

    # ── Feature Extraction ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("🔍 FEATURE EXTRACTION (load best checkpoint)")
    print("="*70 + "\n")

    # Load lại best checkpoint để extract
    best_ckpt = torch.load(CONFIG['ckpt_path'], map_location='cpu')
    model.load_state_dict(best_ckpt['state_dict'])
    print(f"  ✅ Loaded best.pth (epoch {best_ckpt['epoch']}, ValAcc={best_ckpt['best_acc']:.4f})")

    # Tạo dataset với val_tf (không augment) cho cả train lẫn val
    root = CONFIG['gaf2_root']
    train_ds_noaug = GAF2Dataset(root, 'Train', val_tf)
    val_ds_noaug   = GAF2Dataset(root, 'Val',   val_tf)

    feat_dir = CONFIG['feat_dir']
    extract_features(model, train_ds_noaug, 'train', CONFIG['device'], feat_dir,
                     CONFIG['batch_size'], CONFIG['num_workers'])
    extract_features(model, val_ds_noaug,   'val',   CONFIG['device'], feat_dir,
                     CONFIG['batch_size'], CONFIG['num_workers'])

    # ── Zip features ──────────────────────────────────────────────────────
    print("\n📦 Zipping features...")
    feat_zip = '/kaggle/working/vggface_gaf2_features.zip'
    with zipfile.ZipFile(feat_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(feat_dir):
            fpath = os.path.join(feat_dir, fname)
            zf.write(fpath, fname)
    print(f"✅ Saved: {feat_zip}  ({os.path.getsize(feat_zip)/1e6:.1f} MB)")

    print("\n📥 Download từ Kaggle Output tab:")
    print(f"   → {ckpt_zip}")
    print(f"   → {feat_zip}")


if __name__ == '__main__':
    main()