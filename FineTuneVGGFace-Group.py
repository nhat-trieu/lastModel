"""
finetune_vggface_groupemow_only.py
===================================
Fine-tune VGGFace CHỈ trên GroupEmoW face crops
- Auto-save checkpoint mỗi epoch tốt nhất
- Resume nếu Kaggle bị ngắt
- Output: .pth + .zip để tải về
"""

import os, glob, time, zipfile, re
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
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
    # Input: face crops + bbox
    'data_root': '/kaggle/input/datasets/trieung11/full-groupemow-face-cropped/GroupEmoW_Full_Face_Cropped',

    # Pretrained VGGFace (Oxford weights hoặc checkpoint cũ)
    # Nếu có checkpoint cũ thì để vào đây để resume
    'pretrained_path': '/kaggle/input/datasets/trieung11/1821153vgg-face/VGG_FACE.pth',

    'output_dir':  '/kaggle/working/vggface_groupemow',
    'ckpt_path':   '/kaggle/working/vggface_groupemow/best.pth',

    'img_size':    224,
    'batch_size':  64,
    'num_workers': 2,

    'lr':           1e-5,   # đúng theo paper (Adam lr=1e-5)
    'weight_decay': 1e-4,
    'epochs':       50,
    'patience':     10,
    'scheduler_patience': 4,

    # Kaggle time limit: dừng trước 11.5h để kịp save
    'time_limit_sec': 11.5 * 3600,

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

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
        x = self.relu(self.fc6(x)); x = self.drop(x)
        x = self.relu(self.fc7(x))   # 4096-dim
        return x

    def forward(self, x):
        return self.fc8(self.drop(self._features(x)))

    def extract(self, x):
        """Inference-time: không dropout, trả về fc7 4096-dim"""
        return self._features(x)


# ==========================================
# DATASET — GroupEmoW face crops
# ==========================================
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
                    # Bỏ qua file bbox.npy (glob *.jpg nên không lấy .npy)
                    self.samples.append((p, label))

        print(f"  {split}: {len(self.samples)} face crops")
        # Class distribution
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


train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.367, 0.411, 0.507], [1, 1, 1]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.367, 0.411, 0.507], [1, 1, 1]),
])


# ==========================================
# MAIN
# ==========================================
def main():
    print("="*70)
    print("  FINE-TUNE VGGFace — GroupEmoW ONLY")
    print(f"  Device: {CONFIG['device']}")
    print("="*70 + "\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    print("📂 Loading datasets...")
    train_ds = GroupEmoWFaceDataset(CONFIG['data_root'], 'train', train_tf)
    val_ds   = GroupEmoWFaceDataset(CONFIG['data_root'], 'val',   val_tf)

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

    # Resume từ checkpoint nếu có
    resume_path = CONFIG['ckpt_path']
    load_path   = resume_path if os.path.exists(resume_path) else CONFIG['pretrained_path']

    if os.path.exists(load_path):
        print(f"  Loading: {load_path}")
        ckpt = torch.load(load_path, map_location='cpu')
        # Xử lý nhiều format
        sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        # Chỉ load layer có shape khớp — bỏ qua fc8 nếu Oxford pretrained (2622 classes)
        model_sd = model.state_dict()
        sd = {k: v for k, v in sd.items()
              if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  Loaded {len(sd)}/{len(model_sd)} layers")
        if missing:   print(f"  Missing keys: {missing}")
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

    # Resume optimizer state nếu có
    if os.path.exists(resume_path) and 'optimizer_state_dict' in torch.load(resume_path, map_location='cpu'):
        ckpt = torch.load(resume_path, map_location='cpu')
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
                'epoch':               epoch + 1,
                'state_dict':          model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc':            best_acc,
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

    # ── Zip output ────────────────────────────────────────────────────────
    print(f"\n✅ Training done | Best ValAcc = {best_acc:.4f}")
    print("📦 Zipping checkpoint...")
    zip_path = '/kaggle/working/vggface_groupemow_best.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(CONFIG['ckpt_path'], 'vggface_groupemow_best.pth')
    print(f"✅ Saved: {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")
    print("📥 Download từ Kaggle Output tab → vggface_groupemow_best.zip")


if __name__ == '__main__':
    main()
