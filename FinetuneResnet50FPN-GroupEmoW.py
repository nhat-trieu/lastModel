
"""
finetune_and_extract_scene.py
─────────────────────────────────────────────────────────────────────────────
Bước 1: Fine-tune ResNet50-FPN trên GroupEmoW (đúng theo paper)
Bước 2: Extract scene features [2048] cho toàn bộ dataset
Bước 3: Nén thành .zip để tải về
!pip install torch torchvision facenet-pytorch timm --upgrade
─────────────────────────────────────────────────────────────────────────────
"""



import os
import glob
import time
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import roi_align
from torch.utils.data import Dataset, DataLoader

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_ROOT   = '/kaggle/input/groupemowfull/GroupEmoW'
OUTPUT_ROOT = '/kaggle/working/scene_features_final'
CKPT_PATH   = '/kaggle/working/resnet50fpn_groupemow.pth'

LABEL_MAP   = {'negative': 0, 'neutral': 1, 'positive': 2}
NUM_CLASSES = 3
SPLITS      = ['train', 'val', 'test']

FINETUNE_IMG_SIZE = 800
FINETUNE_EPOCHS   = 50
FINETUNE_LR       = 1e-3
FINETUNE_BS       = 8 
FINETUNE_WD       = 1e-4
PATIENCE          = 10

EXTRACT_BS  = 8
ROI_SIZE    = 7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: MODEL (GIỮ NGUYÊN 100%)
# ═══════════════════════════════════════════════════════════════════════════════
class ResNet50FPN_GER(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fpn = resnet_fpn_backbone(
            backbone_name='resnet50',
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
            returned_layers=[1, 2, 3, 4],
            extra_blocks=None,
            trainable_layers=5
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.scene_proj = nn.Sequential(
            nn.Linear(256 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(2048, num_classes)

    def freeze_backbone(self):
        for p in self.fpn.parameters(): p.requires_grad = False
        print("  🔒 Backbone frozen")

    def unfreeze_backbone(self):
        for p in self.fpn.parameters(): p.requires_grad = True
        print("  🔓 Backbone unfrozen — training end-to-end")

    def extract_scene_feat(self, images):
        B, C, H, W = images.shape
        fpn_feats = self.fpn(images)

        rois = torch.zeros(B, 5, device=images.device)
        rois[:, 0] = torch.arange(B, dtype=torch.float32, device=images.device)
        rois[:, 3] = float(W)
        rois[:, 4] = float(H)

        strides    = [4, 8, 16, 32]
        level_keys = ['0', '1', '2', '3']
        pooled_levels = []

        for key, stride in zip(level_keys, strides):
            feat = fpn_feats[key]
            pooled = roi_align(feat, rois, (ROI_SIZE, ROI_SIZE), 1.0 / stride, True)
            pooled = self.global_pool(pooled).flatten(1)
            pooled_levels.append(pooled)

        multi_scale = torch.cat(pooled_levels, dim=1)
        return self.scene_proj(multi_scale)

    def forward(self, images):
        scene_feat = self.extract_scene_feat(images)
        return self.classifier(scene_feat)

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: DATASET (GIỮ NGUYÊN 100%)
# ═══════════════════════════════════════════════════════════════════════════════
class GroupEmoWSceneDataset(Dataset):
    def __init__(self, split, transform):
        self.transform = transform
        self.samples   = []
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_dir): return
        for class_name in os.listdir(split_dir):
            label = LABEL_MAP.get(class_name.lower(), None)
            if label is None: continue
            class_dir = os.path.join(split_dir, class_name)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in glob.glob(os.path.join(class_dir, ext)):
                    self.samples.append((img_path, label))
        print(f"  {split.upper()}: {len(self.samples)} images")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, FINETUNE_IMG_SIZE, FINETUNE_IMG_SIZE)
        return img, label, img_path

train_transform = T.Compose([
    T.Resize((FINETUNE_IMG_SIZE, FINETUNE_IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = T.Compose([
    T.Resize((FINETUNE_IMG_SIZE, FINETUNE_IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 3: FINE-TUNE (Đã thêm Hẹn Giờ & Resume Training)
# ═══════════════════════════════════════════════════════════════════════════════
def finetune():
    print("=" * 65)
    print("  BƯỚC 1: FINE-TUNE ResNet50-FPN trên GroupEmoW")
    print(f"  Device: {DEVICE} | Epochs: {FINETUNE_EPOCHS} | BS: {FINETUNE_BS}")
    print("=" * 65 + "\n")

    train_ds = GroupEmoWSceneDataset('train', train_transform)
    val_ds   = GroupEmoWSceneDataset('val',   eval_transform)
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BS, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=FINETUNE_BS, shuffle=False, num_workers=2, pin_memory=True)

    model = ResNet50FPN_GER(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_acc    = 0.0
    no_improve  = 0

    # KIỂM TRA FILE ĐỂ RESUME MÀ KHÔNG MẤT TRỌNG SỐ
    # Sửa đường dẫn này nếu bạn tạo Dataset chứa file .pth của lần chạy trước
    LOAD_CKPT_PATH = CKPT_PATH 

    if os.path.exists(LOAD_CKPT_PATH):
        print(f"🔄 Đang khôi phục checkpoint từ: {LOAD_CKPT_PATH}")
        checkpoint = torch.load(LOAD_CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch'] + 1
        
        if start_epoch > 2:
            model.unfreeze_backbone()
            optimizer = optim.SGD([
                {'params': model.fpn.parameters(),         'lr': FINETUNE_LR * 0.1},
                {'params': model.scene_proj.parameters(),  'lr': FINETUNE_LR},
                {'params': model.classifier.parameters(),  'lr': FINETUNE_LR},
            ], momentum=0.9, weight_decay=FINETUNE_WD)
        else:
            model.freeze_backbone()
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR, momentum=0.9, weight_decay=FINETUNE_WD)
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, FINETUNE_EPOCHS - start_epoch + 1), eta_min=1e-6)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Khôi phục thành công! Chạy tiếp từ Epoch {start_epoch} (Best Acc cũ: {best_acc:.4f})\n")
    else:
        print("🆕 Không tìm thấy file cũ. Bắt đầu huấn luyện từ số 0.")
        model.freeze_backbone()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR, momentum=0.9, weight_decay=FINETUNE_WD)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)

    print("\n🎯 Training...\n")
    
    # ĐỒNG HỒ HẸN GIỜ: 11.25 tiếng (11 tiếng 15 phút) để Kaggle không xóa file
    KAGGLE_TIME_LIMIT = 11.25 * 3600
    GLOBAL_START_TIME = time.time()

    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        t0 = time.time()

        if epoch == 2 and not os.path.exists(LOAD_CKPT_PATH):
            model.unfreeze_backbone()
            optimizer = optim.SGD([
                {'params': model.fpn.parameters(),         'lr': FINETUNE_LR * 0.1},
                {'params': model.scene_proj.parameters(),  'lr': FINETUNE_LR},
                {'params': model.classifier.parameters(),  'lr': FINETUNE_LR},
            ], momentum=0.9, weight_decay=FINETUNE_WD)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS - 1, eta_min=1e-6)

        model.train()
        train_loss = train_correct = train_total = 0
        for imgs, labels, _ in tqdm(train_loader, desc=f"Ep {epoch:03d}/{FINETUNE_EPOCHS} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss    += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += len(labels)

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels, _ in tqdm(val_loader, desc=f"Ep {epoch:03d}/{FINETUNE_EPOCHS} [Val]  "):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += len(labels)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        scheduler.step()

        print(f"  Ep {epoch:03d} [{time.time() - t0:.0f}s] TrainLoss={train_loss/train_total:.4f} | TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f}")

        # LƯU FILE THEO DẠNG DICTIONARY
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc
            }, CKPT_PATH)
            print(f"  🔥 Best! ValAcc={val_acc:.4f} → saved to {CKPT_PATH}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n🛑 Early stop tại epoch {epoch}")
                break

        # CHỐT CHẶN: NẾU SẮP ĐẾN 12 TIẾNG THÌ DỪNG LẠI NGAY LẬP TỨC
        if (time.time() - GLOBAL_START_TIME) > KAGGLE_TIME_LIMIT:
            print("\n⏳ Sắp hết 12 tiếng của Kaggle! Chủ động dừng để bảo toàn tính mạng...")
            break

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 4: EXTRACT FEATURES (SỬA LỖI ĐỌC FILE CKPT)
# ═══════════════════════════════════════════════════════════════════════════════
def get_output_npy_path(img_path):
    if 'GroupEmoW/' in img_path: rel = img_path.split('GroupEmoW/')[-1]
    else: rel = img_path.split(DATA_ROOT)[-1].lstrip('/')
    stem = os.path.splitext(rel)[0]
    return os.path.join(OUTPUT_ROOT, 'scenes', stem + '.npy')

def extract_features():
    print("=" * 65)
    print("  BƯỚC 2: EXTRACT SCENE FEATURES")
    print("=" * 65 + "\n")

    if not os.path.exists(CKPT_PATH):
        print("⚠️ Không tìm thấy model để Extract. Bỏ qua.")
        return

    print(f"📦 Loading checkpoint: {CKPT_PATH}")
    model = ResNet50FPN_GER(NUM_CLASSES).to(DEVICE)
    
    # [FIX LỖI CHÍ MẠNG Ở ĐÂY]: Giải mã cái Rương (Dictionary) để lấy đúng model_state_dict
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # Fallback lỡ bạn dùng file cũ
        
    model.eval()
    print("✅ Model loaded\n")

    total_ok = total_skip = total_err = 0
    for split in SPLITS:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_dir): continue
        all_imgs = []
        for class_name in os.listdir(split_dir):
            if LABEL_MAP.get(class_name.lower()) is None: continue
            class_dir = os.path.join(split_dir, class_name)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                all_imgs.extend(glob.glob(os.path.join(class_dir, ext)))

        buffer_imgs = []; buffer_paths = []
        def flush_buffer():
            nonlocal total_ok, total_err
            if not buffer_imgs: return
            try:
                tensors = torch.stack([eval_transform(img) for img in buffer_imgs]).to(DEVICE)
                with torch.no_grad():
                    feats = model.extract_scene_feat(tensors)
                for feat, out_path in zip(feats.cpu().numpy(), buffer_paths):
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, feat.astype(np.float32))
                    total_ok += 1
            except Exception as e:
                total_err += len(buffer_imgs)

        for img_path in tqdm(all_imgs, desc=f"  Extracting {split}"):
            out_path = get_output_npy_path(img_path)
            if os.path.exists(out_path):
                total_skip += 1
                continue
            try:
                img = Image.open(img_path).convert('RGB')
                buffer_imgs.append(img)
                buffer_paths.append(out_path)
            except Exception:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, np.zeros(2048, dtype=np.float32))
                total_err += 1
                continue
            if len(buffer_imgs) >= EXTRACT_BS:
                flush_buffer()
                buffer_imgs.clear(); buffer_paths.clear()

        flush_buffer()
        print(f"  ✅ Done {split}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 5: NÉN VÀ TẢI VỀ (GIỮ NGUYÊN 100%)
# ═══════════════════════════════════════════════════════════════════════════════
def zip_and_download():
    zip_path = '/kaggle/working/scene_features_final.zip'
    print(f"\n📦 Đang nén {OUTPUT_ROOT} → {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_ROOT):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, os.path.dirname(OUTPUT_ROOT))
                zf.write(abs_path, rel_path)
    print(f"✅ {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")
    ckpt_zip = '/kaggle/working/resnet50fpn_groupemow_ckpt.zip'
    with zipfile.ZipFile(ckpt_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(CKPT_PATH): zf.write(CKPT_PATH, os.path.basename(CKPT_PATH))
    print(f"✅ Checkpoint: {ckpt_zip}")

def main():
    finetune()         
    extract_features()
    zip_and_download()

if __name__ == '__main__':
    main()