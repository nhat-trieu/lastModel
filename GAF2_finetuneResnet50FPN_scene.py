"""
ResNet50-FPN Scene Branch  ·  GAF 2.0  ·  Colab v3
═══════════════════════════════════════════════════════════
Thay đổi so với v2 (overfit tại Val 66.99%, epoch 17):

NGUYÊN NHÂN OVERFIT:
  - GAF 2.0 Train chỉ có 3630 ảnh (nhỏ hơn GroupEmoW 3x)
  - Augmentation v2 quá nhẹ → model memorize training set nhanh
  - weight_decay=1e-4 quá nhỏ cho dataset nhỏ

FIX:
  1. [AUGMENTATION] Mạnh tay hơn nhiều cho dataset nhỏ:
       - AutoAugment(IMAGENET policy)
       - RandomRotation(±15°)
       - RandomAffine(translate, scale, shear)
       - RandomErasing(p=0.5) — ép model học context toàn cảnh
       - RandomGrayscale(p=0.1) — thêm màu invariance

  2. [WEIGHTS] Places365 pre-trained thay ImageNet1K_v2
       - Load từ CSAILVision/places365 qua torch.hub
       - Inject vào body của resnet_fpn_backbone (strict=False)
       - Fallback tự động về ImageNet nếu hub lỗi

  3. [REGULARIZATION]
       - weight_decay: 1e-4 → 5e-4
       - Dropout spatial_proj: 0.5 → 0.3 (aggregate layer, không cần mạnh)
       - Dropout classifier: giữ 0.5
       - Label Smoothing 0.1 trong CrossEntropyLoss → tránh model quá tự tin

  4. [PATHS] Colab paths — đổi DATA_ROOT theo cấu trúc thực tế của bạn
       Nếu dùng Google Drive: bỏ comment phần drive.mount bên dưới
"""

# ─── GOOGLE DRIVE MOUNT (bỏ comment nếu dataset để trên Drive) ───────────────
# from google.colab import drive
# drive.mount('/content/drive')

import os
import glob
import time
import zipfile
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import roi_align
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Đổi DATA_ROOT theo nơi bạn upload dataset lên Colab:
#   - Unzip thẳng vào /content:  DATA_ROOT = '/content/GAF_2'
#   - Từ Google Drive:           DATA_ROOT = '/content/drive/MyDrive/GAF_2'
DATA_ROOT   = '/root/.cache/kagglehub/datasets/trieung11/gaf2000/versions/1/GAF_2'
OUTPUT_ROOT = '/content/scene_features_v3'
CKPT_PATH   = '/content/resnet50fpn_gaf2000_v3.pth'

# GAF 2.0: PascalCase folders (Positive, Neutral, Negative)
LABEL_MAP   = {'negative': 0, 'neutral': 1, 'positive': 2}
NUM_CLASSES = 3
SPLITS      = ['Train', 'Val']   # GAF 2.0 không có Test

FINETUNE_IMG_SIZE = 800
FINETUNE_EPOCHS   = 50
FINETUNE_LR       = 1e-3        # SGD lr=0.001 theo bài báo
FINETUNE_BS       = 1           # batch_size=1 theo bài báo
FINETUNE_WD       = 5e-4        # ↑ từ 1e-4 — dataset nhỏ cần regularize mạnh hơn
PATIENCE          = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 0: LOAD PLACES365 WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 0: LOAD PLACES365 WEIGHTS (FIXED BY GEMINI)
# ═══════════════════════════════════════════════════════════════════════════════
def load_places365_state_dict():
    import os
    import torch
    
    # Link trực tiếp từ server MIT - Đây là bản ResNet50 chuẩn cho PyTorch
    url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
    dest_path = '/content/resnet50_places365.pth.tar'
    
    # Nếu chạy trên Kaggle thì đổi path
    if not os.path.exists('/content'):
        dest_path = '/kaggle/working/resnet50_places365.pth.tar'

    try:
        if not os.path.exists(dest_path):
            print(f" 📍 Đang tải Places365 từ MIT server...")
            # Sử dụng wget với tùy chọn --no-check-certificate để tránh lỗi SSL
            os.system(f'wget --no-check-certificate {url} -O {dest_path}')
        
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1000000:
            raise Exception("Tải file thất bại hoặc file quá nhỏ (lỗi link).")

        print(" 📍 Đang nạp trọng số vào mô hình...")
        # Đọc file tar
        checkpoint = torch.load(dest_path, map_location='cpu')
        
        # Xử lý state_dict: MIT lưu dưới dạng {'state_dict': {...}} và có prefix 'module.'
        if 'state_dict' in checkpoint:
            raw_sd = checkpoint['state_dict']
        else:
            raw_sd = checkpoint
            
        # Làm sạch key: loại bỏ 'module.' và 'fc.' (vì FPN không dùng lớp phân loại cũ)
        clean_sd = {}
        for k, v in raw_sd.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('fc.'): # Bỏ lớp fully connected của Places365
                continue
            clean_sd[k] = v
            
        print(f" ✅ Places365 đã nạp xong ({len(clean_sd)} layers)!")
        return clean_sd

    except Exception as e:
        print(f" ⚠️ Lỗi tải trọng số: {e}")
        print(" ↩️ Tự động fallback về ImageNet (mặc định của torchvision)...")
        return None


def inject_places365(model: nn.Module, places365_sd):
    """
    resnet_fpn_backbone lưu ResNet layers tại model.fpn.body.*
    Places365 state_dict có keys dạng 'layer1.0.conv1.weight', ...
    → thêm prefix 'body.' rồi load vào model.fpn (strict=False).
    Keys FPN neck sẽ bị missing — bình thường, chúng init từ ImageNet1K_v2.
    """
    if places365_sd is None:
        return   # fallback đã xử lý ở load_places365_state_dict

    prefixed = {'body.' + k: v for k, v in places365_sd.items()}
    missing, unexpected = model.fpn.load_state_dict(prefixed, strict=False)

    # Chỉ log nếu có body keys bị missing (bất thường)
    body_missing = [k for k in missing if not k.startswith('fpn.')]
    if body_missing:
        print(f"  ⚠️  Body keys missing (kiểm tra lại): {body_missing[:5]}")
    else:
        n_fpn_missing = len([k for k in missing if k.startswith('fpn.')])
        print(f"  ✅ Places365 injected vào backbone. "
              f"FPN neck keys (random-init, bình thường): {n_fpn_missing}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class ResNet50FPN_GER(nn.Module):
    """
    Thay đổi so với v2:
      - Dropout spatial_proj: 0.5 → 0.3
        Lý do: đây là lớp aggregate 4 FPN levels (thông tin rất redundant),
        Dropout 0.3 đủ regularize mà không mất spatial signal.
      - Dropout classifier: giữ 0.5 (dễ overfit nhất, dataset nhỏ)
      - Backbone vẫn init bằng ImageNet1K_v2 trước để FPN neck được init đúng,
        sau đó inject Places365 vào phần body (xem build_model).
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.fpn = resnet_fpn_backbone(
            backbone_name='resnet50',
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
            returned_layers=[1, 2, 3, 4],
            extra_blocks=None,
            trainable_layers=5
        )

        # 4 FPN levels × 256ch × 7×7 = 50176 → project → 1024
        self.spatial_proj = nn.Sequential(
            nn.Linear(50176, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)          # ← 0.5 → 0.3
        )

        # Classifier RAN/CARAN — không BN vì batch_size=1
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def freeze_backbone(self):
        for p in self.fpn.parameters():
            p.requires_grad = False
        print("  🔒 Backbone frozen")

    def unfreeze_backbone(self):
        for p in self.fpn.parameters():
            p.requires_grad = True
        print("  🔓 Backbone unfrozen — end-to-end training")

    def extract_scene_feat(self, images):
        B, C, H, W = images.shape
        fpn_feats = self.fpn(images)

        # RoI = toàn bộ ảnh
        rois = torch.zeros(B, 5, device=images.device)
        rois[:, 0] = torch.arange(B, dtype=torch.float32, device=images.device)
        rois[:, 1] = 0.0
        rois[:, 2] = 0.0
        rois[:, 3] = float(W)
        rois[:, 4] = float(H)

        strides    = [4, 8, 16, 32]
        level_keys = ['0', '1', '2', '3']
        pooled_levels = []
        for key, stride in zip(level_keys, strides):
            feat   = fpn_feats[key]
            pooled = roi_align(feat, rois, output_size=(7, 7),
                               spatial_scale=1.0 / stride, aligned=True)
            pooled_levels.append(pooled.flatten(1))   # (B, 12544)

        multi_scale = torch.cat(pooled_levels, dim=1)   # (B, 50176)
        return self.spatial_proj(multi_scale)            # (B, 1024)

    def forward(self, images):
        return self.classifier(self.extract_scene_feat(images))


def build_model(num_classes=NUM_CLASSES, use_places365=True):
    """Khởi tạo model mới, inject Places365 nếu có thể."""
    model = ResNet50FPN_GER(num_classes).to(DEVICE)
    if use_places365:
        p365_sd = load_places365_state_dict()
        inject_places365(model, p365_sd)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: DATASET & AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
class ResizeDirect:
    """Resize thẳng 800×800, không pad — theo bài báo."""
    def __init__(self, size=800):
        self.size = size

    def __call__(self, img):
        return TF.resize(img, (self.size, self.size))


class GAF2000SceneDataset(Dataset):
    """
    GAF 2.0 folder structure:
        GAF_2/Train/Positive/*.jpg
        GAF_2/Train/Neutral/*.jpg
        GAF_2/Train/Negative/*.jpg
        GAF_2/Val/...   (tương tự, không có Test)
    LABEL_MAP dùng .lower() để handle cả PascalCase lẫn lowercase folder names.
    """
    def __init__(self, split, transform):
        self.transform = transform
        self.samples   = []
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_dir):
            print(f"  ⚠️  Không tìm thấy: {split_dir}"); return

        for class_name in os.listdir(split_dir):
            label = LABEL_MAP.get(class_name.lower())
            if label is None:
                continue
            class_dir = os.path.join(split_dir, class_name)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for p in glob.glob(os.path.join(class_dir, '**', ext), recursive=True):
                    self.samples.append((p, label))

        # In phân bố class để detect imbalance
        from collections import Counter
        dist = Counter(lbl for _, lbl in self.samples)
        inv  = {v: k for k, v in LABEL_MAP.items()}
        dist_str = ' | '.join(f"{inv[k]}={dist[k]}" for k in sorted(dist))
        print(f"  {split.upper()}: {len(self.samples)} imgs  [{dist_str}]")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, FINETUNE_IMG_SIZE, FINETUNE_IMG_SIZE)
        return img, label, img_path


# ─── AUGMENTATION ──────────────────────────────────────────────────────────────
#
# Strategy cho dataset nhỏ (3630 train):
#
#   1. AutoAugment(IMAGENET) — tổ hợp augmentation học được từ ImageNet
#      Places365 policy chưa stable trên mọi torchvision version,
#      IMAGENET policy đã được test rộng rãi và hiệu quả với scene data.
#
#   2. RandomRotation(15°) + RandomAffine — ép spatial invariance
#      Rotation ±15° đủ để phá pattern, không làm lệch quá nhiều nội dung ảnh
#
#   3. RandomGrayscale(p=0.1) — thêm color invariance nhẹ
#      10% ảnh train sẽ được chuyển grayscale, giúp model không phụ thuộc màu
#
#   4. RandomErasing(p=0.5, scale=(0.02, 0.20)) — quan trọng nhất
#      Xóa ngẫu nhiên 2-20% diện tích ảnh bằng noise
#      Ép model phải dựa vào context toàn cảnh thay vì local patch
#      ĐẶT SAU ToTensor() + Normalize() vì RandomErasing nhận tensor
#
# eval_transform: không augment, chỉ resize + normalize

train_transform = T.Compose([
    ResizeDirect(FINETUNE_IMG_SIZE),

    # Geometric
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),   # dịch ±8%
        scale=(0.85, 1.15),       # zoom ±15%
        shear=8,
    ),

    # AutoAugment — bao gồm nhiều phép biến đổi màu và geometric
    T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),

    # Màu sắc nhẹ (AutoAugment đã cover, giữ nhỏ tránh xung đột)
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),

    # Grayscale nhẹ — color invariance
    T.RandomGrayscale(p=0.1),

    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # RandomErasing — PHẢI đặt sau Normalize, nhận tensor
    # scale rộng hơn v2 (0.02-0.20) vì dataset nhỏ cần regularize mạnh hơn
    T.RandomErasing(p=0.5, scale=(0.02, 0.20), ratio=(0.3, 3.3), value='random'),
])

eval_transform = T.Compose([
    ResizeDirect(FINETUNE_IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 3: FINE-TUNE
# ═══════════════════════════════════════════════════════════════════════════════
def _save_checkpoint(path, model, optimizer, scheduler,
                     epoch, best_acc, backbone_unfrozen, note=''):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc':             best_acc,
        'backbone_unfrozen':    backbone_unfrozen,
    }, path)
    if note:
        print(note)


def finetune(use_places365=True):
    print("=" * 65)
    print("  FINE-TUNE ResNet50-FPN  ·  GAF 2.0  ·  Colab v3")
    print(f"  Device={DEVICE} | Epochs={FINETUNE_EPOCHS} | BS={FINETUNE_BS}")
    print(f"  WD={FINETUNE_WD} | LabelSmoothing=0.1 | Places365={use_places365}")
    print("=" * 65 + "\n")

    train_ds = GAF2000SceneDataset('Train', train_transform)
    val_ds   = GAF2000SceneDataset('Val',   eval_transform)

    # num_workers=2 trên Colab — tăng lên 4 nếu RAM đủ
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BS, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=FINETUNE_BS, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Label Smoothing 0.1: phạt model khi quá tự tin → chống overfit trực tiếp
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch       = 1
    best_acc          = 0.0
    no_improve        = 0
    backbone_unfrozen = False

    # ── RESUME hay train mới ───────────────────────────────────────────────
    if os.path.exists(CKPT_PATH):
        print(f"🔄 Resume từ: {CKPT_PATH}")
        model = ResNet50FPN_GER(NUM_CLASSES).to(DEVICE)
        ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        best_acc          = ckpt['best_acc']
        start_epoch       = ckpt['epoch'] + 1
        backbone_unfrozen = ckpt.get('backbone_unfrozen', False)

        if backbone_unfrozen:
            model.unfreeze_backbone()
            optimizer = optim.SGD([
                {'params': model.fpn.parameters(),          'lr': FINETUNE_LR * 0.1},
                {'params': model.spatial_proj.parameters(), 'lr': FINETUNE_LR},
                {'params': model.classifier.parameters(),   'lr': FINETUNE_LR},
            ], momentum=0.9, weight_decay=FINETUNE_WD)
        else:
            model.freeze_backbone()
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=FINETUNE_LR, momentum=0.9, weight_decay=FINETUNE_WD
            )
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, FINETUNE_EPOCHS - start_epoch + 1), eta_min=1e-6
        )
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"✅ Resume ep {start_epoch} | Best={best_acc:.4f} | "
              f"Backbone unfrozen={backbone_unfrozen}\n")
    else:
        print("🆕 Train từ đầu.")
        model = build_model(NUM_CLASSES, use_places365=use_places365)
        model.freeze_backbone()
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=FINETUNE_LR, momentum=0.9, weight_decay=FINETUNE_WD
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6
        )

    print("\n🎯 Bắt đầu training...\n")

    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        t0 = time.time()

        # Unfreeze backbone từ epoch 2
        if epoch == 2 and not backbone_unfrozen:
            model.unfreeze_backbone()
            backbone_unfrozen = True
            optimizer = optim.SGD([
                {'params': model.fpn.parameters(),          'lr': FINETUNE_LR * 0.1},
                {'params': model.spatial_proj.parameters(), 'lr': FINETUNE_LR},
                {'params': model.classifier.parameters(),   'lr': FINETUNE_LR},
            ], momentum=0.9, weight_decay=FINETUNE_WD)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=FINETUNE_EPOCHS - 1, eta_min=1e-6
            )
            print(f"  🔓 Epoch {epoch}: Unfreeze backbone — end-to-end fine-tune")

        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{FINETUNE_EPOCHS} [Train]")

        for imgs, labels, _ in pbar:
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

        # ── VALIDATE ───────────────────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels, _ in tqdm(val_loader,
                                        desc=f"Ep {epoch:03d}/{FINETUNE_EPOCHS} [Val]  "):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total   += len(labels)

        if val_total == 0:
            print(f"  ⚠️  val_total=0 — kiểm tra: {os.path.join(DATA_ROOT, 'Val')}")
            break

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        elapsed   = time.time() - t0
        scheduler.step()

        print(f"  Ep {epoch:03d} [{elapsed:.0f}s] "
              f"TrainLoss={train_loss/train_total:.4f} | "
              f"TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            _save_checkpoint(
                CKPT_PATH, model, optimizer, scheduler,
                epoch, best_acc, backbone_unfrozen,
                note=f"  🔥 Best! ValAcc={val_acc:.4f} → {CKPT_PATH}"
            )
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n🛑 Early stop tại epoch {epoch}"); break

    print(f"\n✅ Finetune xong. Best ValAcc = {best_acc:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 4: EXTRACT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def get_output_npy_path(img_path):
    if 'GAF_2/' in img_path:
        rel = img_path.split('GAF_2/')[-1]
    else:
        rel = img_path.split(DATA_ROOT)[-1].lstrip('/')
    stem = os.path.splitext(rel)[0]
    return os.path.join(OUTPUT_ROOT, 'scenes', stem + '.npy')


def extract_features():
    print("=" * 65)
    print("  EXTRACT SCENE FEATURES  ·  GAF 2.0  ·  v3")
    print("=" * 65 + "\n")

    # Ưu tiên best checkpoint, fallback emergency
    load_path = CKPT_PATH if os.path.exists(CKPT_PATH) else None
    if load_path is None:
        print("⚠️ Không tìm thấy checkpoint. Bỏ qua."); return

    print(f"📦 Loading: {load_path}")
    model = ResNet50FPN_GER(NUM_CLASSES).to(DEVICE)
    ckpt  = torch.load(load_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    print("✅ Model loaded\n")

    total_ok = total_skip = total_err = 0
    for split in SPLITS:   # chỉ Train + Val
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_dir): continue

        all_imgs = []
        for class_name in os.listdir(split_dir):
            if LABEL_MAP.get(class_name.lower()) is None: continue
            class_dir = os.path.join(split_dir, class_name)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                all_imgs.extend(glob.glob(os.path.join(class_dir, '**', ext),
                                          recursive=True))

        for img_path in tqdm(all_imgs, desc=f"  Extracting {split}"):
            out_path = get_output_npy_path(img_path)
            if os.path.exists(out_path):
                total_skip += 1; continue
            try:
                img    = Image.open(img_path).convert('RGB')
                tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = model.extract_scene_feat(tensor)   # (1, 1024)
                feat_np = feat.squeeze(0).cpu().numpy().astype(np.float32)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, feat_np)
                total_ok += 1
            except Exception as e:
                print(f"  ⚠️ Lỗi {os.path.basename(img_path)}: {e}")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, np.zeros(1024, dtype=np.float32))
                total_err += 1

        print(f"  ✅ {split} | ok={total_ok} skip={total_skip} err={total_err}\n")

    print(f"\n📊 OK={total_ok} | Skip={total_skip} | Err={total_err}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 5: NÉN VÀ TẢI VỀ
# ═══════════════════════════════════════════════════════════════════════════════
def zip_and_download():
    zip_path  = '/content/scene_features_gaf2000_v3.zip'
    ckpt_zip  = '/content/resnet50fpn_gaf2000_v3_ckpt.zip'
    print(f"\n📦 Nén {OUTPUT_ROOT} → {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_ROOT):
            for f in files:
                abs_p = os.path.join(root, f)
                zf.write(abs_p, os.path.relpath(abs_p, os.path.dirname(OUTPUT_ROOT)))
    print(f"✅ Features: {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")
    with zipfile.ZipFile(ckpt_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(CKPT_PATH):
            zf.write(CKPT_PATH, os.path.basename(CKPT_PATH))
    print(f"✅ Checkpoint: {ckpt_zip}")


# ═══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════
def sanity_check_one_batch():
    print("\n" + "="*65)
    print("  🚀 SANITY CHECK: OVERFIT 1 BATCH")
    print("="*65)
    train_ds = GAF2000SceneDataset('Train', train_transform)
    if len(train_ds) == 0:
        print("❌ Không tìm thấy dữ liệu!"); return

    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    model  = ResNet50FPN_GER(NUM_CLASSES).to(DEVICE)
    model.unfreeze_backbone()
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    imgs, labels, _ = next(iter(loader))
    imgs, labels    = imgs.to(DEVICE), labels.to(DEVICE)
    print(f"  {len(labels)} ảnh | pipeline: 50176 → spatial_proj(0.3) → 1024 → classifier(0.5×3)")

    model.eval()   # tắt Dropout để test gradient thuần túy
    for i in range(100):
        opt.zero_grad()
        logits = model(imgs)
        loss   = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        acc = (logits.argmax(1) == labels).sum().item() / len(labels)
        if i % 10 == 0:
            print(f"  Iter {i:02d} | Loss={loss.item():.4f} | Acc={acc:.4f}")
        if acc == 1.0 and loss.item() < 0.05:
            print("\n  ✅ SANITY CHECK PASS"); return
    print("\n  ❌ SANITY CHECK FAIL — kiểm tra lại kiến trúc.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Đặt use_places365=False nếu Colab không load được torch.hub
    finetune(use_places365=True)
    extract_features()
    zip_and_download()


if __name__ == '__main__':
    main()