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

# FIX #1: Resize thẳng 800×800 theo bài báo, không pad
class ResizeDirect:
    def __init__(self, target_size=800):
        self.target_size = target_size

    def __call__(self, img):
        return TF.resize(img, (self.target_size, self.target_size))

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_ROOT   = '/kaggle/input/datasets/trieung11/gaf2000/GAF_2'
OUTPUT_ROOT = '/kaggle/working/scene_features_final'
CKPT_PATH   = '/kaggle/working/resnet50fpn_gaf2000_v2.pth'

LABEL_MAP   = {'negative': 0, 'neutral': 1, 'positive': 2}
NUM_CLASSES = 3
SPLITS      = ['Train', 'Val']   # GAF_2: Train và Val (không phải Validation)

FINETUNE_IMG_SIZE = 800
FINETUNE_EPOCHS   = 50
FINETUNE_LR       = 1e-3       # SGD lr=0.001 theo bài báo
FINETUNE_BS       = 1          # FIX #2: batch_size=1 theo bài báo
FINETUNE_WD       = 1e-4
PATIENCE          = 10

# ── KAGGLE TIME GUARD ─────────────────────────────────────────────────────────
KAGGLE_TIME_LIMIT   = 11.0 * 3600      # dừng sau 11h, buffer ~1h để lưu file
PERIODIC_SAVE_EVERY = 500              # emergency save mỗi 500 ảnh trong epoch
EMERGENCY_CKPT_PATH = '/kaggle/working/resnet50fpn_gaf2000_emergency.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class ResNet50FPN_GER(nn.Module):
    """
    Fix so với code GAF2 cũ:
    - Bỏ global_pool + scene_proj GAP-style
    - RoIAlign output_size=(7,7) giữ spatial layout (output_size=(1,1) = GAP ngụy trang)
    - 4 levels × 256ch × 7×7 = 50176 → spatial_proj Linear(50176→1024)
    - Classifier RAN/CARAN: 1024→1024→1024→128→3
    - Bỏ BatchNorm1d vì batch_size=1 làm BN crash
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

        # 4 FPN levels × 256 channels × 7×7 spatial = 50176
        self.spatial_proj = nn.Sequential(
            nn.Linear(50176, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
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
        print("  🔓 Backbone unfrozen — training end-to-end")

    def extract_scene_feat(self, images):
        B, C, H, W = images.shape
        fpn_feats = self.fpn(images)

        # RoI = toàn bộ bức ảnh
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
            feat = fpn_feats[key]
            # Giữ 7×7 spatial grid — KHÔNG ép về (1,1) vì đó tương đương GAP
            pooled = roi_align(feat, rois, output_size=(7, 7),
                               spatial_scale=1.0 / stride,
                               aligned=True)
            pooled = pooled.flatten(1)   # (B, 256×7×7 = 12544)
            pooled_levels.append(pooled)

        # Cat 4 levels → (B, 50176) → project → (B, 1024)
        multi_scale = torch.cat(pooled_levels, dim=1)
        return self.spatial_proj(multi_scale)

    def forward(self, images):
        scene_feat = self.extract_scene_feat(images)
        return self.classifier(scene_feat)


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class GAF2000SceneDataset(Dataset):
    """
    Cấu trúc GAF_2:
        Train/Positive/*.jpg
        Train/Neutral/*.jpg
        Train/Negative/*.jpg
        Val/... (tương tự)
    Dùng glob đệ quy (**) để bắt cả dạng thư mục con lồng nhau nếu có.
    """
    def __init__(self, split, transform):
        self.transform = transform
        self.samples   = []
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_dir):
            print(f"  ⚠️  Không tìm thấy thư mục: {split_dir}")
            return

        for class_name in os.listdir(split_dir):
            label = LABEL_MAP.get(class_name.lower(), None)
            if label is None:
                continue
            class_dir = os.path.join(split_dir, class_name)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in glob.glob(os.path.join(class_dir, '**', ext), recursive=True):
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


# FIX #1: ResizeDirect thay ResizeAndPad
train_transform = T.Compose([
    ResizeDirect(FINETUNE_IMG_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


def finetune():
    print("=" * 65)
    print("  BƯỚC 1: FINE-TUNE ResNet50-FPN trên GAF_2  [v2]")
    print(f"  Device: {DEVICE} | Epochs: {FINETUNE_EPOCHS} | BS: {FINETUNE_BS}")
    print("=" * 65 + "\n")

    train_ds = GAF2000SceneDataset('Train', train_transform)
    val_ds   = GAF2000SceneDataset('Val',   eval_transform)

    # FIX #2: num_workers=0 với BS=1 để tránh lỗi multiprocessing trên Kaggle
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BS, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=FINETUNE_BS, shuffle=False,
                              num_workers=0, pin_memory=True)

    model     = ResNet50FPN_GER(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    start_epoch       = 1
    best_acc          = 0.0
    no_improve        = 0
    backbone_unfrozen = False  # FIX #5: dùng flag thay vì điều kiện file tồn tại

    LOAD_CKPT_PATH = CKPT_PATH

    if os.path.exists(LOAD_CKPT_PATH):
        print(f"🔄 Đang khôi phục checkpoint từ: {LOAD_CKPT_PATH}")
        checkpoint        = torch.load(LOAD_CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc          = checkpoint['best_acc']
        start_epoch       = checkpoint['epoch'] + 1
        backbone_unfrozen = checkpoint.get('backbone_unfrozen', False)

        # FIX #5: khôi phục đúng trạng thái freeze từ flag
        if backbone_unfrozen:
            model.unfreeze_backbone()
            optimizer = optim.SGD([
                {'params': model.fpn.parameters(),          'lr': FINETUNE_LR * 0.1},
                {'params': model.spatial_proj.parameters(), 'lr': FINETUNE_LR},  # không bỏ quên
                {'params': model.classifier.parameters(),   'lr': FINETUNE_LR},
            ], momentum=0.9, weight_decay=FINETUNE_WD)
        else:
            model.freeze_backbone()
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=FINETUNE_LR, momentum=0.9, weight_decay=FINETUNE_WD
            )

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, FINETUNE_EPOCHS - start_epoch + 1), eta_min=1e-6
        )
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Khôi phục thành công! Epoch {start_epoch} | "
              f"Best Acc cũ: {best_acc:.4f} | Backbone unfrozen: {backbone_unfrozen}\n")
    else:
        print("🆕 Bắt đầu huấn luyện từ số 0.")
        model.freeze_backbone()
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=FINETUNE_LR, momentum=0.9, weight_decay=FINETUNE_WD
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6
        )

    print("\n🎯 Training...\n")
    print(f"  ⏱️  Time guard: dừng sau {KAGGLE_TIME_LIMIT/3600:.2f}h | "
          f"Emergency save mỗi {PERIODIC_SAVE_EVERY} ảnh\n")

    GLOBAL_START_TIME = time.time()
    time_exhausted    = False

    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        t0 = time.time()

        # FIX #5: unfreeze sau epoch 1, điều kiện rõ ràng qua flag
        if epoch == 2 and not backbone_unfrozen:
            model.unfreeze_backbone()
            backbone_unfrozen = True
            optimizer = optim.SGD([
                {'params': model.fpn.parameters(),          'lr': FINETUNE_LR * 0.1},
                {'params': model.spatial_proj.parameters(), 'lr': FINETUNE_LR},  # không bỏ quên
                {'params': model.classifier.parameters(),   'lr': FINETUNE_LR},
            ], momentum=0.9, weight_decay=FINETUNE_WD)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=FINETUNE_EPOCHS - 1, eta_min=1e-6
            )
            print(f"  🔓 Epoch {epoch}: Unfreeze backbone — fine-tune end-to-end")

        # ── TRAIN ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{FINETUNE_EPOCHS} [Train]")
        for step, (imgs, labels, _) in enumerate(pbar):
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

            # ── PERIODIC EMERGENCY SAVE ─────────────────────────────────────
            if (step + 1) % PERIODIC_SAVE_EVERY == 0:
                elapsed_h = (time.time() - GLOBAL_START_TIME) / 3600
                _save_checkpoint(
                    EMERGENCY_CKPT_PATH, model, optimizer, scheduler,
                    epoch, best_acc, backbone_unfrozen,
                    note=f"  💾 Emergency save @ ep{epoch} step{step+1} "
                         f"[{elapsed_h:.2f}h] → {EMERGENCY_CKPT_PATH}"
                )

            # ── INTRA-EPOCH TIME CHECK ───────────────────────────────────────
            if (time.time() - GLOBAL_START_TIME) > KAGGLE_TIME_LIMIT:
                elapsed_h = (time.time() - GLOBAL_START_TIME) / 3600
                print(f"\n⏳ [{elapsed_h:.2f}h] Sắp hết giờ Kaggle! "
                      f"Lưu emergency và dừng giữa epoch {epoch}...")
                _save_checkpoint(
                    EMERGENCY_CKPT_PATH, model, optimizer, scheduler,
                    epoch, best_acc, backbone_unfrozen,
                    note=f"  💾 Emergency checkpoint → {EMERGENCY_CKPT_PATH}"
                )
                time_exhausted = True
                break

        if time_exhausted:
            break

        # ── VALIDATE ───────────────────────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels, _ in tqdm(val_loader, desc=f"Ep {epoch:03d}/{FINETUNE_EPOCHS} [Val]  "):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += len(labels)

        train_acc       = train_correct / train_total if train_total > 0 else 0.0
        val_acc         = val_correct   / val_total   if val_total   > 0 else 0.0
        if val_total == 0:
            print(f"  ⚠️  val_total=0 — kiểm tra lại đường dẫn Val: {os.path.join(DATA_ROOT, 'Val')}")
            break
        elapsed         = time.time() - t0
        elapsed_total_h = (time.time() - GLOBAL_START_TIME) / 3600
        scheduler.step()

        print(f"  Ep {epoch:03d} [{elapsed:.0f}s | total {elapsed_total_h:.2f}h] "
              f"TrainLoss={train_loss/train_total:.4f} | "
              f"TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            _save_checkpoint(
                CKPT_PATH, model, optimizer, scheduler,
                epoch, best_acc, backbone_unfrozen,
                note=f"  🔥 Best! ValAcc={val_acc:.4f} → saved to {CKPT_PATH}"
            )
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n🛑 Early stop tại epoch {epoch}")
                break

        if (time.time() - GLOBAL_START_TIME) > KAGGLE_TIME_LIMIT:
            elapsed_h = (time.time() - GLOBAL_START_TIME) / 3600
            print(f"\n⏳ [{elapsed_h:.2f}h] Hết giờ sau val. Dừng.")
            break

    if time_exhausted:
        print(f"\n📋 Resume lần sau: load từ {EMERGENCY_CKPT_PATH} "
              f"(hoặc {CKPT_PATH} nếu muốn từ best checkpoint)")
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
    print("  BƯỚC 2: EXTRACT SCENE FEATURES  [v2]")
    print("=" * 65 + "\n")

    if not os.path.exists(CKPT_PATH):
        print("⚠️ Không tìm thấy model để Extract. Bỏ qua.")
        return

    print(f"📦 Loading checkpoint: {CKPT_PATH}")
    model      = ResNet50FPN_GER(NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Model loaded\n")

    total_ok = total_skip = total_err = 0

    for split in SPLITS:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_dir):
            continue

        all_imgs = []
        for class_name in os.listdir(split_dir):
            if LABEL_MAP.get(class_name.lower()) is None:
                continue
            class_dir = os.path.join(split_dir, class_name)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                all_imgs.extend(glob.glob(os.path.join(class_dir, '**', ext), recursive=True))

        # Extract từng ảnh một (batch=1) đồng nhất với training
        for img_path in tqdm(all_imgs, desc=f"  Extracting {split}"):
            out_path = get_output_npy_path(img_path)
            if os.path.exists(out_path):
                total_skip += 1
                continue
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
                print(f"  ⚠️ Lỗi extract {os.path.basename(img_path)}: {e}")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, np.zeros(1024, dtype=np.float32))
                total_err += 1

        print(f"  ✅ Done {split} | ok={total_ok} skip={total_skip} err={total_err}\n")

    print(f"\n📊 Tổng kết: OK={total_ok} | Skip={total_skip} | Err={total_err}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 5: NÉN VÀ TẢI VỀ
# ═══════════════════════════════════════════════════════════════════════════════
def zip_and_download():
    zip_path = '/kaggle/working/scene_features_gaf2000_v2.zip'
    print(f"\n📦 Đang nén {OUTPUT_ROOT} → {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_ROOT):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, os.path.dirname(OUTPUT_ROOT))
                zf.write(abs_path, rel_path)
    print(f"✅ {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")
    ckpt_zip = '/kaggle/working/resnet50fpn_gaf2000_v2_ckpt.zip'
    with zipfile.ZipFile(ckpt_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(CKPT_PATH):
            zf.write(CKPT_PATH, os.path.basename(CKPT_PATH))
    print(f"✅ Checkpoint: {ckpt_zip}")


# ═══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════
def sanity_check_one_batch():
    print("\n" + "="*65)
    print(" 🚀 SANITY CHECK: OVERFIT 1 BATCH")
    print("="*65)

    train_ds = GAF2000SceneDataset('Train', train_transform)
    if len(train_ds) == 0:
        print("❌ Không tìm thấy dữ liệu!"); return

    loader    = DataLoader(train_ds, batch_size=4, shuffle=True)
    model     = ResNet50FPN_GER(NUM_CLASSES).to(DEVICE)
    model.unfreeze_backbone()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    imgs, labels, _ = next(iter(loader))
    imgs, labels    = imgs.to(DEVICE), labels.to(DEVICE)
    print(f"Test với {len(labels)} ảnh | Pipeline: 50176 → spatial_proj → 1024 → classifier")

    # Dùng eval() để tắt Dropout — sanity check chỉ test gradient thông, không test regularization
    model.eval()
    for i in range(100):
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(1) == labels).sum().item() / len(labels)
        if i % 10 == 0:
            print(f"  Iter {i:02d} | Loss={loss.item():.4f} | Acc={acc:.4f}")
        if acc == 1.0 and loss.item() < 0.05:
            print("\n  ✅ SANITY CHECK PASS — gradient thông suốt!")
            print("  (model.eval() dùng để tắt Dropout, training thực dùng model.train())")
            return

    print("\n  ❌ SANITY CHECK FAIL — gradient bị tắc, kiểm tra lại kiến trúc.")


def main():
    finetune()
    if not os.path.exists(CKPT_PATH) and os.path.exists(EMERGENCY_CKPT_PATH):
        print(f"⚠️ Không có best checkpoint. Dùng emergency: {EMERGENCY_CKPT_PATH}")
        import shutil
        shutil.copy(EMERGENCY_CKPT_PATH, CKPT_PATH)
    extract_features()
    zip_and_download()


if __name__ == '__main__':
    main()