"""
GAF3_finetuneExtractVGG_Face_v4.py
=====================================
Pipeline hoàn chỉnh cho GAF3 — A100 optimized:
  1. Detect + Crop khuôn mặt bằng RetinaFace (lưu crop + bbox)
  2. Fine-tune VGGFace (VGG16) trên face crops — 3 class: negative/neutral/positive
  3. Extract features 4096-dim + bbox → .npz per ảnh gốc

Khác GAF2:
  - data_root: GAF_3.0 (tên folder có dấu chấm)
  - SPLITS: ['Train', 'Validation'] (không phải 'Val')
  - GAF3 cấu trúc lồng: Train/Positive/Positive/*.jpg
  - lr=1e-5, weight_decay=1e-4, patience=12 (data nhiều hơn → regularize nhẹ hơn)
  - Bỏ ColorJitter + RandomGrayscale (data đủ nhiều)
  - batch_size=64 (giữ paper baseline, data lớn hơn nên không cần scale mạnh)

Fixes so với file cũ:
  - LABEL_MAP đúng: negative=0, neutral=1, positive=2
  - Bỏ self.eval() trong extract() — caller tự quản lý
  - model.eval() tường minh trong run_extract()
  - torch.no_grad() wrap ở caller thay vì bên trong extract()
  - persistent_workers guard (nw > 0)
  - Weighted CrossEntropyLoss tự động
  - AMP đúng cách (chỉ train loop)
  - A100 TF32 + cudnn.benchmark
"""

import os, glob, re, time, zipfile
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm.auto import tqdm
from collections import defaultdict

import cv2
from retinaface import RetinaFace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

import kagglehub

# ── A100 optimizations ────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

# ==========================================
# CONFIG
# ==========================================
path_gaf3    = kagglehub.dataset_download("trieung11/gaf-3000")
path_vggface = kagglehub.dataset_download("trieung11/1821153vgg-face")
print(f"✅ GAF3    : {path_gaf3}")
print(f"✅ VGGFace : {path_vggface}")

pretrained = path_vggface if path_vggface.endswith('.pth') \
             else f'{path_vggface}/VGG_FACE.pth'

DRIVE_DIR = '/content/drive/MyDrive/GAF3_Project'

CONFIG = {
    # ── Input ─────────────────────────────────────────────────────────────
    'data_root':       f'{path_gaf3}/GAF_3.0',
    'pretrained_path': pretrained,

    # ── Crop dir (trên /content để I/O nhanh) ─────────────────────────────
    'crop_dir':        '/content/GAF3_Face_Cropped',

    # ── Output → Google Drive ─────────────────────────────────────────────
    'output_dir':      DRIVE_DIR,
    'ckpt_path':       f'{DRIVE_DIR}/vgg_best_gaf3.pth',
    'feat_dir':        f'{DRIVE_DIR}/features_vgg_gaf3',

    # ── Detection ─────────────────────────────────────────────────────────
    'detection_threshold': 0.9,

    # ── Training ──────────────────────────────────────────────────────────
    # GAF3 data nhiều hơn GAF2 → lr nhỏ hơn, regularization nhẹ hơn
    # batch_size=128 (A100 optimized, scale từ 64)
    # lr=1.5e-5 (scale nhẹ theo batch 64→128)
    'batch_size':         128,
    'num_workers':        4,
    'img_size':           224,
    'lr':                 1.5e-5,
    'weight_decay':       1e-4,
    'epochs':             50,
    'patience':           12,
    'scheduler_patience': 5,
    'time_limit_sec':     11.5 * 3600,

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

# GAF3: Train + Validation (không có Test split)
SPLITS = ['Train', 'Validation']

# negative=0, neutral=1, positive=2 — nhất quán với GAF2 và GroupEmoW
LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}

for d in [CONFIG['crop_dir'], CONFIG['output_dir'], CONFIG['feat_dir']]:
    os.makedirs(d, exist_ok=True)

print(f"✅ Device    : {CONFIG['device']}")
print(f"✅ Output    → {CONFIG['output_dir']}")
print(f"✅ LABEL_MAP : {LABEL_MAP}")


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
        x = self.relu(self.fc7(x))
        return x

    def forward(self, x):
        return self.fc8(self.drop(self._features(x)))

    def extract(self, x):
        """
        Extract fc7 4096-dim features — no dropout.
        KHÔNG gọi self.eval() — caller phải gọi model.eval() trước.
        Caller wrap torch.no_grad() bên ngoài.
        """
        x = self.relu(self.conv1_1(x)); x = self.relu(self.conv1_2(x)); x = self.pool(x)
        x = self.relu(self.conv2_1(x)); x = self.relu(self.conv2_2(x)); x = self.pool(x)
        x = self.relu(self.conv3_1(x)); x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x)); x = self.pool(x)
        x = self.relu(self.conv4_1(x)); x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x)); x = self.pool(x)
        x = self.relu(self.conv5_1(x)); x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x)); x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc6(x))   # no dropout
        x = self.relu(self.fc7(x))   # 4096-dim
        return x


# ==========================================
# BƯỚC 1: DETECT + CROP
# ==========================================
def _resolve_cls_dir(parent: str, cls_name: str) -> str:
    """
    GAF3 có cấu trúc lồng: Train/Positive/Positive/*.jpg
    Hàm này tự detect và trả về đúng thư mục chứa ảnh.
    """
    inner = os.path.join(parent, cls_name, cls_name)
    return inner if os.path.exists(inner) else os.path.join(parent, cls_name)


def run_face_detection():
    print("=" * 70)
    print("  BƯỚC 1: FACE DETECTION + CROPPING (RetinaFace)")
    print("=" * 70 + "\n")

    existing = glob.glob(os.path.join(CONFIG['crop_dir'], '**/*.jpg'), recursive=True)
    if existing:
        print(f"  📂 Đã có {len(existing)} crop files → bỏ qua detection")
        return

    print("  Loading RetinaFace model...")
    retina_model = RetinaFace.build_model()

    total_scanned  = 0
    total_saved    = 0
    images_no_face = 0

    for split in SPLITS:
        split_dir = os.path.join(CONFIG['data_root'], split)
        if not os.path.exists(split_dir):
            print(f"  ⚠️  Không tìm thấy: {split_dir}")
            continue

        print(f"\n  📂 Split: {split}")

        for cls_name in os.listdir(split_dir):
            if cls_name.lower() not in LABEL_MAP:
                continue

            cls_dir     = _resolve_cls_dir(split_dir, cls_name)
            out_cls_dir = os.path.join(CONFIG['crop_dir'], split, cls_name)
            os.makedirs(out_cls_dir, exist_ok=True)

            img_files = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            for filename in tqdm(img_files, desc=f"    {split}/{cls_name}"):
                total_scanned += 1
                img_path = os.path.join(cls_dir, filename)
                base, ext = os.path.splitext(filename)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img_h, img_w = img.shape[:2]
                    faces = RetinaFace.detect_faces(
                        img, model=retina_model,
                        threshold=CONFIG['detection_threshold']
                    )

                    if not isinstance(faces, dict) or not faces:
                        images_no_face += 1
                        continue

                    for count, (key, face_info) in enumerate(faces.items()):
                        x1, y1, x2, y2 = (int(c) for c in face_info['facial_area'])
                        x1c = max(0, x1); y1c = max(0, y1)
                        x2c = min(img_w, x2); y2c = min(img_h, y2)

                        crop = img[y1c:y2c, x1c:x2c]
                        if crop.size == 0:
                            continue

                        crop_name = f"{base}_face_{count+1}{ext}"
                        crop_path = os.path.join(out_cls_dir, crop_name)

                        if not os.path.exists(crop_path):
                            cv2.imwrite(crop_path, crop)
                            total_saved += 1
                            bbox_info = np.array(
                                [x1, y1, x2, y2, img_w, img_h], dtype=np.float32
                            )
                            np.save(
                                os.path.join(out_cls_dir, f"{base}_face_{count+1}_bbox.npy"),
                                bbox_info
                            )

                except Exception as e:
                    print(f"\n    ❌ Error {img_path}: {e}")

    print(f"\n  ✅ Detection done")
    print(f"     Scanned : {total_scanned}")
    print(f"     Saved   : {total_saved} face crops")
    print(f"     No faces: {images_no_face}")
    print(f"     Output  : {CONFIG['crop_dir']}")


# ==========================================
# BƯỚC 2: DATASET + TRANSFORMS
# ==========================================
class GAF3FaceDataset(Dataset):
    def __init__(self, split, transform):
        self.transform = transform
        self.samples   = []

        split_dir = os.path.join(CONFIG['crop_dir'], split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Không tìm thấy crop_dir cho split '{split}': {split_dir}\n"
                "Hãy chạy run_face_detection() trước."
            )

        for cls_name in os.listdir(split_dir):
            label = LABEL_MAP.get(cls_name.lower())
            if label is None:
                continue
            # Sau khi crop, cấu trúc flat: crop_dir/Train/Positive/*.jpg
            cls_dir = os.path.join(split_dir, cls_name)
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for p in glob.glob(os.path.join(cls_dir, ext)):
                    if '_bbox' not in p:
                        self.samples.append((p, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"Không tìm thấy face crops trong {split_dir}")

        print(f"  {split}: {len(self.samples)} face crops")
        for name, idx in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
            n = sum(1 for _, l in self.samples if l == idx)
            if n > 0:
                print(f"    {name}: {n} ({100*n/len(self.samples):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def get_class_counts(self):
        """Trả về [count_neg, count_neu, count_pos] theo thứ tự label index."""
        counts = [0] * len(LABEL_MAP)
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        return self.transform(img), label


# Normalize: VGGFace mean subtraction (RGB), std=[1,1,1]
_NORM_MEAN = [0.507, 0.411, 0.367]
_NORM_STD  = [1.0,   1.0,   1.0]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),
    # GAF3 data đủ nhiều → không cần ColorJitter + RandomGrayscale
    transforms.ToTensor(),
    transforms.Normalize(_NORM_MEAN, _NORM_STD),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_NORM_MEAN, _NORM_STD),
])


# ==========================================
# BƯỚC 2: FINE-TUNE
# ==========================================
def run_finetune():
    print("\n" + "=" * 70)
    print("  BƯỚC 2: FINE-TUNE VGGFace — GAF3 (A100)")
    print(f"  Device      : {CONFIG['device']}")
    print(f"  Batch size  : {CONFIG['batch_size']}")
    print(f"  LR          : {CONFIG['lr']}")
    print(f"  Ckpt → Drive: {CONFIG['ckpt_path']}")
    print("=" * 70 + "\n")

    print("📂 Loading datasets...")
    train_ds = GAF3FaceDataset('Train',      train_tf)
    val_ds   = GAF3FaceDataset('Validation', val_tf)

    # Weighted CrossEntropyLoss — tự động từ class distribution
    class_counts  = train_ds.get_class_counts()
    total_samples = sum(class_counts)
    class_weights = torch.tensor(
        [total_samples / (len(LABEL_MAP) * c) for c in class_counts],
        dtype=torch.float
    )
    class_weights = (class_weights / class_weights.sum()).to(CONFIG['device'])
    print(f"\n  ⚖️  Class counts (neg/neu/pos): {class_counts}")
    print(f"  ⚖️  Class weights             : {[f'{w:.4f}' for w in class_weights.tolist()]}")

    nw = CONFIG['num_workers']
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'],
        shuffle=True,  num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )

    print("\n📦 Loading model...")
    model = VGG_16(num_classes=3, dropout=0.5).to(CONFIG['device'])

    start_epoch = 0
    best_acc    = 0.0
    no_improve  = 0

    load_path = CONFIG['ckpt_path'] if os.path.exists(CONFIG['ckpt_path']) \
                else CONFIG['pretrained_path']

    if os.path.exists(load_path):
        print(f"  Loading: {load_path}")
        ckpt     = torch.load(load_path, map_location='cpu')
        sd       = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        model_sd = model.state_dict()
        sd = {k: v for k, v in sd.items()
              if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  Loaded {len(sd)}/{len(model_sd)} layers")
        if missing:    print(f"  Missing   : {missing}")
        if unexpected: print(f"  Unexpected: {unexpected}")
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch']
            best_acc    = ckpt.get('best_acc', 0.0)
            print(f"  ✅ Resume từ epoch {start_epoch} | best_acc={best_acc:.4f}")
        else:
            print(f"  ✅ Pretrained weights loaded")
    else:
        print("  ⚠️  Không tìm thấy weights, train từ đầu")

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=CONFIG['scheduler_patience'], min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler    = GradScaler()

    if os.path.exists(CONFIG['ckpt_path']):
        ckpt = torch.load(CONFIG['ckpt_path'], map_location='cpu')
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  ✅ Optimizer state resumed")

    print("\n" + "=" * 70)
    print("🎯 TRAINING START")
    print("=" * 70 + "\n")

    global_start = time.time()

    for epoch in range(start_epoch, CONFIG['epochs']):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = tr_total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch+1:02d} [Train]", leave=False):
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()

            with autocast():
                out  = model(imgs)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_loss    += loss.item() * len(labels)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total   += len(labels)

        # ── Validate — no AMP needed ───────────────────────────────────────
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

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            torch.save({
                'epoch':                epoch + 1,
                'state_dict':           model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc':             best_acc,
            }, CONFIG['ckpt_path'])
            print(f"  🔥 Best! ValAcc={val_acc:.4f} → saved to Drive")
        else:
            no_improve += 1
            print(f"  ⏳ No improve {no_improve}/{CONFIG['patience']}")
            if no_improve >= CONFIG['patience']:
                print("\n🛑 Early stopping")
                break

        if time.time() - global_start > CONFIG['time_limit_sec']:
            elapsed = (time.time() - global_start) / 3600
            print(f"\n⏳ {elapsed:.1f}h elapsed — stopping")
            break

    print(f"\n✅ Training done | Best ValAcc = {best_acc:.4f}")
    print(f"   Checkpoint: {CONFIG['ckpt_path']}")


# ==========================================
# BƯỚC 3: EXTRACT FEATURES + BBOX
# ==========================================
def extract_features_with_bbox(model, split):
    """
    Group face crops theo ảnh gốc, extract fc7 4096-dim + bbox thật.
    Output: feat_dir/<split>/<cls_lower>/<orig_name>.npz
    Caller phải gọi model.eval() trước khi gọi hàm này.
    """
    split_dir = os.path.join(CONFIG['crop_dir'], split)
    if not os.path.exists(split_dir):
        print(f"  ⚠️  Not found: {split_dir}")
        return 0

    groups      = defaultdict(list)
    dummy_count = 0

    for cls_name in os.listdir(split_dir):
        if LABEL_MAP.get(cls_name.lower()) is None:
            continue

        # Sau khi crop, cấu trúc flat: crop_dir/Train/Positive/*.jpg
        cls_dir = os.path.join(split_dir, cls_name)

        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            img_files.extend(glob.glob(os.path.join(cls_dir, ext)))

        for img_path in img_files:
            basename = os.path.basename(img_path)
            if '_bbox' in basename:
                continue
            match = re.match(r'^(.+?)_face_(\d+)\.(jpg|jpeg|png)$', basename, re.IGNORECASE)
            if not match:
                continue
            orig_name = match.group(1)

            stem      = os.path.splitext(img_path)[0]
            bbox_path = stem + '_bbox.npy'
            if os.path.exists(bbox_path):
                raw  = np.load(bbox_path)
                bbox = raw[:4].astype(np.float32)
            else:
                bbox = np.array([0, 0, 1, 1], dtype=np.float32)
                dummy_count += 1

            groups[(cls_name, orig_name)].append((img_path, bbox))

    if dummy_count > 0:
        print(f"  ⚠️  {dummy_count} faces không có bbox → dummy [0,0,1,1]")

    total_saved = 0
    for (cls_name, orig_name), face_list in tqdm(groups.items(), desc=f"  Extract [{split}]"):
        out_dir  = os.path.join(CONFIG['feat_dir'], split, cls_name.lower())
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{orig_name}.npz')

        if os.path.exists(out_path):
            total_saved += 1
            continue

        face_list.sort(key=lambda x: x[0])

        imgs_tensor, bboxes = [], []
        for img_path, bbox in face_list:
            try:
                img = Image.open(img_path).convert('RGB')
                imgs_tensor.append(val_tf(img))
                bboxes.append(bbox)
            except Exception as e:
                print(f"    ⚠️  Skip {img_path}: {e}")

        if len(imgs_tensor) == 0:
            continue

        batch = torch.stack(imgs_tensor).to(CONFIG['device'])
        with torch.no_grad():
            feats = model.extract(batch).cpu().numpy()   # (N, 4096)
        boxes = np.array(bboxes, dtype=np.float32)       # (N, 4)

        np.savez(out_path, features=feats, boxes=boxes)
        total_saved += 1

    return total_saved


def run_extract():
    print("\n" + "=" * 70)
    print("  BƯỚC 3: FEATURE EXTRACTION — load best checkpoint từ Drive")
    print("=" * 70 + "\n")

    if not os.path.exists(CONFIG['ckpt_path']):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {CONFIG['ckpt_path']}\n"
            "Hãy chạy run_finetune() trước."
        )

    model     = VGG_16(num_classes=3, dropout=0.5).to(CONFIG['device'])
    best_ckpt = torch.load(CONFIG['ckpt_path'], map_location='cpu')
    model.load_state_dict(best_ckpt['state_dict'])
    model.eval()   # tường minh — extract() không tự gọi eval()
    print(f"  ✅ Loaded checkpoint "
          f"(epoch {best_ckpt['epoch']}, ValAcc={best_ckpt['best_acc']:.4f})")

    total_npz = 0
    for split in SPLITS:
        print(f"\n📂 Extracting [{split}]...")
        n = extract_features_with_bbox(model, split)
        print(f"  ✅ {split}: {n} .npz files")
        total_npz += n
    print(f"\n✅ Total: {total_npz} .npz files")

    # Verify 1 sample
    samples = glob.glob(os.path.join(CONFIG['feat_dir'], '**/*.npz'), recursive=True)
    if samples:
        s = np.load(samples[0])
        print(f"\n📋 Sample: {os.path.basename(samples[0])}")
        print(f"   features: {s['features'].shape}  ← (num_faces, 4096)")
        print(f"   boxes:    {s['boxes'].shape}      ← (num_faces, 4)")
        print(f"   boxes[0]: {s['boxes'][0]}")

    # Zip → Drive
    print("\n📦 Zipping features → Drive...")
    feat_zip = os.path.join(CONFIG['output_dir'], 'vggface_gaf3_features.zip')
    with zipfile.ZipFile(feat_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(CONFIG['feat_dir']):
            for f in files:
                abs_p = os.path.join(root, f)
                rel_p = os.path.relpath(abs_p, CONFIG['feat_dir'])
                zf.write(abs_p, rel_p)
    print(f"✅ {feat_zip}  ({os.path.getsize(feat_zip)/1e6:.1f} MB)")
    print(f"\n📥 Tất cả output tại: {CONFIG['output_dir']}")


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 70)
    print("  VGGFace GAF3 v4 — DETECT → FINETUNE → EXTRACT")
    print(f"  Device     : {CONFIG['device']}")
    print(f"  Batch size : {CONFIG['batch_size']}")
    print(f"  LR         : {CONFIG['lr']}")
    print(f"  LABEL_MAP  : {LABEL_MAP}")
    print("=" * 70 + "\n")

    run_face_detection()   # tự bỏ qua nếu crops đã có
    run_finetune()
    run_extract()


main()