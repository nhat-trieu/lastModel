# ==========================================
# TRÍCH XUẤT FACE - GAF3
# Chạy file này TRƯỚC, nó tạo ra gaf3_split.csv
# ==========================================

import subprocess
subprocess.run(["pip", "install", "facenet-pytorch", "timm", "-q"], check=True)

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import shutil
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG
# ==========================================
CONFIG = {
    'checkpoint_path': '/kaggle/input/datasets/trieung11/finetunevggface11/vggface_epoch09_acc0.6945.pth',
    'data_root':       '/kaggle/input/datasets/trieung11/gaf-3000/GAF_3.0',
    'output_dir':      '/kaggle/working/gaf3_face_features',
    'csv_path':        '/kaggle/working/gaf3_split.csv',

    'val_size':        0.2,
    'random_seed':     42,

    'min_face_size':   20,
    'min_conf':        0.8,
    'max_faces':       16,

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==========================================
# VGGFace MODEL (giữ nguyên từ code cũ)
# ==========================================
class VGG_16(nn.Module):
    def __init__(self, num_classes=3, dropout=0.0):
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
        self.fc6     = nn.Linear(512 * 7 * 7, 4096)
        self.fc7     = nn.Linear(4096, 4096)
        self.fc8     = nn.Linear(4096, num_classes)
        self.relu    = nn.ReLU(inplace=True)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.conv1_1(x)); x = self.relu(self.conv1_2(x)); x = self.pool(x)
        x = self.relu(self.conv2_1(x)); x = self.relu(self.conv2_2(x)); x = self.pool(x)
        x = self.relu(self.conv3_1(x)); x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x)); x = self.pool(x)
        x = self.relu(self.conv4_1(x)); x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x)); x = self.pool(x)
        x = self.relu(self.conv5_1(x)); x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x)); x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))  # 4096-dim
        return x


def load_vggface(checkpoint_path, device):
    print(f"📦 Loading VGGFace: {checkpoint_path}")
    model = VGG_16(num_classes=3, dropout=0.0)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    print("  ✅ VGGFace loaded, fc7 = 4096-dim")
    return model


face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.367, 0.411, 0.507], [1, 1, 1]),
])


# ==========================================
# BƯỚC 1: XÂY DỰNG SPLIT CSV
# ==========================================
def find_images(split_folder, gaf_split_name):
    records  = []
    emotions = ['Positive', 'Neutral', 'Negative']
    for emotion in emotions:
        in_dir = os.path.join(split_folder, emotion, emotion)
        if not os.path.exists(in_dir):
            in_dir = os.path.join(split_folder, emotion)
        if not os.path.exists(in_dir):
            print(f"  ⚠️  Không tìm thấy: {in_dir}")
            continue
        img_files = sorted(
            glob.glob(os.path.join(in_dir, '*.jpg')) +
            glob.glob(os.path.join(in_dir, '*.png')) +
            glob.glob(os.path.join(in_dir, '*.jpeg'))
        )
        for p in img_files:
            records.append({
                'img_path':       p,
                'filename':       os.path.basename(p),
                'emotion':        emotion,
                'original_split': gaf_split_name,
            })
    return records


def build_split_csv(data_root, csv_path, val_size=0.2, seed=42):
    print("\n" + "="*60)
    print("📊 BƯỚC 1: XÂY DỰNG SPLIT CSV")
    print("="*60)

    train_records = find_images(os.path.join(data_root, 'Train'),      'Train')
    test_records  = find_images(os.path.join(data_root, 'Validation'), 'Validation')

    print(f"Train gốc            : {len(train_records)} ảnh")
    print(f"Validation gốc → test: {len(test_records)} ảnh")

    df_train_all = pd.DataFrame(train_records)
    labels = df_train_all['emotion'].values

    train_idx, val_idx = train_test_split(
        range(len(df_train_all)),
        test_size=val_size,
        stratify=labels,
        random_state=seed
    )

    df_train_all['new_split'] = ''
    df_train_all.iloc[list(train_idx), df_train_all.columns.get_loc('new_split')] = 'train'
    df_train_all.iloc[list(val_idx),   df_train_all.columns.get_loc('new_split')] = 'val'

    df_test              = pd.DataFrame(test_records)
    df_test['new_split'] = 'test'

    df_all = pd.concat([df_train_all, df_test], ignore_index=True)
    df_all[['filename', 'emotion', 'original_split', 'new_split']].to_csv(csv_path, index=False)

    print("\n📋 THỐNG KÊ SAU KHI CHIA:")
    print(f"{'Split':<8} {'Positive':>12} {'Neutral':>12} {'Negative':>12} {'Total':>8}")
    print("-" * 56)
    for split in ['train', 'val', 'test']:
        sub    = df_all[df_all['new_split'] == split]
        counts = sub['emotion'].value_counts()
        pos    = counts.get('Positive', 0)
        neu    = counts.get('Neutral',  0)
        neg    = counts.get('Negative', 0)
        total  = len(sub)
        print(f"{split:<8} {pos:>5}({pos/total*100:.0f}%) "
              f"{neu:>5}({neu/total*100:.0f}%) "
              f"{neg:>5}({neg/total*100:.0f}%)  {total:>6}")

    print(f"\n✅ CSV lưu tại: {csv_path}")
    return df_all


# ==========================================
# BƯỚC 2: TRÍCH XUẤT FACE
# ==========================================
def process_image(img_path, mtcnn, model, device):
    img_pil = Image.open(img_path).convert('RGB')
    W, H    = img_pil.size

    boxes_raw, probs = mtcnn.detect(img_pil)
    if boxes_raw is None or len(boxes_raw) == 0:
        return np.zeros((0, 4096), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    mask      = probs > CONFIG['min_conf']
    boxes_raw = boxes_raw[mask]
    probs     = probs[mask]
    if len(boxes_raw) == 0:
        return np.zeros((0, 4096), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    if len(boxes_raw) > CONFIG['max_faces']:
        top_idx   = np.argsort(probs)[::-1][:CONFIG['max_faces']]
        boxes_raw = boxes_raw[top_idx]

    face_tensors = []
    valid_boxes  = []
    for box in boxes_raw:
        x1, y1, x2, y2 = [int(b) for b in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img_pil.crop((x1, y1, x2, y2))
        face_tensors.append(face_transform(crop))
        valid_boxes.append([x1, y1, x2, y2])

    if len(face_tensors) == 0:
        return np.zeros((0, 4096), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    batch = torch.stack(face_tensors).to(device)
    with torch.no_grad():
        feats = model(batch).cpu().numpy()

    return feats.astype(np.float32), np.array(valid_boxes, dtype=np.float32)


def extract_features(df_all, mtcnn, model, output_dir, device):
    print("\n" + "="*60)
    print("🔍 BƯỚC 2: TRÍCH XUẤT FACE FEATURES")
    print("="*60)

    total_saved  = 0
    total_noface = 0

    for split in ['train', 'val', 'test']:
        df_split = df_all[df_all['new_split'] == split]
        print(f"\n📂 [{split.upper()}] — {len(df_split)} ảnh")

        for emotion in ['Positive', 'Neutral', 'Negative']:
            df_emo = df_split[df_split['emotion'] == emotion]
            if len(df_emo) == 0:
                continue

            out_dir = os.path.join(output_dir, split, emotion)
            os.makedirs(out_dir, exist_ok=True)

            saved  = 0
            noface = 0

            for _, row in tqdm(df_emo.iterrows(), total=len(df_emo),
                               desc=f"  {emotion}", leave=False):
                base     = os.path.splitext(row['filename'])[0]
                out_path = os.path.join(out_dir, base + '.npz')

                if os.path.exists(out_path):
                    saved += 1
                    continue

                try:
                    feats, boxes = process_image(row['img_path'], mtcnn, model, device)
                    if len(feats) == 0:
                        noface += 1
                    np.savez_compressed(out_path, features=feats, boxes=boxes)
                    saved += 1
                except Exception as e:
                    print(f"\n    ❌ {row['filename']}: {e}")

            print(f"  ✅ {emotion:10s}: {saved} files | No-face: {noface}")
            total_saved  += saved
            total_noface += noface

    print(f"\n✅ HOÀN TẤT FACE — Tổng: {total_saved} files | No-face: {total_noface}")


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 70)
    print("🚀 GAF3 — CHIA TẬP 80/20 + TRÍCH XUẤT FACE (VGGFace)")
    print(f"   Seed: {CONFIG['random_seed']}  |  Device: {CONFIG['device']}")
    print("=" * 70)

    df_all = build_split_csv(
        CONFIG['data_root'],
        CONFIG['csv_path'],
        val_size=CONFIG['val_size'],
        seed=CONFIG['random_seed'],
    )

    print("\n" + "="*60)
    print("⚙️  KHỞI TẠO MODEL")
    print("="*60)
    mtcnn = MTCNN(
        keep_all=True,
        device=CONFIG['device'],
        min_face_size=CONFIG['min_face_size'],
        post_process=False,
    )
    model = load_vggface(CONFIG['checkpoint_path'], CONFIG['device'])

    extract_features(df_all, mtcnn, model, CONFIG['output_dir'], CONFIG['device'])

    samples = glob.glob(os.path.join(CONFIG['output_dir'], '**/*.npz'), recursive=True)
    if samples:
        s = np.load(samples[0])
        print(f"\n📋 Sample: {os.path.basename(samples[0])}")
        print(f"   features: {s['features'].shape}  ← (num_faces, 4096)")
        print(f"   boxes:    {s['boxes'].shape}")

    print("\n⏳ Nén output...")
    shutil.make_archive('/kaggle/working/gaf3_face_features', 'zip',
                        '/kaggle/working/gaf3_face_features')
    print("✅ gaf3_face_features.zip")
    print("✅ gaf3_split.csv  ← xem/kiểm tra split bất cứ lúc nào")


if __name__ == '__main__':
    main()