# ==========================================
# TRÍCH XUẤT OBJECT - GAF3
# ⚠️  Nên chạy SAU file face để dùng chung gaf3_split.csv
#     Nếu CSV chưa có, script tự tạo lại
# ==========================================

import subprocess
subprocess.run(["pip", "install", "timm", "-q"], check=True)

import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
import glob
from tqdm.auto import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
import timm
import shutil
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG
# ==========================================
CONFIG = {
    'csv_path':       '/kaggle/working/gaf3_split.csv',
    'data_root':      '/kaggle/input/datasets/trieung11/gaf-3000/GAF_3.0',
    'output_dir':     '/kaggle/working/gaf3_object_features',
    'max_objects':    10,
    'conf_threshold': 0.5,
    'val_size':       0.2,   # Dùng nếu phải tự tạo lại CSV
    'random_seed':    42,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# ==========================================
# TỰ TẠO CSV NẾU CHƯA CÓ
# ==========================================
def find_images(split_folder, gaf_split_name):
    records  = []
    emotions = ['Positive', 'Neutral', 'Negative']
    for emotion in emotions:
        in_dir = os.path.join(split_folder, emotion, emotion)
        if not os.path.exists(in_dir):
            in_dir = os.path.join(split_folder, emotion)
        if not os.path.exists(in_dir):
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


def ensure_csv(data_root, csv_path, val_size=0.2, seed=42):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"📋 Đọc CSV có sẵn: {csv_path} ({len(df)} ảnh)")
        # Cần khôi phục img_path vì CSV không lưu path tuyệt đối
        df['img_path'] = df.apply(lambda r: restore_img_path(r, data_root), axis=1)
        return df

    print("⚠️  Không tìm thấy gaf3_split.csv → tự tạo lại...")
    train_records = find_images(os.path.join(data_root, 'Train'),      'Train')
    test_records  = find_images(os.path.join(data_root, 'Validation'), 'Validation')

    df_train_all = pd.DataFrame(train_records)
    train_idx, val_idx = train_test_split(
        range(len(df_train_all)),
        test_size=val_size,
        stratify=df_train_all['emotion'].values,
        random_state=seed
    )
    df_train_all['new_split'] = ''
    df_train_all.iloc[list(train_idx), df_train_all.columns.get_loc('new_split')] = 'train'
    df_train_all.iloc[list(val_idx),   df_train_all.columns.get_loc('new_split')] = 'val'

    df_test              = pd.DataFrame(test_records)
    df_test['new_split'] = 'test'

    df_all = pd.concat([df_train_all, df_test], ignore_index=True)
    df_all[['filename', 'emotion', 'original_split', 'new_split']].to_csv(csv_path, index=False)
    print(f"✅ CSV đã tạo: {csv_path}")
    return df_all


def restore_img_path(row, data_root):
    gaf_split = row['original_split']
    emotion   = row['emotion']
    filename  = row['filename']
    p = os.path.join(data_root, gaf_split, emotion, emotion, filename)
    if os.path.exists(p):
        return p
    p = os.path.join(data_root, gaf_split, emotion, filename)
    if os.path.exists(p):
        return p
    return None


# ==========================================
# KHỞI TẠO MODEL
# ==========================================
print(f"🚀 Loading models... Device: {CONFIG['device']}")

detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
detector = detector.to(CONFIG['device']).eval()
print("  ✅ Faster R-CNN loaded")

feature_extractor = timm.create_model('seresnet50', pretrained=True, num_classes=0)
feature_extractor = feature_extractor.to(CONFIG['device']).eval()
print("  ✅ SE-ResNet50 loaded (2048-dim)")

obj_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
to_tensor = transforms.ToTensor()


# ==========================================
# XỬ LÝ 1 ẢNH
# ==========================================
def process_image(img_path, device):
    img_pil = Image.open(img_path).convert('RGB')
    W, H   = img_pil.size

    img_t = to_tensor(img_pil).to(device)
    with torch.no_grad():
        pred = detector([img_t])[0]

    boxes_det  = pred['boxes']
    scores_det = pred['scores']
    obj_feats   = []
    valid_boxes = []

    if len(boxes_det) > 0:
        keep       = scores_det > CONFIG['conf_threshold']
        kept_boxes = boxes_det[keep][:CONFIG['max_objects']]

        batch_crops = []
        for box in kept_boxes:
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if x2 > x1 and y2 > y1:
                crop = img_pil.crop((x1, y1, x2, y2))
                batch_crops.append(obj_transform(crop))
                valid_boxes.append([x1, y1, x2, y2])

        if batch_crops:
            batch_t = torch.stack(batch_crops).to(device)
            with torch.no_grad():
                obj_feats = feature_extractor(batch_t).cpu().numpy()

    if len(obj_feats) == 0:
        return (np.zeros((0, 2048), dtype=np.float32),
                np.zeros((0, 4),    dtype=np.float32))

    return (np.array(obj_feats,   dtype=np.float32),
            np.array(valid_boxes, dtype=np.float32))


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 70)
    print("📦 GAF3 — TRÍCH XUẤT OBJECT (Faster-RCNN + SE-ResNet50)")
    print("=" * 70)

    df = ensure_csv(CONFIG['data_root'], CONFIG['csv_path'],
                    CONFIG['val_size'], CONFIG['random_seed'])

    total_saved = 0

    for split in ['train', 'val', 'test']:
        df_split = df[df['new_split'] == split]
        print(f"\n📂 [{split.upper()}] — {len(df_split)} ảnh")

        for emotion in ['Positive', 'Neutral', 'Negative']:
            df_emo = df_split[df_split['emotion'] == emotion]
            if len(df_emo) == 0:
                continue

            out_dir = os.path.join(CONFIG['output_dir'], split, emotion)
            os.makedirs(out_dir, exist_ok=True)

            saved  = 0
            no_obj = 0

            for _, row in tqdm(df_emo.iterrows(), total=len(df_emo),
                               desc=f"  {emotion}", leave=False):
                base     = os.path.splitext(row['filename'])[0]
                out_path = os.path.join(out_dir, base + '.npz')

                if os.path.exists(out_path):
                    saved += 1
                    continue

                img_path = row.get('img_path') or restore_img_path(row, CONFIG['data_root'])
                if not img_path or not os.path.exists(img_path):
                    print(f"\n    ⚠️  Không tìm thấy: {row['filename']}")
                    continue

                try:
                    feats, boxes = process_image(img_path, CONFIG['device'])
                    if len(feats) == 0:
                        no_obj += 1
                    np.savez_compressed(out_path, features=feats, boxes=boxes)
                    saved += 1
                except Exception as e:
                    print(f"\n    ❌ {row['filename']}: {e}")

            print(f"  ✅ {emotion:10s}: {saved} files | No-object: {no_obj}")
            total_saved += saved

    print(f"\n✅ HOÀN TẤT OBJECT — Tổng: {total_saved} files")

    samples = glob.glob(os.path.join(CONFIG['output_dir'], '**/*.npz'), recursive=True)
    if samples:
        s = np.load(samples[0])
        print(f"\n📋 Sample: {os.path.basename(samples[0])}")
        print(f"   features: {s['features'].shape}  ← (num_objects, 2048)")

    print("\n⏳ Nén output...")
    shutil.make_archive('/kaggle/working/gaf3_object_features', 'zip',
                        '/kaggle/working/gaf3_object_features')
    print("✅ gaf3_object_features.zip")


if __name__ == '__main__':
    main()