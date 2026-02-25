# ==========================================
# TRÍCH XUẤT SCENE - GAF3
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
    'csv_path':      '/kaggle/working/gaf3_split.csv',
    'data_root':     '/kaggle/input/datasets/trieung11/gaf-3000/GAF_3.0',
    'scene_weights': '/kaggle/input/resnet50-scene-combined/resnet50_scene_combined.pth',
    'output_dir':    '/kaggle/working/gaf3_scene_features',
    'val_size':      0.2,
    'random_seed':   42,
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


def ensure_csv(data_root, csv_path, val_size=0.2, seed=42):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"📋 Đọc CSV có sẵn: {csv_path} ({len(df)} ảnh)")
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


# ==========================================
# KHỞI TẠO MODEL SE-ResNet50
# ==========================================
def build_scene_extractor(weights_path, device):
    print("🚀 Loading SE-ResNet50...")
    if os.path.exists(weights_path):
        print(f"  📦 Fine-tuned weights: {weights_path}")
        model = timm.create_model('seresnet50', pretrained=False, num_classes=3)
        ckpt  = torch.load(weights_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
            print("  ✅ Nạp weights thành công!")
        except RuntimeError:
            print("  ⚠️  Thử strict=False...")
            model.load_state_dict(state_dict, strict=False)
    else:
        print(f"  ⚠️  Không thấy weights → dùng ImageNet pretrained")
        model = timm.create_model('seresnet50', pretrained=True, num_classes=3)

    model.reset_classifier(num_classes=0)  # bỏ FC → 2048-dim
    model = model.to(device).eval()
    print("  ✅ SE-ResNet50 ready, output = 2048-dim")
    return model


scene_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 70)
    print("🌆 GAF3 — TRÍCH XUẤT SCENE (SE-ResNet50)")
    print(f"   Device: {CONFIG['device']}")
    print("=" * 70)

    df        = ensure_csv(CONFIG['data_root'], CONFIG['csv_path'],
                           CONFIG['val_size'], CONFIG['random_seed'])
    extractor = build_scene_extractor(CONFIG['scene_weights'], CONFIG['device'])

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

            saved = 0

            for _, row in tqdm(df_emo.iterrows(), total=len(df_emo),
                               desc=f"  {emotion}", leave=False):
                base     = os.path.splitext(row['filename'])[0]
                out_path = os.path.join(out_dir, base + '.npy')

                if os.path.exists(out_path):
                    saved += 1
                    continue

                img_path = row.get('img_path') or restore_img_path(row, CONFIG['data_root'])
                if not img_path or not os.path.exists(img_path):
                    print(f"\n    ⚠️  Không tìm thấy: {row['filename']}")
                    continue

                try:
                    img  = Image.open(img_path).convert('RGB')
                    t    = scene_transform(img).unsqueeze(0).to(CONFIG['device'])
                    with torch.no_grad():
                        feat = extractor(t).cpu().numpy()[0]  # (2048,)
                    np.save(out_path, feat)
                    saved += 1
                except Exception as e:
                    print(f"\n    ❌ {row['filename']}: {e}")

            print(f"  ✅ {emotion:10s}: {saved} files")
            total_saved += saved

    print(f"\n✅ HOÀN TẤT SCENE — Tổng: {total_saved} files")

    samples = glob.glob(os.path.join(CONFIG['output_dir'], '**/*.npy'), recursive=True)
    if samples:
        s = np.load(samples[0])
        print(f"\n📋 Sample: {os.path.basename(samples[0])}")
        print(f"   shape: {s.shape}  ← (2048,)")

    print("\n⏳ Nén output...")
    shutil.make_archive('/kaggle/working/gaf3_scene_features', 'zip',
                        '/kaggle/working/gaf3_scene_features')
    print("✅ gaf3_scene_features.zip")


if __name__ == '__main__':
    main()