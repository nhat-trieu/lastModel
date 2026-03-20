import subprocess
subprocess.run(["pip", "install", "timm", "-q"], check=True)

import os
import torch
import numpy as np
from PIL import Image
import glob
from tqdm.auto import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
import timm
import shutil
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG
# ==========================================
CONFIG = {
    'data_root':      '/kaggle/input/datasets/trieung11/gaf-3000/GAF_3.0',
    'output_dir':     '/kaggle/working/gaf3_object_features',
    'max_objects':    10,
    'conf_threshold': 0.5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

# GAF3: Train và Validation, không dùng Test (không có nhãn)
SPLITS   = ['Train', 'Validation']
EMOTIONS = ['Positive', 'Neutral', 'Negative']

os.makedirs(CONFIG['output_dir'], exist_ok=True)


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
def process_image(img_path):
    img_pil = Image.open(img_path).convert('RGB')
    W, H    = img_pil.size
    device  = CONFIG['device']

    img_t = to_tensor(img_pil).to(device)
    with torch.no_grad():
        pred = detector([img_t])[0]

    boxes_det  = pred['boxes']
    scores_det = pred['scores']
    obj_feats  = []
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

    return (np.array(obj_feats,    dtype=np.float32),
            np.array(valid_boxes,  dtype=np.float32))


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 70)
    print("📦 GAF3 — TRÍCH XUẤT OBJECT (Faster-RCNN + SE-ResNet50)")
    print("=" * 70)

    total_saved = 0

    for split in SPLITS:
        print(f"\n📂 [{split.upper()}]")

        for emotion in EMOTIONS:
            in_dir = os.path.join(CONFIG['data_root'], split, emotion)
            if not os.path.exists(in_dir):
                print(f"  ⚠️  Không tìm thấy: {in_dir}")
                continue

            img_files = sorted(
                glob.glob(os.path.join(in_dir, '**', '*.jpg'),  recursive=True) +
                glob.glob(os.path.join(in_dir, '**', '*.png'),  recursive=True) +
                glob.glob(os.path.join(in_dir, '**', '*.jpeg'), recursive=True)
            )
            if not img_files:
                print(f"  ⚠️  Không có ảnh trong: {in_dir}")
                continue

            out_dir = os.path.join(CONFIG['output_dir'], split, emotion)
            os.makedirs(out_dir, exist_ok=True)

            saved  = 0
            no_obj = 0

            for img_path in tqdm(img_files, desc=f"  {split}/{emotion}", leave=False):
                base     = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(out_dir, base + '.npz')

                if os.path.exists(out_path):
                    saved += 1
                    continue

                try:
                    feats, boxes = process_image(img_path)
                    if len(feats) == 0:
                        no_obj += 1
                    np.savez_compressed(out_path, features=feats, boxes=boxes)
                    saved += 1
                except Exception as e:
                    print(f"\n    ❌ {os.path.basename(img_path)}: {e}")

            print(f"  ✅ {split}/{emotion:10s}: {saved} files | No-object: {no_obj}")
            total_saved += saved

    print(f"\n✅ HOÀN TẤT OBJECT — Tổng: {total_saved} files")

    # Kiểm tra sample output
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