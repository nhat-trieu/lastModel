"""
extract_face_features_bbox_gaf2.py
===================================
Extract face features 4096-dim từ VGGFace đã finetune trên GAF2.
- GAF2 chỉ có Train/Val (KHÔNG có Test)
- Cấu trúc thư mục: <root>/Train|Val / Positive|Negative|Neutral / *_face_<N>.jpg
- Output: .npz chứa features (N,4096) + boxes (N,4) cho mỗi ảnh gốc
"""

import os, glob, re, zipfile
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms

# ==========================================
# CONFIG
# ==========================================
CONFIG = {
    'ckpt_path':  '/kaggle/input/datasets/vggface_gaf2/best.pth',  # ← Sửa lại đường dẫn best.pth GAF2

    'data_root':  '/kaggle/input/datasets/wawuwaa/tr-facecropgaf2/GAF2_Full_Face_Cropped',

    'output_dir': '/kaggle/working/face_features_bbox_gaf2',

    'batch_size':  64,
    'num_workers': 2,
    'img_size':    224,

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

# GAF2 chỉ có Train và Val — KHÔNG có Test
SPLITS    = ['Train', 'Val']
LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==========================================
# MODEL (giống y hệt lúc train GAF2)
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

    def extract(self, x):
        """Inference-time: trả về fc7 4096-dim, KHÔNG dropout"""
        # Gọi hàm này khi model đã ở eval() — dropout tự tắt
        x = self.relu(self.conv1_1(x)); x = self.relu(self.conv1_2(x)); x = self.pool(x)
        x = self.relu(self.conv2_1(x)); x = self.relu(self.conv2_2(x)); x = self.pool(x)
        x = self.relu(self.conv3_1(x)); x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x)); x = self.pool(x)
        x = self.relu(self.conv4_1(x)); x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x)); x = self.pool(x)
        x = self.relu(self.conv5_1(x)); x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x)); x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc6(x))   # KHÔNG drop — inference
        x = self.relu(self.fc7(x))   # [B, 4096]
        return x

# ==========================================
# TRANSFORMS (giống val_tf lúc train GAF2)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.507, 0.411, 0.367], [1, 1, 1]),  # RGB order, giống train
])

# ==========================================
# LOAD MODEL
# ==========================================
def load_model():
    print(f"📦 Loading checkpoint: {CONFIG['ckpt_path']}")
    if not os.path.exists(CONFIG['ckpt_path']):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {CONFIG['ckpt_path']}")

    model = VGG_16(num_classes=3, dropout=0.5).to(CONFIG['device'])
    ckpt  = torch.load(CONFIG['ckpt_path'], map_location='cpu')
    sd    = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()  # Tắt dropout/batchnorm cho inference
    print(f"  ✅ Model loaded")
    if 'epoch' in ckpt:
        print(f"  📊 Epoch={ckpt['epoch']} | BestValAcc={ckpt.get('best_acc', 'N/A')}")
    return model

# ==========================================
# HELPER: build output path từ img_path
# ==========================================
def get_output_path(img_path, split, cls_name):
    """
    Input:  .../Train/Positive/Picture_00003_face_10.jpg
    Output: output_dir/train/positive/Picture_00003.npz
    Note:   split và cls_name được lower() để nhất quán
    """
    basename = os.path.basename(img_path)
    match = re.match(r'^(.+?)_face_(\d+)\.(jpg|jpeg|png)$', basename, re.IGNORECASE)
    orig_name = match.group(1) if match else os.path.splitext(basename)[0]

    out_dir = os.path.join(CONFIG['output_dir'], split.lower(), cls_name.lower())
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{orig_name}.npz'), orig_name

# ==========================================
# EXTRACT MỘT SPLIT
# ==========================================
def extract_split(model, split):
    """
    Duyệt toàn bộ split (Train hoặc Val), group các face crop theo ảnh gốc,
    extract feature, load bbox .npy, lưu .npz.
    """
    split_dir = os.path.join(CONFIG['data_root'], split)
    if not os.path.exists(split_dir):
        print(f"  ⚠️ Không tìm thấy: {split_dir}")
        return 0

    # ── Group: (cls_name, orig_name) → [(img_path, bbox), ...] ──
    groups = defaultdict(list)

    for cls_name in os.listdir(split_dir):
        if LABEL_MAP.get(cls_name.lower()) is None:
            continue
        cls_dir = os.path.join(split_dir, cls_name)

        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            img_files.extend(glob.glob(os.path.join(cls_dir, ext)))

        for img_path in img_files:
            basename = os.path.basename(img_path)
            match = re.match(r'^(.+?)_face_(\d+)\.(jpg|jpeg|png)$', basename, re.IGNORECASE)
            if not match:
                continue  # Bỏ qua file không đúng định dạng _face_N
            orig_name = match.group(1)

            # Load bbox nếu có file .npy kèm theo
            stem      = os.path.splitext(img_path)[0]
            bbox_path = stem + '_bbox.npy'
            if os.path.exists(bbox_path):
                raw_bbox = np.load(bbox_path)
                bbox = raw_bbox[:4].astype(np.float32)  # [x1, y1, x2, y2]
            else:
                bbox = np.array([0, 0, 1, 1], dtype=np.float32)  # fallback

            groups[(cls_name, orig_name)].append((img_path, bbox))

    # ── Extract từng group ──
    total_saved = 0
    all_groups  = list(groups.items())

    for (cls_name, orig_name), face_list in tqdm(all_groups, desc=f"  {split}"):
        out_path, _ = get_output_path(face_list[0][0], split, cls_name)

        # Skip nếu đã extract rồi
        if os.path.exists(out_path):
            total_saved += 1
            continue

        # Sort theo face index để thứ tự nhất quán
        face_list.sort(key=lambda x: x[0])

        imgs_tensor = []
        bboxes      = []

        for img_path, bbox in face_list:
            try:
                img = Image.open(img_path).convert('RGB')
                imgs_tensor.append(transform(img))
                bboxes.append(bbox)
            except Exception as e:
                print(f"    ⚠️ Skip {img_path}: {e}")

        if len(imgs_tensor) == 0:
            continue

        batch = torch.stack(imgs_tensor).to(CONFIG['device'])
        with torch.no_grad():
            feats = model.extract(batch).cpu().numpy()  # (N, 4096)

        boxes_arr = np.array(bboxes, dtype=np.float32)  # (N, 4)

        np.savez(out_path, features=feats, boxes=boxes_arr)
        total_saved += 1

    return total_saved

# ==========================================
# MAIN
# ==========================================
def main():
    print("="*70)
    print("  EXTRACT FACE FEATURES — GAF2 (Train + Val only, no Test)")
    print(f"  Device: {CONFIG['device']}")
    print("="*70 + "\n")

    model = load_model()

    total = 0
    for split in SPLITS:
        print(f"\n📂 Processing {split}...")
        n = extract_split(model, split)
        print(f"  ✅ {split}: {n} .npz files saved")
        total += n

    print(f"\n✅ Total: {total} .npz files")
    print(f"📁 Output: {CONFIG['output_dir']}")

    # ── Kiểm tra sample output ──
    samples = glob.glob(os.path.join(CONFIG['output_dir'], '**/*.npz'), recursive=True)
    if samples:
        s = np.load(samples[0])
        print(f"\n📋 Sample: {os.path.basename(samples[0])}")
        print(f"   features: {s['features'].shape}  ← (num_faces, 4096)")
        print(f"   boxes:    {s['boxes'].shape}      ← (num_faces, 4) [x1,y1,x2,y2]")

    # ── Zip toàn bộ output ──
    print("\n📦 Zipping features...")
    zip_path = '/kaggle/working/face_features_bbox_gaf2.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(CONFIG['output_dir']):
            for f in files:
                abs_p = os.path.join(root, f)
                rel_p = os.path.relpath(abs_p, CONFIG['output_dir'])
                zf.write(abs_p, rel_p)
    print(f"✅ {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")
    print("📥 Download từ Kaggle Output tab → face_features_bbox_gaf2.zip")

if __name__ == '__main__':
    main()