# ==========================================
# CELL: Extract Face Features using Fine-tuned VGGFace
# Output: .npz files chứa features + boxes cho mỗi ảnh
# ==========================================

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
from tqdm.auto import tqdm

# ==========================================
# CONFIG
# ==========================================
CONFIG = {
    # ✅ Đổi path này thành checkpoint finetune tốt nhất của bạn
    'checkpoint_path': '/kaggle/input/datasets/trieung11/finetunevggface11/vggface_epoch09_acc0.6945.pth',

    # Input: ảnh crop khuôn mặt (đã crop sẵn từ bước trước)
    'face_crops_dir': '/kaggle/input/datasets/trieung11/face-crops-combined',

    # Output: lưu features .npz
    'output_dir': '/kaggle/working/face_features',

    'img_size': 224,
    'batch_size': 64,
    'num_workers': 0,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print("=" * 70)
print("🔍 EXTRACT FACE FEATURES - Fine-tuned VGGFace")
print("=" * 70)
print(f"Device: {CONFIG['device']}")


# ==========================================
# MODEL - Lấy feature tại fc7 (4096-dim)
# ==========================================
class VGG_16(nn.Module):
    def __init__(self, num_classes=3, dropout=0.6):
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
        x = self.relu(self.fc7(x))   # ✅ Lấy output fc7 = 4096-dim
        # Không đi qua fc8 (classifier)
        return x

    def extract_feature(self, x):
        """Alias rõ ràng hơn"""
        return self.forward(x)


def load_finetuned_model(checkpoint_path):
    print(f"\n📦 Loading fine-tuned checkpoint: {checkpoint_path}")
    model = VGG_16(num_classes=3, dropout=0.0)  # dropout=0 khi inference

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Tương thích cả 2 dạng save
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        epoch = ckpt.get('epoch', '?')
        val_acc = ckpt.get('val_acc', '?')
        print(f"  ✅ Checkpoint epoch={epoch}, val_acc={val_acc:.4f}" if isinstance(val_acc, float) else f"  ✅ Loaded checkpoint")
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"  ✅ Model loaded, fc7 output = 4096-dim")
    return model


# ==========================================
# TRANSFORM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.367, 0.411, 0.507], [1, 1, 1]),
])


# ==========================================
# EXTRACT FEATURES
# Cấu trúc input:  face_crops_dir/{dataset}/{split}/{emotion}/img_name_face{N}.jpg
# Cấu trúc output: output_dir/{dataset}/{split}/{emotion}/img_name.npz
#                  mỗi npz chứa:
#                    features: shape (num_faces, 4096)
#                    boxes:    shape (num_faces, 4) - [x1,y1,x2,y2] (dummy nếu đã crop)
# ==========================================
def extract_for_split(model, dataset, split, emotion, device):
    """
    Gom tất cả face crops của cùng 1 ảnh gốc lại thành 1 file .npz
    """
    in_dir = os.path.join(CONFIG['face_crops_dir'], dataset, split, emotion)
    out_dir = os.path.join(CONFIG['output_dir'], dataset, split, emotion)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(in_dir):
        return 0

    # Lấy tất cả ảnh
    all_imgs = sorted(glob.glob(os.path.join(in_dir, '*.jpg')) +
                      glob.glob(os.path.join(in_dir, '*.png')) +
                      glob.glob(os.path.join(in_dir, '*.jpeg')))

    if len(all_imgs) == 0:
        return 0

    # ✅ Group các face crops thuộc cùng 1 ảnh gốc
    # Naming convention: {image_name}_face{N}.jpg
    # → group by image_name (bỏ phần _face{N})
    from collections import defaultdict
    import re

    groups = defaultdict(list)
    for img_path in all_imgs:
        basename = os.path.basename(img_path)
        # Tách phần _face0, _face1, ... ra để lấy tên ảnh gốc
        match = re.match(r'^(.+?)_face(\d+)\.(jpg|png|jpeg)$', basename, re.IGNORECASE)
        if match:
            orig_name = match.group(1)
        else:
            # Không có _face suffix → coi cả file là 1 group
            orig_name = os.path.splitext(basename)[0]
        groups[orig_name].append(img_path)

    saved = 0
    for orig_name, face_paths in groups.items():
        out_path = os.path.join(out_dir, f'{orig_name}.npz')
        if os.path.exists(out_path):
            saved += 1
            continue

        # Load tất cả face của ảnh này
        tensors = []
        valid_paths = []
        for fp in sorted(face_paths):
            try:
                img = Image.open(fp).convert('RGB')
                tensors.append(transform(img))
                valid_paths.append(fp)
            except Exception as e:
                print(f"  ⚠️ Skip {fp}: {e}")

        if len(tensors) == 0:
            continue

        # Batch inference
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model(batch).cpu().numpy()  # (num_faces, 4096)

        # Tạo dummy boxes (vì ảnh đã crop sẵn, không có tọa độ gốc)
        # Nếu bạn có bounding box thật từ MTCNN, thay vào đây
        num_faces = len(feats)
        dummy_boxes = np.zeros((num_faces, 4), dtype=np.float32)
        # Đặt box = [0, 0, 1, 1] normalized (dummy)
        dummy_boxes[:, 2] = 1.0
        dummy_boxes[:, 3] = 1.0

        np.savez(out_path, features=feats, boxes=dummy_boxes)
        saved += 1

    return saved


# ==========================================
# MAIN
# ==========================================
print("\n[1/2] Loading model...")
model = load_finetuned_model(CONFIG['checkpoint_path'])
model = model.to(CONFIG['device'])

print("\n[2/2] Extracting features...")
datasets  = ['groupemow']
splits    = ['train', 'val', 'test']
emotions  = ['Positive', 'Neutral', 'Negative']

total = 0
for ds in datasets:
    for sp in splits:
        for em in emotions:
            n = extract_for_split(model, ds, sp, em, CONFIG['device'])
            if n > 0:
                print(f"  ✅ {ds}/{sp}/{em}: {n} files")
                total += n

print(f"\n✅ Done! Total: {total} .npz files")
print(f"📁 Output: {CONFIG['output_dir']}")

# Kiểm tra 1 file mẫu
sample_files = glob.glob(os.path.join(CONFIG['output_dir'], '**/*.npz'), recursive=True)
if sample_files:
    sample = np.load(sample_files[0])
    print(f"\n📋 Sample file: {os.path.basename(sample_files[0])}")
    print(f"   features shape: {sample['features'].shape}  ← (num_faces, 4096)")
    print(f"   boxes shape:    {sample['boxes'].shape}")
    print(f"\n✅ Features ready để đưa vào ConGNN model!")