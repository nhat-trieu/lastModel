import os
import glob
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import roi_align


DATA_ROOT   = '/kaggle/input/groupemowfull/GroupEmoW'
OUTPUT_ROOT = '/kaggle/working/scene_features_final'

# SỬA DÒNG NÀY: Trỏ tới file .pth trong thư mục input của Kaggle
CKPT_PATH   = '/kaggle/input/datasets/trieung11/resnet50fpn-groupemow/resnet50fpn_groupemow.pth'
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP   = {'negative': 0, 'neutral': 1, 'positive': 2}
NUM_CLASSES = 3
SPLITS      = ['train', 'val', 'test']
FINETUNE_IMG_SIZE = 800
EXTRACT_BS  = 8
ROI_SIZE    = 7
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. KHAI BÁO MODEL (Y xì đúc để đọc được file .pth)
class ResNet50FPN_GER(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fpn = resnet_fpn_backbone(backbone_name='resnet50', weights=None, returned_layers=[1, 2, 3, 4], trainable_layers=5)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.scene_proj = nn.Sequential(nn.Linear(256 * 4, 2048), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.classifier = nn.Linear(2048, num_classes)

    def extract_scene_feat(self, images):
        B, C, H, W = images.shape
        fpn_feats = self.fpn(images)
        rois = torch.zeros(B, 5, device=images.device)
        rois[:, 0] = torch.arange(B, dtype=torch.float32, device=images.device)
        rois[:, 3] = float(W); rois[:, 4] = float(H)
        pooled_levels = [self.global_pool(roi_align(fpn_feats[k], rois, (ROI_SIZE, ROI_SIZE), 1.0 / s, True)).flatten(1) 
                         for k, s in zip(['0', '1', '2', '3'], [4, 8, 16, 32])]
        return self.scene_proj(torch.cat(pooled_levels, dim=1))

eval_transform = T.Compose([
    T.Resize((FINETUNE_IMG_SIZE, FINETUNE_IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_output_npy_path(img_path):
    if 'GroupEmoW/' in img_path: rel = img_path.split('GroupEmoW/')[-1]
    else: rel = img_path.split(DATA_ROOT)[-1].lstrip('/')
    return os.path.join(OUTPUT_ROOT, 'scenes', os.path.splitext(rel)[0] + '.npy')

# 2. HÀM EXTRACT VÀ NÉN
def main():
    print(f"📦 Đang load bộ tạ tinh hoa từ: {CKPT_PATH}")
    model = ResNet50FPN_GER(NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    
    # Mở rương lấy đồ
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Đã nạp tạ ở Epoch: {checkpoint.get('epoch', 'Unknown')} | Best Acc: {checkpoint.get('best_acc', 'Unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("\n🚀 BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG...")
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
            if not buffer_imgs: return
            try:
                tensors = torch.stack([eval_transform(img) for img in buffer_imgs]).to(DEVICE)
                with torch.no_grad(): feats = model.extract_scene_feat(tensors)
                for feat, out_path in zip(feats.cpu().numpy(), buffer_paths):
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, feat.astype(np.float32))
            except Exception as e: print(f"Lỗi nhẹ bỏ qua: {e}")

        for img_path in tqdm(all_imgs, desc=f"  Extracting {split}"):
            out_path = get_output_npy_path(img_path)
            if os.path.exists(out_path): continue
            try:
                buffer_imgs.append(Image.open(img_path).convert('RGB'))
                buffer_paths.append(out_path)
            except Exception:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, np.zeros(2048, dtype=np.float32))
                
            if len(buffer_imgs) >= EXTRACT_BS:
                flush_buffer()
                buffer_imgs.clear(); buffer_paths.clear()
        flush_buffer()

    print("\n📦 Đang nén thành file ZIP...")
    zip_path = '/kaggle/working/scene_features_final.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_ROOT):
            for file in files:
                abs_path = os.path.join(root, file)
                zf.write(abs_path, os.path.relpath(abs_path, os.path.dirname(OUTPUT_ROOT)))
    print(f"🎉 XONG! File ZIP đã sẵn sàng ở phần Output: {zip_path}")

if __name__ == '__main__':
    main()