# ==========================================
# BƯỚC 2.5: TRÍCH XUẤT OBJECTS + PRIORS (NPZ + BOXES + ZIP)
# ==========================================
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import glob
from tqdm.auto import tqdm
import warnings
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
import timm
import shutil

warnings.filterwarnings('ignore')

# 1. Cấu hình
CONFIG = {
    'data_root': '/kaggle/input/groupemowfull/GroupEmoW',
    'base_output': '/kaggle/working/features_congnn',
    'output_dir': '/kaggle/working/features_congnn/objects',
    'max_objects': 10, # Có thể tăng thêm nếu muốn lấy nhiều vật thể hơn
    'conf_threshold': 0.5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# 2. Chuẩn bị Model
print("🚀 Đang tải model Object Detection & Feature Extraction...")
# Detector lấy từ torchvision (Faster R-CNN)
detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(CONFIG['device'])
detector.eval()

# Feature Extractor (SE-ResNet50)
feature_extractor = timm.create_model('seresnet50', pretrained=True, num_classes=0).to(CONFIG['device'])
feature_extractor.eval()

obj_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def find_emotion_data_path(root_path, split_keyword):
    for root, dirs, files in os.walk(root_path):
        if split_keyword.lower() in os.path.basename(root).lower():
            lower_dirs = [d.lower() for d in dirs]
            if 'positive' in lower_dirs and 'negative' in lower_dirs:
                return root
    return None

# 3. Main Loop
def run_object_extraction_with_priors():
    print(f"🔧 Device: {CONFIG['device']}")
    splits = ['train', 'val', 'test']
    emotions = ['Negative', 'Neutral', 'Positive']
    
    for split in splits:
        real_path = find_emotion_data_path(CONFIG['data_root'], split)
        if not real_path: continue
        
        print(f"\n📦 Đang xử lý OBJECTS tập {split.upper()}...")
        
        for emotion in emotions:
            emo_dir = next((os.path.join(real_path, d) for d in os.listdir(real_path) 
                          if d.lower() == emotion.lower()), None)
            if not emo_dir: continue
            
            save_path = os.path.join(CONFIG['output_dir'], split, emotion.lower())
            os.makedirs(save_path, exist_ok=True)
            
            img_files = glob.glob(os.path.join(emo_dir, '*'))
            img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_path in tqdm(img_files, desc=f" {emotion}", leave=False):
                base = os.path.splitext(os.path.basename(img_path))[0]
                dst_file = os.path.join(save_path, base + '.npz')
                
                if os.path.exists(dst_file): continue
                
                try:
                    img_pil = Image.open(img_path).convert('RGB')
                    width, height = img_pil.size
                    
                    # Phát hiện vật thể
                    img_tensor = transforms.ToTensor()(img_pil).to(CONFIG['device'])
                    with torch.no_grad():
                        prediction = detector([img_tensor])[0]
                    
                    boxes = prediction['boxes']
                    scores = prediction['scores']
                    
                    # Lọc theo ngưỡng tin cậy
                    keep = scores > CONFIG['conf_threshold']
                    boxes = boxes[keep]
                    scores = scores[keep]
                    
                    # Chỉ lấy tối đa N vật thể to nhất/quan trọng nhất
                    if len(boxes) > CONFIG['max_objects']:
                        boxes = boxes[:CONFIG['max_objects']]
                        scores = scores[:CONFIG['max_objects']]

                    valid_feats = []
                    valid_boxes = []
                    valid_scores = []
                    valid_areas = []

                    if len(boxes) > 0:
                        batch_crops = []
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.tolist()
                            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
                            
                            if x2 > x1 and y2 > y1:
                                crop = img_pil.crop((x1, y1, x2, y2))
                                batch_crops.append(obj_transform(crop))
                                
                                # Tính diện tích vật thể
                                area = (x2 - x1) * (y2 - y1)
                                valid_boxes.append([x1, y1, x2, y2])
                                valid_scores.append(scores[i].item())
                                valid_areas.append(area)
                        
                        if batch_crops:
                            # Trích xuất đặc trưng hàng loạt
                            batch_tensor = torch.stack(batch_crops).to(CONFIG['device'])
                            with torch.no_grad():
                                # SE-ResNet50 trích xuất ra vector (num_objs, 2048)
                                feats = feature_extractor(batch_tensor)
                                valid_feats = feats.cpu().numpy()

                            # Tính toán Priors cho Object
                            valid_areas = np.array(valid_areas, dtype=np.float32)
                            valid_scores = np.array(valid_scores, dtype=np.float32)
                            priors = valid_areas * valid_scores
                            
                            # Chuẩn hóa Prior về [0, 1]
                            if priors.max() > 0:
                                priors = priors / (priors.max() + 1e-9)
                            
                            np.savez_compressed(dst_file, 
                                features=valid_feats.astype(np.float32),
                                boxes=np.array(valid_boxes, dtype=np.float32),
                                confidences=valid_scores.astype(np.float32),
                                areas=valid_areas,
                                priors=priors.astype(np.float32))
                        else:
                            # Không có crop nào hợp lệ
                            self_save_empty(dst_file)
                    else:
                        # Không tìm thấy vật thể
                        self_save_empty(dst_file)
                except Exception as e:
                    # print(f"Lỗi: {e}")
                    continue

    # 4. Nén folder kết quả
    print("\n📦 Đang nén folder OBJECTS...")
    zip_path = '/kaggle/working/objects_features_priors'
    shutil.make_archive(zip_path, 'zip', 
                        root_dir=CONFIG['base_output'], 
                        base_dir='objects')
    print(f"✅ Xong! File nén tại: {zip_path}.zip")

def self_save_empty(dst_file):
    """Hàm phụ để lưu file rỗng khi không detect được gì"""
    np.savez_compressed(dst_file, 
        features=np.zeros((0, 2048), dtype=np.float32),
        boxes=np.zeros((0, 4), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        areas=np.zeros(0, dtype=np.float32),
        priors=np.zeros(0, dtype=np.float32))

if __name__ == "__main__":
    run_object_extraction_with_priors()