# ==========================================
# BƯỚC 2.5: TRÍCH XUẤT ĐẶC TRƯNG OBJECT (NPZ + BOXES + ZIP)
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
    'data_root': '/kaggle/input/mydata',
    'base_output': '/kaggle/working/features_congnn',
    'output_dir': '/kaggle/working/features_congnn/SiteGroEmo_objects',
    'max_objects': 10,
    'conf_threshold': 0.5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# 2. Chuẩn bị Model
print("🚀 Đang tải model Object Detection & Feature Extraction...")
detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(CONFIG['device'])
detector.eval()

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
def run_object_extraction():
    print(f"🔧 Device: {CONFIG['device']}")
    splits = ['train', 'val', 'test']
    emotions = ['Negative', 'Neutral', 'Positive']
    
    for split in splits:
        real_path = find_emotion_data_path(CONFIG['data_root'], split)
        if not real_path: continue
        
        print(f"\n📦 Đang xử lý OBJECTS: {split.upper()}...")
        
        for emotion in emotions:
            emo_dir = next((os.path.join(real_path, d) for d in os.listdir(real_path) 
                          if d.lower() == emotion.lower()), None)
            if not emo_dir: continue
            
            save_path = os.path.join(CONFIG['output_dir'], split, emotion)
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
                    
                    img_tensor = transforms.ToTensor()(img_pil).to(CONFIG['device'])
                    with torch.no_grad():
                        prediction = detector([img_tensor])[0]
                    
                    boxes = prediction['boxes']
                    scores = prediction['scores']
                    
                    obj_feats, valid_boxes = [], []
                    
                    if len(boxes) > 0:
                        keep = scores > CONFIG['conf_threshold']
                        keep_boxes = boxes[keep][:CONFIG['max_objects']]
                        
                        batch_crops = []
                        for box in keep_boxes:
                            x1, y1, x2, y2 = box.tolist()
                            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
                            
                            if x2 > x1 and y2 > y1:
                                crop = img_pil.crop((x1, y1, x2, y2))
                                batch_crops.append(obj_transform(crop))
                                valid_boxes.append([x1, y1, x2, y2])
                        
                        if batch_crops:
                            batch_tensor = torch.stack(batch_crops).to(CONFIG['device'])
                            with torch.no_grad():
                                obj_feats = feature_extractor(batch_tensor).cpu().numpy()
                    
                    np.savez_compressed(dst_file, 
                                      features=np.array(obj_feats, dtype=np.float32),
                                      boxes=np.array(valid_boxes, dtype=np.float32))
                except: continue

    # 4. Nén kết quả
    print("\n📦 Đang nén folder OBJECTS...")
    
    # Lấy tên thư mục con cuối cùng từ đường dẫn output_dir
    folder_to_zip = os.path.basename(CONFIG['output_dir']) # Kết quả là 'SiteGroEmo_objects'
    
    shutil.make_archive('/kaggle/working/objects_extracted', 'zip', 
                        root_dir=CONFIG['base_output'], 
                        base_dir=folder_to_zip)
    
    zip_path = '/kaggle/working/objects_extracted.zip'
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024*1024)
        print(f"✅ Xong! File nặng: {size_mb:.2f} MB")

if __name__ == "__main__":
    run_extraction_code = run_object_extraction()