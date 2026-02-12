import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import glob
from tqdm.auto import tqdm
from torchvision import models, transforms
from facenet_pytorch import MTCNN 
import shutil
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
CONFIG = {
    'data_root': '/kaggle/input/groupemowfull',
    'face_weights': '/kaggle/input/resnet50-face-best-112-v2/resnet50_face_best_112_v2.pth',
    'output_dir': '/kaggle/working/features_congnn_priors',
    'min_conf': 0.8,
    'min_face_size': 30,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==========================================
# 2. KHỞI TẠO MODEL (GIỮ KIẾN TRÚC THEO WEIGHTS 112x112)
# ==========================================
def build_resnet50_extractor(weights_path):
    # Kiến trúc này dành riêng cho weights face fine-tuned của ông
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 3) # Giả định weights cũ có lớp cuối là 3 emotions
    
    if os.path.exists(weights_path):
        print(f"✅ Đang nạp trọng số Face fine-tuned từ: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=CONFIG['device']))
    else:
        print("⚠️ CẢNH BÁO: Không thấy file weights! Model sẽ chạy với random weights.")

    # Bỏ lớp FC cuối để lấy 2048 đặc trưng (Global Average Pooling output)
    extractor = nn.Sequential(*list(model.children())[:-1])
    extractor = extractor.to(CONFIG['device'])
    extractor.eval()
    return extractor

# ==========================================
# 3. QUY TRÌNH TRÍCH XUẤT (ALL FACES + PRIORS)
# ==========================================
def run_extraction_full_priors():
    # Khởi tạo MTCNN lấy toàn bộ mặt
    mtcnn = MTCNN(
        keep_all=True, 
        device=CONFIG['device'], 
        min_face_size=CONFIG['min_face_size'],
        post_process=False
    )
    
    face_extractor = build_resnet50_extractor(CONFIG['face_weights'])
    
    # Image Net normalization (thông số chuẩn cho ResNet)
    face_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    splits = ['train', 'val', 'test']
    emotions = ['Negative', 'Neutral', 'Positive']

    for split in splits:
        # Tìm đường dẫn split (không phân biệt hoa thường)
        split_path = None
        for root, dirs, files in os.walk(CONFIG['data_root']):
            if split.lower() == os.path.basename(root).lower():
                split_path = root
                break
        if not split_path: continue

        print(f"\n👤 Đang xử lý FACES tập {split.upper()}...")
        
        for emotion in emotions:
            emo_dir = os.path.join(split_path, emotion)
            if not os.path.exists(emo_dir): continue
            
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
                    
                    # 1. Phát hiện mặt và lấy xác suất
                    boxes, probs = mtcnn.detect(img_pil)
                    
                    if boxes is not None:
                        # Lọc theo threshold
                        mask = (probs > CONFIG['min_conf'])
                        boxes = boxes[mask]
                        probs = probs[mask]
                        
                        if len(boxes) > 0:
                            # Tính diện tích
                            areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
                            
                            # Sắp xếp mặt to lên trước
                            sort_idx = np.argsort(areas)[::-1]
                            boxes = boxes[sort_idx]
                            probs = probs[sort_idx]
                            areas = areas[sort_idx]
                            
                            face_batch = []
                            valid_boxes = []
                            valid_probs = []
                            valid_areas = []
                            
                            for i, box in enumerate(boxes):
                                x1, y1, x2, y2 = [int(b) for b in box]
                                # Crop an toàn
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    face_crop = img_pil.crop((x1, y1, x2, y2))
                                    face_batch.append(face_tf(face_crop))
                                    valid_boxes.append([x1, y1, x2, y2])
                                    valid_probs.append(probs[i])
                                    valid_areas.append(areas[i])
                            
                            if len(face_batch) > 0:
                                # 2. Trích xuất đặc trưng
                                with torch.no_grad():
                                    batch_tensor = torch.stack(face_batch).to(CONFIG['device'])
                                    feats = face_extractor(batch_tensor)
                                    f_feats = torch.flatten(feats, 1).cpu().numpy()
                                
                                # 3. Tính Priors (Area * Confidence)
                                valid_areas = np.array(valid_areas, dtype=np.float32)
                                valid_probs = np.array(valid_probs, dtype=np.float32)
                                priors = valid_areas * valid_probs
                                
                                # Chuẩn hóa priors về [0, 1] trong nội bộ ảnh
                                if priors.max() > 0:
                                    priors = priors / (priors.max() + 1e-9)

                                # Lưu đầy đủ thông tin
                                np.savez_compressed(dst_file, 
                                    features=f_feats.astype(np.float32),
                                    boxes=np.array(valid_boxes, dtype=np.float32),
                                    confidences=valid_probs,
                                    areas=valid_areas,
                                    priors=priors
                                )
                                continue

                    # Trường hợp không có mặt nào thỏa mãn
                    np.savez_compressed(dst_file, 
                        features=np.zeros((0, 2048), dtype=np.float32),
                        boxes=np.zeros((0, 4), dtype=np.float32),
                        confidences=np.zeros(0, dtype=np.float32),
                        areas=np.zeros(0, dtype=np.float32),
                        priors=np.zeros(0, dtype=np.float32)
                    )

                except Exception as e:
                    print(f"❌ Lỗi ảnh {base}: {e}")
                    continue

    # ==========================================
    # 4. NÉN FILE SAU KHI HOÀN THÀNH
    # ==========================================
    print("\n⏳ Đang nén file kết quả...")
    zip_path = '/kaggle/working/faces_priors_final'
    shutil.make_archive(zip_path, 'zip', CONFIG['output_dir'])
    print(f"✅ HOÀN TẤT! File nén tại: {zip_path}.zip")

if __name__ == "__main__":
    run_extraction_full_priors()