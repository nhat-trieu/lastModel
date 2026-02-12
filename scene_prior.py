import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import glob
from tqdm.auto import tqdm
from torchvision import transforms
import timm 
import shutil
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
CONFIG = {
    'data_root': '/kaggle/input/groupemowfull', 
    'scene_weights': '/kaggle/input/resnet50-scene-combined/resnet50_scene_combined.pth', 
    'output_dir': '/kaggle/working/features_congnn/scenes', # Thư mục đồng bộ với các phần khác
    'zip_name': '/kaggle/working/scene_features_final',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==========================================
# 2. HÀM KHỞI TẠO MODEL SE-RESNET50 (CHUẨN IPR-MPNN)
# ==========================================
def get_congnn_scene_extractor():
    print(f"🚀 Khởi tạo SE-ResNet50 Extractor...")
    
    # Khởi tạo model seresnet50 (Squeeze-and-Excitation giúp focus bối cảnh tốt hơn)
    model = timm.create_model('seresnet50', pretrained=False, num_classes=3)
    
    # Load weights custom của ông
    if os.path.exists(CONFIG['scene_weights']):
        print(f"✅ Đang nạp weights từ: {CONFIG['scene_weights']}")
        checkpoint = torch.load(CONFIG['scene_weights'], map_location=CONFIG['device'])
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        try:
            model.load_state_dict(state_dict)
            print("  ✨ Nạp weights thành công!")
        except RuntimeError:
            print(f"  ⚠️ Đang nạp weights với strict=False...")
            model.load_state_dict(state_dict, strict=False)
    else:
        print("  ❌ Không thấy weights. Sử dụng ImageNet mặc định.")
        model = timm.create_model('seresnet50', pretrained=True, num_classes=3)

    # Xóa lớp phân loại để lấy feature vector 2048 chiều
    model.reset_classifier(num_classes=0) 
    model = model.to(CONFIG['device'])
    model.eval()
    return model

# ==========================================
# 3. THỰC THI TRÍCH XUẤT (SAVE NPZ + PRIORS)
# ==========================================
def run_scene_extraction():
    extractor = get_congnn_scene_extractor()
    
    # Transform chuẩn của SE-ResNet
    scene_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    splits = ['train', 'val', 'test']
    emotions = ['Negative', 'Neutral', 'Positive']
    
    for split in splits:
        split_path = None
        for root, dirs, files in os.walk(CONFIG['data_root']):
            if split.lower() == os.path.basename(root).lower():
                split_path = root
                break
        
        if not split_path: continue
            
        print(f"\n📂 Đang xử lý SCENE tập {split.upper()}...")
        out_root = os.path.join(CONFIG['output_dir'], split)
        
        for emotion in emotions:
            emo_dir = next((os.path.join(split_path, d) for d in os.listdir(split_path) 
                          if d.lower() == emotion.lower()), None)
            if not emo_dir: continue
            
            save_path = os.path.join(out_root, emotion.lower())
            os.makedirs(save_path, exist_ok=True)
            
            img_files = glob.glob(os.path.join(emo_dir, '*'))
            img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_path in tqdm(img_files, desc=f" {emotion}", leave=False):
                base = os.path.splitext(os.path.basename(img_path))[0]
                dst_path = os.path.join(save_path, base + '.npz') # CHUYỂN SANG .NPZ
                
                if os.path.exists(dst_path): continue
                
                try:
                    img_pil = Image.open(img_path).convert('RGB')
                    width, height = img_pil.size
                    
                    # A. Trích xuất đặc trưng hình ảnh
                    img_tensor = scene_tf(img_pil).unsqueeze(0).to(CONFIG['device'])
                    with torch.no_grad():
                        feat = extractor(img_tensor)
                        feat_np = feat.cpu().numpy() # Shape (1, 2048)
                    
                    # B. Thiết lập thông tin Cấu trúc (Prior) cho Scene node
                    # Box là toàn bộ ảnh: [x1, y1, x2, y2]
                    box = np.array([[0, 0, width, height]], dtype=np.float32)
                    # Độ tin cậy cho Scene node mặc định là 1.0
                    conf = np.array([1.0], dtype=np.float32)
                    # Diện tích toàn ảnh
                    area = np.array([width * height], dtype=np.float32)
                    # Prior cực kỳ quan trọng cho IPR-MPNN: Scene luôn là 1.0
                    prior = np.array([1.0], dtype=np.float32)
                    
                    # C. Lưu NPZ nén (Đồng bộ với Face và Object)
                    np.savez_compressed(dst_path, 
                        features=feat_np.astype(np.float32),
                        boxes=box,
                        confidences=conf,
                        areas=area,
                        priors=prior
                    )
                except Exception: 
                    continue

    # ==========================================
    # 4. NÉN KẾT QUẢ
    # ==========================================
    print("\n📦 Đang nén kết quả...")
    # Nén folder chứa các folder con train/val/test
    shutil.make_archive(CONFIG['zip_name'], 'zip', CONFIG['output_dir'])
    print(f"✅ Hoàn tất! File lưu tại: {CONFIG['zip_name']}.zip")

if __name__ == "__main__":
    run_scene_extraction()