import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import glob
from tqdm.auto import tqdm
from torchvision import transforms
import timm  # Đảm bảo đã cài: pip install timm
import shutil
import warnings

warnings.filterwarnings('ignore')

# 1. Cấu hình
CONFIG = {
    'data_root': '/kaggle/input/mydata', 
    'scene_weights': '/kaggle/input/resnet50-scene-combined/resnet50_scene_combined.pth', 
    'output_dir': '/kaggle/working/features_scene_seresnet',
    'zip_name': '/kaggle/working/SiteGroEmo_scene_features',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# 2. Hàm khởi tạo Model SE-ResNet50 chuẩn bài báo
def get_congnn_scene_extractor():
    print(f"🚀 Khởi tạo SE-ResNet50 (Chuẩn ConGNN)...")
    
    # Bước A: Khởi tạo model SE-ResNet50 với 3 lớp đầu ra (như lúc bạn fine-tune)
    # Dùng seresnet50 từ timm để có cơ chế Squeeze-and-Excitation
    model = timm.create_model('seresnet50', pretrained=False, num_classes=3)
    
    # Bước B: Load weights của bạn
    if os.path.exists(CONFIG['scene_weights']):
        print(f"✅ Đang nạp weights từ: {CONFIG['scene_weights']}")
        checkpoint = torch.load(CONFIG['scene_weights'], map_location=CONFIG['device'])
        
        # Xử lý nếu checkpoint chứa 'state_dict'
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        try:
            model.load_state_dict(state_dict)
            print("  ✨ Nạp weights thành công!")
        except RuntimeError as e:
            print(f"  ⚠️ Cảnh báo: Kiến trúc weights có thể khác SE-ResNet. Đang thử nạp mềm (strict=False)...")
            model.load_state_dict(state_dict, strict=False)
    else:
        print("  ❌ Không tìm thấy weights. Sử dụng ImageNet mặc định.")
        model = timm.create_model('seresnet50', pretrained=True, num_classes=3)

    # Bước C: Chuyển thành Extractor (Lấy 2048 đặc trưng)
    # num_classes=0 trong timm sẽ tự động xóa lớp FC và trả về feature vector
    model.reset_classifier(num_classes=0) 
    model = model.to(CONFIG['device'])
    model.eval()
    return model

# 3. Thực thi trích xuất
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
        # Tìm đường dẫn thực tế (hỗ trợ cấu trúc thư mục lồng nhau)
        split_path = None
        for root, dirs, files in os.walk(CONFIG['data_root']):
            if split.lower() == os.path.basename(root).lower():
                split_path = root
                break
        
        if not split_path: continue
            
        print(f"\n📂 Đang xử lý tập {split.upper()}...")
        out_root = os.path.join(CONFIG['output_dir'], 'scenes', split)
        
        for emotion in emotions:
            emo_dir = next((os.path.join(split_path, d) for d in os.listdir(split_path) 
                          if d.lower() == emotion.lower()), None)
            if not emo_dir: continue
            
            os.makedirs(os.path.join(out_root, emotion), exist_ok=True)
            img_files = glob.glob(os.path.join(emo_dir, '*'))
            img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_path in tqdm(img_files, desc=f" {emotion}", leave=False):
                base = os.path.splitext(os.path.basename(img_path))[0]
                dst_path = os.path.join(out_root, emotion, base + '.npy')
                
                if os.path.exists(dst_path): continue
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = scene_tf(img).unsqueeze(0).to(CONFIG['device'])
                    
                    with torch.no_grad():
                        feat = extractor(img_tensor)
                        # feat lúc này đã là (1, 2048) nhờ model.reset_classifier(0)
                        feat_np = feat.cpu().numpy()[0]
                    
                    np.save(dst_path, feat_np)
                except Exception: continue

    print("\n📦 Đang nén kết quả...")
    shutil.make_archive(CONFIG['zip_name'], 'zip', CONFIG['output_dir'])
    print(f"✅ Hoàn tất! File lưu tại: {CONFIG['zip_name']}.zip")

if __name__ == "__main__":
    run_scene_extraction()