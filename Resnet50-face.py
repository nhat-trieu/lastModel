import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import glob
from tqdm.auto import tqdm
import warnings

# ✅ FIX: Tắt DecompressionBomb warning
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIG
# ==========================================
CONFIG = {
    'data_root': '/kaggle/input/groupemow-dataset/GroupEmow',
    'resnet_weights': '/kaggle/input/resnet50-finetuned-groupemow/resnet50_finetuned_groupemow.pth',
    'output_dir': '/kaggle/working/features_congnn_resnet50',
    
    'max_faces': 16,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

print(f"🔧 Device: {CONFIG['device']}")

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("📦 Loading models...")

# === MTCNN for Face Detection ===
try:
    from facenet_pytorch import MTCNN
except:
    os.system('pip install facenet_pytorch -q')
    from facenet_pytorch import MTCNN

mtcnn = MTCNN(
    keep_all=True,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.8],
    device=CONFIG['device']
)
print("  ✅ MTCNN loaded")

# === ResNet50 Feature Extractor ===
def build_resnet50_extractor():
    """Load fine-tuned ResNet50 and remove classifier"""
    # Build same architecture as training
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 3)  # Must match training
    
    # Load weights
    if os.path.exists(CONFIG['resnet_weights']):
        model.load_state_dict(torch.load(CONFIG['resnet_weights'], map_location='cpu'))
        print(f"  ✅ Loaded fine-tuned weights from {CONFIG['resnet_weights']}")
    else:
        print(f"  ⚠️ Weights not found! Using ImageNet pretrained")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Remove classifier (keep only feature extractor)
    extractor = nn.Sequential(*list(model.children())[:-1])  # Remove fc layer
    extractor.eval()
    return extractor

resnet_extractor = build_resnet50_extractor().to(CONFIG['device'])
print("  ✅ ResNet50 extractor ready (output: 2048D)")

# ==========================================
# 3. PREPROCESSING
# ==========================================
face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_face(img_rgb, box):
    """Crop and preprocess face for ResNet50"""
    x1, y1, x2, y2 = [int(b) for b in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Crop face
    face = img_rgb[y1:y2, x1:x2]
    face_pil = Image.fromarray(face)
    face_tensor = face_transform(face_pil)
    
    return face_tensor

# ==========================================
# 4. EXTRACTION LOOP
# ==========================================
def extract_faces():
    print("="*80)
    print("🔬 EXTRACTING FACE FEATURES WITH RESNET50 (2048D)")
    print("="*80)
    
    splits = ['train', 'val', 'test']
    emotions = ['Negative', 'Neutral', 'Positive']
    
    stats = {
        'total': 0,
        'detected': 0,
        'no_faces': 0,
        'errors': 0
    }
    
    for split in splits:
        # Find split directory
        split_path = None
        for root, dirs, files in os.walk(CONFIG['data_root']):
            if split.lower() in os.path.basename(root).lower():
                lower = [d.lower() for d in os.listdir(root)]
                if 'positive' in lower:
                    split_path = root
                    break
        
        if not split_path:
            print(f"⚠️ Skip {split}")
            continue
        
        print(f"\n📂 Processing {split.upper()}")
        out_dir = os.path.join(CONFIG['output_dir'], 'faces', split)
        
        for emotion in emotions:
            emo_path = next((os.path.join(split_path, d)
                           for d in os.listdir(split_path)
                           if d.lower() == emotion.lower()), None)
            
            if not emo_path:
                continue
            
            os.makedirs(os.path.join(out_dir, emotion), exist_ok=True)
            
            img_files = glob.glob(os.path.join(emo_path, '*'))
            img_files = [f for f in img_files 
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"  > {emotion}: {len(img_files)} images")
            
            for img_path in tqdm(img_files, desc=f"  {emotion}", leave=False):
                stats['total'] += 1
                
                base = os.path.splitext(os.path.basename(img_path))[0]
                dst = os.path.join(out_dir, emotion, base + '.npz')
                
                try:
                    # Read image
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        stats['errors'] += 1
                        continue
                    
                    # Resize if too large (avoid DecompressionBomb)
                    h, w = img_bgr.shape[:2]
                    if h * w > 10000000:  # >10MP
                        scale = (10000000 / (h * w)) ** 0.5
                        new_h, new_w = int(h * scale), int(w * scale)
                        img_bgr = cv2.resize(img_bgr, (new_w, new_h))
                    
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    # Detect faces
                    boxes, probs = mtcnn.detect(img_pil)
                    
                    face_features = []
                    valid_boxes = []
                    
                    if boxes is not None and probs is not None:
                        # Filter by confidence
                        keep = probs > 0.8
                        boxes = boxes[keep]
                        
                        if len(boxes) > 0:
                            # Sort by size
                            areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
                            top_indices = areas.argsort()[::-1][:CONFIG['max_faces']]
                            boxes = boxes[top_indices]
                            
                            # Process faces in batches
                            batch = []
                            for box in boxes:
                                face_tensor = preprocess_face(img_rgb, box)
                                if face_tensor is not None:
                                    batch.append(face_tensor)
                                    valid_boxes.append(box)
                            
                            # Extract features
                            if batch:
                                with torch.no_grad():
                                    batch = torch.stack(batch).to(CONFIG['device'])
                                    features = resnet_extractor(batch)
                                    # Shape: (N, 2048, 1, 1) → (N, 2048)
                                    features = features.squeeze(-1).squeeze(-1)
                                    face_features = features.cpu().numpy()
                                
                                stats['detected'] += 1
                    
                    # Save (even if no faces - important for dataset consistency)
                    if len(face_features) == 0:
                        stats['no_faces'] += 1
                        face_features = np.zeros((1, 2048), dtype=np.float32)
                        valid_boxes = np.zeros((1, 4), dtype=np.float32)
                    
                    np.savez_compressed(
                        dst,
                        features=np.array(face_features, dtype=np.float32),
                        boxes=np.array(valid_boxes, dtype=np.float32)
                    )
                    
                except Exception as e:
                    stats['errors'] += 1
                    print(f"    ❌ Error on {os.path.basename(img_path)}: {e}")
                    continue
    
    # Print statistics
    print("\n" + "="*80)
    print("📊 EXTRACTION STATISTICS")
    print("="*80)
    print(f"Total images processed: {stats['total']}")
    print(f"With faces detected:    {stats['detected']} ({100*stats['detected']/stats['total']:.1f}%)")
    print(f"No faces found:         {stats['no_faces']} ({100*stats['no_faces']/stats['total']:.1f}%)")
    print(f"Errors:                 {stats['errors']} ({100*stats['errors']/stats['total']:.1f}%)")
    print(f"\n💾 Features saved to: {CONFIG['output_dir']}/faces/")
    print("="*80)

# ==========================================
# 5. VALIDATION
# ==========================================
def validate_features():
    """Check quality of extracted features"""
    print("\n🔍 Validating extracted features...")
    
    sample_dir = os.path.join(CONFIG['output_dir'], 'faces', 'train', 'Positive')
    if not os.path.exists(sample_dir):
        print("❌ No features found!")
        return
    
    files = glob.glob(os.path.join(sample_dir, '*.npz'))[:10]
    
    print(f"\nChecking {len(files)} sample files:")
    for f in files:
        data = np.load(f)
        feats = data['features']
        boxes = data['boxes']
        print(f"  {os.path.basename(f)}: {len(feats)} faces, shape={feats.shape}")
        
        # Check for issues
        if np.isnan(feats).any():
            print(f"    ⚠️ Contains NaN!")
        if feats.std() < 0.01:
            print(f"    ⚠️ Low variance!")
    
    print("\n✅ Feature validation complete!")

# ==========================================
# 6. COMPRESS TO ZIP
# ==========================================
def compress_to_zip():
    """Compress extracted features to ZIP for easy dataset creation"""
    import shutil
    from pathlib import Path
    
    print("\n" + "="*80)
    print("🗜️ COMPRESSING FEATURES TO ZIP")
    print("="*80)
    
    source_folder = CONFIG['output_dir']
    output_zip = '/kaggle/working/face_features_resnet50'
    
    # Check if folder exists
    if not os.path.exists(source_folder):
        print(f"❌ ERROR: Folder not found: {source_folder}")
        return
    
    # Count files and calculate size
    total_files = sum(1 for _ in Path(source_folder).rglob('*') if _.is_file())
    total_size = sum(f.stat().st_size for f in Path(source_folder).rglob('*') if f.is_file())
    size_mb = total_size / (1024**2)
    
    print(f"\n📂 Source: {source_folder}")
    print(f"   Files: {total_files}")
    print(f"   Size: {size_mb:.2f} MB")
    
    # Compress
    print(f"\n🗜️ Compressing to: {output_zip}.zip")
    print("   This may take 1-3 minutes...")
    
    try:
        shutil.make_archive(output_zip, 'zip', source_folder)
        
        # Verify
        zip_file = f"{output_zip}.zip"
        if os.path.exists(zip_file):
            zip_size = os.path.getsize(zip_file) / (1024**2)
            compression_ratio = (1 - zip_size/size_mb) * 100 if size_mb > 0 else 0
            
            print(f"\n✅ Compression complete!")
            print(f"📦 Output: {zip_file}")
            print(f"   Size: {zip_size:.2f} MB")
            print(f"   Compression: {compression_ratio:.1f}%")
            
            # Preview structure
            print(f"\n📁 Dataset structure:")
            print("face_features_resnet50.zip")
            print("└── faces/")
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(source_folder, 'faces', split)
                if os.path.exists(split_path):
                    for emotion in os.listdir(split_path):
                        emotion_path = os.path.join(split_path, emotion)
                        if os.path.isdir(emotion_path):
                            count = len([f for f in os.listdir(emotion_path) if f.endswith('.npz')])
                            print(f"    ├── {split}/{emotion}/ ({count} files)")
            
            return True
        else:
            print("\n❌ ERROR: ZIP file not created!")
            return False
            
    except Exception as e:
        print(f"\n❌ Compression failed: {e}")
        return False

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Step 1: Extract features
    extract_faces()
    
    # Step 2: Validate
    validate_features()
    
    # Step 3: Compress to ZIP
    compress_success = compress_to_zip()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ALL STEPS COMPLETE!")
    print("="*80)
    
    if compress_success:
        print("\n📋 NEXT STEPS:")
        print("1. Save this notebook (Quick Save)")
        print("2. Go to Output tab → Find 'face_features_resnet50.zip'")
        print("3. Click 'Create New Dataset'")
        print("4. Name: 'face-features-resnet50-groupemow'")
        print("5. Update ConGNN config:")
        print("   CONFIG = {")
        print("       'face_dir': '/kaggle/input/face-features-resnet50-groupemow',")
        print("       'face_dim': 2048,  # Changed from 4096")
        print("       'hidden_dim': 512,")
        print("       # ... rest unchanged")
        print("   }")
        print("\n🎯 Expected results after training ConGNN:")
        print("   Face Accuracy: 70-75%")
        print("   Context Accuracy: 80-85%")
        print("   Whole Accuracy: 87-89%")
    else:
        print("\n⚠️ Compression failed. You can:")
        print("1. Run the compress_to_zip() function again")
        print("2. Or manually compress /kaggle/working/features_congnn_resnet50/")
    
    print("="*80)