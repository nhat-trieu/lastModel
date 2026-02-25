# finetune_vggface_groupemow.py
"""
Fine-tune VGGFace on GroupEmoW + SiteGroEmo Face Crops
=======================================================
Task: 3-class emotion classification (Positive, Neutral, Negative)
Backbone: VGGFace (4096-dim features)
Output: Fine-tuned .pth weights for feature extraction

Author: [Your Name]
Date: 2024
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import glob
import warnings
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIG
# ==========================================
CONFIG = {
    # Paths
    'data_root': '/kaggle/input/datasets/trieung11/face-crops-combined',
    'output_dir': '/kaggle/working/vggface_finetuned',
    'checkpoint_dir': '/kaggle/working/checkpoints',
    
    # Training params
    'batch_size': 64,
    'num_epochs': 30,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'scheduler_patience': 3,
    'early_stop_patience': 7,
    
    # Model params
    'num_classes': 3,  # Positive, Neutral, Negative
    'freeze_backbone': False,  # Fine-tune toàn bộ
    'dropout': 0.5,
    
    # Data params
    'img_size': 224,
    'num_workers': 2,
    
    # Device
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 42
}

# Set seed for reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

print(f"🔧 Device: {CONFIG['device']}")
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

# ==========================================
# 2. VGGFACE MODEL
# ==========================================
class VGG_16(nn.Module):
    """
    VGGFace architecture (VGG-16 variant)
    Modified for 3-class emotion classification
    """
    def __init__(self, num_classes=3, dropout=0.5):
        super(VGG_16, self).__init__()
        
        # ==========================================
        # CONVOLUTIONAL LAYERS (FEATURE EXTRACTOR)
        # ==========================================
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
        
        # ==========================================
        # FULLY CONNECTED LAYERS
        # ==========================================
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)  # ✅ Emotion classifier
        
        # Activations
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)
        
        # Conv block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)
        
        # Conv block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)
        
        # Conv block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)
        
        # Conv block 5
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))  # fc7 = 4096-dim features
        x = self.dropout(x)
        x = self.fc8(x)  # Classifier output
        
        return x
    
    def get_features(self, x):
        """
        Extract 4096-dim features from fc7
        (use this after fine-tuning for feature extraction)
        """
        # Forward through conv layers
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)
        
        # FC up to fc7
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))  # ✅ Return here (4096-dim)
        
        return x

def load_pretrained_vggface():
    """
    Load VGGFace pre-trained weights
    Download from: http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz
    """
    print("📦 Loading VGGFace pre-trained model...")
    model = VGG_16(num_classes=3, dropout=CONFIG['dropout'])
    
    # Try to load Oxford VGGFace weights
    pretrained_path = '/kaggle/working/VGG_FACE.pth'
    
    if os.path.exists(pretrained_path):
        print(f"✅ Loading pre-trained weights from: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        
        # Get model state dict
        model_dict = model.state_dict()
        
        # Filter out fc8 (classifier layer - we'll train from scratch)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and not k.startswith('fc8')}
        
        # Update model dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print(f"  ✨ Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
        print(f"  🔧 fc8 (classifier) initialized randomly for fine-tuning")
    else:
        print(f"⚠️ WARNING: Pre-trained weights not found!")
        print(f"  Searched at: {pretrained_path}")
        print(f"  Training from SCRATCH (not recommended)")
        print(f"  Please upload VGGFace pre-trained weights to Kaggle dataset")
    
    return model

# ==========================================
# 3. DATASET
# ==========================================
class FaceCropsDataset(Dataset):
    """
    Dataset for face crops from GroupEmoW + SiteGroEmo
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir: /kaggle/input/.../face-crops-combined
        split: 'train', 'val', or 'test'
        """
        self.transform = transform
        self.samples = []
        self.emotion_to_idx = {
            'Positive': 0,
            'Neutral': 1,
            'Negative': 2
        }
        
        # Scan both datasets
        datasets = ['groupemow', 'sitegroemo']
        emotions = ['Positive', 'Neutral', 'Negative']
        
        print(f"📂 Loading {split} split from {root_dir}...")
        
        for dataset in datasets:
            for emotion in emotions:
                emotion_dir = os.path.join(root_dir, dataset, split, emotion)
                
                if not os.path.exists(emotion_dir):
                    print(f"  ⚠️ Not found: {emotion_dir}")
                    continue
                
                # Get all image files
                img_files = glob.glob(os.path.join(emotion_dir, '*'))
                img_files = [f for f in img_files 
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                # Add to samples
                for img_path in img_files:
                    self.samples.append({
                        'path': img_path,
                        'label': self.emotion_to_idx[emotion],
                        'emotion': emotion,
                        'dataset': dataset
                    })
                
                print(f"  ✅ {dataset}/{split}/{emotion}: {len(img_files)} images")
        
        print(f"  📊 Total {split} samples: {len(self.samples)}\n")
        
        # Class distribution
        labels = [s['label'] for s in self.samples]
        for emotion, idx in self.emotion_to_idx.items():
            count = labels.count(idx)
            print(f"    {emotion}: {count} ({100*count/len(labels):.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            img = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (224, 224), color='black')
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        return img, sample['label']

# ==========================================
# 4. DATA TRANSFORMS
# ==========================================
# VGGFace standard transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.367035294117647, 0.41083294117647057, 0.5066129411764705],
        std=[1, 1, 1]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.367035294117647, 0.41083294117647057, 0.5066129411764705],
        std=[1, 1, 1]
    )
])

# ==========================================
# 5. TRAINING FUNCTIONS
# ==========================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, dataloader, criterion, device):
    """
    Validate model
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Neutral', 'Negative'],
                yticklabels=['Positive', 'Neutral', 'Negative'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved to: {save_path}")

# ==========================================
# 6. MAIN TRAINING LOOP
# ==========================================
def train_vggface():
    """
    Main training function
    """
    print("="*80)
    print("🚀 FINE-TUNING VGGFACE ON GROUPEMOW + SITEGROEMO")
    print("="*80)
    
    # ==========================================
    # Load datasets
    # ==========================================
    print("\n[STEP 1/5] Loading datasets...")
    train_dataset = FaceCropsDataset(
        CONFIG['data_root'], 
        split='train', 
        transform=train_transform
    )
    val_dataset = FaceCropsDataset(
        CONFIG['data_root'], 
        split='val', 
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    print(f"\n📊 Dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # ==========================================
    # Load model
    # ==========================================
    print("\n[STEP 2/5] Loading VGGFace model...")
    model = load_pretrained_vggface()
    model = model.to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # ==========================================
    # Loss and optimizer
    # ==========================================
    print("\n[STEP 3/5] Setting up training...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=CONFIG['scheduler_patience']
    )
    
    print(f"  Loss: CrossEntropyLoss")
    print(f"  Optimizer: Adam (lr={CONFIG['lr']})")
    print(f"  Scheduler: ReduceLROnPlateau")
    print(f"  Early stopping patience: {CONFIG['early_stop_patience']}")
    
    # ==========================================
    # Training loop
    # ==========================================
    print("\n[STEP 4/5] Training...")
    print("="*80)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\n📅 Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device']
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, CONFIG['device']
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': CONFIG
            }
            
            save_path = os.path.join(CONFIG['checkpoint_dir'], 'vggface_best.pth')
            torch.save(checkpoint, save_path)
            print(f"✅ Saved best model (Val Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
            
            # Plot confusion matrix for best model
            cm_path = os.path.join(CONFIG['output_dir'], 'confusion_matrix_best.png')
            plot_confusion_matrix(val_labels, val_preds, cm_path)
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{CONFIG['early_stop_patience']}")
        
        # Early stopping
        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"\n⛔ Early stopping triggered!")
            break
    
    # ==========================================
    # Save final results
    # ==========================================
    print("\n[STEP 5/5] Saving final results...")
    
    # Save final model
    final_path = os.path.join(CONFIG['output_dir'], 'vggface_finetuned_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"  💾 Final model saved to: {final_path}")
    
    # Save best model to output (for easy download)
    best_output_path = os.path.join(CONFIG['output_dir'], 'vggface_finetuned_best.pth')
    best_checkpoint = torch.load(os.path.join(CONFIG['checkpoint_dir'], 'vggface_best.pth'))
    torch.save(best_checkpoint['state_dict'], best_output_path)
    print(f"  🏆 Best model saved to: {best_output_path}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Curve')
    plt.grid(True)
    
    plt.tight_layout()
    curves_path = os.path.join(CONFIG['output_dir'], 'training_curves.png')
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"  📈 Training curves saved to: {curves_path}")
    
    # Print final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"🏆 Best Validation F1 Score: {best_val_f1:.4f}")
    print(f"\n📂 Output files:")
    print(f"  - vggface_finetuned_best.pth (use this for extraction!)")
    print(f"  - vggface_finetuned_final.pth")
    print(f"  - confusion_matrix_best.png")
    print(f"  - training_curves.png")
    print("="*80)
    
    return model, history

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Train
    model, history = train_vggface()
    
    print("\n🎉 All done! Download 'vggface_finetuned_best.pth' from Output tab")
