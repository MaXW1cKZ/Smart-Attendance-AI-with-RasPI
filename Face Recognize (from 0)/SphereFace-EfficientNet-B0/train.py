import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# Import models from model.py
from model import FaceRecognitionModel

# --- Configuration ---
DATA_DIR = 'dataset' # ชี้ไปที่โฟลเดอร์แม่ของ train/test
MODEL_SAVE_DIR = 'saved_model'
MODEL_NAME = 'sphereface_efficientnet_b0_from_scratch.pth'
CLASS_MAPPING_NAME = 'class_mapping.json'
PLOT_SAVE_PATH = 'training_history_sphereface_from_scratch.png'


# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 150 # ตั้งไว้สูงเพื่อเทรนจาก 0
LEARNING_RATE = 0.0001 # LR อาจจะน้อยกว่า ArcFace จาก 0 เล็กน้อย

def plot_training_history(history):
    """ฟังก์ชันสำหรับวาดกราฟ Accuracy และ Loss แล้วบันทึกเป็นไฟล์"""
    plt.figure(figsize=(14, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"\n✅ Training history plot saved to {PLOT_SAVE_PATH}")
    plt.close()

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. สร้าง Transforms และ Datasets สำหรับ Train และ Validation ---
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 2. สร้าง Model และส่วนประกอบอื่นๆ ---
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")
    
    class_mapping = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open(os.path.join(MODEL_SAVE_DIR, CLASS_MAPPING_NAME), 'w') as f:
        json.dump(class_mapping, f)
    print("Saved class mapping to JSON.")

    # Initialize model with pretrained=False
    model = FaceRecognitionModel(num_classes=num_classes, backbone_name='efficientnet_b0').to(device)
    
    # SphereFace's output is designed to work with NLLLoss
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # เพิ่ม weight_decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.00001) # เพิ่ม eta_min
    
    # --- 3. Training & Validation Loop ---
    print("Starting training with SphereFace Loss...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # The model's forward pass now includes the SphereFace calculation
            output_from_sphereface = model(images, labels)
            
            # We use NLLLoss after log_softmax because the SphereFace layer outputs logits-like values
            log_probs = F.log_softmax(output_from_sphereface, dim=-1)
            loss = criterion(log_probs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(log_probs.data, 1) # Calculate accuracy based on log_probs
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                images, labels = images.to(device), labels.to(device)
                
                output_from_sphereface = model(images, labels)
                log_probs = F.log_softmax(output_from_sphereface, dim=-1)
                loss = criterion(log_probs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(log_probs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # -- Log metrics for this epoch --
        train_loss_epoch = running_loss / len(train_dataset)
        val_loss_epoch = val_loss / len(val_dataset)
        train_acc_epoch = correct_train / total_train
        val_acc_epoch = correct_val / total_val
        
        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['train_acc'].append(train_acc_epoch)
        history['val_acc'].append(val_acc_epoch)
        
        scheduler.step() # ปรับ Learning Rate

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f} | LR: {current_lr:.6f}")

    # --- Save the trained model ---
    # We only need to save the backbone for testing, as the head is for training only
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    torch.save(model.backbone.state_dict(), model_path)
    print(f"\nTraining finished. Model backbone saved to {model_path}")

    # --- Plot the training history ---
    plot_training_history(history)

    # --- Calculate and display model size ---
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")
    
if __name__ == '__main__':
    train()
        