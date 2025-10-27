# Face Recognize\ArcFace-EfficientNet-B0\train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import models from model.py
from model import FaceRecognitionModel
    
# --- Configuration ---
DATA_DIR = 'dataset_new' # ชี้ไปที่โฟลเดอร์แม่ของ train/test
MODEL_SAVE_DIR = 'saved_model'
MODEL_NAME = 'efficientnet_b0_arcface.pth'
CLASS_MAPPING_NAME = 'class_mapping.json'
PLOT_SAVE_PATH = 'training_history_arcface_finetune.png'

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 0.001

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
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([ # แยก transform สำหรับ validation
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_transform) # ใช้ dataset/test เป็น validation
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # DataLoader สำหรับ validation

    # --- 2. สร้าง Model และส่วนประกอบอื่นๆ ---
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")
    
    # Save class mapping (ใช้จาก train_dataset)
    class_mapping = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open(os.path.join(MODEL_SAVE_DIR, CLASS_MAPPING_NAME), 'w') as f:
        json.dump(class_mapping, f)
    print("Saved class mapping to JSON.")

    model = FaceRecognitionModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.00001)
    
    # --- 3. Training & Validation Loop ---
    print("Starting training with ArcFace Loss...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train() # ตั้งค่าโมเดลเป็นโหมด train
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
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
                outputs = model(images, labels)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
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
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f} | LR: {current_lr:.6f}")
    
    # --- Save the trained model ---
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