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
DATA_DIR = 'dataset' 
MODEL_SAVE_DIR = 'saved_model'
MODEL_NAME = 'resnet18_cosface.pth' 
CLASS_MAPPING_NAME = 'class_mapping.json'
PLOT_SAVE_PATH = 'training_history_resnet18_cosface.png'

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
BACKBONE_NAME = 'resnet18'
PRETRAINED = True

def plot_training_history(history, best_epoch):
    """ฟังก์ชันวาดกราฟที่เพิ่มการ Plot จุด Best Epoch"""
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch+1}')
    plt.title(f'Model Accuracy ({BACKBONE_NAME} + Cosface)')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch+1}')
    plt.title(f'Model Loss ({BACKBONE_NAME} + Cosface)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"\n✅ Training history plot saved to {PLOT_SAVE_PATH}")
    plt.close()

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")

    # Data Augmentation and Normalization
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'validation'), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes); print(f"Found {num_classes} classes: {train_dataset.classes}")
    
    class_mapping = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open(os.path.join(MODEL_SAVE_DIR, CLASS_MAPPING_NAME), 'w') as f: json.dump(class_mapping, f)
    print("Saved class mapping to JSON.")

    model = FaceRecognitionModel(num_classes=num_classes, backbone_name=BACKBONE_NAME).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.00001)
    
    print(f"Starting training {BACKBONE_NAME} with CosFace Loss...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    best_val_accuracy = 0.0
    best_epoch = -1

    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
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
        val_loss, correct_val, total_val = 0.0, 0, 0
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

        if val_acc_epoch > best_val_accuracy:
            best_val_accuracy = val_acc_epoch
            best_epoch = epoch
            # บันทึกเฉพาะโมเดลที่ดีที่สุดเท่านั้น
            model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
            torch.save(model.backbone.state_dict(), model_path)
            print(f"✨ New best model saved at epoch {epoch+1} with Val Acc: {best_val_accuracy:.4f}")
            

        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f} | LR: {current_lr:.6f}")
        
    print(f"\nTraining finished. Best model was saved from epoch {best_epoch+1}.")

    # 1. Plot the complete training history and mark the best epoch
    #    (แก้ไข: ต้องส่ง best_epoch เข้าไปด้วย)
    plot_training_history(history, best_epoch)
    
    # 2. Calculate and display the FINAL model size (which is the best model)
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    # ใส่ check กรณีที่เทรนแล้ว val_acc ไม่ดีขึ้นเลย (best_epoch = -1)
    if os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Final Model Size: {model_size_mb:.2f} MB")
    else:
        print("No model was saved as validation accuracy did not improve.")
    # ---^^^ REVISED END OF FUNCTION ^^^---
    
    # (ลบโค้ดบล็อกที่ซ้ำซ้อนท้ายไฟล์ออกทั้งหมด)

if __name__ == '__main__':
    train()