import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the simple classifier model
from model import SimpleClassifierModel

# --- Configuration ---
DATA_DIR = 'dataset' 
MODEL_SAVE_DIR = 'saved_model'
MODEL_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k_crossentropy.pth' 
CLASS_MAPPING_NAME = 'class_mapping.json'
PLOT_SAVE_PATH = 'training_history_tf_efficientnetv2_s.in21k_ft_in1k_crossentropy.png' 

# Hyperparameters
IMG_SIZE = 224 # ขนาดภาพ
BATCH_SIZE = 16 # ปรับตามความเหมาะสมของ GPU
EPOCHS = 50 # จำนวน epoch สำหรับการฝึก
LEARNING_RATE = 0.0001 # เริ่มต้นด้วยค่า learning rate ที่ต่ำ
BACKBONE_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k' # ชื่อ backbone ที่ใช้
PRETRAINED = True # ใช้ ImageNet weights สำหรับ fine-tuning

#--- ฟังก์ชันช่วยเหลือ ---
def plot_training_history(history):
    """ฟังก์ชันสำหรับวาดกราฟ Accuracy และ Loss แล้วบันทึกเป็นไฟล์"""
    plt.figure(figsize=(14, 6))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Model Accuracy ({BACKBONE_NAME} + CrossEntropy)')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss ({BACKBONE_NAME} + CrossEntropy)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"\n✅ Training history plot saved to {PLOT_SAVE_PATH}")
    plt.close()

#--- Main Training Function ---
def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. สร้าง Transforms และ Datasets สำหรับ Train และ Validation ---
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5), # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    model = SimpleClassifierModel(num_classes=num_classes, backbone_name=BACKBONE_NAME, pretrained=PRETRAINED).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # เพิ่ม weight_decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.00001)
    
    # --- 3. Training & Validation Loop ---
    print(f"Starting training {BACKBONE_NAME} with CrossEntropy Loss (Stage 1)...")
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
            outputs = model(images) # Model now outputs logits directly
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
                outputs = model(images)
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
    # Save the entire model state (backbone + classification head)
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining finished. Model saved to {model_path}")

    # --- Plot the training history ---
    plot_training_history(history)
    
    # --- Calculate and display model size ---
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")

if __name__ == '__main__':
    train()
