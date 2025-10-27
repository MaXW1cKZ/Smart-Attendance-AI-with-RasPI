import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import json
from tqdm import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt

# Import models from model.py
from model import FaceNetModel

# --- Configuration ---
DATA_DIR = 'dataset'
MODEL_SAVE_DIR = 'saved_model'
MODEL_NAME = 'facenet_efficientnet_b0_from_scratch.pth'
CLASS_MAPPING_NAME = 'class_mapping.json'
PLOT_SAVE_PATH = 'training_history_facenet_from_scratch.png'

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.0001
EMBEDDING_SIZE = 512 
TRIPLET_MARGIN = 0.5

#--- Custom Triplet Dataset ---
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root_dir)
        self.class_to_indices = self._get_class_to_indices()
        if len(self.class_to_indices) < 2:
            raise ValueError(f"Directory {root_dir} needs at least 2 classes to create triplets. Found {len(self.class_to_indices)}.")

    def _get_class_to_indices(self):
        class_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in class_to_indices: class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices

    def __getitem__(self, index):
        anchor_path, anchor_label = self.dataset.samples[index]
        anchor_img = Image.open(anchor_path).convert('RGB')
        
        positive_indices = self.class_to_indices[anchor_label]
        positive_index = random.choice(positive_indices)
        while positive_index == index and len(positive_indices) > 1: # Ensure positive is not the same image as anchor
            positive_index = random.choice(positive_indices)
        positive_path, _ = self.dataset.samples[positive_index]
        positive_img = Image.open(positive_path).convert('RGB')
        
        available_negative_labels = list(filter(lambda x: x != anchor_label, self.class_to_indices.keys()))
        if not available_negative_labels:
             raise ValueError(f"Cannot find negative sample: No other classes available for label {anchor_label} in {self.root_dir}.")
        negative_label = random.choice(available_negative_labels)
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_path, _ = self.dataset.samples[negative_index]
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.dataset.samples)

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Triplet Loss (FaceNet train from 0)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print(f"\n✅ Training history plot saved to {PLOT_SAVE_PATH}")
    plt.close()

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- สร้าง Transforms และ Datasets สำหรับ Train และ Validation ---
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5), # เพิ่ม data augmentation
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
    
    train_dataset = TripletDataset(root_dir=os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = TripletDataset(root_dir=os.path.join(DATA_DIR, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # --- สร้าง Model และส่วนประกอบอื่นๆ ---
    class_mapping = {v: k for k, v in train_dataset.dataset.class_to_idx.items()}
    with open(os.path.join(MODEL_SAVE_DIR, CLASS_MAPPING_NAME), 'w') as f: json.dump(class_mapping, f)
    print(f"Found {len(class_mapping)} classes. Saved class mapping to JSON.")

    model = FaceNetModel(embedding_size=EMBEDDING_SIZE, pretrained=False).to(device) # pretrained=False for from scratch
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # เพิ่ม weight_decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.00001)
    
    # --- Training & Validation Loop ---
    print("Starting training with Triplet Loss...")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for anchor, positive, negative in progress_bar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_emb, positive_emb, negative_emb = model(anchor), model(positive), model(negative)
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_emb, positive_emb, negative_emb = model(anchor), model(positive), model(negative)
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                val_loss += loss.item()
        
        train_loss_epoch = running_loss / len(train_loader)
        val_loss_epoch = val_loss / len(val_loader)
        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | LR: {current_lr:.6f}")

    # --- Save the trained model ---
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining finished. Model saved to {model_path}")
    
    plot_training_history(history)

    # --- Calculate and display model size ---
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")


if __name__ == '__main__':
    train()