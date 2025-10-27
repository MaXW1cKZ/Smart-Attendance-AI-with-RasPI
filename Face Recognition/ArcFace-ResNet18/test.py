import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json
import time  # <--- เพิ่มบรรทัดนี้เข้ามา
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import timm
from sklearn.metrics import classification_report

# --- Configuration ---
TEST_DATA_DIR = 'dataset/test'
TRAIN_DATA_DIR = 'dataset/train'
MODEL_SAVE_DIR = 'saved_model'
MODEL_NAME = 'resnet18_arcface.pth'
CLASS_MAPPING_NAME = 'class_mapping.json'

# Hyperparameters (ควรตรงกับตอนเทรน)
IMG_SIZE = 224
BACKBONE_NAME = 'resnet18'

def test():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Class Mapping ---
    # Load class mapping
    class_mapping_path = os.path.join(MODEL_SAVE_DIR, CLASS_MAPPING_NAME)
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    print(f"Loaded {num_classes} classes.")

    # Load backbone model
    backbone = timm.create_model(BACKBONE_NAME, pretrained=False, num_classes=0)
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    backbone.load_state_dict(torch.load(model_path, map_location=device))
    backbone.to(device)
    backbone.eval()
    print("Model loaded successfully for testing.")
    
    # Data transformation for testing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 1. Create Gallery Embeddings from Training Set ---
    print("Creating gallery embeddings from training data...")
    gallery_embeddings = {}
    gallery_labels = {}
    
    train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            embeddings = F.normalize(backbone(images))
            for i in range(len(labels)):
                label = labels[i].item()
                if label not in gallery_embeddings:
                    gallery_embeddings[label] = []
                gallery_embeddings[label].append(embeddings[i])

    # Average the embeddings for each class to get a single representative vector
    for label, embs in gallery_embeddings.items():
        gallery_embeddings[label] = torch.stack(embs).mean(0)
        gallery_labels[label] = class_mapping[str(label)]

    print("Gallery created.")
    
    # --- 2. Evaluate on Test Set ---
    print("Evaluating on test data...")
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    true_labels = []
    pred_labels = []
    inference_times = []

    with torch.no_grad():
        for image, label in test_loader:
            true_labels.append(label.item())
            image = image.to(device)

            # Measure inference time
            start_time = time.time()
            query_embedding = F.normalize(backbone(image))
            end_time = time.time()
            inference_times.append(end_time - start_time)

            # Compare with gallery
            similarities = {}
            for class_idx, gallery_emb in gallery_embeddings.items():
                # Cosine similarity
                sim = F.cosine_similarity(query_embedding, gallery_emb.unsqueeze(0))
                similarities[class_idx] = sim.item()
            
            # Predict the class with the highest similarity
            predicted_class_idx = max(similarities, key=similarities.get)
            pred_labels.append(predicted_class_idx)

    # --- 3. Report Results ---
    print("\n--- Evaluation Results ---")
    
    # Accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Inference Time
    avg_inference_time_ms = (sum(inference_times) / len(inference_times)) * 1000
    print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms per image")

    # Model Size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")

    print("\n--- Detailed Classification Report ---")
    class_names = [class_mapping[str(i)] for i in range(num_classes)]
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print(report)

    # Confusion Matrix
    class_names = [class_mapping[str(i)] for i in sorted(class_mapping, key=lambda x: int(x))]
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (ArcFace + ResNet18)')
    plt.show()

if __name__ == '__main__':
    test()
