import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json
import numpy as np
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import timm
import seaborn as sns
import matplotlib.pyplot as plt

# Import the simple classifier model
from model import SimpleClassifierModel

# --- Configuration ---
TEST_DATA_DIR = 'dataset/test'
MODEL_SAVE_DIR = 'saved_model'
MODEL_NAME = 'resnet50_crossentropy.pth' # ต้องตรงกับ MODEL_NAME ใน train.py
CLASS_MAPPING_NAME = 'class_mapping.json'

# Hyperparameters (ควรตรงกับตอนเทรน)
IMG_SIZE = 224
BACKBONE_NAME = 'resnet50'
# PRETRAINED here will be False, as we load OUR OWN trained weights
# (refer to previous explanation about pretrained=False in test.py)


def test():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Class Mapping ---
    class_mapping_path = os.path.join(MODEL_SAVE_DIR, CLASS_MAPPING_NAME)
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    print(f"Loaded {num_classes} classes.")

    # Load the model structure (pretrained=False as we load our weights)
    model = SimpleClassifierModel(num_classes=num_classes, backbone_name=BACKBONE_NAME, pretrained=False).to(device)
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model {MODEL_NAME} loaded successfully.")
    
    # Data transformation for testing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Evaluate on Test Set ---
    print("Evaluating on test data...")
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) # Batch size 1 for accurate inference time per image

    true_labels = []
    pred_labels = []
    inference_times = []

    with torch.no_grad():
        for image, label in test_loader:
            true_labels.append(label.item())
            image = image.to(device)

            # Measure inference time
            start_time = time.time()
            outputs = model(image)
            end_time = time.time()
            inference_times.append(end_time - start_time)

            _, predicted = torch.max(outputs.data, 1)
            pred_labels.append(predicted.item())

    # --- Report Results ---
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

    # Confusion Matrix
    class_names = [class_mapping[str(i)] for i in sorted(class_mapping, key=lambda x: int(x))]
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({BACKBONE_NAME} + CrossEntropy)')
    plt.show()


if __name__ == '__main__':
    test()
