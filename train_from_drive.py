"""
Train from Google Drive - Local Script
Mount your Google Drive and train without uploading data separately.

Setup:
1. Get your Google Drive API credentials:
   - Go to https://console.cloud.google.com/
   - Create a project and enable Google Drive API
   - Create OAuth 2.0 credentials
   - Download client_secret.json

2. Install: pip install pydrive2 torch torchvision scikit-learn matplotlib seaborn

3. Run this script - it will authenticate once, then cache credentials
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from io import BytesIO

# ==================== CONFIGURATION ====================

# Google Drive folder ID (the part after /folders/ in the share link)
# Example: https://drive.google.com/drive/folders/1ABC123xyz → folder_id = "1ABC123xyz"
DRIVE_FOLDER_ID = "YOUR_FOLDER_ID_HERE"

# Or use direct path if mounted via google.colab
COLAB_DATA_PATH = "/content/drive/MyDrive/date final.v1i.folder"

NUM_CLASSES = 10
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

CLASS_NAMES = [
    "1. Potassium Deficiency",
    "10. Dubas",
    "2. Manganese Deficiency",
    "3. Magnesium Deficiency",
    "4. Black Scorch",
    "5. Leaf Spots",
    "6. Fusarium Wilt",
    "7. Rachis Blight",
    "8. Parlatoria Blanchardi",
    "9. Healthy sample"
]

# =======================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DriveDataset(Dataset):
    """Dataset that reads from Google Drive folder"""

    def __init__(self, base_path, split, transform=None):
        self.base_path = base_path
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

        split_dir = os.path.join(base_path, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found!")
            return

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))

        print(f"Loaded {len(self.samples)} samples from {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(augment=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


def create_model(num_classes, pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), 100.0 * correct / total, all_preds, all_labels


def main():
    print("=" * 60)
    print("Date Palm Disease - Google Drive Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Check if running in Colab
    if os.path.exists("/content/drive"):
        print("\nRunning in Google Colab")
        data_path = COLAB_DATA_PATH
    else:
        print("\nRunning locally - Google Drive must be mounted")
        print("Mount Drive manually or use train_local.py instead")
        data_path = COLAB_DATA_PATH

    if not os.path.exists(data_path):
        print(f"\nERROR: Data not found at {data_path}")
        print("Make sure Google Drive is mounted and path is correct")
        return

    print(f"Data path: {data_path}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = DriveDataset(data_path, "train", transform=get_transforms(augment=True))
    val_dataset = DriveDataset(data_path, "valid", transform=get_transforms(augment=False))
    test_dataset = DriveDataset(data_path, "test", transform=get_transforms(augment=False))

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    print("\nCreating MobileNetV2 model...")
    model = create_model(NUM_CLASSES).to(DEVICE)

    # Class weights
    counts = np.zeros(NUM_CLASSES)
    for _, label in train_dataset.samples:
        counts[label] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    class_weights = torch.FloatTensor(weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=0.01
    )

    # Training
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS}: "
              f"Train: {train_loss:.4f} ({train_acc:.2f}%) | "
              f"Val: {val_loss:.4f} ({val_acc:.2f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"  -> Saved best model ({val_acc:.2f}%)")

    # Fine-tuning
    print("\n" + "=" * 60)
    print("Fine-tuning top layers...")
    print("=" * 60)

    for param in model.features[-5:].parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE / 10, weight_decay=0.01
    )

    for epoch in range(5):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)

        print(f"FT {epoch+1}/5: Train: {train_loss:.4f} ({train_acc:.2f}%) | "
              f"Val: {val_loss:.4f} ({val_acc:.2f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')

    # Evaluate
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    checkpoint = torch.load('best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, DEVICE)

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4))

    # Confusion Matrix
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(test_labels, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xticks(rotation=45, ha='right')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\nConfusion matrix saved!")

    # Training curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved!")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
