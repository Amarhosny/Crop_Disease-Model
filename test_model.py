"""
Test the trained model on individual images
Use this to verify the model isn't overfitting
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

NUM_CLASSES = 10
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def create_model(num_classes):
    model = models.mobilenet_v2(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


def predict_image(model, image_path, top_k=3):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, indices = torch.topk(probabilities, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i].item()
        results.append({
            'class': CLASS_NAMES[idx],
            'confidence': confidences[0][i].item() * 100
        })

    return results


def test_on_folder(model, folder_path):
    """Test model on all images in a folder and calculate accuracy"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    correct = 0
    total = 0

    print(f"\nTesting on folder: {folder_path}")
    print("-" * 60)

    # Folder structure should be: folder/class_name/image.jpg
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.exists(class_dir):
            continue

        class_correct = 0
        class_total = 0

        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(image_extensions):
                img_path = os.path.join(class_dir, img_name)
                results = predict_image(model, img_path, top_k=1)
                predicted_class = results[0]['class']

                class_total += 1
                total += 1

                if predicted_class == class_name:
                    class_correct += 1
                    correct += 1

        if class_total > 0:
            class_acc = 100.0 * class_correct / class_total
            print(f"{class_name}: {class_correct}/{class_total} ({class_acc:.1f}%)")

    print("-" * 60)
    print(f"Overall Accuracy: {correct}/{total} ({100.0 * correct / total:.2f}%)")

    return correct / total if total > 0 else 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model.pth')
    parser.add_argument('--image', type=str, help='Test single image')
    parser.add_argument('--folder', type=str, help='Test folder (train/valid/test)')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = create_model(NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model (Val Acc: {checkpoint['val_acc']:.2f}%)")
    print(f"Device: {DEVICE}")

    if args.image:
        # Single image prediction
        results = predict_image(model, args.image, top_k=3)
        print(f"\nPredicting: {args.image}")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['class']}: {r['confidence']:.2f}%")

    elif args.folder:
        # Test on folder
        test_on_folder(model, args.folder)


if __name__ == "__main__":
    main()
