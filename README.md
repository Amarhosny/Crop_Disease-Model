This README is designed to showcase your impressive **95.12% accuracy** and the technical sophistication of your graduation project. It highlights your use of **Transfer Learning** and the **MobileNetV2** architecture.

***

# Date Palm Disease Classification using Deep Learning

This project implements a robust deep learning solution to identify 9 diseases and nutritional deficiencies in Date Palm leaves, achieving a **95.12% test accuracy**.

## 📊 Performance Summary
*   **Overall Accuracy:** 95.12%
*   **Model Architecture:** MobileNetV2 (Transfer Learning)
*   **Classes:** 10 (9 Disease/Deficiency types + Healthy samples)
*   **Best F1-Score:** 0.99 (Dubas)

## 📁 Dataset Structure
The model was trained using a dataset structured into `train`, `valid`, and `test` splits, including:
1. Potassium Deficiency
2. Manganese Deficiency
3. Magnesium Deficiency
4. Black Scorch
5. Leaf Spots
6. Fusarium Wilt
7. Rachis Blight
8. Parlatoria Blanchardi
9. Healthy sample
10. Dubas



## 🛠️ Technical Stack
*   **Framework:** PyTorch
*   **Base Model:** MobileNetV2 (Pre-trained on ImageNet)
*   **Optimizer:** AdamW
*   **Loss Function:** CrossEntropyLoss with **Class Weights** (to handle data imbalance)
*   **Environment:** Google Colab (GPU Accelerated)

## 🚀 Key Features
*   **Live Training Monitor:** Real-time visualization of Loss and Accuracy curves during training.
*   **Two-Phase Training:** 
    1.  **Phase 1:** Training the custom classifier head while freezing the backbone.
    2.  **Phase 2:** Fine-tuning the top layers of MobileNetV2 for domain-specific feature extraction.
*   **Robust Preprocessing:** Integrated image augmentation (Rotations, Flips, Color Jitter) to improve generalization.

## 📈 Evaluation Results
The final evaluation on the unseen test set yielded the following classification report:

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| 10. Dubas | 1.00 | 0.98 | 0.99 |
| 9. Healthy sample | 0.99 | 0.96 | 0.98 |
| **Average** | **0.95** | **0.95** | **0.95** |

<img width="1120" height="989" alt="image" src="https://github.com/user-attachments/assets/ecbbabd8-0159-4607-989a-f2f8d0d3c62e" />



## 💻 Usage
### Testing the Model
To run the model on your own images, ensure `best_model.pth` is in your directory and run:
```python
# Example prediction snippet
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
# ... (load image and transform)
output = model(image_tensor)
prediction = CLASS_NAMES[output.argmax(dim=1).item()]
```

## 🎓 Graduation Project Credits
**Project Title:** Date Palm Disease Detection System  
**Accuracy Achieved:** 95.12%  
**Model Type:** Convolutional Neural Network (CNN)
