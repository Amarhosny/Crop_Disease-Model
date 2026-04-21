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


| **Class Name**               | **Precision** | **Recall** | **F1-Score** | **Support** |
| ---------------------------- | ------------- | ---------- | ------------ | ----------- |
| **1. Potassium Deficiency**  | 0.8846        | 0.8734     | 0.8790       | 79          |
| **10. Dubas**                | 1.0000        | 0.9821     | 0.9910       | 224         |
| **2. Manganese Deficiency**  | 0.9756        | 0.9756     | 0.9756       | 41          |
| **3. Magnesium Deficiency**  | 0.8800        | 0.9167     | 0.8980       | 48          |
| **4. Black Scorch**          | 0.8000        | 1.0000     | 0.8889       | 8           |
| **5. Leaf Spots**            | 1.0000        | 0.8667     | 0.9286       | 60          |
| **6. Fusarium Wilt**         | 0.6400        | 0.8000     | 0.7111       | 20          |
| **7. Rachis Blight**         | 0.8846        | 1.0000     | 0.9388       | 23          |
| **8. Parlatoria Blanchardi** | 0.9316        | 0.9820     | 0.9561       | 111         |
| **9. Healthy sample**        | 0.9865        | 0.9648     | 0.9755       | 227         |
|                              |               |            |              |             |
| **Accuracy**                 |               |            | **0.9512**   | **841**     |
| **Macro Average**            | 0.8983        | 0.9361     | 0.9143       | 841         |
| **Weighted Average**         | 0.9548        | 0.9512     | 0.9521       | 841         |




<img width="400" height="232" alt="image" src="https://github.com/user-attachments/assets/ecbbabd8-0159-4607-989a-f2f8d0d3c62e" />



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
