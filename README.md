# Diabetic Retinopathy Detection using Deep Learning

## Overview
This project implements a deep learning-based approach to detect Diabetic Retinopathy (DR) in high-resolution retinal fundus images. Using a pre-trained ResNet50 model, the system classifies retinal images as either having Diabetic Retinopathy (DR) or not (No_DR) and provides explainable predictions using Grad-CAM. The project is deployed as a user-friendly *Streamlit app*.

---

## Features
- *High Accuracy*: Achieved 96.97% test accuracy using ResNet50 pre-trained model.
- *Explainability*: Grad-CAM highlights regions in the retinal images that influenced the model's prediction, enhancing clinical interpretability.
- *User Interface*: A Streamlit app allows users to upload retinal images and view predictions along with Grad-CAM visualizations.

---

## Dataset
### Data Description
This dataset comprises a large collection of high-resolution retinal images captured under various imaging conditions. Each image is assessed by a medical professional and labeled as:

- *Diabetic Retinopathy (DR)*: 0
- *No Diabetic Retinopathy (No_DR)*: 1

#### Class-to-Index Mapping
- {'DR': 0, 'No_DR': 1}

### Data Split
- *Training samples*: 2076
- *Validation samples*: 531
- *Test samples*: 231

### Preprocessing
The preprocessing pipeline includes the following steps:

- *Resize:* All images are resized to 224x224 pixels to match the input size of the ResNet50 model.
- *Normalization:* Pixel values are normalized using calculated mean and standard deviation values.
- *To Tensor:* Images are converted to PyTorch tensors for compatibility with the deep learning model.

---

## Model
### Architecture
- *Base Model:* ResNet50 (pre-trained on ImageNet).
- *Feature Extraction Layers:* All pre-trained layers were frozen to retain learned features.
- *Modified Fully Connected Layers:*
  - Input: Features from ResNet50 (num_features = 2048).
  - Added a dense layer with 128 neurons and ReLU activation for better learning.
  - Dropout layer with a rate of 0.3 for regularization.
  - Output: A dense layer with 2 neurons for binary classification (DR or No_DR).
- *Optimizer:* Adam with a learning rate of 0.001, applied only to the modified fully connected layers.
- *Loss Function:* CrossEntropyLoss for multi-class classification.

### Training Results
| Epoch | Training Loss | Training Accuracy |
|-------|---------------|-------------------|
| 1     | 18.9141       | 88.10%            |
| 2     | 12.6329       | 93.45%            |
| 3     | 11.3515       | 93.83%            |
| 4     | 12.9329       | 93.16%            |
| 5     | 9.7793        | 94.85%            |
| 6     | 9.3528        | 95.57%            |
| 7     | 9.2785        | 95.81%            |
| 8     | 7.7981        | 96.19%            |
| 9     | 8.3323        | 96.15%            |
| 10    | 8.1565        | 96.05%            |

### Evaluation Metrics
- *Validation Accuracy*: 95.86%
- *Test Accuracy*: 96.97%

---

## Explainability
### Grad-CAM Implementation

Grad-CAM (Gradient-weighted Class Activation Mapping) is a method for providing visual explanations for deep learning models. This section describes how Grad-CAM was implemented in the project to enhance the explainability of predictions.

#### Key Features of the Grad-CAM Implementation

1. **Gradient Capture:**
   - Gradients flowing back from the output layer were captured during the backpropagation phase.
   - These gradients were used to calculate the importance of each feature map in the last convolutional layer.

2. **Activation Maps:**
   - Activations from the final convolutional layer were retrieved to understand the spatial regions of interest.
   - These activations were weighted by the average gradient to compute a class-specific heatmap.

3. **ReLU Activation:**
   - A ReLU activation function was applied to retain only positive contributions to the class prediction, ensuring clearer and more interpretable visualizations.

4. **Heatmap Normalization:**
   - The generated heatmap was normalized to a range of [0, 1], facilitating better overlay visualization with the input image.

5. **Overlay Visualization:**
   - The Grad-CAM heatmap was overlaid on the original retinal image using a colormap (e.g., 'jet') for intuitive interpretation.
   - This overlay highlights the regions most influential in the model's decision-making process.

### Visual Interpretability

The Grad-CAM implementation provides insights into the model's predictions by emphasizing areas in the retinal image that correspond to the presence of diabetic retinopathy. This enhances trust in the AI system, particularly for clinical applications, where explainability is critical.

The Grad-CAM heatmaps are displayed in the app, allowing users to:
- Identify affected regions in retinal images.
- Validate model predictions with visual evidence.

Grad-CAM serves as an essential tool for bridging the gap between AI models and human interpretability, particularly in medical imaging tasks.

---

## Deployment
### Streamlit App
The Streamlit app provides a simple interface for:
1. *Image Upload*: Users can upload a retinal fundus image.
2. *Prediction*: The app predicts if the image shows signs of Diabetic Retinopathy.
3. *Visualization*: Displays the Grad-CAM heatmap for explainability.

### App Walkthrough
The following screenshots demonstrate the functionality of the Streamlit app:

1. *Home Page:* The app's landing page with an option to upload a retinal image.
   
   ![HomePage](https://github.com/user-attachments/assets/91429bdd-c256-46d8-9419-bba96996e815)

2. *Image Upload:* Users can upload a retinal fundus image for prediction.

   ![ImageUpload](https://github.com/user-attachments/assets/684a43ba-5bd7-4293-bad2-f6ef7c22ebae)

   
3. *Prediction and Heatmap:* Displays the prediction (DR / No_DR) along with the Grad-CAM heatmap highlighting the regions influencing the prediction.
   
   ![PredictionandHaetmap](https://github.com/user-attachments/assets/8a65bd76-001a-41c9-917a-18fcc52dbc3c)

---

## Acknowledgments
- *Dataset*: [Diagnosis of Diabetic Retinopathy (Kaggle)](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy/data).
- *Pre-trained Model*: ResNet50 from Keras Applications.

---

## Future Enhancements
- Extend the application to classify multiple stages of Diabetic Retinopathy.
- Integrate additional explainable AI techniques.
- Deploy the application to the cloud for wider accessibility.

---

## The EyeCatchers Team
- [Fatima Noor](https://www.linkedin.com/in/fatima-nur/)
- [Mashal Fatima](https://www.linkedin.com/in/mashalfatima/)
- [Maryam Nawaz](https://www.linkedin.com/in/maryammnawaz)

---
