# Explainable AI (XAI) Implementation: CNN with SHAP and LIME

## 📌 Project Overview
This project demonstrates the implementation of **Explainable AI (XAI)** techniques applied to a Convolutional Neural Network (CNN). By utilizing state-of-the-art interpretability libraries, we move beyond "black-box" predictions to provide visual and statistical evidence for model decisions in handwritten digit classification.

## 🎯 Objectives
* **Transparency**: Understand why a CNN identifies a specific digit.
* **Interpretability**: Apply global and local XAI techniques (SHAP, LIME, Grad-CAM).
* **Trust & Fairness**: Evaluate model bias and ensure reliable performance across all classes.

## 👥 Group Members
* **Abhinav Badhe** (202301040100)
* **Chetanraje Gund** (202301040080)
* **Omkar Waghmare** (202301040010)
* **Virendra Pandule** (202301040078)

---

## 🛠️ Tech Stack
* **Deep Learning**: TensorFlow 2.x / Keras 3
* **Dataset**: MNIST (70,000 images of handwritten digits)
* **XAI Frameworks**: 
    * `SHAP` (SHapley Additive exPlanations)
    * `LIME` (Local Interpretable Model-agnostic Explanations)
* **Visualization**: Matplotlib, Seaborn, OpenCV (for Grad-CAM)

---

## 🚀 Key Implementation Parts
The project is structured into 15 logical parts, covering the full lifecycle of an XAI project:

1.  **Model Architecture**: A GPU-optimized CNN with Dropout layers to prevent overfitting.
2.  **Global Explanations**: Using SHAP summary plots to identify which pixels generally influence the model the most.
3.  **Local Explanations**:
    * **LIME**: Segmenting images into "superpixels" to show which areas support a specific prediction.
    * **SHAP Force Plots**: Visualizing the "push and pull" of individual pixel features.
4.  **Grad-CAM**: Generating class-activation heatmaps to see the "focus" of the final convolutional layer.
5.  **Fairness Analysis**: Detailed classification reports and confusion matrices to detect class-wise bias.

---

## 📊 Comparative Analysis of XAI Techniques

| Technique | Scope | Best For | Complexity |
| :--- | :--- | :--- | :--- |
| **Feature Importance** | Global | General pixel significance | Low |
| **SHAP Summary** | Global | Overall model behavior | High |
| **LIME** | Local | Individual image segments | Medium |
| **SHAP Force Plot** | Local | Individual pixel impact | Medium |
| **Grad-CAM** | Local | CNN feature map visualization | Medium |

---

## ⚙️ Setup & Installation
1. Clone the repository or download the notebook.
2. Install dependencies:
   ```bash
   pip install tensorflow shap lime opencv-python matplotlib scikit-image