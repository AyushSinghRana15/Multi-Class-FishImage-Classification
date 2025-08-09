# Multi-Class-FishImage-Classification
This project focuses on building an AI-powered fish species classification system capable of accurately identifying 11 different fish and seafood classes from images. The work covers the complete machine learning lifecycle: from data preprocessing, model training, performance evaluation, to building an interactive deployment app.
# 🐟 Fish Species Classification

## 📌 Project Overview
This project is an **AI-powered fish species image classifier** built using **TensorFlow/Keras** for model training and **Streamlit** for deployment.  
It can classify **11 fish and seafood species** from images with high accuracy, using both a **Custom CNN** and **five pre-trained networks** (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).  

The **main focuses** of this project are:
- **Data Preprocessing & Augmentation** for robust training
- **Model Training**
  - CNN from scratch
  - Transfer learning with fine-tuning for five architectures
- **Model Evaluation** using accuracy, precision, recall, F1-score, and confusion matrices
- **Interactive Deployment** with:
  - Image upload **or** sample image selection
  - **Top-3 predictions with confidence scores**
  - Class-wise confidence bar chart
  - Descriptions of each fish species

---

## 📂 Dataset
The dataset is organized into:

data/
train/
class_1/ 
class_2/
…
val/
class_1/
… 
test/ 
class_1/
…
**Classes:**
animal fish 
animal fish bass
fish sea_food black_sea_sprat 
fish sea_food gilt_head_bream 
fish sea_food hourse_mackerel 
fish sea_food red_mullet 
fish sea_food red_sea_bream 
fish sea_food sea_bass 
fish sea_food shrimp 
fish sea_food striped_red_mullet 
fish sea_food trout


---

## 🛠 Installation

1. **Clone this repository**

git clone https://github.com/yourusername/fish-classifier.git cd fish-classifier


2. **Install dependencies**

Requirements include:
- TensorFlow
- Keras
- NumPy
- Pillow
- scikit-learn
- matplotlib
- seaborn
- streamlit

---

## 🚀 How to Run the Project

### **1️⃣ Train Models (optional if using saved .h5 models)**
Train your models on the dataset:

python train_models.py

This will:
- Train the **CNN from scratch** and five pretrained models
- Fine-tune top layers
- Save best models as `.h5` files in the project directory

---

### **2️⃣ Evaluate Models**
Run evaluation on the test set to get accuracy, precision, recall, F1-score, and confusion matrices:


---

### **3️⃣ Run Streamlit App**
Start the interactive web app:

streamlit run fish_class.py

Features:
- Upload or select sample fish images
- Display **Top-3 predictions** with confidence scores
- Show bar chart of class-wise confidence
- Expandable **fish species information** section

---

### **4️⃣ Sample Images**
Place optional testing images inside:

sample_fish_images/

These will appear in the app for quick testing without uploads.

---

## 📊 Results Summary
- **Best Performing Models:** ResNet50 & EfficientNetB0
- **Top Accuracy:** ~94% on test data
- Pretrained models significantly **outperformed custom CNN**
- Confidence visualization helps in interpreting model results

---

## 📌 Future Improvements
- More diverse dataset to reduce class bias
- Real-time webcam capture
- Cloud deployment for public access
- Lightweight models for mobile inference

---

## 🙌 Acknowledgements
- Dataset contributors
- TensorFlow/Keras for deep learning framework
- Streamlit for simple, interactive deployment

---
