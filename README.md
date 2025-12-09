# 🚦 Real-Time Accident Information System  
A Machine Learning–powered accident detection and severity classification system using deep learning and computer vision.

---

## 📌 Overview  
Road accidents cause over 1.3 million deaths every year, and delayed reporting remains a major global issue. Existing systems rely heavily on manual reporting, slow surveillance review, and inefficient insurance workflows.

This project solves that by providing an **AI-based real-time accident severity classification system** using images captured from CCTV, dashcams, or roadside monitoring units. The system uses multiple deep learning models, with **ResNet50 achieving the highest accuracy of 89%**.

---

## 🎯 Key Features  
- 🔍 **Accident Detection** from static images  
- 🧭 **Severity Classification** into 4 levels:  
  - Non-Accident  
  - Less Severe  
  - Moderately Severe  
  - Highly Severe  
- 🤖 **Trained Deep Learning Models:** CNN, EfficientNetB0, MobileNetV2, ResNet50  
- 🏆 **Best Model:** ResNet50 (89% accuracy)  
- 🛠️ **Regression Module** for continuous severity scoring  
- 🚑 **Potential for real-time emergency alerts**  
- 🧾 **Insurance assistance workflow support**

---

## 📂 Dataset  
A custom-labelled dataset of accident and non-accident images categorized into four classes.

| Split       | Highly Severe | Moderately Severe | Less Severe | Non-Accident | Total |
|-------------|----------------|-------------------|-------------|--------------|-------|
| Train       | 742            | 543               | 787         | 913          | 2985  |
| Validation  | 158            | 116               | 169         | 194          | 637   |
| Test        | 159            | 116               | 169         | 196          | 640   |

---

## 🧪 Methodology  

### **1. Preprocessing**
- Resizing images to 224×224  
- Normalization  
- Data augmentation: rotation, flipping, zoom  

### **2. Models Used**
- Basic CNN — 78% accuracy  
- EfficientNetB0 — 82%  
- MobileNetV2 — 87%  
- **ResNet50 — 89% (best performer)**  

All models trained using:
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Epochs: 25–50  
- Batch size: 32  

### **3. Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- ROC–AUC  

---

## 📊 Results

### ✅ **Model Performance Comparison**

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Basic CNN      | 78%      | 0.75      | 0.74   | 0.74     |
| EfficientNetB0 | 82%      | 0.79      | 0.78   | 0.78     |
| MobileNetV2    | 87%      | 0.86      | 0.85   | 0.85     |
| **ResNet50**   | **89%**  | **0.88**  | **0.87**| **0.88** |

### 🏆 Final Chosen Model: **ResNet50**

It offered:
- Strong feature extraction  
- Stable training  
- Best accuracy and F1 score  
- Robustness to image variations  

---

## 🏗️ System Architecture (Flow)

1. Input surveillance/dashcam image  
2. Preprocessing (resize, normalize)  
3. Deep learning model prediction  
4. Severity classification  
5. Optional:  
   - Emergency alert trigger  
   - Insurance data logging  

---

## 🛠️ Tools & Technologies  
- **Python 3.8+**  
- **TensorFlow / Keras**  
- **OpenCV**  
- **NumPy, Pandas**  
- **Matplotlib, Seaborn**  
- **Scikit-Learn**  
- Training performed on **Google Colab Pro (NVIDIA T4 GPU)**  

---

## ⚙️ Installation & Usage

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/Real-Time-Accident-Information-System.git
cd Real-Time-Accident-Information-System
