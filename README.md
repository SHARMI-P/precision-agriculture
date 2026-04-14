# 🌿 Precision Agriculture System

An AI-powered web application for **plant disease detection** and **crop yield prediction** built using Deep Learning and Machine Learning.

---

## 🚀 Live Demo

Run locally using Streamlit — see setup instructions below.

---

## 📌 Features

| Tab | Feature | Description |
|-----|---------|-------------|
| 🔬 Disease Detection | CNN Model | Upload a leaf image → detects disease with confidence score |
| 🌾 Yield Prediction | ML Ensemble | Enter soil & climate data → predicts crop yield in hg/ha |
| 🧪 Fertilizer Optimizer | NPK Optimizer | Recommends optimal N, P, K doses to maximize yield |
| 📊 SHAP Insights | Explainable AI | Feature importance plots for yield prediction |

---

## 🤖 Models

| Model | Task | Accuracy |
|-------|------|----------|
| MobileNetV2 (CNN) | Plant Disease Detection | **93.55%** |
| Random Forest / XGBoost | Crop Yield Prediction | R² evaluated via main.py |

### Disease Detection
- **72 disease classes** across 20+ crops
- Trained on **1,16,147 images** (PlantVillage dataset)
- Transfer Learning + Fine-tuning (2 phases)

### Crops Covered
Apple, Bell Pepper, Blueberry, Cassava, Cherry, Coffee, Corn, Grape, Orange, Peach, Potato, Raspberry, Rice, Rose, Soybean, Squash, Strawberry, Sugarcane, Tomato, Watermelon

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/SHARMI-P/precision-agriculture.git
cd precision-agriculture
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Files (Required)

> ⚠️ Model files are too large for GitHub. Download from Google Drive:

📥 **[Download Models from Google Drive](https://drive.google.com/drive/folders/1EkbqMN-53W1_E4zp0aAV7ANKsgkOgLDG?usp=sharing)**

Download both files and place them inside the `models/` folder:
```
precision-agriculture/
└── models/
    ├── cnn_disease_model.h5     ← download from Drive
    ├── best_model.pkl           ← download from Drive
    ├── class_names.pkl          ✅ already in repo
    ├── scaler.pkl               ✅ already in repo
    ├── le_area.pkl              ✅ already in repo
    ├── le_item.pkl              ✅ already in repo
    └── feature_cols.pkl         ✅ already in repo
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
precision-agriculture/
├── app.py                  # Streamlit web application
├── main.py                 # Full yield prediction pipeline
├── requirements.txt        # Python dependencies
├── models/                 # Trained model files
├── src/
│   ├── cnn_model.py        # MobileNetV2 model definition
│   ├── train_cnn.py        # CNN training script
│   ├── image_preprocess.py # Image data generators
│   ├── split_dataset.py    # Dataset splitting
│   ├── preprocess.py       # Data cleaning & fusion
│   ├── feature_eng.py      # Feature engineering
│   ├── train.py            # ML model training
│   ├── evaluate.py         # Model evaluation & plots
│   ├── explainability.py   # SHAP analysis
│   └── optimizer.py        # Fertilizer NPK optimizer
└── notebooks/
    ├── 01_eda_yield.ipynb
    ├── 02_eda_soil.ipynb
    └── 03_fusion_check.ipynb
```

---

## 🧪 Run Full Pipeline

```bash
# Step 1: Run yield prediction pipeline (training + SHAP)
python main.py

# Step 2: Train CNN (requires GPU)
python src/train_cnn.py

# Step 3: Launch web app
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras (MobileNetV2)
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainability:** SHAP
- **Web App:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 📊 CNN Training Results

| Phase | Epochs | Best Val Accuracy |
|-------|--------|------------------|
| Phase 1 (Head Training) | 20 | 89.81% |
| Phase 2 (Fine-tuning) | 10 | **93.55%** |

---

## 👩‍💻 Author

**SHARMI P**
GitHub: [@SHARMI-P](https://github.com/SHARMI-P)