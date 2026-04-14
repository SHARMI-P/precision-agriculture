# 🌾 Multi-Source Precision Agriculture
### Crop Yield Prediction, Disease Detection & Fertilizer Optimization using Soil-Climate-Yield Fusion with Explainable AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-189AB4?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.2-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.4-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.17.1-8CAAE6?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-0.51.0-FF6B6B?style=for-the-badge)

[![R²](https://img.shields.io/badge/R²%20Score-0.9774-brightgreen?style=for-the-badge)]()
[![MAE](https://img.shields.io/badge/MAE-3576.36%20hg%2Fha-blue?style=for-the-badge)]()
[![RMSE](https://img.shields.io/badge/RMSE-8286.78%20hg%2Fha-orange?style=for-the-badge)]()
[![CNN](https://img.shields.io/badge/CNN%20Accuracy-93.55%25-brightgreen?style=for-the-badge)]()
[![Disease Classes](https://img.shields.io/badge/Disease%20Classes-72-purple?style=for-the-badge)]()
[![Records](https://img.shields.io/badge/Dataset-26183%20Records-yellow?style=for-the-badge)]()
[![Features](https://img.shields.io/badge/Features-16%20Engineered-purple?style=for-the-badge)]()

</div>

---

## 📌 Abstract

A production-grade machine learning pipeline that fuses heterogeneous agricultural datasets — crop yield records and soil nutrient profiles — to predict crop yield with **97.74% R² accuracy** and recommend optimal fertilizer doses. Now extended with a **CNN-based plant disease detection system** achieving **93.55% accuracy across 72 disease classes** using the PlantVillage dataset. The system integrates domain-knowledge feature engineering, ensemble learning, SHAP-based explainability, gradient-free numerical optimization, and deep learning for visual disease diagnosis — all served through an interactive Streamlit dashboard.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION LAYER                           │
│  yield_df.csv (28,242 rows) + soil_crop.csv (2,200 rows)           │
│  PlantVillage Dataset (50,000+ leaf images, 72 disease classes)    │
└──────────────────────┬──────────────────────────┬───────────────────┘
                       │                          │
                       ▼                          ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│     PREPROCESSING LAYER      │   │       CNN TRAINING PIPELINE      │
│  • Duplicate removal         │   │  • Image resize to 224×224       │
│  • IQR outlier filtering     │   │  • Data augmentation             │
│  • Median imputation         │   │  • MobileNetV2 / Custom CNN      │
│  • Crop-level soil fusion    │   │  • 72-class softmax output       │
│  → fused_dataset (26,183)    │   │  → cnn_disease_model.h5 saved   │
└──────────────────────┬───────┘   └──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                         │
│  NPK Ratio • Soil Fertility Score • Climate Index                   │
│  pH Deviation • Pesticide Efficiency • Decade Encoding              │
│  LabelEncoding (Area, Item) • StandardScaler                        │
│  → 16 features | 80/20 train-test split                            │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING LAYER                            │
│                                                                     │
│  ┌───────────────┐  ┌─────────────┐  ┌──────────────────┐          │
│  │ Random Forest │  │   XGBoost   │  │Gradient Boosting │          │
│  │  R²=0.9774 ✅ │  │  R²=0.9732  │  │   R²=0.9479      │          │
│  └───────────────┘  └─────────────┘  └──────────────────┘          │
│           → Best model saved: Random Forest                         │
└──────────────┬──────────────────────────┬───────────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│   EXPLAINABILITY     │   │        OPTIMIZATION          │
│  SHAP TreeExplainer  │   │  scipy.optimize (L-BFGS-B)   │
│  Global + Local      │   │  Optimal N,P,K → Max Yield   │
└──────────────────────┘   └──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT DASHBOARD                            │
│  🔬 Disease Detection | 🌾 Yield Prediction                        │
│  🧪 Fertilizer Optimizer | 📊 SHAP Insights                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Datasets

| Dataset | File | Rows / Size | Key Columns |
|---|---|---|---|
| Crop Yield | `yield_df.csv` | 28,242 rows | Area, Item, Year, rainfall, avg_temp, pesticides, hg/ha_yield |
| Soil Nutrients | `soil_crop.csv` | 2,200 rows | N, P, K, ph, humidity, temperature, label |
| **Fused** | `fused_dataset.csv` | **26,183 rows** | All 15 columns merged on crop name |
| Plant Disease | PlantVillage | 50,000+ images | 72 disease classes across 14 crops |

> ⚠️ Download `yield_df.csv` and `soil_crop.csv` from Kaggle. Place in `data/raw/` and rename `Crop_recommendation.csv` → `soil_crop.csv`.
> Download the PlantVillage dataset and place extracted images in `data/plantvillage/`.

---

## 📁 Project Structure

```
precision_agriculture/
│
├── data/
│   ├── raw/
│   │   ├── yield_df.csv                ← Crop yield dataset
│   │   └── soil_crop.csv               ← Soil nutrients dataset
│   ├── processed/
│   │   ├── cleaned_yield.csv           ← Auto-generated
│   │   ├── cleaned_soil.csv            ← Auto-generated
│   │   └── fused_dataset.csv           ← Auto-generated
│   ├── plantvillage/                   ← Leaf images by class folder
│   └── plantvillage_split.zip          ← Zipped image archive
│
├── notebooks/
│   ├── 01_eda_yield.ipynb              ← Yield data exploration
│   ├── 02_eda_soil.ipynb               ← Soil data exploration
│   └── 03_fusion_check.ipynb           ← Fusion validation
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                   ← Cleaning & fusion
│   ├── feature_eng.py                  ← Feature creation & scaling
│   ├── train.py                        ← RF, XGBoost, GBM training
│   ├── evaluate.py                     ← Metrics & evaluation plots
│   ├── explainability.py               ← SHAP analysis
│   ├── cnn_model.py                    ← CNN disease model training
│   └── optimizer.py                    ← Fertilizer optimizer
│
├── models/
│   ├── best_model.pkl                  ← Trained Random Forest
│   ├── scaler.pkl                      ← Feature scaler
│   ├── feature_cols.pkl                ← Feature column names
│   ├── le_area.pkl                     ← Area label encoder
│   ├── le_item.pkl                     ← Crop label encoder
│   ├── cnn_disease_model.h5            ← Trained CNN model
│   └── class_names.pkl                 ← 72 disease class names
│
├── plots/                              ← All generated charts
├── app.py                              ← Streamlit dashboard
├── main.py                             ← Full pipeline runner
└── requirements.txt
```

---

## ⚙️ Setup

```bash
# Clone
git clone https://github.com/SHARMI-P/precision-agriculture.git
cd precision-agriculture

# Virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Step 1 — Run full ML pipeline (only needed once or when retraining)
python main.py

# Step 2 — Launch dashboard
streamlit run app.py
```

> ✅ Once `main.py` has been run, all models are saved in `models/`. On subsequent runs just use `streamlit run app.py` directly.

---

## 🖥️ Dashboard Features

### 🔬 Tab 1 — Plant Disease Detection (CNN)
- Upload any leaf image (JPG/PNG)
- CNN model classifies it into one of **72 disease classes** across 14 crops
- Displays **Top 3 predictions** with confidence scores
- Identifies plant name and disease name separately
- Model accuracy: **93.55%**

### 🌾 Tab 2 — Yield Prediction
- Input crop, region, year, rainfall, temperature, N, P, K, pH, humidity, pesticides
- Predicts crop yield in **hg/ha** using the best trained model (Random Forest)

### 🧪 Tab 3 — Fertilizer Optimizer
- Input current N, P, K values
- Uses **L-BFGS-B optimization** to find the NPK combination that maximizes predicted yield
- Shows optimal N, P, K values and the delta change from current values

### 📊 Tab 4 — SHAP Insights
- Displays global feature importance (SHAP bar chart)
- Beeswarm plot showing feature impact direction
- Dependence plots for top features

---

## 🧠 Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| `npk_ratio` | `N / (P + K + 1)` | Nitrogen dominance in soil |
| `soil_fertility_score` | `0.4N + 0.3P + 0.3K` | Weighted soil health index |
| `climate_index` | `rain×0.5 + (30-\|temp-25\|)×0.3 + humidity×0.2` | Climate suitability score |
| `ph_deviation` | `\|ph - 7.0\|` | Distance from neutral pH |
| `pesticide_efficiency` | `yield / (pesticides + 1)` | Output per pesticide unit |
| `decade` | `(Year // 10) × 10` | Long-term trend capture |

---

## 🤖 Model Configurations

```python
# Random Forest (Best)
RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)

# XGBoost
XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8)

# Gradient Boosting
GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

# CNN Disease Model
Input: 224×224 RGB leaf image
Output: 72-class softmax
Accuracy: 93.55%
```

---

## 📊 Results

| Model | R² | MAE (hg/ha) | RMSE (hg/ha) | MAPE |
|---|---|---|---|---|
| ✅ **Random Forest** | **0.9774** | **3576.36** | **8286.78** | 14.82% |
| XGBoost | 0.9732 | 5112.61 | 9018.77 | — |
| Gradient Boosting | 0.9479 | 7584.87 | 12574.38 | — |

| CNN Model | Accuracy | Classes |
|---|---|---|
| Plant Disease CNN | **93.55%** | 72 disease classes |

---

## 🔬 SHAP Explainability

```
For any prediction ŷ:
ŷ = base_value + Σ SHAP_value(feature_i)
```

- `Item_enc` dominates with ~65% importance — yield is highly crop-specific
- `climate_index` outperforms raw `rainfall` and `avg_temp` individually
- `N`, `P`, `K` contribute via engineered features more than individually

---

## ⚗️ Fertilizer Optimizer

```python
result = minimize(
    fun    = lambda npk: -model.predict(build_feature_vector(npk)),
    x0     = [current_N, current_P, current_K],
    method = 'L-BFGS-B',
    bounds = [(0,140), (0,145), (0,205)]   # agronomic bounds kg/ha
)
```

**Sample Output:**
```
Current  : N=40.0, P=40.0, K=40.0
Optimal  : N=82.3, P=51.7, K=38.2
Max Yield: 68,450 hg/ha
Change   : N +42.3, P +11.7, K -1.8
```

---

## 📈 Generated Plots

| File | Description |
|---|---|
| `model_comparison.png` | R² across all models |
| `actual_vs_predicted.png` | Predicted vs actual scatter |
| `residuals.png` | Residual distribution |
| `feature_importance.png` | Top 15 feature scores |
| `all_models_comparison.png` | R², MAE, RMSE side-by-side |
| `shap_bar.png` | SHAP global importance |
| `shap_beeswarm.png` | SHAP impact direction |
| `shap_dependence.png` | Top feature dependence |
| `shap_force_plot.png` | Single prediction breakdown |

---

## 🛠️ Tech Stack

| Category | Library | Version |
|---|---|---|
| Deep Learning | TensorFlow / Keras | 2.x |
| Machine Learning | scikit-learn | 1.8.0 |
| Boosting | XGBoost | 3.2.0 |
| Explainability | SHAP | 0.51.0 |
| Optimization | SciPy | 1.17.1 |
| Data | Pandas + NumPy | 3.0.2 / 2.4.4 |
| Visualization | Matplotlib + Seaborn | 3.10.8 / 0.13.2 |
| Dashboard | Streamlit | 1.56.0 |
| Image Processing | Pillow (PIL) | — |
| Language | Python | 3.13 |

---

## 👤 Author

**SHARMI-P** — [GitHub](https://github.com/SHARMI-P/precision-agriculture)