# 🌾 Multi-Source Precision Agriculture
### Crop Yield Prediction & Fertilizer Optimization using Soil-Climate-Yield Fusion with Explainable AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
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
[![Records](https://img.shields.io/badge/Dataset-26183%20Records-yellow?style=for-the-badge)]()
[![Features](https://img.shields.io/badge/Features-16%20Engineered-purple?style=for-the-badge)]()

</div>

---

## 📌 Abstract

A production-grade machine learning pipeline that fuses heterogeneous agricultural datasets — crop yield records and soil nutrient profiles — to predict crop yield with **97.74% R² accuracy** and recommend optimal fertilizer doses. The system integrates domain-knowledge feature engineering, ensemble learning, SHAP-based explainability, and gradient-free numerical optimization, served through an interactive Streamlit dashboard.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                      │
│   yield_df.csv (28,242 rows) + soil_crop.csv (2,200 rows)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                        │
│  • Duplicate removal      • IQR outlier filtering           │
│  • Median imputation      • String normalization            │
│  • Crop-level soil aggregation (groupby mean)               │
│  • Left merge on crop name → fused_dataset (26,183 rows)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                FEATURE ENGINEERING LAYER                    │
│  NPK Ratio • Soil Fertility Score • Climate Index           │
│  pH Deviation • Pesticide Efficiency • Decade Encoding      │
│  LabelEncoding (Area, Item) • StandardScaler                │
│  → 16 features | 80/20 train-test split                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING LAYER                       │
│                                                             │
│  ┌───────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ Random Forest │  │   XGBoost   │  │Gradient Boosting │  │
│  │  R²=0.9774 ✅ │  │  R²=0.9732  │  │   R²=0.9479      │  │
│  └───────────────┘  └─────────────┘  └──────────────────┘  │
│           → Best model saved: Random Forest                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐   ┌─────────────────────────────┐
│  EXPLAINABILITY     │   │      OPTIMIZATION           │
│  SHAP TreeExplainer │   │  scipy.optimize (L-BFGS-B)  │
│  Global + Local     │   │  Optimal N, P, K → MaxYield │
└─────────────────────┘   └─────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                      │
│      Yield Prediction | Fertilizer Optimizer | SHAP        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Datasets

| Dataset | File | Rows | Key Columns |
|---|---|---|---|
| Crop Yield | `yield_df.csv` | 28,242 | Area, Item, Year, rainfall, avg_temp, pesticides, hg/ha_yield |
| Soil Nutrients | `soil_crop.csv` | 2,200 | N, P, K, ph, humidity, temperature, label |
| **Fused** | `fused_dataset.csv` | **26,183** | All 15 columns merged on crop name |

> ⚠️ Download datasets from Kaggle. Place in `data/raw/` and rename `Crop_recommendation.csv` → `soil_crop.csv`

---

## 📁 Project Structure

```
precision_agriculture/
│
├── data/
│   ├── raw/                        ← Place datasets here
│   └── processed/                  ← Auto-generated CSVs
│
├── src/
│   ├── preprocess.py               ← Cleaning, fusion
│   ├── feature_eng.py              ← Feature creation, scaling
│   ├── train.py                    ← RF, XGBoost, GBM
│   ├── evaluate.py                 ← Metrics, plots
│   ├── explainability.py           ← SHAP analysis
│   └── optimizer.py                ← Fertilizer optimizer
│
├── models/                         ← Saved model, scaler, encoders
├── plots/                          ← All generated charts
├── notebooks/                      ← EDA notebooks
├── app.py                          ← Streamlit dashboard
├── main.py                         ← Pipeline runner
└── requirements.txt
```

---

## ⚙️ Setup

```bash
# Clone
git clone https://github.com/SHARMI-P/precision-agriculture.git
cd precision-agriculture

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Run full pipeline
python main.py

# Launch dashboard
streamlit run app.py
```

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
```

---

## 📊 Results

| Model | R² | MAE (hg/ha) | RMSE (hg/ha) | MAPE |
|---|---|---|---|---|
| ✅ **Random Forest** | **0.9774** | **3576.36** | **8286.78** | 14.82% |
| XGBoost | 0.9732 | 5112.61 | 9018.77 | — |
| Gradient Boosting | 0.9479 | 7584.87 | 12574.38 | — |

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
| Machine Learning | scikit-learn | 1.8.0 |
| Boosting | XGBoost | 3.2.0 |
| Explainability | SHAP | 0.51.0 |
| Optimization | SciPy | 1.17.1 |
| Data | Pandas + NumPy | 3.0.2 / 2.4.4 |
| Visualization | Matplotlib + Seaborn | 3.10.8 / 0.13.2 |
| Dashboard | Streamlit | 1.56.0 |
| Language | Python | 3.13 |

---


