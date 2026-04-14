# main.py

from src.preprocess import clean_yield_data, clean_soil_data, fuse_datasets
from src.feature_eng import engineer_features
from src.train import train_models
from src.evaluate import run_full_evaluation
from src.explainability import run_shap_analysis
import os

# ── Create directories ────────────────────────────────────────────────────────
os.makedirs('data/raw',       exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models',         exist_ok=True)
os.makedirs('plots',          exist_ok=True)

print("=" * 50)
print("  Precision Agriculture — Full Pipeline")
print("=" * 50)

# ── Step 1: Clean datasets ────────────────────────────────────────────────────
print("\nSTEP 1: Cleaning datasets...")
clean_yield_data()
clean_soil_data()
print("  Done.")

# ── Step 2: Fuse datasets ─────────────────────────────────────────────────────
print("\nSTEP 2: Fusing datasets...")
fuse_datasets()
print("  Done.")

# ── Step 3: Feature engineering ───────────────────────────────────────────────
print("\nSTEP 3: Feature engineering...")
X_train, X_test, y_train, y_test, X_tr_raw, X_te_raw, df, feature_cols = engineer_features()
print("  Done.")

# ── Step 4: Train models ──────────────────────────────────────────────────────
print("\nSTEP 4: Training models...")
best_model, all_models, results = train_models(
    X_train, X_test, y_train, y_test, feature_cols
)
print("  Done.")

# ── Step 4b: Evaluate models ──────────────────────────────────────────────────
print("\nSTEP 4b: Evaluating models...")
metrics, comparison = run_full_evaluation(
    best_model, all_models,
    X_test, y_test, feature_cols,
    model_name=results.iloc[0]['Model']
)
print("  Done.")

# ── Step 5: SHAP explainability ───────────────────────────────────────────────
print("\nSTEP 5: Running SHAP analysis...")
run_shap_analysis(best_model, X_tr_raw, X_te_raw, feature_cols)
print("  Done.")

print("\n" + "=" * 50)
print("  All steps complete!")
print("  Run: streamlit run app.py")
print("=" * 50)