# main.py

from src.preprocess import clean_yield_data, clean_soil_data, fuse_datasets
from src.feature_eng import engineer_features
from src.train import train_models
from src.evaluate import run_full_evaluation
from src.explainability import run_shap_analysis
import os

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 50)
print("STEP 1: Cleaning datasets")
clean_yield_data()
clean_soil_data()

print("\nSTEP 2: Fusing datasets")
fuse_datasets()

print("\nSTEP 3: Feature engineering")
X_train, X_test, y_train, y_test, X_tr_raw, X_te_raw, df, feature_cols = engineer_features()

print("\nSTEP 4: Training models")
best_model, all_models, results = train_models(
    X_train, X_test, y_train, y_test, feature_cols
)

print("\nSTEP 4b: Evaluating models")        # ← NEW
metrics, comparison = run_full_evaluation(   # ← NEW
    best_model, all_models,                  # ← NEW
    X_test, y_test, feature_cols,            # ← NEW
    model_name=results.iloc[0]['Model']      # ← NEW
)                                            # ← NEW

print("\nSTEP 5: SHAP explainability")
run_shap_analysis(best_model, X_train, X_test, feature_cols)

print("\nAll steps complete!")