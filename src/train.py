# src/train.py

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np, pickle, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(X_train, X_test, y_train, y_test, feature_cols):

    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=8, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbosity=0
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42
        )
    }

    results = []
    trained = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({'Model': name, 'R²': round(r2,4),
                        'MAE': round(mae,2), 'RMSE': round(rmse,2)})
        trained[name] = model
        print(f"  R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    # Save best model
    best_name = results_df.iloc[0]['Model']
    best_model = trained[best_name]
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest model: {best_name} saved.")

    # Plot comparison
    plt.figure(figsize=(9,5))
    sns.barplot(x='Model', y='R²', data=results_df, palette='viridis')
    plt.title('Model Comparison — R² Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.show()

    return best_model, trained, results_df