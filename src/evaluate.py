# src/evaluate.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error, mean_absolute_percentage_error
)


def evaluate_model(model, X_test, y_test, model_name="Best Model"):
    """
    Full evaluation of a single model with all metrics and plots.
    """
    y_pred = model.predict(X_test)

    # --- Metrics ---
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse  = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # as %

    print(f"\n{'='*50}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*50}")
    print(f"  R² Score  : {r2:.4f}")
    print(f"  MAE       : {mae:.2f} hg/ha")
    print(f"  RMSE      : {rmse:.2f} hg/ha")
    print(f"  MSE       : {mse:.2f}")
    print(f"  MAPE      : {mape:.2f}%")
    print(f"{'='*50}\n")

    metrics = {
        'Model' : model_name,
        'R2'    : round(r2, 4),
        'MAE'   : round(mae, 2),
        'RMSE'  : round(rmse, 2),
        'MSE'   : round(mse, 2),
        'MAPE'  : round(mape, 2)
    }

    return y_pred, metrics


def plot_actual_vs_predicted(y_test, y_pred, model_name="Best Model"):
    """
    Scatter plot of actual vs predicted yield values.
    A perfect model would have all points on the diagonal line.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue', edgecolors='k', linewidths=0.3)

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Yield (hg/ha)', fontsize=12)
    plt.ylabel('Predicted Yield (hg/ha)', fontsize=12)
    plt.title(f'Actual vs Predicted — {model_name}', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png', dpi=150)
    plt.show()
    print("Saved: plots/actual_vs_predicted.png")


def plot_residuals(y_test, y_pred, model_name="Best Model"):
    """
    Residual plot — helps detect bias or heteroscedasticity.
    Residuals should be randomly scattered around zero.
    """
    residuals = np.array(y_test) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, color='darkorange', edgecolors='k', linewidths=0.3)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Predicted Yield (hg/ha)', fontsize=11)
    axes[0].set_ylabel('Residual (Actual − Predicted)', fontsize=11)
    axes[0].set_title('Residuals vs Predicted', fontsize=13)

    # Residual distribution
    axes[1].hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Residual Value', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=13)

    plt.suptitle(f'Residual Analysis — {model_name}', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('plots/residuals.png', dpi=150)
    plt.show()
    print("Saved: plots/residuals.png")


def plot_feature_importance(model, feature_cols, top_n=15):
    """
    Built-in feature importance from tree-based models (not SHAP).
    Works for Random Forest, XGBoost, and Gradient Boosting.
    """
    if not hasattr(model, 'feature_importances_'):
        print("This model does not support feature_importances_. Skipping.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_cols[i] for i in indices]
    top_values   = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_values, y=top_features, palette='viridis')
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances (Built-in)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=150)
    plt.show()
    print("Saved: plots/feature_importance.png")


def plot_all_models_comparison(trained_models, X_test, y_test):
    """
    Compare all trained models side-by-side on R², MAE, and RMSE.
    Pass the `trained` dict returned by train_models().
    """
    records = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        records.append({
            'Model': name,
            'R²'   : round(r2_score(y_test, y_pred), 4),
            'MAE'  : round(mean_absolute_error(y_test, y_pred), 2),
            'RMSE' : round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        })

    df = pd.DataFrame(records)
    print("\n=== Full Model Comparison ===")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, color in zip(axes, ['R²', 'MAE', 'RMSE'],
                                  ['steelblue', 'darkorange', 'seagreen']):
        sns.barplot(x='Model', y=metric, data=df, ax=ax, palette=[color]*len(df))
        ax.set_title(f'Model Comparison — {metric}', fontsize=13)
        ax.set_xlabel('')
        if metric == 'R²':
            ax.set_ylim(0, 1)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.suptitle('Model Evaluation Summary', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('plots/all_models_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plots/all_models_comparison.png")

    return df


def run_full_evaluation(model, trained_models, X_test, y_test, feature_cols, model_name="Best Model"):
    """
    Master function — call this from main.py to run everything at once.
    """
    print("\nRunning full evaluation pipeline...")

    # 1. Metrics
    y_pred, metrics = evaluate_model(model, X_test, y_test, model_name)

    # 2. Actual vs Predicted
    plot_actual_vs_predicted(y_test, y_pred, model_name)

    # 3. Residuals
    plot_residuals(y_test, y_pred, model_name)

    # 4. Feature importance
    plot_feature_importance(model, feature_cols)

    # 5. All models comparison
    comparison_df = plot_all_models_comparison(trained_models, X_test, y_test)

    print("\nEvaluation complete. All plots saved to plots/")
    return metrics, comparison_df