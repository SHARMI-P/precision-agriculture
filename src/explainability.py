# src/explainability.py

import shap
import matplotlib.pyplot as plt
import numpy as np

def run_shap_analysis(model, X_train, X_test, feature_cols):

    print("Running SHAP analysis...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # --- Plot 1: Summary bar plot ---
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_cols,
                      plot_type='bar', show=False)
    plt.title('SHAP — Global Feature Importance')
    plt.tight_layout()
    plt.savefig('plots/shap_bar.png')
    plt.close()
    print("Saved: plots/shap_bar.png")

    # --- Plot 2: Beeswarm plot ---
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_cols, show=False)
    plt.title('SHAP — Feature Impact on Yield Prediction')
    plt.tight_layout()
    plt.savefig('plots/shap_beeswarm.png')
    plt.close()
    print("Saved: plots/shap_beeswarm.png")

    # --- Plot 3: Dependence plot for top feature ---
    top_feature_idx = np.argsort(np.abs(shap_values).mean(0))[-1]
    shap.dependence_plot(top_feature_idx, shap_values, X_test,
                         feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_dependence.png')
    plt.close()
    print("Saved: plots/shap_dependence.png")

    # --- Plot 4: Force plot for single prediction ---
    try:
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X_test[0],
            feature_names=feature_cols,
            matplotlib=True,
            show=False
        )
        plt.savefig('plots/shap_force_plot.png', bbox_inches='tight')
        plt.close()
        print("Saved: plots/shap_force_plot.png")
    except Exception as e:
        print(f"Force plot skipped: {e}")

    print("SHAP analysis complete.")
    return shap_values, explainer