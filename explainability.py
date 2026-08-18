"""
explainability.py

Generates SHAP / LIME / feature-importance explanations for the
fake profile classifier so predictions are interpretable by
moderators and end users.
"""

import shap
import pandas as pd


def explain_with_shap(model, X: pd.DataFrame, output_path: str = "results/shap_summary.png"):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    import matplotlib.pyplot as plt
    plt.savefig(output_path, bbox_inches="tight")
    return shap_values


def feature_importance(model, feature_names) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_")
    return pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


if __name__ == "__main__":
    print("Run this after train.py — pass the trained model and feature matrix.")
