"""
train.py

Trains and benchmarks classical ML and deep learning models for
fake profile detection: Random Forest, SVM, Logistic Regression,
CNN, LSTM (GNN/BERT fine-tuning to be integrated separately).
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000),
}


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1_score": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        }
    return results


if __name__ == "__main__":
    df = pd.read_csv("data/processed/instagram_accounts_features.csv")
    target_col = "is_fake"  # update to match actual label column

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    results_df = pd.DataFrame(results).T
    results_df.to_csv("results/model_comparison.csv")
    print(results_df)
