import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import engineer_features, FEATURE_COLS

DATA_PATH = "../data/subscribers.csv"
MODEL_PATH = "../model/churn_model.pkl"


def load_data(path):
    df = pd.read_csv(path)
    df = engineer_features(df)
    return df


def train(df):
    X = df[FEATURE_COLS]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost works well out of the box for this kind of tabular data
    # keeping it simple for now, not doing a full hyperparam search
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- evaluation on test set ---")
    print(classification_report(y_test, y_pred, target_names=["retained", "churned"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return model, X_test


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nmodel saved to {path}")


def plot_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, show=False)
    plt.tight_layout()

    os.makedirs("../plots", exist_ok=True)
    plt.savefig("../plots/shap_summary.png", dpi=150, bbox_inches="tight")
    print("shap plot saved to plots/shap_summary.png")
    plt.close()


if __name__ == "__main__":
    print("loading data...")
    df = load_data(DATA_PATH)
    print(f"loaded {len(df)} rows, churn rate: {df['churned'].mean():.2%}")

    print("\ntraining model...")
    model, X_test = train(df)

    save_model(model, MODEL_PATH)
    plot_shap(model, X_test)
