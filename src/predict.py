import argparse
import pickle
import sys

import pandas as pd
import shap

from features import engineer_features, FEATURE_COLS

MODEL_PATH = "../model/churn_model.pkl"


def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"no model found at {path}. run train.py first.")
        sys.exit(1)


def build_input(args):
    # putting everything into a dataframe so we can reuse engineer_features
    row = {
        "plan_type": args.plan,
        "days_since_login": args.days_since_login,
        "emails_opened_30d": args.emails_opened,
        "sessions_per_week": args.sessions_per_week,
        "support_tickets_90d": args.support_tickets,
        "billing_failures": args.billing_failures,
        "account_age_days": args.account_age_days,
    }
    df = pd.DataFrame([row])
    df = engineer_features(df)
    return df[FEATURE_COLS]


def get_top_reason(model, X_input, feature_cols):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_input)[0]

    # find the feature pushing the prediction highest
    top_idx = shap_vals.argmax()
    top_feature = feature_cols[top_idx]

    labels = {
        "days_since_login": "hasn't logged in recently",
        "emails_opened_30d": "low email engagement",
        "sessions_per_week": "low session activity",
        "billing_failures": "billing failure on record",
        "support_tickets_90d": "high support ticket volume",
        "plan_encoded": "plan type",
        "login_staleness": "account activity has dropped off",
        "engagement_score": "low overall engagement",
        "is_inactive": "account flagged as inactive",
        "account_age_days": "account age",
    }

    return labels.get(top_feature, top_feature)


def main():
    parser = argparse.ArgumentParser(description="predict churn probability for a subscriber")
    parser.add_argument("--plan", choices=["free", "basic", "pro"], required=True)
    parser.add_argument("--days_since_login", type=int, required=True)
    parser.add_argument("--emails_opened", type=int, default=0)
    parser.add_argument("--sessions_per_week", type=int, default=1)
    parser.add_argument("--support_tickets", type=int, default=0)
    parser.add_argument("--billing_failures", type=int, default=0)
    parser.add_argument("--account_age_days", type=int, default=180)

    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    X_input = build_input(args)

    prob = model.predict_proba(X_input)[0][1]
    top_reason = get_top_reason(model, X_input, FEATURE_COLS)

    risk_label = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"

    print(f"\nChurn probability : {prob:.0%}")
    print(f"Risk level        : {risk_label}")
    print(f"Top reason        : {top_reason}")


if __name__ == "__main__":
    main()
