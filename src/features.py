import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # encode plan type - order matters here (free < basic < pro engagement-wise)
    plan_map = {"free": 0, "basic": 1, "pro": 2}
    df["plan_encoded"] = df["plan_type"].map(plan_map)

    # this ratio gives a sense of how stale the account is
    # a new account with no logins is different from an old account with no logins
    df["login_staleness"] = df["days_since_login"] / (df["account_age_days"] + 1)

    # combined engagement signal
    df["engagement_score"] = df["emails_opened_30d"] + df["sessions_per_week"] * 2

    # flag people who basically never log in
    df["is_inactive"] = (df["days_since_login"] > 14).astype(int)

    return df


FEATURE_COLS = [
    "plan_encoded",
    "days_since_login",
    "emails_opened_30d",
    "sessions_per_week",
    "support_tickets_90d",
    "billing_failures",
    "account_age_days",
    "login_staleness",
    "engagement_score",
    "is_inactive",
]
