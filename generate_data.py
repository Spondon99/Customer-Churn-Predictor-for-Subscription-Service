import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000


def generate_subscribers(n):
    # plan type affects a lot of other stuff so doing it first
    plan = np.random.choice(["free", "basic", "pro"], size=n, p=[0.5, 0.3, 0.2])

    days_since_login = np.where(
        plan == "free",
        np.random.randint(0, 60, n),
        np.where(plan == "basic", np.random.randint(0, 30, n), np.random.randint(0, 15, n)),
    )

    emails_opened_30d = np.where(
        plan == "free",
        np.random.randint(0, 5, n),
        np.where(plan == "basic", np.random.randint(0, 10, n), np.random.randint(2, 15, n)),
    )

    # sessions per week roughly correlates with how engaged someone is
    sessions_per_week = np.where(
        plan == "free",
        np.random.randint(0, 4, n),
        np.where(plan == "basic", np.random.randint(1, 7, n), np.random.randint(2, 10, n)),
    )

    support_tickets = np.random.poisson(lam=0.5, size=n)
    billing_failures = np.where(plan == "free", 0, np.random.binomial(1, 0.1, n))

    account_age_days = np.random.randint(30, 730, n)

    # churn logic - trying to make it realistic, not just random
    churn_score = (
        0.03 * days_since_login
        - 0.04 * emails_opened_30d
        - 0.05 * sessions_per_week
        + 0.15 * billing_failures
        + 0.08 * support_tickets
        - 0.002 * account_age_days
        + np.where(plan == "free", 0.3, np.where(plan == "basic", 0.1, -0.1))
    )

    churn_prob = 1 / (1 + np.exp(-churn_score))
    churned = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "plan_type": plan,
            "days_since_login": days_since_login,
            "emails_opened_30d": emails_opened_30d,
            "sessions_per_week": sessions_per_week,
            "support_tickets_90d": support_tickets,
            "billing_failures": billing_failures,
            "account_age_days": account_age_days,
            "churned": churned,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_subscribers(N)
    df.to_csv("subscribers.csv", index=False)
    print(f"saved {len(df)} rows to subscribers.csv")
    print(f"churn rate: {df['churned'].mean():.2%}")
    print(df.head())
