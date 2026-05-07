import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_CUSTOMERS = 10000

customer_ids = np.arange(1, N_CUSTOMERS + 1)

ages = np.random.normal(38, 10, N_CUSTOMERS).clip(18, 75).astype(int)

salaries = np.random.normal(18000, 12000, N_CUSTOMERS).clip(3000, 120000)

account_balances = salaries * np.random.uniform(1.5, 8.0, N_CUSTOMERS)

avg_card_spend = salaries * np.random.uniform(0.1, 0.6, N_CUSTOMERS)

tenure_months = np.random.randint(1, 180, N_CUSTOMERS)

digital_engagement = np.random.uniform(0, 100, N_CUSTOMERS)

missed_payments = np.random.poisson(0.3, N_CUSTOMERS)

is_sme_owner = np.random.choice([0, 1], size=N_CUSTOMERS, p=[0.9, 0.1])

international_spend_ratio = np.random.uniform(0, 1, N_CUSTOMERS)

# -----------------------------
# Propensity Targets
# -----------------------------

personal_loan_prob = (
    (salaries / salaries.max()) * 0.4
    + (digital_engagement / 100) * 0.3
    + (1 - missed_payments / (missed_payments.max() + 1)) * 0.3
)

personal_loan_conversion = (
    np.random.uniform(0, 1, N_CUSTOMERS) < personal_loan_prob
).astype(int)

fx_card_prob = (
    international_spend_ratio * 0.6
    + (salaries / salaries.max()) * 0.4
)

fx_card_conversion = (
    np.random.uniform(0, 1, N_CUSTOMERS) < fx_card_prob
).astype(int)

churn_prob = (
    (1 - digital_engagement / 100) * 0.5
    + (missed_payments / (missed_payments.max() + 1)) * 0.5
)

churn_risk = (
    np.random.uniform(0, 1, N_CUSTOMERS) < churn_prob
).astype(int)

# -----------------------------
# Final Dataset
# -----------------------------

df = pd.DataFrame({
    "customer_id": customer_ids,
    "age": ages,
    "salary": salaries.round(2),
    "account_balance": account_balances.round(2),
    "avg_card_spend": avg_card_spend.round(2),
    "tenure_months": tenure_months,
    "digital_engagement_score": digital_engagement.round(2),
    "missed_payments": missed_payments,
    "is_sme_owner": is_sme_owner,
    "international_spend_ratio": international_spend_ratio.round(2),
    "personal_loan_conversion": personal_loan_conversion,
    "fx_card_conversion": fx_card_conversion,
    "churn_risk": churn_risk
})

output_path = Path("data/customer_propensity_data.csv")

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(df.head())