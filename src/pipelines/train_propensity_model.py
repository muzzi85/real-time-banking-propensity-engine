import mlflow
import mlflow.sklearn

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier


# -----------------------------
# Load Dataset
# -----------------------------

data_path = Path("data/customer_propensity_data.csv")

df = pd.read_csv(data_path)

print(f"Loaded dataset shape: {df.shape}")


# -----------------------------
# Features & Target
# -----------------------------

TARGET = "personal_loan_conversion"

FEATURES = [
    "age",
    "salary",
    "account_balance",
    "avg_card_spend",
    "tenure_months",
    "digital_engagement_score",
    "missed_payments",
    "is_sme_owner",
    "international_spend_ratio"
]

X = df[FEATURES]

y = df[TARGET]


# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")


# -----------------------------
# MLflow Setup
# -----------------------------

mlflow.set_experiment("banking_propensity_models")

# -----------------------------
# Hyperparameter Search Space
# -----------------------------

param_grid = [
    {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.05
    },
    {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1
    },
    {
        "n_estimators": 200,
        "max_depth": 7,
        "learning_rate": 0.05
    }
]

best_roc_auc = 0
best_model = None


# -----------------------------
# Training Loop
# -----------------------------

for params in param_grid:

    with mlflow.start_run():

        print("\nTraining model with params:")
        print(params)

        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        prediction_probs = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)

        roc_auc = roc_auc_score(y_test, prediction_probs)

        # -----------------------------
        # MLflow Logging
        # -----------------------------

        mlflow.log_params(params)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="propensity_model"
        )

        print(f"Accuracy: {accuracy:.4f}")

        print(f"ROC-AUC: {roc_auc:.4f}")

        # -----------------------------
        # Track Best Model
        # -----------------------------

        if roc_auc > best_roc_auc:

            best_roc_auc = roc_auc

            best_model = model

print("\nBest ROC-AUC Achieved:")
print(best_roc_auc)