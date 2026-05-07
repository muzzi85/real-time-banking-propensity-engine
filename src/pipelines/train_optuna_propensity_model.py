import optuna
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# -----------------------------
# Load Dataset
# -----------------------------

data_path = Path("data/customer_propensity_data.csv")

df = pd.read_csv(data_path)

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


# -----------------------------
# MLflow Experiment
# -----------------------------

mlflow.set_experiment("optuna_banking_propensity_models")


# -----------------------------
# Optuna Objective Function
# -----------------------------

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42
    }

    run_name = (
        f"xgb_depth_{params['max_depth']}"
        f"_lr_{params['learning_rate']:.3f}"
    )

    with mlflow.start_run(run_name=run_name):

        model = XGBClassifier(**params)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        prediction_probs = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)

        roc_auc = roc_auc_score(y_test, prediction_probs)

        precision = precision_score(y_test, predictions)

        recall = recall_score(y_test, predictions)

        f1 = f1_score(y_test, predictions)

        # -----------------------------
        # Log Parameters
        # -----------------------------

        mlflow.log_params(params)

        # -----------------------------
        # Log Metrics
        # -----------------------------

        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_metric("precision", precision)

        mlflow.log_metric("recall", recall)

        mlflow.log_metric("f1_score", f1)

        # -----------------------------
        # Log Model
        # -----------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="propensity_model"
        )

        print("\nTrial Complete")

        print(params)

        print(f"ROC-AUC: {roc_auc:.4f}")

        return roc_auc


# -----------------------------
# Run Optimization
# -----------------------------

study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=10)


# -----------------------------
# Best Trial Summary
# -----------------------------

print("\nBest Trial")

print(f"Best ROC-AUC: {study.best_value:.4f}")

print("Best Parameters:")

print(study.best_params)

print(study.best_params)

# -----------------------------
# Train Final Best Model
# -----------------------------

best_model = XGBClassifier(
    **study.best_params,
    random_state=42
)

best_model.fit(X_train, y_train)

model_path = Path("src/models/artifacts/best_propensity_model.pkl")

joblib.dump(best_model, model_path)

print(f"\nBest model saved to: {model_path}")
