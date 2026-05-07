from fastapi import FastAPI

import joblib
import pandas as pd

from pathlib import Path

from src.api.schemas import CustomerFeatures


# -----------------------------
# Load Trained Model
# -----------------------------

model_path = Path("src/models/artifacts/best_propensity_model.pkl")

model = joblib.load(model_path)

print("Best propensity model loaded successfully")


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(
    title="Banking Propensity Engine",
    description="AI-powered banking propensity prediction API",
    version="1.0"
)


# -----------------------------
# Health Endpoint
# -----------------------------

@app.get("/health")

def health_check():

    return {"status": "healthy"}


# -----------------------------
# Next Best Offer Logic
# -----------------------------

def determine_next_best_offer(
    probability,
    salary,
    is_sme_owner,
    international_spend_ratio,
    digital_engagement_score
):

    if probability > 0.8 and is_sme_owner == 1:

        return "Premium SME Loan Package"

    elif probability > 0.75 and international_spend_ratio > 0.6:

        return "FX Credit Card"

    elif salary > 30000 and digital_engagement_score > 70:

        return "Premium Banking Upgrade"

    elif probability < 0.4:

        return "Retention Campaign"

    else:

        return "Standard Personal Loan Offer"
    
# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")

def predict(customer: CustomerFeatures):

    input_df = pd.DataFrame([customer.dict()])

    prediction_probability = model.predict_proba(input_df)[0][1]

    prediction = int(prediction_probability > 0.5)

    next_best_offer = determine_next_best_offer(
        probability=prediction_probability,
        salary=customer.salary,
        is_sme_owner=customer.is_sme_owner,
        international_spend_ratio=customer.international_spend_ratio,
        digital_engagement_score=customer.digital_engagement_score
    )
    response = {
        "personal_loan_conversion_probability": round(float(prediction_probability), 4),
        "predicted_conversion": prediction,
        "next_best_offer": next_best_offer
    }

    return response