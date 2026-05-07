# Real-Time Banking Propensity Engine

AI-powered banking propensity and Next Best Offer (NBA) platform.

---

## Business Problem

Modern banks generate massive amounts of customer behavioural data:
- transactions,
- mobile app activity,
- salary deposits,
- product usage,
- engagement signals.

This platform predicts:
- personal loan conversion,
- SME loan conversion,
- premium account upgrades,
- FX card likelihood,
- churn risk.

The system then exposes predictions through production-grade APIs to support:
- personalized banking,
- customer retention,
- AI-driven marketing,
- real-time decisioning.

---

## Platform Components

### ML Platform
- Propensity models
- Feature engineering
- MLflow model registry
- Experiment tracking

### Serving Layer
- FastAPI inference APIs
- Real-time scoring
- Model versioning

### Monitoring
- Drift detection
- Prediction monitoring
- Observability metrics

---

## Tech Stack

- Python
- FastAPI
- MLflow
- Scikit-learn
- XGBoost
- Docker

---

## Future Roadmap

- Kafka streaming
- Spark feature pipelines
- Real-time fraud signals
- RAG + LLM decisioning
- Next Best Action engine


---

## Engineering Stack

Already looks like a real ML platform environment:

- VSCode
- WSL
- Python
- Git
- GitHub
- SSH
- Docker
- MLflow-ready architecture---

# Banking AI Propensity & Next Best Offer Platform

Production-style AI banking platform for customer propensity modeling and Next Best Offer (NBA) decisioning.

The platform simulates how enterprise banking AI systems:
- train propensity models,
- optimize experiments,
- track ML lifecycle,
- serve real-time predictions,
- orchestrate personalized offers.

---

# Business Use Cases

The platform predicts and serves:

- Personal loan conversion
- FX credit card propensity
- Premium banking upgrades
- Customer retention campaigns
- SME lending opportunities

---

# Platform Architecture

```text
Synthetic Banking Data
        ↓
ML Training Pipeline
        ↓
Optuna Hyperparameter Optimization
        ↓
MLflow Experiment Tracking
        ↓
Best Model Artifact
        ↓
FastAPI Inference API
        ↓
Next Best Offer Decision Engine
        ↓
Containerized Docker Service
```

---

# Engineering Stack

- Python
- FastAPI
- XGBoost
- Optuna
- MLflow
- Pandas
- Scikit-learn
- Docker
- GitHub
- WSL
- VSCode

---

# Core Components

## 1. Synthetic Banking Data Generation

Creates realistic synthetic banking customers with:
- salaries
- balances
- engagement signals
- SME ownership
- spending behaviour
- churn indicators

---

## 2. Propensity Modeling

Trains XGBoost models for:
- loan conversion prediction
- customer targeting
- NBA scoring

Includes:
- train/test pipelines
- evaluation metrics
- ROC-AUC optimization

---

## 3. Hyperparameter Optimization

Uses Optuna for:
- intelligent parameter search
- automated experimentation
- model performance optimization

---

## 4. MLflow Experiment Tracking

Tracks:
- model parameters
- metrics
- artifacts
- experiment lineage

---

## 5. FastAPI Inference Service

Production-style REST API for:
- real-time predictions
- schema validation
- Swagger documentation
- live scoring

Endpoints:
- `/health`
- `/predict`

---

## 6. Next Best Offer (NBA) Engine

Business rules layer that converts ML predictions into:
- personalized banking offers
- retention campaigns
- premium upgrade recommendations
- SME lending actions

---

## 7. Dockerized Deployment

Containerized deployment-ready inference service using Docker.

---

# Example Prediction Request

```json
{
  "age": 35,
  "salary": 25000,
  "account_balance": 120000,
  "avg_card_spend": 8000,
  "tenure_months": 48,
  "digital_engagement_score": 82,
  "missed_payments": 0,
  "is_sme_owner": 1,
  "international_spend_ratio": 0.7
}
```

---

# Example Prediction Response

```json
{
  "personal_loan_conversion_probability": 0.8421,
  "predicted_conversion": 1,
  "next_best_offer": "Premium SME Loan Package"
}
```

---

# Run Locally

## Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Generate synthetic data

```bash
python src/pipelines/generate_synthetic_data.py
```

## Train optimized model

```bash
python src/pipelines/train_optuna_propensity_model.py
```

## Run FastAPI service

```bash
uvicorn src.api.main:app --reload
```

## Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

---

# Run With Docker

## Build container

```bash
docker build -t banking-propensity-api .
```

## Run container

```bash
docker run -p 8000:8000 banking-propensity-api
```

---

# Future Extensions

- Kafka streaming inference
- Real-time feature store
- Snowflake integration
- Databricks pipelines
- RAG-powered customer intelligence
- LLM-driven recommendation engine
- Model drift monitoring
- Kubernetes deployment---

# Example AI Decision Response

Example response from the `/predict` API endpoint:

```json
{
  "personal_loan_conversion_probability": 0.3572,
  "predicted_conversion": 0,
  "next_best_offer": "Retention Campaign",
  "feature_importance": {
    "age": 0.0894,
    "salary": 0.1194,
    "account_balance": 0.1019,
    "avg_card_spend": 0.1025,
    "tenure_months": 0.0972,
    "digital_engagement_score": 0.1694,
    "missed_payments": 0.1412,
    "is_sme_owner": 0.0838,
    "international_spend_ratio": 0.0952
  }
}
```

---

# Response Interpretation

## Conversion Probability

```text
"personal_loan_conversion_probability": 0.3572
```

The AI model predicts that the customer has approximately a 35.7% probability of converting to the proposed banking product.

---

## Binary Prediction

```text
"predicted_conversion": 0
```

The platform applies a decision threshold of 0.5:

- Above 0.5 → likely conversion
- Below 0.5 → unlikely conversion

Since the probability is below the threshold, the customer is classified as unlikely to convert.

---

## Next Best Offer (NBA)

```text
"next_best_offer": "Retention Campaign"
```

Instead of aggressively pushing a new lending product, the decision engine recommends a retention-focused strategy.

Typical retention campaigns in banking may include:
- loyalty incentives
- cashback offers
- fee waivers
- engagement campaigns
- account upgrade incentives

This simulates how enterprise banking AI systems combine:
- machine learning predictions
- customer segmentation
- business rules
- campaign orchestration

to determine the most appropriate customer action.

---

## Feature Importance

The API also returns model explainability information:

```text
"feature_importance"
```

This shows which customer features are most influential globally across the trained XGBoost model.

Example:
- `digital_engagement_score` has strong influence on conversion likelihood
- `missed_payments` significantly impacts customer scoring
- `salary` contributes strongly to product propensity

This provides transparency and explainability for AI-driven banking decisions.