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