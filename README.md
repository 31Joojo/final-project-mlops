# MLOps Final Project

## Authors : BAUDET Quentin, CARDONA Quentin & LARMAILLARD-NOIREN Joris

## Final Project — End-to-End MLOps Pipeline (DVC + MLflow Registry + CI/CD + Cloud Deploy)

This repository implements a production-grade MLOps system covering the full lifecycle: **data versioning**, **training & evaluation**, **MLflow tracking + Model Registry**, **automated promotion with quality gates**, **reproducible deployments**, and a **public web application** serving predictions.  
The focus is on **MLOps execution quality** (traceability, reproducibility, automation), not on maximizing model performance.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Global Architecture](#global-architecture)
- [Repository Structure](#repository-structure)
- [Model & Prediction Contract](#model--prediction-contract)
- [Data Versioning (DVC)](#data-versioning-dvc)
- [Experiment Tracking & Model Registry (MLflow on DagsHub)](#experiment-tracking--model-registry-mlflow-on-dagshub)
- [CI/CD Workflows](#cicd-workflows)
- [Automated Model Promotion & Quality Gates](#automated-model-promotion--quality-gates)
- [Cloud Environments (Render)](#cloud-environments-render)
- [Local Setup](#local-setup)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Project Overview
The system is composed of:
- **Web UI (Next.js)**: user interface to enter features, dropdowns and inputs, and display predictions.
- **Inference API (FastAPI)**: serves `/health`, `/options`, and `/predict`, and loads the model from **MLflow Model Registry**.
- **Training pipeline (Python / scikit-learn)**: trains and evaluates a baseline model, logs metrics and artifacts to MLflow, and registers model versions to the Registry.
- **Model promotion pipeline (GitHub Actions)**: triggered on `staging`, trains a candidate model, deploys to staging, runs quality gates, and promotes the model to **Production** in the Registry if all gates pass.

---

## Global Architecture
**Serving path**
1. User interacts with the **Next.js** application.
2. Next.js proxies API calls (`/api/options`, `/api/predict`) to the FastAPI service.
3. FastAPI loads the model from **MLflow Model Registry** using a stage/alias (staging/prod).
4. FastAPI returns predictions along with model metadata (name, stage, version).

**Training & promotion path**
1. Raw dataset is versioned with **DVC** and stored in a remote backend (DagsHub Storage).
2. Training logs runs to **MLflow Tracking** (DagsHub) with full traceability:
   - metrics/parameters/artifacts,
   - `git_commit`,
   - `dvc_data_rev`.
3. On `staging`, the promotion workflow:
   - trains and registers a candidate model,
   - deploys staging,
   - runs quality gates,
   - promotes to **Production** if PASS.

---

## Repository Structure
```bash
.
├── README.md
├── apps
│  ├── api
│  │  ├── Dockerfile
│  │  ├── app
│  │  │  ├── __init__.py
│  │  │  ├── main.py
│  │  │  ├── model_loader.py
│  │  │  ├── predictor.py
│  │  │  ├── schema.py
│  │  │  └── settings.py
│  │  ├── requirements-dev.txt
│  │  └── requirements.txt
│  └── web
│      ├── Dockerfile
│      ├── app
│      │  ├── api
│      │  │  ├── options
│      │  │  │  └── route.ts
│      │  │  └── predict
│      │  │      └── route.ts
│      │  ├── globals.css
│      │  ├── layout.tsx
│      │  └── page.tsx
│      ├── next-env.d.ts
│      ├── next.config.js
│      ├── package-lock.json
│      ├── package.json
│      ├── public
│      └── tsconfig.json
├── data
│  └── raw
│      ├── ChocolateSales.pbit
│      ├── ChocolateSales.pbix
│      ├── ChocolateSales.pdf
│      ├── chocolate_sales.csv
│      └── chocolate_sales.csv.dvc
├── docker-compose.yml
├── ml
│  ├── __init__.py
│  ├── promotion
│  │  ├── __init__.py
│  │  ├── gates_and_promote.py
│  │  └── train_and_register.py
│  ├── requirements.txt
│  └── training
│      ├── __init__.py
│      ├── train.py
│      └── utils.py
├── mlflow.db
├── pytest.ini
└── tests
    ├── conftest.py
    ├── e2e
    │  ├── test_prod_live.py
    │  └── test_staging_live.py
    ├── integration
    │  ├── test_api_registry_driven.py
    │  ├── test_health.py
    │  ├── test_options.py
    │  ├── test_predict.py
    │  └── test_training_smoke.py
    └── unit
        ├── test_amount_cleaning.py
        ├── test_dvc_rev.py
        ├── test_model_loader.py
        ├── test_predictor.py
        ├── test_schema_validation.py
        └── test_settings.py

19 directories, 51 files
```

---


---

## Model & Prediction Contract

### Business interpretation
The model returns the probability that a given sale configuration will be classified as **"high amount"** (above the threshold defined during training). This is used as a proxy for a "high-value sale".

### Input features (UI/API)
The UI and API share a stable contract:
- `sales_person` (categorical)
- `country` (categorical)
- `product` (categorical)
- `boxes_shipped` (numeric)
- `date` (ISO date string)

### Feature engineering
Before inference, the API aligns the payload with the training pipeline:
- maps fields to training column names (e.g., `Sales Person`, `Country`, `Product`, `Boxes Shipped`)
- derives calendar features from `date` (e.g., `year`, `month`, `dayofweek`)

### API endpoints
- `GET /health`: service health and model metadata (`loaded`, `model_name`, `model_stage`, `model_version`)
- `GET /options`: returns dropdown values (sales persons, countries, products) extracted from the loaded model pipeline
- `POST /predict`: returns prediction and probability and model metadata

---

## Data Versioning (DVC)
The raw dataset is tracked with DVC:
- Local path: `data/raw/chocolate_sales.csv`  
- DVC pointer: `data/raw/chocolate_sales.csv.dvc`

The dataset is stored in a remote backend (DagsHub Storage). Credentials are configured locally or via CI secrets and are **never committed**.

Typical commands:
```bash
### Track dataset
dvc add data/raw/chocolate_sales.csv
git add data/raw/chocolate_sales.csv.dvc .gitignore
git commit -m "Track dataset with DVC"

### Push/pull data to/from remote
dvc push
dvc pull
```

---

## Experiment Tracking & Model Registry (MLflow on DagsHub)
MLflow is used for:
- **Experiments tracking**: metrics, parameters, artifacts, input example, signature
- **Traceability**: tags include git_commit and dvc_data_rev
- **Model Registry**: central source of truth for deployments

Required environment variables:
- `MLFLOW_TRACKING_URI` (DagsHub MLflow endpoint)
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

The registered model name is:
- `chocolate_sales_logreg`

---

## CI/CD Workflows
The CI/CD strategy is branch-driven and follows the repository governance model.

### PR → `dev` (CI)
- Runs unit tests and integration tests 
- Builds Docker images for API and web (no push)
- Enforced by branch protection rules

### Merge → `staging`
- Runs full test suite (excluding live E2E where needed)
- Deploys to Render staging via deploy hooks 
- Waits for `/health` and `/options` 
- Runs live E2E tests against the `staging` URL

### `staging` → `main` (CD production)
- Runs tests 
- Deploys to Render production 
- Waits for `/health`
- Runs live E2E tests against the `production` URL

---

## Automated Model Promotion & Quality Gates
A promotion pipeline is triggered on each push/merge to the `staging branch:

1. Train_and_register.py
- pulls data via DVC in CI
- trains a candidate model
- logs run to MLflow
- registers a new model version in the Registry

2. Deploy staging (Render hook)

3. gates_and_promote.py
- executes quality gates (automated)
- tags gate results on the model version
- promotes to Production in MLflow Registry if PASS
- if FAIL: no promotion; production remains unchanged

### Implemented quality gates
- Metric gate: candidate must meet or exceed a baseline or minimum threshold for F1
- Contract/schema gate: model loads successfully and prediction contract remains valid 
- Latency gate : p95 response time must remain below a configured limit

This mechanism ensures that production is only updated when the candidate model satisfies measurable quality constraints.

---

## Cloud Environments (Render)
The project is deployed on Render with separate staging and production services:

### Staging
- API (staging): `https://final-project-mlops-api-staging.onrender.com`
- Web (staging): `https://final-project-mlops-web-staging.onrender.com`

### Production
Production services run on the `main` branch and are configured to serve only the model in the MLflow Registry Production stage:
- API (prod): `https://api-prod-xlib.onrender.com`
- Web (prod): `https://web-prod-ie6y.onrender.com`

### Environment variables (Render)
API
- `MODEL_NAME=chocolate_sales_logreg`
- `MODEL_STAGE=Production` (prod) or `@staging` / Staging (`staging`)
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
- `CORS_ALLOW_ORIGINS=https://web-prod-ie6y.onrender.com`

Web
- `API_URL=https://api-prod-xlib.onrender.com`

---

## Local Setup
1. Prerequisites :
- Python 3.11+ 
- Node 18+ 
- Docker + Docker Compose 
- DVC installed 
- Access to DagsHub (MLflow + Storage)

2. Configure environment variables :
Create a local .env (not committed) using the provided examples:
- `.env.example.dev`
- `.env.example.staging`
- `.env.example.prod`
At minimum for local API registry loading:
```bash
export MODEL_NAME="chocolate_sales_logreg"
export MODEL_STAGE="@staging"

export MLFLOW_TRACKING_URI="https://dagshub.com/31Joojo/final-project-mlops.mlflow"
export MLFLOW_TRACKING_USERNAME="31Joojo"
export MLFLOW_TRACKING_PASSWORD="..."
```

3. Pull data (DVC) :
```bash
dvc pull
```

4. Run locally with Docker Compose :
```bash
docker compose up --build
```

Services:
- Web: `http://localhost:3000`
- API: `http://localhost:8000`

---

## Reproducibility
This project is reproducible by design:
- **Data**: every training run references an explicit DVC revision (`dvc_data_rev`)
- **Code**: every run logs the git commit hash (`git_commit`)
- **Model**: each version is registered in MLflow Registry and can be loaded by stage or alias

To reproduce a training run:
1. Checkout the corresponding git commit. 
2. `dvc pull` the dataset version used in the run. 
3. Re-run the training script:
```bash
python -m ml.training.train
```
4. Compare metrics in MLflow and validate the registered model version.

---

## Testing
Tests are organized in three categories:
- **Unit tests**: core logic (schema validation, predictor, settings, DVC rev utilities)
- **Integration tests**: FastAPI routes with TestClient (registry-driven behavior mocked in CI)
- **E2E live tests**: run against staging/prod URLs:
  - `tests/e2e/test_staging_live.py`
  - `tests/e2e/test_prod_live.py`

Run locally:
```bash
pytest -q
pytest -q -m unit
pytest -q -m integration
pytest -q -m e2e
```

---

## Troubleshooting
1. API fails to load the model in Docker/cloud
- Ensure `MLFLOW_TRACKING_*` variables are set correctly.
- Verify `MODEL_NAME` and `MODEL_STAGE` match the Registry (stage/alias).
- In CI/dev mode, model loading can be mocked to avoid external network dependencies.

2. DVC pull fails in CI
- Confirm DagsHub Storage credentials are set as secrets (username + token).
- Ensure remote configuration uses basic auth and is applied locally during the workflow.

3. E2E tests flake due to deploy delay
Increase wait/retry timeouts.
Ensure the deploy hook completes and `/health` becomes ready before running `/predict`.
