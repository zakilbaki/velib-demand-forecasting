# Velib Demand Forecasting

End-to-end ML / MLOps project to predict **next-hour bike availability** for Paris Velib stations.

## Stack

Python, PostgreSQL, pandas, scikit-learn, MLflow, FastAPI, Docker Compose.

## What the project does

1. Ingests live Velib snapshots from the **CityBikes API**
2. Stores station history in **PostgreSQL**
3. Rebuilds a supervised hourly forecasting dataset
4. Trains and evaluates models with **MLflow**
5. Promotes a validated model for serving
6. Serves predictions through **FastAPI**

## Architecture

```text
CityBikes API
  -> ingestion pipeline
  -> PostgreSQL
  -> dataset builder
  -> training / evaluation
  -> model candidates
  -> promoted current model
  -> FastAPI
```

## Repository Layout

```text
src/
  ingestion/   # API client, mapping, validation, persistence
  features/    # training dataset builder
  training/    # CV training + final evaluation
  serving/     # API, feature reconstruction, model promotion
  db.py
  modeling_config.py

init_sql/
  init.sql

serving_models/
  current/     # model currently served
  candidates/  # evaluated candidates
```

## Data Layer

- `stations`: static station metadata
- `availability_history`: station availability snapshots over time

The ingestion pipeline is idempotent:
- station metadata is upserted
- availability uses `(station_id, timestamp)` to avoid duplicate snapshots

Main ingestion command:

```bash
python -m src.main
```

## Dataset and Features

The ML dataset is built from PostgreSQL, not directly from the raw API payloads.

Dataset logic:
- aggregate station history to the **hourly** level
- keep the **last observation of each hour**
- create `free_bikes_next_hour`
- keep only true `T -> T+1h` transitions

Official modeling window:
- `2026-01-01` to `2026-03-16`

Best feature set:
- `free_bikes_current`
- `empty_slots_current`
- `hour_of_day`
- `is_weekend`
- `latitude`
- `longitude`
- `free_bikes_t_minus_1`
- `free_bikes_t_minus_2`
- `free_bikes_t_minus_3`
- `delta_1h`
- `delta_3h`

## Modeling

Selected model:
- `GradientBoostingRegressor`

Performance:
- CV RMSE: `3.0244`
- CV MAE: `2.0670`
- CV R²: `0.9325`
- Test RMSE: `2.8223`
- Test MAE: `1.9663`
- Test R²: `0.9395`

## MLflow

MLflow is used for:
- experiment tracking
- parameters and metrics
- model artifacts

Local setup:
- `mlflow.db` stores run metadata
- `mlruns/` stores run artifacts

## Serving Workflow

The project separates:
- **evaluation** of model candidates
- **promotion** of the model actually served

### Evaluate a candidate

```bash
python -m src.training.evaluate_regression
```

This exports a candidate bundle to:

```text
serving_models/candidates/<run_id>/
```

### Promote a candidate

```bash
python -m src.serving.promote_model --run-id <run_id>
```

This copies the chosen bundle to:

```text
serving_models/current/
```

### API routes

- `GET /health`
- `POST /predict`
- `POST /predict/station-state`

The `station-state` route rebuilds lag and geography features from PostgreSQL before prediction.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d postgres
python -m src.main
python -m src.training.train_regression
python -m src.training.evaluate_regression
python -m src.serving.promote_model --run-id <run_id>
docker compose up -d --build api
```

Test the API:

```bash
curl http://127.0.0.1:8000/health
```

## Scope

This is a strong **local MLOps project** with:
- ingestion
- PostgreSQL storage
- MLflow tracking
- explicit model promotion
- Dockerized FastAPI serving

Not included yet:
- remote model registry
- cloud deployment
- CI/CD
- production monitoring
