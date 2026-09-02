# Recommendation Engine

A production-oriented recommendation system project designed to grow from exploratory analysis into a reusable ML service.

## Project Scope

The planned system will cover:

- Exploratory data analysis
- Popularity-based, content-based, and collaborative filtering
- Hybrid recommendations
- Recommender-system evaluation
- Reusable Python pipelines and model serialization
- FastAPI and Streamlit interfaces
- Automated testing, Docker, GitHub Actions, and deployment readiness

This initial phase establishes the project structure and engineering foundations. Recommendation algorithms are intentionally not implemented yet.

## Setup

The project targets Python 3.11 or newer and uses the `recsys` virtual environment.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\recsys\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Repository Layout

- `app/`: Streamlit application components
- `api/`: FastAPI application layer
- `configs/`: Versioned configuration files
- `data/`: Raw, intermediate, processed, and external datasets
- `models/`: Serialized model artifacts
- `notebooks/`: Exploratory and experimental notebooks
- `reports/`: Figures and evaluation metrics
- `src/`: Reusable package code
- `tests/`: Automated tests
- `.github/workflows/`: CI/CD workflows

## Development Checks

```powershell
python -m pytest
ruff check .
python -m build
```

## Data and Artifact Policy

Large datasets and generated artifacts remain outside version control. The tracked `.gitkeep` files preserve the expected directory structure until real inputs and outputs are added.
