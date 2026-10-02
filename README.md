# Wine Cultivar MLOps Pipeline with CI/CD & MLflow Automation

[![CI/CD MLOps Quality Gate](https://github.com/fast-nu/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/fast-nu/wine-mlops-pipeline/actions)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.17.2-0194E2.svg)
![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-black.svg)

<<<<<<< HEAD
An enterprise-grade (Main line update), reproducible MLOps pipeline for multi-class chemical cultivar classification using the 13-feature Wine dataset.
=======
An enterprise-grade (Branch conflict-simulation), reproducible MLOps pipeline for multi-class chemical cultivar classification using the 13-feature Wine dataset.
>>>>>>> conflict-simulation

## 🎯 Architecture & Features
- **Local Automation via Makefile**: Standardized developer workflow (`install`, `lint`, `test`, `train`, `clean`).
- **Modular Codebase**: Strict separation between data ingestion (`src/data.py`), model training & tracking (`src/train.py`), and inference verification (`src/evaluate.py`).
- **MLflow Tracking & Model Registry**: Systematic 5-fold cross-validation tuning across Random Forest and Gradient Boosting classifier families, signature logging, and champion version registration with `@champion` alias.
- **Automated MLOps Quality Gate**: CI pipeline in GitHub Actions validating linting, unit data contracts, F1-score performance threshold (>= 0.88), latency benchmarks (<= 30 ms), and output schema integrity.
- **Git Branching & Conflict Resolution**: Engineered feature branching and deterministic merge conflict handling.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
make install
```

### 2. Code Quality & Linting
```bash
make lint
```

### 3. Run Automated Tests & Quality Gate
```bash
make test
```

### 4. Train Models & Track with MLflow
```bash
make train
```

### 5. Evaluate Champion Model
```bash
python src/evaluate.py
```

### 6. Launch MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
