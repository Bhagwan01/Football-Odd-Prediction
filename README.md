# ⚽ Football Odds Intelligence & Match Analytics

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-15%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Ready-FF4B4B.svg)](dashboard/app.py)

A professional, end-to-end Data Science, Machine Learning, and Quantitative Analytics portfolio project analyzing historical football betting odds, bookmaker behavior, odds movements, and match outcomes across **63,597 matches** and **~37 million odds observations** (2018–2026).

---

## 🎯 Key Project Highlights

- **Scale:** 63,597 historical football fixtures across global competitions; 251,326 multi-bookmaker opening & closing odds quotes; ~37M tick-level odds movements.
- **Sharp Bookmaker Benchmark:** Statistically demonstrated that **Pinnacle** operates as the sharpest benchmark bookmaker with a **3.34%** median overround, compared to **5.95%** for Bet365 and **6.23%** for SBOBET (Kruskal-Wallis $H = 107,374.91$, $p < 10^{-300}$).
- **Home Advantage Quantification:** One-sample Z-test confirms global Home Win rate of **43.81%** ($Z = 56.06$, $p \approx 0.0$, 95% CI: [43.43%, 44.20%]).
- **Leakage-Free ML Benchmarking:** Strict chronological out-of-time evaluation (Train: 42,328 matches < 2024; Test: 20,747 matches 2024–2026). Our **Calibrated Logistic Regression** matches/edges the bookmaker consensus baseline with **52.85% Accuracy** and **0.9705 Log Loss**.
- **Market Anomaly Radar:** Unsupervised Isolation Forest flagged 1,262 erratic fixtures; discovered that in anomalous fixtures, the **Draw rate collapses by 72.6%** (from 25.7% down to 7.05%), indicating extreme market polarization.
- **Interactive Streamlit Dashboard:** 6-module interactive analytics application featuring an interactive odds probability engine.
- **100% Test Coverage:** 15 unit tests across data loaders, cleaning pipelines, feature engineering, statistical hypothesis testing, ML models, and anomaly detectors.

---

## 🏗️ Project Architecture

```
football dashboard/
├── data/
│   ├── raw/                       # Raw Apache Parquet datasets (read-only)
│   └── processed/                 # Cleaned datasets & ML feature matrices
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_machine_learning.ipynb
│   └── 07_model_evaluation.ipynb
├── src/
│   ├── data/                      # loader.py, cleaner.py, profiling.py
│   ├── features/                  # builder.py
│   ├── models/                    # trainer.py, anomaly.py
│   ├── evaluation/                # metrics.py, statistical_tests.py
│   └── visualization/             # plots.py
├── models/
│   ├── best_model.joblib          # Trained ML model artifact
│   └── isolation_forest.joblib    # Unsupervised anomaly model
├── dashboard/
│   └── app.py                     # Interactive Streamlit Web Application
├── tests/                         # Pytest unit testing suite (15 tests)
├── reports/
│   ├── figures/                   # Publication-grade charts & diagrams
│   └── analysis/                  # Statistical JSONs & final project report
├── requirements.txt
├── README.md
├── plan.md
└── .gitignore
```

---

## 📊 Machine Learning Model Benchmarks

Evaluated on an **out-of-time test set of 20,747 fixtures (2024–2026)**:

| Model | Accuracy (%) | Multi-Class Log Loss | Brier Score | Macro F1 |
|---|---|---|---|---|
| **Bookmaker Consensus Baseline** | 52.74% | 0.9706 | 0.5781 | 0.4468 |
| **Calibrated Logistic Regression** | **52.85%** | **0.9705** | **0.5783** | **0.4452** |
| **Random Forest Classifier** | 52.53% | 0.9718 | 0.5791 | 0.4485 |
| **XGBoost Classifier** | 52.69% | 0.9730 | 0.5797 | 0.4534 |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.12+ (tested on Python 3.14.6)
- Virtual environment

### 2. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd "football dashboard"

# Activate your virtual environment
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
```bash
pytest tests/
```
All 15 tests should pass.

### 4. Run the Data & Modeling Pipelines
```bash
# 1. Profile raw data
python src/data/profiling.py

# 2. Run data cleaning pipeline
python src/data/cleaner.py

# 3. Run statistical hypothesis tests & generate figures
python src/evaluation/run_statistical_analysis.py

# 4. Engineer ML feature matrix
python src/features/builder.py

# 5. Train & benchmark supervised ML models
python src/models/trainer.py

# 6. Fit unsupervised anomaly detection (Isolation Forest)
python src/models/anomaly.py
```

### 5. Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📈 Visual Reports & Artifacts

The pipeline generates publication-grade visualizations in `reports/figures/`:
- `missing_values.png` — Dataset missingness audit
- `initial_distributions.png` — Match outcomes, goal frequencies, and yearly coverage
- `closing_odds_distribution.png` — Decimal odds distributions
- `bookmaker_margins.png` — Overround margin boxplots and density curves (Pinnacle vs Bet365 vs SBOBET vs Singbet)
- `odds_drift_distribution.png` — 1X2 market odds drift percentages
- `league_home_advantage.png` — Competition rankings by home win percentage
- `model_comparison.png` — ML accuracy, log loss, and Brier score benchmarks
- `confusion_matrices.png` — Multi-class confusion heatmaps on the out-of-time test set
- `calibration_curves.png` — Reliability diagrams comparing ML probabilities vs Bookmaker consensus
- `anomaly_distribution.png` — Isolation Forest decision score distribution and draw rate collapse

---

## 👤 Author

**Mohd Afzal**  
B.Tech Electronics and Communication Engineering (3rd Year)  
Interest: Data Science, Machine Learning, and Quantitative Analytics  

## 📄 License
This project is for educational and portfolio purposes.
