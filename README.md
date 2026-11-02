# ⚽ ODDSPULSE — Football Odds Intelligence & Match Analytics

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-15%20passed%20(100%25)-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-ODDSPULSE%20Platform-FF4B4B.svg)](dashboard/app.py)
[![Design System](https://img.shields.io/badge/Theme-Dark%20%26%20Lime%20Green-C5F23B.svg)](dashboard/app.py)

**ODDSPULSE** is a high-performance quantitative sports analytics platform developed to analyze market efficiency, bookmaker pricing power, odds momentum, and predictive machine learning models across **63,597 matches** and **~37 million odds ticks** from 2018 to 2026 across four global bookmakers (**Pinnacle, Bet365, SBOBET, Singbet**).

---

## 🎨 Premium Sports Analytics Design System

The platform has been redesigned into a sleek, dark-themed sports intelligence interface inspired by modern professional football analytics systems:

- **Main Background:** `#080D16` (Deep navy/black)
- **Secondary Background:** `#101927` (Charcoal navy)
- **Card Background:** `#172333` (Card surface with borders `#253449`)
- **Primary Accent:** `#C5F23B` (Vibrant lime green)
- **Secondary Accent:** `#27B8D1` (Cyan blue)
- **Typography:** Modern clean sans-serif with monospace metric callouts
- **Information Architecture:** 8 specialized analytical modules with interactive Plotly visualizations and zero data leakage.

---

## 🧭 8 Integrated Platform Modules

1. **📊 Executive Overview:** Top-line KPI metrics (63,597 matches, ~37M ticks, 4 bookmakers, 33 competitions), Match & Market Spotlight card, and live model forecast vs. bookmaker consensus bar chart.
2. **🔍 Match Explorer:** Search fixtures by team name, filter by league, season, or outcome; interactive single-match inspector featuring full bookmaker quote breakdown (Open vs Close odds, margin %, fair implied probabilities).
3. **📈 Odds Analytics:** Closing decimal odds distributions, fair implied probability violin plots, odds drift momentum histograms, and league-level home advantage rankings.
4. **🏦 Bookmaker Analysis:** Margin audit comparing Pinnacle (3.34% median), Singbet (5.45%), Bet365 (5.95%), and SBOBET (6.23%); statistical verification via Kruskal-Wallis ($H = 107,374.91, p < 10^{-300}$).
5. **🤖 Predictive Analytics:** Interactive decimal odds probability calculator with automatic overround margin removal, and strict chronological validation disclosures (evaluated on 20,747 test matches).
6. **🚨 Anomaly Radar:** Unsupervised Isolation Forest (2% contamination, 1,262 fixtures flagged); visual proof of the empirical **Draw rate collapse** from 25.7% to 7.05% (-72.6%) in polarized anomalous markets.
7. **📋 Model Performance:** Rigorous benchmark suite comparing Bookmaker Baseline, Calibrated Logistic Regression, Random Forest, and XGBoost with accuracy bars, interactive confusion matrix heatmaps, and log loss / Brier score evaluations.
8. **ℹ️ About Project:** System architecture, LaTeX mathematical formulations, data lineage, and developer profile.

---

## 📊 Machine Learning Model Benchmarks

Evaluated strictly on an **out-of-time test set of 20,747 fixtures (2024–2026)** to prevent lookahead leakage:

| Model | Accuracy (%) | Multi-Class Log Loss | Brier Score | Macro F1 |
|---|---|---|---|---|
| **Bookmaker Consensus Baseline** | 52.74% | 0.9706 | 0.5781 | 0.4468 |
| **Calibrated Logistic Regression** | **52.85%** | **0.9705** | **0.5783** | **0.4452** |
| **Random Forest Classifier** | 52.53% | 0.9718 | 0.5791 | 0.4485 |
| **XGBoost Classifier** | 52.69% | 0.9730 | 0.5797 | 0.4534 |

---

## 🏗️ Project Architecture

```
Football-Odd-Prediction/
├── .streamlit/
│   └── config.toml                # Custom dark theme configuration
├── dashboard/
│   └── app.py                     # Streamlit Interactive Web Application (ODDSPULSE)
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
│   ├── best_model.joblib          # Trained ML model artifact (Logistic Regression)
│   └── isolation_forest.joblib    # Unsupervised anomaly model
├── tests/                         # Pytest unit testing suite (15 tests)
├── reports/
│   ├── figures/                   # 12 Publication-grade charts & diagrams
│   └── analysis/                  # Statistical JSONs & final project report
├── requirements.txt
├── README.md
├── plan.md
└── .gitignore
```

---

## 🚀 Quick Start Guide

### 1. Run Automated Unit Tests (15 Tests)
```bash
pytest tests/
```
All 15 tests should pass with 100% success rate.

### 2. Launch ODDSPULSE Dashboard
```bash
streamlit run dashboard/app.py
```

### 3. Re-run Full Data & Modeling Pipeline
```bash
# 1. Run data cleaning pipeline:
python src/data/cleaner.py

# 2. Run feature engineering pipeline:
python src/features/builder.py

# 3. Train & benchmark supervised ML models:
python src/models/trainer.py

# 4. Fit unsupervised anomaly detection:
python src/models/anomaly.py

# 5. Run statistical hypothesis tests:
python src/evaluation/run_statistical_analysis.py
```

---

## 👤 Author

**Mohd Afzal**  
B.Tech Electronics and Communication Engineering (3rd Year)  
Interest: Data Science, Quantitative Sports Analytics, Machine Learning  
GitHub: [Bhagwan01/Football-Odd-Prediction](https://github.com/Bhagwan01/Football-Odd-Prediction)

## 📄 License
This project is for educational and portfolio purposes under the MIT License.
