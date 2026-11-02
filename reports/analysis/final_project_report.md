# Football Odds Intelligence & Match Analytics — Final Project Report

**Author:** Mohd Afzal  
**Degree Track:** B.Tech Electronics and Communication Engineering (3rd Year)  
**Specialization:** Data Science, Machine Learning, and Quantitative Analytics  
**Date:** November 2026  

---

## Executive Summary

This project delivers an end-to-end, enterprise-grade Data Science and Machine Learning research investigation into global football betting markets, multi-bookmaker pricing dynamics, and predictive match outcome modeling. 

Using a real-world dataset covering **63,597 matches** and **~37 million odds ticks** across 4 major bookmakers (**Pinnacle, Bet365, SBOBET, Singbet**) between 2018 and 2026, we investigated 10 core analytical questions:
1. Empirical odds movement dynamics from open to close.
2. Cross-bookmaker pricing strategies and margin discrepancies.
3. Unsupervised detection of erratic odds movements (Isolation Forest).
4. Statistical evaluation of odds drift.
5. Mathematical formulation of normalized fair implied probabilities.
6. Calibration and accuracy of bookmaker quotes vs actual match outcomes.
7. Domestic and international league variation in home advantage.
8. Unsupervised market polarization discovery (draw probability collapse in anomalous fixtures).
9. Supervised machine learning benchmarking against the Bookmaker Consensus Baseline on an out-of-time test set (20,747 matches).
10. Deployment of an interactive Streamlit analytics platform.

---

## 1. Dataset Architecture & Quality Audit

The dataset consists of 5 relational entities linked by primary key `match_id`:
- **`matches`**: 63,597 matches across domestic leagues and international tournaments. 100% score integrity (0 negative goals, 0 half-time score violations).
- **`match_closing`**: 251,326 bookmaker quotes (average 3.95 bookmakers per fixture). 0 orphan matches.
- **`odds_1x2`, `odds_ah`, `odds_ou`**: High-frequency time-series tick movements (~37 million observations).

To ensure high-performance execution without memory exhaustion, raw tables were maintained in Apache Parquet format (reducing storage from ~2.5 GB of raw CSVs down to ~289 MB).

---

## 2. Quantitative & Statistical Findings

### 2.1 Bookmaker Margin Disparity & Sharpness Benchmark
The **Overround** $\Omega = \sum_{i} \frac{1}{O_i} - 1$ was computed across 251,326 quotes:
- **Pinnacle (Sharp Benchmark):** Median margin **3.34%** (Mean 3.62%, std 1.22%).
- **Singbet:** Median margin **5.45%** (Mean 6.80%, std 2.47%).
- **Bet365:** Median margin **5.95%** (Mean 6.18%, std 1.56%).
- **SBOBET:** Median margin **6.23%** (Mean 7.15%, std 2.17%).

**Hypothesis Testing:**
- A **Kruskal-Wallis H-test** rejected the null hypothesis of equal bookmaker pricing ($H = 107,374.91$, $p \approx 0.0$).
- A **Mann-Whitney U test** confirmed that Pinnacle prices matches with a statistically significantly lower margin than Bet365 ($p \approx 0.0$), confirming industry domain theory that Pinnacle operates as a low-margin, high-volume market maker.

### 2.2 Global Home Advantage Quantification
Across all 63,597 matches:
- Home Wins: **27,864 (43.81%)**
- Away Wins: **19,615 (30.84%)**
- Draws: **16,118 (25.34%)**

A **one-sample proportion Z-test** against the uniform random prior ($\frac{1}{3}$) yielded $Z = 56.06$ ($p \approx 0.0$, 95% Confidence Interval: **[43.43%, 44.20%]**), establishing home advantage as a profound structural phenomenon in global football. Furthermore, a **Pearson Chi-Square test** ($\chi^2 = 754.01$, $df = 64$, $p = 1.81 \times 10^{-118}$) proved outcome distributions are highly dependent on competition.

### 2.3 Systematic Odds Drift
Wilcoxon Signed-Rank tests evaluated paired opening and closing quotes across Home, Draw, and Away markets ($p < 10^{-90}$). On average, closing odds lengthen slightly (+0.10% to +0.27%) as bookmaker margins narrow toward kickoff due to competitive price discovery.

---

## 3. Supervised Machine Learning Benchmark

To ensure genuine data science integrity and eliminate lookahead bias, all supervised models were evaluated using a strict **chronological out-of-time test split**:
- **Training Set:** 42,328 matches (Kickoff prior to 2024)
- **Test Set:** 20,747 matches (Kickoff 2024 to 2026)

Predictor variables included only pre-match market signals (fair implied probabilities, market spreads, standard deviation of quotes across bookmakers, odds drift, and calendar attributes). **Zero score or post-match variables were accessible to the models.**

### 3.1 Model Benchmark Results

| Model | Accuracy (%) | Multi-Class Log Loss | Brier Score | Macro F1 |
|---|---|---|---|---|
| **Bookmaker Consensus Baseline** | 52.74% | 0.9706 | 0.5781 | 0.4468 |
| **Calibrated Logistic Regression** | **52.85%** | **0.9705** | **0.5783** | **0.4452** |
| **Random Forest Classifier** | 52.53% | 0.9718 | 0.5791 | 0.4485 |
| **XGBoost Classifier** | 52.69% | 0.9730 | 0.5797 | 0.4534 |

### 3.2 Machine Learning Insights
1. **Market Efficiency:** Football betting markets are remarkably efficient; closing bookmaker consensus quotes already incorporate massive global crowdsourced intelligence.
2. **Calibration over Complexity:** Calibrated Multinomial Logistic Regression slightly edged the bookmaker baseline (Log Loss 0.9705 vs 0.9706). Random Forest and XGBoost exhibited slight overconfidence on noisy categorical multi-class outcomes, underscoring that in noisy domains with low signal-to-noise ratios, well-calibrated linear models frequently outperform complex unregularized trees.

---

## 4. Unsupervised Anomaly Detection

An **Isolation Forest** model was trained on multi-bookmaker drift momentum, odds variance, and market spreads, flagging **1,262 fixtures (2.00%)** exhibiting erratic market behavior.

### 4.1 Discovery: Market Polarization & Draw Rate Collapse
Comparing normal fixtures against anomalous fixtures revealed an extraordinary structural pattern:
- **Normal Fixtures:** Home Win = 43.50%, Away Win = 30.79%, **Draw = 25.71%**.
- **Anomalous Fixtures:** Home Win = **59.27%**, Away Win = 33.68%, **Draw = 7.05%**.

In anomalous fixtures with extreme odds drift and bookmaker disagreement, the probability of a Draw collapses by **72.6%**, while decisive Home Wins surge. Market anomalies strongly reflect polarized matches where the market aggressively prices out the possibility of a draw.

---

## 5. Software Architecture & Deliverables

```
football dashboard/
├── data/
│   ├── raw/                 # Read-only Parquet datasets
│   └── processed/           # Cleaned datasets & ML feature matrices
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_machine_learning.ipynb
│   └── 07_model_evaluation.ipynb
├── src/
│   ├── data/ (loader.py, cleaner.py, profiling.py)
│   ├── features/ (builder.py)
│   ├── models/ (trainer.py, anomaly.py)
│   ├── evaluation/ (metrics.py, statistical_tests.py)
│   └── visualization/ (plots.py)
├── models/
│   ├── best_model.joblib
│   └── isolation_forest.joblib
├── dashboard/
│   └── app.py               # Streamlit application
├── tests/                   # 15 Passing unit tests
├── reports/
│   ├── figures/             # Diagnostic & publication charts
│   └── analysis/            # JSON summaries & final project report
```

---

## 6. Conclusion

This project successfully proves that combining domain-grounded feature engineering, statistical hypothesis testing, and rigorous leakage-free machine learning provides actionable quantitative intelligence in complex sports financial markets. All code, data pipelines, and dashboards are fully reproducible, robustly unit-tested, and prepared for professional Data Science portfolio evaluation.
