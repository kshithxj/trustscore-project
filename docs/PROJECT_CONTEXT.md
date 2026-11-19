# Project Context – Home Credit Default Risk

> **Purpose**: This document gives future sessions (or collaborators) full
> context so they can pick up work without re-reading the original brief.

---

## 1. Project Summary

Build a **fairness-aware credit scoring** pipeline on the Kaggle
*Home Credit Default Risk* dataset.  The pipeline must:

* Predict loan default (`TARGET = 1`) from application and bureau data.
* Evaluate two models: a baseline (**Model A**) and an improved model with
  behavioural features (**Model B**).
* Audit fairness across protected segments (age, gender, etc.) using
  Fairlearn / AIF360 metrics.
* Detect proxy leakage of protected attributes.
* Generate SHAP explanations and apply `ThresholdOptimizer` for equitable
  decision thresholds.
* Produce a reproducible report with metrics, figures, and a poster.

---

## 2. Rules & Constraints

1. **Reproducibility** – every run uses `seed = 42`; all random state is
   set via `src/utils.set_seed()`.
2. **No data leakage** – splits are created once and persisted; validation
   and test sets are never used for feature engineering.
3. **Decline rate** – the top 15 % riskiest applicants are declined
   (`decline_rate = 0.15`).
4. **Bootstrap confidence intervals** – 200 iterations for metric CIs.
5. **Allowed libraries** – see `requirements.txt` (NumPy, pandas,
   scikit-learn, LightGBM, XGBoost, Fairlearn, AIF360, SHAP, matplotlib,
   seaborn, SciPy, statsmodels, PyYAML, python-dotenv, JupyterLab, pytest).
6. **Config-driven** – all tuneable values live in `configs/config.yaml`.
7. **Modular code** – each `src/` sub-package owns one concern; notebooks
   call library code, not the other way around.

---

## 3. Phase Plan

| Phase | Title | Key Deliverables |
|-------|-------|-----------------|
| **0** | Project setup | Folder structure, config, requirements, README, smoke tests |
| **1** | Data audit + segments + split | EDA notebook, segment definitions, persisted train/valid/test splits |
| **2** | Model A | Baseline LightGBM/XGBoost, ROC-AUC, classification report |
| **3** | Behavioural features + Model B | Bureau/balance features, retrained model, comparison metrics |
| **4** | Fairness + proxy-leakage test | Demographic parity, equalised odds, proxy-leakage regression |
| **5** | SHAP + ThresholdOptimizer | SHAP summary/dependence plots, fairness-adjusted thresholds |
| **6** | Results freeze, report, poster v2 | Final metrics JSON, PDF report, poster PDF |

---

## 4. Folder Layout

```
miniproject/
├── configs/config.yaml
├── data/{raw/home_credit, processed, splits}
├── src/{data, features, models, fairness, explain, utils.py}
├── reports/{metrics, figures, predictions}
├── notebooks/
├── tests/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. Key Configuration Values

| Key | Value |
|-----|-------|
| `seed` | 42 |
| `target_column` | TARGET |
| `id_column` | SK_ID_CURR |
| `split.train / valid / test` | 0.6 / 0.2 / 0.2 |
| `decline_rate` | 0.15 |
| `bootstrap_n` | 200 |
