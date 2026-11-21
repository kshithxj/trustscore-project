# TrustScore: Fairness-Aware Credit Scoring on Home Credit Default Risk

A reproducible, fairness-audited credit scoring pipeline evaluating whether behavioral lender-history improves default prediction over traditional bureau-style underwriting, how benefits distribute across credit segments (`full_file`, `thin_file`, `no_file`), and how group fairness tradeoffs behave under threshold mitigation.

---

## 1. Quick Start & One-Command Reproduction

To reproduce all results, metrics, figures, and tables from raw data from scratch:

```bash
# 1. Activate environment
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 2. Run the complete pipeline end-to-end
python -m src.run_all

# Optional: skip SHAP computation for a faster smoke run
python -m src.run_all --skip-shap

# 3. Run the full test suite
pytest -v
```

---

## 1.1 Demo Dashboard

Interactive Streamlit demonstration dashboard exploring model performance, flip dynamics, policy simulation, fairness tradeoffs, proxy leakage, SHAP applicant explanations, post-processing mitigation, and live scoring of held-out test applicants:

```bash
# 1. Install dependencies
pip install streamlit

# 2. Precompute policy frontier table and SHAP applicant sample
python -m src.app.precompute

# 3. Launch the dashboard
streamlit run src/app/streamlit_app.py
```

---

## 2. Dataset Source & Setup

- **Dataset Source**: [Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
- **Download Date**: TODO: add date
- **Placement**: Place the unzipped raw CSV files into `data/raw/home_credit/`:
  ```
  data/raw/home_credit/
  ├── application_train.csv
  ├── application_test.csv
  ├── bureau.csv
  ├── bureau_balance.csv
  ├── credit_card_balance.csv
  ├── installments_payments.csv
  ├── POS_CASH_balance.csv
  └── previous_application.csv
  ```
  *(Note: All data files, models, and predictions are strictly git-ignored).*

---

## 3. Key Documentation & Reports

- **Results Summary**: [reports/results_summary.md](reports/results_summary.md) — Comprehensive numeric breakdown across all 7 study dimensions.
- **Results Note**: [reports/results_note.md](reports/results_note.md) — One-page synthesis answering the three core project research questions.
- **Project Limitations**: [docs/limitations.md](docs/limitations.md) — Plain-language discussion of institutional, legal, and statistical limitations.
- **Data Dictionary**: [docs/data_dictionary.md](docs/data_dictionary.md) — Definitions of all base and engineered features.
- **Reproducibility Audit**: [reports/metrics/reproducibility_check.json](reports/metrics/reproducibility_check.json) — Exact numeric diffs ($\le 10^{-9}$) across reruns.

---

## 4. Repository Layout

```
miniproject/
├── configs/
│   ├── config.yaml               # Master pipeline config (seeds, splits, paths)
│   └── model_params.yaml         # XGBoost hyperparameters
├── data/                         # [Git-ignored] Raw, processed, and split files
│   ├── raw/home_credit/          # Unzipped Kaggle CSV tables
│   ├── processed/                # applicant_base, eval_attrs, behavioral_features
│   └── splits/                   # train_ids, valid_ids, test_ids (60/20/20 stratified)
├── docs/
│   ├── PROJECT_CONTEXT.md        # Technical architecture and problem statement
│   ├── data_dictionary.md        # Feature schemas and descriptions
│   └── limitations.md            # Plain-language project limitations
├── models/                       # [Git-ignored] Serialized model artifacts
│   ├── model_a.json              # Bureau baseline XGBoost model
│   └── model_b.json              # TrustScore behavioral XGBoost model
├── reports/
│   ├── figures/                  # 300 DPI figures (AUC, fairness gaps, SHAP, mitigation)
│   ├── metrics/                  # JSON performance, fairness, and audit artifacts
│   ├── tables/                   # CSV result tables and matched tradeoff tables
│   ├── results_summary.md        # 7-section comprehensive study summary
│   ├── results_note.md           # 3-question executive summary with TODO lines
│   └── shap_notes.md             # Local explanations and proxy leakage analysis
├── src/
│   ├── data/                     # Ingestion, validation guards, and splitting
│   ├── features/                 # Bureau and behavioral feature engineering
│   ├── models/                   # Training pipelines for Model A and Model B
│   ├── fairness/                 # Decision policies, audits, proxy tests, mitigation
│   ├── explainability/           # TreeSHAP computation and figure generation
│   ├── make_summary.py           # Generator for reports/results_summary.md
│   ├── make_figures.py           # Publication figure generator
│   ├── make_note.py              # Generator for reports/results_note.md
│   ├── run_all.py                # Master one-command reproduction runner
│   └── utils.py                  # Seed and YAML config helpers
└── tests/                        # Comprehensive pytest test suite (Phases 1-6)
```
