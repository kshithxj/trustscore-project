# Home Credit Default Risk – Fairness-Aware Credit Scoring

A machine-learning pipeline for predicting loan default on the
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
dataset, with an emphasis on **model fairness**, **explainability**, and
**threshold optimisation**.

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> && cd miniproject

# 2. Create a virtual environment (Python 3.10+)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the smoke tests
pytest tests/ -v
```

## Where to Place the Kaggle Data

Download the CSVs from the
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
competition page and place them **unzipped** in:

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

> The `data/` folder is git-ignored; you must download the files yourself.

## Folder Layout

```
miniproject/
├── configs/
│   └── config.yaml            # master configuration
├── data/
│   ├── raw/home_credit/       # original Kaggle CSVs (git-ignored)
│   ├── processed/             # cleaned / engineered data
│   └── splits/                # train / valid / test splits
├── src/
│   ├── data/                  # data loading & preprocessing
│   ├── features/              # feature engineering
│   ├── models/                # model training & evaluation
│   ├── fairness/              # fairness audits & mitigation
│   ├── explain/               # SHAP / explainability
│   └── utils.py               # shared helpers (load_config, set_seed)
├── reports/
│   ├── metrics/               # JSON / CSV evaluation results
│   ├── figures/               # plots (PNG, PDF)
│   └── predictions/           # submission files (git-ignored)
├── notebooks/                 # exploratory / presentation notebooks
├── tests/                     # pytest test suite
├── docs/                      # project documentation
├── requirements.txt           # pinned dependencies (Python 3.10)
├── .gitignore
└── README.md
```

## Configuration

All tuneable parameters live in [`configs/config.yaml`](configs/config.yaml):
seed, split ratios, decline rate, bootstrap iterations, and folder paths.

## Project Phases

| Phase | Description |
|-------|-------------|
| 0 | Project setup (this phase) |
| 1 | Data audit, segment analysis, train/valid/test split |
| 2 | Model A – baseline classifier |
| 3 | Behavioural features + Model B |
| 4 | Fairness audit + proxy-leakage test |
| 5 | SHAP explanations + ThresholdOptimizer |
| 6 | Results freeze, final report, poster v2 |
