# SHAP Analysis Notes: Model B Interpretability and Proxy Leakage

## 1. Local Explanation for Median Declined Thin-File Applicant

- **Applicant SK_ID_CURR**: `333107`
- **Segment**: `thin_file` (lender behavioral history present; no bureau history)
- **Model B Predicted Default Probability**: `0.215282`
- **Validation Decline Threshold**: `0.144659`
- **Decision**: **DECLINED** (Score `0.215282` $\ge$ Threshold `0.144659`)
- **True Outcome (TARGET)**: `0` (Did not default)
- **Base Value (Margin Log-Odds)**: `-2.4508`

### Top 8 Contributing Features (Margin Log-Odds Scale):

| Rank | Feature Name | Feature Value | SHAP Value (Log-Odds) | Impact on Risk |
|---|---|---|---|---|
| 1 | `EXT_SOURCE_2` | `0.034333358255672485` | `+1.1069` | Increases Risk (+) |
| 2 | `DAYS_EMPLOYED` | `-4421.0` | `-0.1630` | Decreases Risk (-) |
| 3 | `AMT_CREDIT` | `755190.0` | `+0.1342` | Increases Risk (+) |
| 4 | `pos_n_records` | `35.0` | `-0.1249` | Decreases Risk (-) |
| 5 | `AMT_ANNUITY` | `30078.0` | `+0.1214` | Increases Risk (+) |
| 6 | `ORGANIZATION_TYPE` | `Other` | `-0.1018` | Decreases Risk (-) |
| 7 | `EXT_SOURCE_3` | `nan` | `+0.0949` | Increases Risk (+) |
| 8 | `prev_mean_amt_annuity` | `8808.75` | `+0.0837` | Increases Risk (+) |

### Plain-English Narrative:
Applicant `333107` is a thin-file borrower who was declined under TrustScore Model B's 15% decline policy. Their predicted default probability was `0.2153`, slightly exceeding the validation threshold of `0.1447`. The applicant actually had TARGET = `0`, representing a false positive (unnecessary rejection of a non-defaulting borrower). 

Examining their top SHAP contributions on the log-odds scale reveals that the primary drivers pushing this applicant over the risk threshold were:
1. `EXT_SOURCE_2` (value: `0.034333358255672485`, SHAP: `+1.1069`): this feature exerted the largest upward pressure on the default log-odds.
2. `DAYS_EMPLOYED` (value: `-4421.0`, SHAP: `-0.1630`).
3. `AMT_CREDIT` (value: `755190.0`, SHAP: `+0.1342`).

Despite positive offsets from features such as `prev_mean_amt_annuity` (SHAP `+0.0837`), the net contribution pushed the log-odds score into the top 15% riskiest tranche, causing the loan decline.

---

## 2. Potential Demographic Proxy Features Among Top SHAP Contributors

Matching the top 20 Model B features by mean |SHAP| against known demographic proxies and top-gain features from `reports/metrics/proxy.json`:

- **Suspicious Proxy Attributes Monitored**: `OCCUPATION_TYPE`, `NAME_INCOME_TYPE`, `DAYS_EMPLOYED`, `FLAG_OWN_CAR`, `OWN_CAR_AGE`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `ORGANIZATION_TYPE`, plus top-gain features from 5-fold proxy CV.
- **Top SHAP Features Matching Proxy List**:
  - **`EXT_SOURCE_1`**: Mean |SHAP| = `0.1551` | Gender Proxy: No | Age Proxy: Yes (Top 5 CV Gain)
  - **`ORGANIZATION_TYPE`**: Mean |SHAP| = `0.1115` | Gender Proxy: No | Age Proxy: No
  - **`DAYS_EMPLOYED`**: Mean |SHAP| = `0.0962` | Gender Proxy: No | Age Proxy: Yes (Top 5 CV Gain)
  - **`OCCUPATION_TYPE`**: Mean |SHAP| = `0.0870` | Gender Proxy: Yes (Top 5 CV Gain) | Age Proxy: No
  - **`OWN_CAR_AGE`**: Mean |SHAP| = `0.0764` | Gender Proxy: Yes (Top 5 CV Gain) | Age Proxy: No

### Analysis:
While demographic attributes (gender, age) are strictly excluded from Model B inputs, legitimate credit variables such as `OCCUPATION_TYPE`, `NAME_INCOME_TYPE`, and `DAYS_EMPLOYED` appear prominently in the model's top SHAP explanations. These exact variables were demonstrated in Phase 4 to carry high proxy information for gender (OOF AUC 0.896) and age (OOF AUC 0.932). Consequently, disparities in group approval rates (Demographic Parity difference ~0.070 for gender and ~0.109 for binary age) are largely driven through these high-importance socio-economic proxies rather than direct attribute conditioning.
