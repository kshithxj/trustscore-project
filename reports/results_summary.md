# TrustScore Project Results Summary

*Split: TEST | Decision Policy: 15% validation decline threshold (unless otherwise noted) | Definitions: Positive class = Default (TARGET=1); Prediction = DECLINE (predicted 1); Approval (1 - decline) is the favorable outcome.*

## 1. Data Overview

- **Total Applicants**: 307,511
- **Overall Default Rate**: 0.080729 (8.07%)
- **Stratified Split**: 60% Train (184,506), 20% Validation (61,502), 20% Test (61,503)
- **Stratification Strategy**: Stratified across target_x_segment combinations with seed 42.

### Segment Counts & Default Rates (Full Dataset):
| Segment | Applicants | Share (%) | Default Rate | Description |
|---|---|---|---|---|
| `full_file` | 263,491 | 85.69% | 0.077301 | Has both Bureau records and prior Home Credit history |
| `thin_file` | 41,820 | 13.60% | 0.102822 | Has prior Home Credit history but NO Bureau records |
| `no_file` | 2,200 | 0.72% | 0.071364 | Has neither Bureau records nor prior Home Credit history |

### Demographic Group Sizes on TEST Split:
| Dimension | Group | Test Size | Defaults | Default Rate |
|---|---|---|---|---|
| gender | `F` | 40,541 | 2,856 | 0.070447 |
| gender | `M` | 20,961 | 2,109 | 0.100615 |
| binary_age | `<median` | 30,914 | 3,046 | 0.098531 |
| binary_age | `>=median` | 30,589 | 1,919 | 0.062735 |
| age_bin | `<30` | 8,979 | 1,037 | 0.115492 |
| age_bin | `30-44` | 24,922 | 2,243 | 0.090001 |
| age_bin | `45-59` | 20,555 | 1,345 | 0.065434 |
| age_bin | `60+` | 7,047 | 340 | 0.048247 |

## 2. Model Performance: Model A (Bureau Baseline) vs Model B (TrustScore)

*Split: TEST | Decision Policy: 15% validation decline threshold (unless otherwise noted) | Definitions: Positive class = Default (TARGET=1); Prediction = DECLINE (predicted 1); Approval (1 - decline) is the favorable outcome.*

| Segment / Slice | Model A Test AUC (95% CI) | Model B Test AUC (95% CI) | Paired Difference (B - A) [95% CI] | Precision Flag |
|---|---|---|---|---|
| **Overall (All Applicants)** | 0.751652 [0.745044, 0.758377] | 0.766810 [0.759674, 0.773116] | **+0.015163** [+0.012190, +0.017969] | Standard |
| `full_file` | 0.754255 [0.746639, 0.761343] | 0.767157 [0.759294, 0.774797] | **+0.013084** [+0.010011, +0.016078] | Standard |
| `thin_file` | 0.723296 [0.707265, 0.739383] | 0.755320 [0.739041, 0.772823] | **+0.032395** [+0.022979, +0.040515] | Standard |
| `no_file` | 0.814970 [0.739093, 0.883703] | 0.836659 [0.771039, 0.894991] | **+0.022282** [-0.026211, +0.072480] | Low precision (31 test defaults) |

- **Top Gain Features (Model B)**: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `AMT_CREDIT`, `prev_mean_amt_annuity`
- **Behavioral Share of Model B Importance**: 29.26% in thin_file, 23.70% in full_file.

## 3. Fairness Audit at Fixed 15% Validation Decline Policy

*Split: TEST | Decision Policy: 15% validation decline threshold (unless otherwise noted) | Definitions: Positive class = Default (TARGET=1); Prediction = DECLINE (predicted 1); Approval (1 - decline) is the favorable outcome.*

### Headline Demographic Gaps (Test Split):
| Attribute | Metric | Model A Gap (95% CI) | Model B Gap (95% CI) | Paired Difference (B - A) [95% CI] |
|---|---|---|---|---|
| gender | `dp_difference` | 0.073101 [0.067153, 0.078832] | 0.069729 [0.064048, 0.076108] | **-0.003372** [-0.007787, +0.001117] |
| gender | `dp_ratio` | 0.916467 [0.910006, 0.923275] | 0.920365 [0.913236, 0.926772] | **+0.003898** [-0.001185, +0.008771] |
| gender | `eo_difference` | 0.080805 [0.061200, 0.107428] | 0.085073 [0.059659, 0.111006] | **+0.004268** [-0.012895, +0.021096] |
| binary_age | `dp_difference` | 0.117602 [0.112394, 0.123459] | 0.108936 [0.103464, 0.114532] | **-0.008666** [-0.012859, -0.004574] |
| binary_age | `dp_ratio` | 0.870670 [0.864389, 0.876360] | 0.879841 [0.874070, 0.885856] | **+0.009171** [+0.004782, +0.013564] |
| binary_age | `eo_difference` | 0.191178 [0.161750, 0.217032] | 0.177717 [0.145592, 0.206470] | **-0.013461** [-0.036436, +0.007192] |
| four_age_bins | `dp_difference` | 0.258799 [0.248476, 0.266718] | 0.243943 [0.232775, 0.252682] | **-0.014856** [-0.022875, -0.007225] |
| four_age_bins | `dp_ratio` | 0.729694 [0.721814, 0.739568] | 0.744605 [0.735844, 0.755349] | **+0.014911** [+0.007481, +0.023065] |
| four_age_bins | `eo_difference` | 0.423385 [0.371351, 0.472963] | 0.400819 [0.347754, 0.452399] | **-0.022566** [-0.060950, +0.019691] |

## 4. Policy Robustness Across Decline Rates

*Split: TEST | Decision Policy: 15% validation decline threshold (unless otherwise noted) | Definitions: Positive class = Default (TARGET=1); Prediction = DECLINE (predicted 1); Approval (1 - decline) is the favorable outcome.*

| Target Decline Rate | Attribute | Model A Threshold | Model B Threshold | Model A DP Diff | Model B DP Diff | Model A EO Diff | Model B EO Diff |
|---|---|---|---|---|---|---|---|
| 10% | gender | 0.182096 | 0.184932 | 0.053275 | 0.050809 | 0.068263 | 0.076718 |
| 10% | binary_age | 0.182096 | 0.184932 | 0.084538 | 0.081910 | 0.169650 | 0.163933 |
| 15% | gender | 0.143708 | 0.144659 | 0.073101 | 0.069729 | 0.080805 | 0.085073 |
| 15% | binary_age | 0.143708 | 0.144659 | 0.117602 | 0.108936 | 0.191178 | 0.177717 |
| 20% | gender | 0.118561 | 0.118196 | 0.086679 | 0.084983 | 0.089065 | 0.089350 |
| 20% | binary_age | 0.118561 | 0.118196 | 0.142608 | 0.130693 | 0.205572 | 0.186182 |

## 5. Proxy Leakage Analysis (5-Fold CV XGBoost Proxy Models)

| Target Attribute | S1 AUC (App Features) | S2 AUC (Behavioral Features) | S3 AUC (Combined) | Majority Class Acc | Top Proxy Predictors |
|---|---|---|---|---|---|
| `gender` | 0.8961 +/- 0.0012 | 0.5749 +/- 0.0015 | 0.8974 +/- 0.0014 | 0.6584 | FLAG_OWN_CAR, OCCUPATION_TYPE, FLAG_DOCUMENT_8 |
| `binary_age` | 0.9325 +/- 0.0007 | 0.6364 +/- 0.0020 | 0.9342 +/- 0.0007 | 0.5002 | NAME_INCOME_TYPE, DAYS_EMPLOYED, CNT_CHILDREN |

## 6. SHAP Explanations

- **Scale**: margin (log-odds)
- **Method**: shap.TreeExplainer
- **Behavioral Importance Share**: full_file = 23.70%, thin_file = 29.26%, no_file = 14.34%
- **Top Behavioral Feature**: `prev_mean_amt_annuity` (Mean |SHAP| = 0.1058)

### Local Explanation: Median Declined Thin-File Applicant (SK_ID_CURR `333107`):
- **Model B Score**: 0.215282 (Validation Threshold = 0.144659, Outcome = No Default)
- **Base Value (Log-Odds)**: -2.4508
| Rank | Feature | Feature Value | SHAP Value (Log-Odds) |
|---|---|---|---|
| 1 | `EXT_SOURCE_2` | 0.034333358255672485 | +1.1069 |
| 2 | `DAYS_EMPLOYED` | -4421.0 | -0.1630 |
| 3 | `AMT_CREDIT` | 755190.0 | +0.1342 |
| 4 | `pos_n_records` | 35.0 | -0.1249 |
| 5 | `AMT_ANNUITY` | 30078.0 | +0.1214 |
| 6 | `ORGANIZATION_TYPE` | Other | -0.1018 |
| 7 | `EXT_SOURCE_3` | nan | +0.0949 |
| 8 | `prev_mean_amt_annuity` | 8808.75 | +0.0837 |

## 7. Fairness Mitigation and Matched-Decline-Rate Controls

*Split: TEST | Decision Policy: 15% validation decline threshold (unless otherwise noted) | Definitions: Positive class = Default (TARGET=1); Prediction = DECLINE (predicted 1); Approval (1 - decline) is the favorable outcome.*

### Mitigation Tradeoff Table with Realized Decline Rates:
| Sensitive Attribute | Policy Variant | Realized Decline Rate (95% CI) | Default Capture (TPR) | Balanced Accuracy | DP Difference | EO Difference | Single Threshold Used |
|---|---|---|---|---|---|---|---|
| gender | equalized_odds (mitigated) | 0.3808 [0.3767, 0.3850] | 0.7255 | 0.6875 | 0.0118 | 0.0116 | Group-Specific (Opt) |
| gender | equalized_odds (control) | 0.3804 [0.3770, 0.3840] | 0.7392 | 0.6951 | 0.1100 | 0.1028 | 0.066873 |
| gender | demographic_parity (mitigated) | 0.3503 [0.3472, 0.3543] | 0.7039 | 0.6924 | 0.0023 | 0.0400 | Group-Specific (Opt) |
| gender | demographic_parity (control) | 0.3521 [0.3482, 0.3561] | 0.7146 | 0.6972 | 0.1062 | 0.0985 | 0.072445 |
| binary_age | equalized_odds (mitigated) | 0.3091 [0.3051, 0.3128] | 0.6487 | 0.6847 | 0.0089 | 0.0050 | Group-Specific (Opt) |
| binary_age | equalized_odds (control) | 0.3105 [0.3070, 0.3143] | 0.6731 | 0.6972 | 0.1634 | 0.1722 | 0.081462 |
| binary_age | demographic_parity (mitigated) | 0.3369 [0.3333, 0.3406] | 0.6846 | 0.6891 | 0.0043 | 0.0205 | Group-Specific (Opt) |
| binary_age | demographic_parity (control) | 0.3377 [0.3338, 0.3415] | 0.7007 | 0.6975 | 0.1680 | 0.1721 | 0.075264 |

### Paired Differences vs Matched Single-Threshold Control (Mitigated minus Control):
| Run | DP Difference Diff (95% CI) | EO Difference Diff (95% CI) | Balanced Accuracy Diff (95% CI) | Default Capture Diff (95% CI) |
|---|---|---|---|---|
| `gender_equalized_odds` | **-0.0983** [-0.1020, -0.0945] | **-0.0912** [-0.1030, -0.0630] | -0.0077 [-0.0116, -0.0049] | -0.0137 [-0.0211, -0.0085] |
| `gender_demographic_parity` | **-0.1039** [-0.1108, -0.0898] | **-0.0585** [-0.0839, -0.0316] | -0.0048 [-0.0075, -0.0020] | -0.0107 [-0.0155, -0.0055] |
| `binary_age_equalized_odds` | **-0.1545** [-0.1585, -0.1505] | **-0.1672** [-0.1776, -0.1282] | -0.0125 [-0.0163, -0.0081] | -0.0244 [-0.0316, -0.0158] |
| `binary_age_demographic_parity` | **-0.1637** [-0.1735, -0.1511] | **-0.1516** [-0.1703, -0.1288] | -0.0083 [-0.0125, -0.0046] | -0.0161 [-0.0233, -0.0088] |
