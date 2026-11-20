# TrustScore Project: Key Results and Findings

*All metrics reported on the 20% stratified test split (61,503 applicants) using fixed seeds. Definitions: Positive class = default (TARGET=1); prediction = decline (score $\ge$ threshold); approval is the favorable outcome.*

---

## 1. Does behavioral lender-history improve prediction overall?

Yes. Adding 20 behavioral features derived from prior loan applications, installment repayments, POS cash balances, and credit cards yields a statistically significant increase in discriminatory power across the test population:
- **Model A (Bureau-style Baseline)**: Test ROC-AUC = `0.751652` (95% CI: `[0.745044, 0.758377]`).
- **Model B (TrustScore)**: Test ROC-AUC = `0.766810` (95% CI: `[0.759674, 0.773116]`).
- **Paired Performance Gain (B minus A)**: $\Delta = +0.015163$ (95% CI: `[+0.012190, +0.017969]`, $P(B > A) = 1.0$).

Behavioral features contribute `23.70%` of total feature importance in the `full_file` segment. Key drivers include repayment discipline (`inst_frac_late`, `pos_n_records`) and prior credit size/annuity ratios (`prev_mean_amt_annuity`, `prev_mean_amt_credit`).

TODO (my interpretation):

---

## 2. Does behavioral history help more for thin-file applicants, and what happens for no-file applicants?

Behavioral features provide the largest relative benefit to `thin_file` borrowers who lack external credit bureau records:
- **`thin_file` Segment (8,364 test applicants)**:
  - Model A AUC = `0.723296` (95% CI: `[0.707265, 0.739383]`).
  - Model B AUC = `0.755320` (95% CI: `[0.739041, 0.772823]`).
  - Paired Gain $\Delta = +0.032395$ (95% CI: `[+0.022979, +0.040515]`, $P(B > A) = 1.0$).
  - Behavioral features account for `29.26%` of Model B's importance in this segment, demonstrating that internal repayment records can serve as an effective substitute for absent bureau files.

- **`no_file` Segment (440 test applicants, 31 defaults)**:
  - Model A AUC = `0.814970` (95% CI: `[0.739093, 0.883703]`).
  - Model B AUC = `0.836659` (95% CI: `[0.771039, 0.894991]`).
  - Paired Difference $\Delta = +0.022282$ (95% CI: `[-0.026211, +0.072480]`, $P(B > A) = 0.825$).
  - Because `no_file` applicants lack both bureau and prior lender records, all behavioral features are missing (NaN). Model B falls back entirely on application features; performance is nearly identical to Model A, and the small sample size (`31` defaults) results in wide confidence intervals and low statistical precision.

TODO (my interpretation):

---

## 3. What does it do to group fairness and proxy leakage, and what does mitigation change when compared at a matched decline rate?

### Fairness Disparities and Proxy Leakage:
- Under the operational 15% validation decline policy, Model B slightly reduces Demographic Parity differences compared to Model A:
  - Gender DP difference: `0.073101` (Model A) vs `0.069729` (Model B), $\Delta = -0.003372$.
  - Binary Age DP difference: `0.117602` (Model A) vs `0.108936` (Model B), $\Delta = -0.008666$.
  - Equalized Odds differences remain essentially unchanged for gender (`0.080805` vs `0.085073`) and slightly narrower for age (`0.191178` vs `0.177717`).
- **Proxy Leakage**: Standard application features (S1) strongly encode protected attributes (`0.8961` AUC for gender, `0.9325` AUC for age) through occupation, income type, and car ownership. Behavioral lender features (S2) exhibit dramatically lower proxy encoding (`0.5749` for gender, `0.6364` for age) and add negligible proxy leakage when combined.

### Mitigation vs Matched-Rate Control:
Postprocessing with Fairlearn's `ThresholdOptimizer` (balanced accuracy objective) shifts the decision threshold, resulting in a higher overall decline rate (~31%-38%) rather than 15%. Comparing the mitigated policies against a single-threshold control matched to the exact same validation decline rate demonstrates:
- **Gender (Equalized Odds)**:
  - Shifting from 15% to 38% decline captures ~72%-74% of defaults in both models.
  - Mitigated vs Matched Control paired EO difference: `-0.091207` (95% CI: `[-0.103022, -0.063012]`).
  - Balanced accuracy paired difference: `-0.007653` (95% CI: `[-0.011631, -0.004912]`).
- **Binary Age (Equalized Odds)**:
  - Mitigated vs Matched Control paired EO difference: `-0.167222` (95% CI: `[-0.177561, -0.128226]`).
  - Balanced accuracy paired difference: `-0.012504` (95% CI: `[-0.016336, -0.008146]`).

ThresholdOptimizer successfully compresses group disparity gaps by setting group-specific cutoffs, but does so by actively trading off overall accuracy and requiring protected demographic attributes at underwriting time.

TODO (my interpretation):
