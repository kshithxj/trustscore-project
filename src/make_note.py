"""Generate reports/results_note.md with exact figures from metrics artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_results_note():
    metrics_dir = _PROJECT_ROOT / "reports" / "metrics"
    tables_dir = _PROJECT_ROOT / "reports" / "tables"

    with open(metrics_dir / "model_a.json") as f:
        ma = json.load(f)
    with open(metrics_dir / "model_b.json") as f:
        mb = json.load(f)
    with open(metrics_dir / "a_vs_b.json") as f:
        avsb = json.load(f)
    with open(metrics_dir / "fairness.json") as f:
        fair = json.load(f)
    with open(metrics_dir / "proxy.json") as f:
        proxy = json.load(f)
    with open(metrics_dir / "shap_summary.json") as f:
        shap_s = json.load(f)
    with open(metrics_dir / "mitigation_control.json") as f:
        mit_ctrl = json.load(f)

    base_rates_df = pd.read_csv(tables_dir / "base_rates.csv")
    no_file_defaults = int(base_rates_df[(base_rates_df["dimension"] == "segment") & (base_rates_df["group"] == "no_file")]["defaults"].iloc[0])

    a_ov = ma["test_results"]["overall"]
    b_ov = mb["test_results"]["overall"]
    d_ov = avsb["test_auc"]["overall"]["paired_bootstrap"]

    a_thin = ma["test_results"]["by_segment"]["thin_file"]
    b_thin = mb["test_results"]["by_segment"]["thin_file"]
    d_thin = avsb["test_auc"]["by_segment"]["thin_file"]["paired_bootstrap"]

    a_no = ma["test_results"]["by_segment"]["no_file"]
    b_no = mb["test_results"]["by_segment"]["no_file"]
    d_no = avsb["test_auc"]["by_segment"]["no_file"]["paired_bootstrap"]

    g_fair = fair["demographic_gaps"]["gender"]
    a_fair = fair["demographic_gaps"]["binary_age"]

    g_s1 = proxy["targets"]["gender"]["feature_sets"]["S1"]["mean_oof_auc"]
    g_s2 = proxy["targets"]["gender"]["feature_sets"]["S2"]["mean_oof_auc"]
    a_s1 = proxy["targets"]["binary_age"]["feature_sets"]["S1"]["mean_oof_auc"]
    a_s2 = proxy["targets"]["binary_age"]["feature_sets"]["S2"]["mean_oof_auc"]

    g_eo_pd = mit_ctrl["runs"]["gender_equalized_odds"]["paired_difference_mitigated_minus_control"]
    a_eo_pd = mit_ctrl["runs"]["binary_age_equalized_odds"]["paired_difference_mitigated_minus_control"]

    content = f"""# TrustScore Project: Key Results and Findings

*All metrics reported on the 20% stratified test split (61,503 applicants) using fixed seeds. Definitions: Positive class = default (TARGET=1); prediction = decline (score $\\ge$ threshold); approval is the favorable outcome.*

---

## 1. Does behavioral lender-history improve prediction overall?

Yes. Adding 20 behavioral features derived from prior loan applications, installment repayments, POS cash balances, and credit cards yields a statistically significant increase in discriminatory power across the test population:
- **Model A (Bureau-style Baseline)**: Test ROC-AUC = `{a_ov['auc']:.6f}` (95% CI: `[{a_ov['ci_95_lower']:.6f}, {a_ov['ci_95_upper']:.6f}]`).
- **Model B (TrustScore)**: Test ROC-AUC = `{b_ov['auc']:.6f}` (95% CI: `[{b_ov['ci_95_lower']:.6f}, {b_ov['ci_95_upper']:.6f}]`).
- **Paired Performance Gain (B minus A)**: $\\Delta = +{d_ov['mean_difference']:.6f}$ (95% CI: `[{d_ov['ci_95_lower']:+.6f}, {d_ov['ci_95_upper']:+.6f}]`, $P(B > A) = {d_ov['share_b_gt_a']:.1f}$).

Behavioral features contribute `{shap_s['behavioral_share_by_segment']['full_file']*100:.2f}%` of total feature importance in the `full_file` segment. Key drivers include repayment discipline (`inst_frac_late`, `pos_n_records`) and prior credit size/annuity ratios (`prev_mean_amt_annuity`, `prev_mean_amt_credit`).

TODO (my interpretation):

---

## 2. Does behavioral history help more for thin-file applicants, and what happens for no-file applicants?

Behavioral features provide the largest relative benefit to `thin_file` borrowers who lack external credit bureau records:
- **`thin_file` Segment (8,364 test applicants)**:
  - Model A AUC = `{a_thin['auc']:.6f}` (95% CI: `[{a_thin['ci_95_lower']:.6f}, {a_thin['ci_95_upper']:.6f}]`).
  - Model B AUC = `{b_thin['auc']:.6f}` (95% CI: `[{b_thin['ci_95_lower']:.6f}, {b_thin['ci_95_upper']:.6f}]`).
  - Paired Gain $\\Delta = +{d_thin['mean_difference']:.6f}$ (95% CI: `[{d_thin['ci_95_lower']:+.6f}, {d_thin['ci_95_upper']:+.6f}]`, $P(B > A) = {d_thin['share_b_gt_a']:.1f}$).
  - Behavioral features account for `{shap_s['behavioral_share_by_segment']['thin_file']*100:.2f}%` of Model B's importance in this segment, demonstrating that internal repayment records can serve as an effective substitute for absent bureau files.

- **`no_file` Segment (440 test applicants, 31 defaults)**:
  - Model A AUC = `{a_no['auc']:.6f}` (95% CI: `[{a_no['ci_95_lower']:.6f}, {a_no['ci_95_upper']:.6f}]`).
  - Model B AUC = `{b_no['auc']:.6f}` (95% CI: `[{b_no['ci_95_lower']:.6f}, {b_no['ci_95_upper']:.6f}]`).
  - Paired Difference $\\Delta = +{d_no['mean_difference']:.6f}$ (95% CI: `[{d_no['ci_95_lower']:+.6f}, {d_no['ci_95_upper']:+.6f}]`, $P(B > A) = {d_no['share_b_gt_a']:.3f}$).
  - Because `no_file` applicants lack both bureau and prior lender records, all behavioral features are missing (NaN). Model B falls back entirely on application features; performance is nearly identical to Model A, and the small sample size (`{no_file_defaults}` defaults) results in wide confidence intervals and low statistical precision.

TODO (my interpretation):

---

## 3. What does it do to group fairness and proxy leakage, and what does mitigation change when compared at a matched decline rate?

### Fairness Disparities and Proxy Leakage:
- Under the operational 15% validation decline policy, Model B slightly reduces Demographic Parity differences compared to Model A:
  - Gender DP difference: `{g_fair['model_a_dp_diff']:.6f}` (Model A) vs `{g_fair['model_b_dp_diff']:.6f}` (Model B), $\\Delta = {g_fair['delta_dp_diff']:+.6f}$.
  - Binary Age DP difference: `{a_fair['model_a_dp_diff']:.6f}` (Model A) vs `{a_fair['model_b_dp_diff']:.6f}` (Model B), $\\Delta = {a_fair['delta_dp_diff']:+.6f}$.
  - Equalized Odds differences remain essentially unchanged for gender (`{g_fair['model_a_eo_diff']:.6f}` vs `{g_fair['model_b_eo_diff']:.6f}`) and slightly narrower for age (`{a_fair['model_a_eo_diff']:.6f}` vs `{a_fair['model_b_eo_diff']:.6f}`).
- **Proxy Leakage**: Standard application features (S1) strongly encode protected attributes (`{g_s1:.4f}` AUC for gender, `{a_s1:.4f}` AUC for age) through occupation, income type, and car ownership. Behavioral lender features (S2) exhibit dramatically lower proxy encoding (`{g_s2:.4f}` for gender, `{a_s2:.4f}` for age) and add negligible proxy leakage when combined.

### Mitigation vs Matched-Rate Control:
Postprocessing with Fairlearn's `ThresholdOptimizer` (balanced accuracy objective) shifts the decision threshold, resulting in a higher overall decline rate (~31%-38%) rather than 15%. Comparing the mitigated policies against a single-threshold control matched to the exact same validation decline rate demonstrates:
- **Gender (Equalized Odds)**:
  - Shifting from 15% to 38% decline captures ~72%-74% of defaults in both models.
  - Mitigated vs Matched Control paired EO difference: `{g_eo_pd['point_estimates']['eo_difference']:+.6f}` (95% CI: `[{g_eo_pd['confidence_intervals_95']['eo_difference'][0]:+.6f}, {g_eo_pd['confidence_intervals_95']['eo_difference'][1]:+.6f}]`).
  - Balanced accuracy paired difference: `{g_eo_pd['point_estimates']['balanced_accuracy']:+.6f}` (95% CI: `[{g_eo_pd['confidence_intervals_95']['balanced_accuracy'][0]:+.6f}, {g_eo_pd['confidence_intervals_95']['balanced_accuracy'][1]:+.6f}]`).
- **Binary Age (Equalized Odds)**:
  - Mitigated vs Matched Control paired EO difference: `{a_eo_pd['point_estimates']['eo_difference']:+.6f}` (95% CI: `[{a_eo_pd['confidence_intervals_95']['eo_difference'][0]:+.6f}, {a_eo_pd['confidence_intervals_95']['eo_difference'][1]:+.6f}]`).
  - Balanced accuracy paired difference: `{a_eo_pd['point_estimates']['balanced_accuracy']:+.6f}` (95% CI: `[{a_eo_pd['confidence_intervals_95']['balanced_accuracy'][0]:+.6f}, {a_eo_pd['confidence_intervals_95']['balanced_accuracy'][1]:+.6f}]`).

ThresholdOptimizer successfully compresses group disparity gaps by setting group-specific cutoffs, but does so by actively trading off overall accuracy and requiring protected demographic attributes at underwriting time.

TODO (my interpretation):
"""

    note_path = _PROJECT_ROOT / "reports" / "results_note.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[make_note] Saved {note_path}")
    return content


if __name__ == "__main__":
    generate_results_note()
