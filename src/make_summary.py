"""Generate comprehensive results summary report in markdown from metrics artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFINITIONS_NOTE = (
    "*Split: TEST | Decision Policy: 15% validation decline threshold (unless otherwise noted) | "
    "Definitions: Positive class = Default (TARGET=1); Prediction = DECLINE (predicted 1); "
    "Approval (1 - decline) is the favorable outcome.*"
)


def generate_results_summary() -> str:
    metrics_dir = _PROJECT_ROOT / "reports" / "metrics"
    tables_dir = _PROJECT_ROOT / "reports" / "tables"

    # 1. Load artifacts
    with open(metrics_dir / "phase1_audit.json", "r") as f:
        p1 = json.load(f)
    with open(metrics_dir / "model_a.json", "r") as f:
        ma = json.load(f)
    with open(metrics_dir / "model_b.json", "r") as f:
        mb = json.load(f)
    with open(metrics_dir / "a_vs_b.json", "r") as f:
        avsb = json.load(f)
    with open(metrics_dir / "fairness.json", "r") as f:
        fair = json.load(f)
    with open(metrics_dir / "proxy.json", "r") as f:
        proxy = json.load(f)
    with open(metrics_dir / "shap_summary.json", "r") as f:
        shap_s = json.load(f)
    with open(metrics_dir / "mitigation.json", "r") as f:
        mit = json.load(f)
    with open(metrics_dir / "mitigation_control.json", "r") as f:
        mit_ctrl = json.load(f)

    base_rates_df = pd.read_csv(tables_dir / "base_rates.csv")
    fair_gaps_df = pd.read_csv(tables_dir / "fairness_gaps.csv")
    fair_dec_df = pd.read_csv(tables_dir / "fairness_by_decline_rate.csv")
    tradeoff_matched_df = pd.read_csv(tables_dir / "tradeoff_matched.csv")

    doc: list[str] = []

    # Title
    doc.append("# TrustScore Project Results Summary\n")
    doc.append(f"{DEFINITIONS_NOTE}\n")

    # Section 1: Data
    doc.append("## 1. Data Overview\n")
    doc.append(f"- **Total Applicants**: {p1['n_applicants']:,}")
    doc.append(f"- **Overall Default Rate**: {p1['target_mean']:.6f} ({p1['target_mean']*100:.2f}%)")
    doc.append("- **Stratified Split**: 60% Train (184,506), 20% Validation (61,502), 20% Test (61,503)")
    doc.append(f"- **Stratification Strategy**: Stratified across {p1['stratification_used']} combinations with seed 42.\n")

    doc.append("### Segment Counts & Default Rates (Full Dataset):")
    doc.append("| Segment | Applicants | Share (%) | Default Rate | Description |")
    doc.append("|---|---|---|---|---|")
    seg_desc = {
        "full_file": "Has both Bureau records and prior Home Credit history",
        "thin_file": "Has prior Home Credit history but NO Bureau records",
        "no_file": "Has neither Bureau records nor prior Home Credit history",
    }
    for seg, count in p1["segment_counts"]["overall"].items():
        share = count / p1["n_applicants"]
        dr = p1["default_rate_by_segment"][seg]
        doc.append(
            f"| `{seg}` | {count:,} | {share*100:.2f}% | {dr:.6f} | {seg_desc.get(seg, '')} |"
        )
    doc.append("")

    doc.append("### Demographic Group Sizes on TEST Split:")
    doc.append("| Dimension | Group | Test Size | Defaults | Default Rate |")
    doc.append("|---|---|---|---|---|")
    for _, r in base_rates_df.iterrows():
        if r["dimension"] in ["gender", "binary_age", "age_bin"]:
            doc.append(f"| {r['dimension']} | `{r['group']}` | {int(r['size']):,} | {int(r['defaults']):,} | {r['default_rate']:.6f} |")
    doc.append("")

    # Section 2: Model A vs Model B
    doc.append("## 2. Model Performance: Model A (Bureau Baseline) vs Model B (TrustScore)\n")
    doc.append(f"{DEFINITIONS_NOTE}\n")
    doc.append("| Segment / Slice | Model A Test AUC (95% CI) | Model B Test AUC (95% CI) | Paired Difference (B - A) [95% CI] | Precision Flag |")
    doc.append("|---|---|---|---|---|")
    
    a_ov = ma["test_results"]["overall"]
    b_ov = mb["test_results"]["overall"]
    d_ov = avsb["test_auc"]["overall"]["paired_bootstrap"]
    doc.append(
        f"| **Overall (All Applicants)** | {a_ov['auc']:.6f} [{a_ov['ci_95_lower']:.6f}, {a_ov['ci_95_upper']:.6f}] | "
        f"{b_ov['auc']:.6f} [{b_ov['ci_95_lower']:.6f}, {b_ov['ci_95_upper']:.6f}] | "
        f"**+{d_ov['mean_difference']:.6f}** [{d_ov['ci_95_lower']:+.6f}, {d_ov['ci_95_upper']:+.6f}] | Standard |"
    )
    for seg in ["full_file", "thin_file", "no_file"]:
        a_s = ma["test_results"]["by_segment"][seg]
        b_s = mb["test_results"]["by_segment"][seg]
        d_s = avsb["test_auc"]["by_segment"][seg]["paired_bootstrap"]
        flag = "Low precision (31 test defaults)" if seg == "no_file" else "Standard"
        doc.append(
            f"| `{seg}` | {a_s['auc']:.6f} [{a_s['ci_95_lower']:.6f}, {a_s['ci_95_upper']:.6f}] | "
            f"{b_s['auc']:.6f} [{b_s['ci_95_lower']:.6f}, {b_s['ci_95_upper']:.6f}] | "
            f"**+{d_s['mean_difference']:.6f}** [{d_s['ci_95_lower']:+.6f}, {d_s['ci_95_upper']:+.6f}] | {flag} |"
        )
    doc.append("")
    doc.append("- **Top Gain Features (Model B)**: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `AMT_CREDIT`, `prev_mean_amt_annuity`")
    doc.append(f"- **Behavioral Share of Model B Importance**: {shap_s['behavioral_share_by_segment']['thin_file']*100:.2f}% in thin_file, {shap_s['behavioral_share_by_segment']['full_file']*100:.2f}% in full_file.\n")

    # Section 3: Fairness
    doc.append("## 3. Fairness Audit at Fixed 15% Validation Decline Policy\n")
    doc.append(f"{DEFINITIONS_NOTE}\n")
    doc.append("### Headline Demographic Gaps (Test Split):")
    doc.append("| Attribute | Metric | Model A Gap (95% CI) | Model B Gap (95% CI) | Paired Difference (B - A) [95% CI] |")
    doc.append("|---|---|---|---|---|")
    ov_gaps = fair_gaps_df[fair_gaps_df["scope"] == "overall"]
    for _, r in ov_gaps.iterrows():
        doc.append(
            f"| {r['attribute']} | `{r['metric']}` | "
            f"{r['model_a_gap']:.6f} [{r['model_a_ci_lower']:.6f}, {r['model_a_ci_upper']:.6f}] | "
            f"{r['model_b_gap']:.6f} [{r['model_b_ci_lower']:.6f}, {r['model_b_ci_upper']:.6f}] | "
            f"**{r['delta_b_minus_a']:+.6f}** [{r['delta_ci_lower']:+.6f}, {r['delta_ci_upper']:+.6f}] |"
        )
    doc.append("")

    # Section 4: Robustness
    doc.append("## 4. Policy Robustness Across Decline Rates\n")
    doc.append(f"{DEFINITIONS_NOTE}\n")
    doc.append("| Target Decline Rate | Attribute | Model A Threshold | Model B Threshold | Model A DP Diff | Model B DP Diff | Model A EO Diff | Model B EO Diff |")
    doc.append("|---|---|---|---|---|---|---|---|")
    for _, r in fair_dec_df.iterrows():
        doc.append(
            f"| {r['target_decline_rate']*100:.0f}% | {r['attribute']} | {r['model_a_threshold']:.6f} | "
            f"{r['model_b_threshold']:.6f} | {r['model_a_dp_diff']:.6f} | {r['model_b_dp_diff']:.6f} | "
            f"{r['model_a_eo_diff']:.6f} | {r['model_b_eo_diff']:.6f} |"
        )
    doc.append("")

    # Section 5: Proxy Leakage
    doc.append("## 5. Proxy Leakage Analysis (5-Fold CV XGBoost Proxy Models)\n")
    doc.append("| Target Attribute | S1 AUC (App Features) | S2 AUC (Behavioral Features) | S3 AUC (Combined) | Majority Class Acc | Top Proxy Predictors |")
    doc.append("|---|---|---|---|---|---|")
    for tgt in ["gender", "binary_age"]:
        t_data = proxy["targets"][tgt]
        s1 = t_data["feature_sets"]["S1"]
        s2 = t_data["feature_sets"]["S2"]
        s3 = t_data["feature_sets"]["S3"]
        top_f = ", ".join([x["feature"] for x in t_data["top_5_features_by_gain_for_best_set"]["top_5"][:3]])
        doc.append(
            f"| `{tgt}` | {s1['mean_auc']:.4f} +/- {s1['std_auc']:.4f} | "
            f"{s2['mean_auc']:.4f} +/- {s2['std_auc']:.4f} | "
            f"{s3['mean_auc']:.4f} +/- {s3['std_auc']:.4f} | "
            f"{t_data['majority_class_accuracy']:.4f} | {top_f} |"
        )
    doc.append("")

    # Section 6: SHAP Explanations
    doc.append("## 6. SHAP Explanations\n")
    doc.append(f"- **Scale**: {shap_s['metadata']['scale']}")
    doc.append(f"- **Method**: {shap_s['metadata']['method']}")
    doc.append(f"- **Behavioral Importance Share**: full_file = {shap_s['behavioral_share_by_segment']['full_file']*100:.2f}%, thin_file = {shap_s['behavioral_share_by_segment']['thin_file']*100:.2f}%, no_file = {shap_s['behavioral_share_by_segment']['no_file']*100:.2f}%")
    doc.append(f"- **Top Behavioral Feature**: `{shap_s['top_behavioral_feature']['feature']}` (Mean |SHAP| = {shap_s['top_behavioral_feature']['mean_abs_shap']:.4f})\n")

    ch = shap_s["chosen_local_applicant"]
    doc.append(f"### Local Explanation: Median Declined Thin-File Applicant (SK_ID_CURR `{ch['SK_ID_CURR']}`):")
    doc.append(f"- **Model B Score**: {ch['score']:.6f} (Validation Threshold = {ch['threshold']:.6f}, Outcome = {'Default' if ch['target']==1 else 'No Default'})")
    doc.append(f"- **Base Value (Log-Odds)**: {ch['base_value_log_odds']:.4f}")
    doc.append("| Rank | Feature | Feature Value | SHAP Value (Log-Odds) |")
    doc.append("|---|---|---|---|")
    for idx_k, item in enumerate(ch["top_8_contributions"], 1):
        doc.append(f"| {idx_k} | `{item['feature']}` | {item['feature_value']} | {item['shap_value']:+.4f} |")
    doc.append("")

    # Section 7: Mitigation & Matched-Rate Control
    doc.append("## 7. Fairness Mitigation and Matched-Decline-Rate Controls\n")
    doc.append(f"{DEFINITIONS_NOTE}\n")
    doc.append("### Mitigation Tradeoff Table with Realized Decline Rates:")
    doc.append("| Sensitive Attribute | Policy Variant | Realized Decline Rate (95% CI) | Default Capture (TPR) | Balanced Accuracy | DP Difference | EO Difference | Single Threshold Used |")
    doc.append("|---|---|---|---|---|---|---|---|")
    for _, r in tradeoff_matched_df.iterrows():
        th_val = f"{r['control_threshold']:.6f}" if pd.notna(r["control_threshold"]) else "Group-Specific (Opt)"
        doc.append(
            f"| {r['sensitive_feature']} | {r['constraint']} ({r['policy_variant']}) | "
            f"{r['decline_rate']:.4f} [{r['decline_rate_ci_lower']:.4f}, {r['decline_rate_ci_upper']:.4f}] | "
            f"{r['default_capture']:.4f} | {r['balanced_accuracy']:.4f} | {r['dp_difference']:.4f} | "
            f"{r['eo_difference']:.4f} | {th_val} |"
        )
    doc.append("")

    doc.append("### Paired Differences vs Matched Single-Threshold Control (Mitigated minus Control):")
    doc.append("| Run | DP Difference Diff (95% CI) | EO Difference Diff (95% CI) | Balanced Accuracy Diff (95% CI) | Default Capture Diff (95% CI) |")
    doc.append("|---|---|---|---|---|")
    for r_key, r_data in mit_ctrl["runs"].items():
        pd_m = r_data["paired_difference_mitigated_minus_control"]
        pt = pd_m["point_estimates"]
        ci = pd_m["confidence_intervals_95"]
        doc.append(
            f"| `{r_key}` | **{pt['dp_difference']:+.4f}** [{ci['dp_difference'][0]:+.4f}, {ci['dp_difference'][1]:+.4f}] | "
            f"**{pt['eo_difference']:+.4f}** [{ci['eo_difference'][0]:+.4f}, {ci['eo_difference'][1]:+.4f}] | "
            f"{pt['balanced_accuracy']:+.4f} [{ci['balanced_accuracy'][0]:+.4f}, {ci['balanced_accuracy'][1]:+.4f}] | "
            f"{pt['default_capture']:+.4f} [{ci['default_capture'][0]:+.4f}, {ci['default_capture'][1]:+.4f}] |"
        )
    doc.append("")

    summary_text = "\n".join(doc)
    out_path = _PROJECT_ROOT / "reports" / "results_summary.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[make_summary] Saved {out_path}")
    return summary_text


if __name__ == "__main__":
    generate_results_summary()
