# Project Limitations: TrustScore Study

This document outlines the methodological, operational, and ethical limitations of the TrustScore credit risk pipeline, as evaluated on the Home Credit Default Risk benchmark.

## 1. Single Lender Context
The empirical evaluation is conducted entirely on a single historical dataset from one consumer finance provider (Home Credit). Operating characteristics, customer demographics, and underwriting policies observed here may not generalize to other consumer lending institutions, geographies, or economic cycles.

## 2. Definition of "Behavioral" vs. "Alternative" Data
The supplementary data tables utilized in this study (`previous_application`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance`) reflect previous retail lending relationships with the same institution. They do not comprise non-financial alternative data sources such as utility payments, telecommunication records, or open-banking cash-flow streams. The project evaluates internal relationship banking records rather than broad third-party alternative data.

## 3. Endogeneity and Historical Institutional Bias
Certain behavioral features, notably `prev_approved_rate` and `prev_refused_rate`, reflect the lender's own prior credit decisions rather than autonomous borrower behavior. These variables inherently encode historical underwriting standards, and any legacy biases present in past approval decisions may be reinforced by downstream machine learning models.

## 4. Protected Demographic Attribute Limitations
- **Gender**: Encoded as binary (`F` vs `M`) in the primary dataset; 4 applicants nationwide have missing/unrecorded gender recorded as `XNA` and were necessarily excluded from gender fairness evaluations.
- **Age**: Age operates as both a direct actuarial risk factor and a complex proxy for career maturity, accumulated wealth, and life stage. Categorization into binary median splits or coarse age brackets facilitates audit comparability but simplifies a continuous, non-linear biological and socioeconomic phenomenon.

## 5. Proxy Leakage and Feature Redundancy
Demographic attributes (gender and age) were strictly excluded from model feature sets. However, empirical proxy tests demonstrate that standard application features (S1) predict gender with ~0.896 OOF AUC and age with ~0.932 OOF AUC via occupational type, income category, and vehicle ownership. Simply omitting protected attributes does not prevent models from reconstructing demographic disparities through legitimate correlated predictors.

## 6. Sample Size Constraints in the `no_file` Segment
The `no_file` subpopulation (applicants lacking both bureau and prior lender records) comprises only 440 individuals and 31 default events in the test split. Because of small sample counts, evaluation metrics in this segment have wide confidence intervals and must be interpreted as descriptive rather than definitive.

## 7. Multiple Hypothesis Testing
The audit performs multiple simultaneous subgroup comparisons across segments, gender, and age categories. Sub-segment analyses (e.g., segment x demographic intersections) are exploratory and have not been adjusted with family-wise error rate corrections.

## 8. Policy Dependency of Fairness Metrics
Disparity metrics (Demographic Parity difference, Equalized Odds difference) are evaluated at a specific operational threshold (e.g., top 15% decline rate). Fairness gaps shift substantially across different target approval volumes and economic cutoffs, as demonstrated in the robustness sweeps.

## 9. Legal and Ethical Constraints on Decision-Time Mitigation
Postprocessing mitigation using Fairlearn's `ThresholdOptimizer` enforces parity by applying separate, group-specific cutoffs to protected demographic groups at decision time. In credit underwriting, using protected demographic characteristics (such as gender or age) to set differentiated credit criteria is explicitly restricted or unlawful under fair lending statutes (e.g., the U.S. Equal Credit Opportunity Act / Regulation B) in many jurisdictions. Furthermore, the optimizer optimizes for balanced accuracy, shifting the aggregate decline rate from 15% to over 30%. These experiments represent academic proof-of-concept simulations rather than commercially deployable underwriting policies.

## 10. Non-Causal Nature of Models
All models evaluated (Model A and Model B) are correlational predictive algorithms trained on observational cross-sectional data. No causal claims are made regarding whether improving behavioral indicators causes lower credit default. This project is an academic proof-of-concept for responsible credit scoring methodologies.
