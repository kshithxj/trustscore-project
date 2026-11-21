Among individual lender-history sources added to Model A, installments_payments provides the largest predictive lift with a paired ΔAUC of +0.006811 over the bureau-only baseline.
Removing installments_payments from Model B causes the steepest performance drop of -0.004680 AUC, confirming it as the most critical behavioral component.
On thin-file applicants, Model B delivers a total AUC lift of +0.032024 compared to Model A, while the single addition of installments_payments achieves +0.011738.
Regarding demographic fairness, adding credit_card_balance shifts the binary age equalized odds gap by +0.018692, indicating that behavioral sources modestly impact age disparity alongside accuracy gains.
