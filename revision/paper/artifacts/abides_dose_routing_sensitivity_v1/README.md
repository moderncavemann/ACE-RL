# ABIDES Dose And Routing Sensitivity v1 Audit Bundle

This is the repo-visible audit export of the preregistered secondary
sensitivity panel. It contains three realized-dose groups and five unique
routing-intensity groups. The 10-policy-seed panel must not replace the v5
20-seed primary result or be used to select a new primary operating point.

Run `python revision/experiments/verify_abides_dose_routing_sensitivity.py`
to recompute per-seed aggregates, paired t intervals, exact sign tests,
two-way paired bootstrap intervals, Holm adjustment, budget diagnostics,
per-step path integrity, and the frozen routing classification.
