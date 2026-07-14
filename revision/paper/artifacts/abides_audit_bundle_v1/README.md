# ACE-RL ABIDES Audit Bundle v1

This bundle is the committed source for the domain-benchmark tables and
inference. `per_path.csv` is the primitive result table; `per_seed.csv` is
derived from it. `step_records.jsonl.gz` contains the paired per-decision
audit trail. Run `python revision/experiments/verify_abides_audit_bundle.py
--bundle revision/paper/artifacts/abides_audit_bundle_v1` to verify hashes,
recompute statistics, and rerun integrity checks.

The Brero threshold adaptation is calibration-unmatched and is therefore an
operating-point comparison, not a same-budget superiority test.

The frozen convergence integrity check failed. Efficacy statistics are recomputed for auditability but are not confirmatory evidence.
