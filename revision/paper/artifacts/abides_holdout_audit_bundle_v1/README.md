# ACE-RL ABIDES Holdout Audit Bundle v1

This bundle contains the standard-learner and two pre-frozen ecology holdouts.
`per_path.csv` is the primitive table. `per_seed.csv` and `statistics.json`
are derived only from that table. `step_records.jsonl.gz` contains 200,000
decision records with holdout and learner identifiers.

The official learner source is `marlbenchmark/on-policy` at pinned commit
`de66d7a4b23fac2513f56f96f73b3f5cb96695ac`. The ecology policies use the
repository's compact MAPPO-style intervention-adapter and are not renamed as
standard MAPPO. Compact ecology records expose realized flow shares rather
than latent routing probabilities; the step records label that provenance.

The cross-group audit uses an equal-weight mean over the ten evaluation paths
inside each policy seed. The compact ecology runner's preregistered source
summary pools low/high-risk steps instead. The verifier reconstructs that
source estimand as a separate integrity check; both definitions preserve the
reported direction, and they are not silently treated as identical numbers.

Inference uses policy-seed paired t intervals, exact sign tests, and a
10,000-replicate two-way paired bootstrap with RNG seed 20260714. Secondary
contrasts use Holm adjustment. Groups are never pooled to hide a reversal.
Run `python revision/experiments/verify_abides_holdout_audit_bundle.py` to
verify source correspondence, hashes, grids, statistics, and paper exports.
