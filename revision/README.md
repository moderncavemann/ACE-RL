# Revision Workspace

This directory is the clean work area for the revised ACE-RL project.

Use the local-only `archive/v1_rejected_submission/` as a source archive, not as
active code. Migrate files here only after deciding they support the revised
claim.

Suggested structure:

- `src/` - revised implementation.
- `experiments/` - runnable experiment scripts. Generated outputs go under
  `experiments/outputs/` and are ignored by git.
- `paper/` - revised manuscript assets.
- `notes/` - review responses, experiment plans, and evidence audits.

## Toy LOB Sequence

The toy stage follows a strict order:

1. Calibrate canonical parameters:
   `python3 revision/experiments/calibrate_toy_lob_params.py`
2. Run no-intervention observation only:
   `python3 revision/experiments/run_toy_lob_observation.py --episodes 1500 --eval-episodes 50 --n-sweep 2,5,10 --n-sweep-episodes 1000`
3. Calibrate the risk-aware gate from no-intervention records:
   `python3 revision/experiments/calibrate_toy_lob_gate.py --observation-dir revision/experiments/outputs/observation --output revision/experiments/outputs/gate_calibration/gate_config.json`
4. If observation passes, run simple baselines plus the diagnostic oracle gate:
   `python3 revision/experiments/run_toy_lob_baselines.py --seeds 0,1,2,3,4,5,6,7,8,9 --random-rate auto --gate-config revision/experiments/outputs/gate_calibration/gate_config.json`
5. If baselines justify a gate, run gate ablations:
   `python3 revision/experiments/run_toy_lob_ablation.py --gate-config revision/experiments/outputs/gate_calibration/gate_config.json`
6. Aggregate and plot:
   `python3 revision/experiments/aggregate_toy_lob_results.py --input revision/experiments/outputs/baselines/summary.json`
   `python3 revision/experiments/plot_toy_lob_results.py --observation-dir revision/experiments/outputs/observation --baseline-summary revision/experiments/outputs/baselines/summary.json --gate-config revision/experiments/outputs/gate_calibration/gate_config.json`

Do not use intervention results as paper evidence until the no-intervention
observation shows the toy environment actually exhibits the target problem.
The `oracle_gate` baseline is diagnostic only: it uses the true risk-adjusted
collusion gap to test whether routing would work with correct timing, and is not
a deployable method.

The tabular learner state includes risk, average spread, own spread, market
minimum spread, and spread range. The last two terms are required so a one-agent
undercut can be visible in an N-agent market.
