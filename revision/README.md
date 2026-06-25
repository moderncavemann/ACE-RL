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

## Formal Toy Sweep

Use separate seeds for threshold calibration and intervention evaluation. A
minimum formal pass is:

1. Calibration observation:
   `python3 revision/experiments/run_toy_lob_observation.py --episodes 1000 --eval-episodes 50 --seeds 0,1,2,3,4 --jobs 5 --output-dir revision/experiments/outputs/observation_calibration_sweep_v1`
2. Freeze gate threshold:
   `python3 revision/experiments/calibrate_toy_lob_gate.py --observation-dir revision/experiments/outputs/observation_calibration_sweep_v1 --output revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json`
3. Evaluate held-out baselines:
   `python3 revision/experiments/run_toy_lob_baselines.py --episodes 1000 --eval-episodes 50 --seeds 5,6,7,8,9,10,11,12,13,14 --jobs 10 --random-rate auto --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output revision/experiments/outputs/baselines_sweep_v1/summary.json`
4. Aggregate held-out baselines:
   `python3 revision/experiments/aggregate_toy_lob_results.py --input revision/experiments/outputs/baselines_sweep_v1/summary.json --output-csv revision/experiments/outputs/aggregated_sweep_v1/toy_lob_baseline_aggregate.csv --output-json revision/experiments/outputs/aggregated_sweep_v1/toy_lob_baseline_aggregate.json`
5. Plot held-out diagnostics:
   `python3 revision/experiments/plot_toy_lob_results.py --observation-dir revision/experiments/outputs/observation_calibration_sweep_v1 --baseline-summary revision/experiments/outputs/baselines_sweep_v1/summary.json --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output-dir revision/experiments/outputs/figures_sweep_v1`
6. If the baseline sweep passes, run held-out ablations:
   `python3 revision/experiments/run_toy_lob_ablation.py --episodes 1000 --eval-episodes 50 --seeds 5,6,7,8,9,10,11,12,13,14 --jobs 10 --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output revision/experiments/outputs/ablation_sweep_v1/summary.json`

Promote to a stronger final sweep only after this pass is stable. The next
scale-up should use 20 held-out evaluation seeds and 1500-3000 training
episodes.

## Toy Confirmation Sweep

The first full toy confirmation sweep used the frozen gate from
`gate_calibration_sweep_v1` and held out seeds `15-34`:

1. Confirmation baselines:
   `python3 revision/experiments/run_toy_lob_baselines.py --episodes 1500 --eval-episodes 100 --seeds 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34 --jobs 32 --random-rate auto --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output revision/experiments/outputs/baselines_confirmation_sweep_v1/summary.json`
2. Confirmation ablations:
   `python3 revision/experiments/run_toy_lob_ablation.py --episodes 1500 --eval-episodes 100 --seeds 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34 --jobs 32 --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output revision/experiments/outputs/ablation_confirmation_sweep_v1/summary.json`
3. Aggregate:
   `python3 revision/experiments/aggregate_toy_lob_results.py --input revision/experiments/outputs/baselines_confirmation_sweep_v1/summary.json --output-csv revision/experiments/outputs/aggregated_confirmation_sweep_v1/toy_lob_baseline_aggregate.csv --output-json revision/experiments/outputs/aggregated_confirmation_sweep_v1/toy_lob_baseline_aggregate.json`
   `python3 revision/experiments/aggregate_toy_lob_results.py --input revision/experiments/outputs/ablation_confirmation_sweep_v1/summary.json --output-csv revision/experiments/outputs/aggregated_confirmation_sweep_v1/toy_lob_ablation_aggregate.csv --output-json revision/experiments/outputs/aggregated_confirmation_sweep_v1/toy_lob_ablation_aggregate.json`
4. Plot:
   `python3 revision/experiments/plot_toy_lob_results.py --observation-dir revision/experiments/outputs/observation_calibration_sweep_v1 --baseline-summary revision/experiments/outputs/baselines_confirmation_sweep_v1/summary.json --ablation-summary revision/experiments/outputs/ablation_confirmation_sweep_v1/summary.json --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output-dir revision/experiments/outputs/figures_confirmation_sweep_v1`

The repo-visible summary is
`revision/notes/toy_lob_confirmation_sweep_v1_summary.md`.

## Toy Robustness Sweep

After confirmation, run controlled robustness extensions before moving to full
LOB or MAPPO/CRPO:

`python3 revision/experiments/run_toy_lob_robustness.py --episodes 800 --eval-episodes 50 --seeds 35,36,37,38,39,40,41,42,43,44 --scenarios base_n10,n5,n20,cost_wide,risk_fast,long_horizon --jobs 32 --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output revision/experiments/outputs/robustness_sweep_v1/summary.json`

The first robustness summary is
`revision/notes/toy_lob_robustness_sweep_v1_summary.md`.
