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

## N=20 Follow-up

The first targeted N=20 follow-up keeps the frozen calibration threshold and
checks whether the larger-N caveat is training length or feature-distribution
shift:

1. N=20 longer-training follow-up:
   `python3 revision/experiments/run_toy_lob_robustness.py --episodes 1500 --eval-episodes 100 --seeds 45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64 --scenarios n20 --methods no_intervention,always_on,threshold,gated_intervention,spread_only_gate --jobs 32 --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output revision/experiments/outputs/n20_followup_1500_v1/summary.json`
2. N=10 and N=20 no-intervention diagnostics:
   `python3 revision/experiments/run_toy_lob_observation.py --episodes 800 --eval-episodes 50 --seeds 45,46,47,48,49,50,51,52,53,54 --jobs 10 --n-agents 10 --output-dir revision/experiments/outputs/observation_n10_diagnostic_v1`
   `python3 revision/experiments/run_toy_lob_observation.py --episodes 800 --eval-episodes 50 --seeds 45,46,47,48,49,50,51,52,53,54 --jobs 10 --n-agents 20 --output-dir revision/experiments/outputs/observation_n20_diagnostic_v1`
3. N diagnostic plots:
   `python3 revision/experiments/plot_toy_lob_n_diagnostics.py --inputs n10=revision/experiments/outputs/observation_n10_diagnostic_v1,n20=revision/experiments/outputs/observation_n20_diagnostic_v1 --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --robustness-summary revision/experiments/outputs/n20_followup_1500_v1/summary.json --robustness-scenarios n20 --output-dir revision/experiments/outputs/n20_diagnostics_v1`

The repo-visible summary is
`revision/notes/toy_lob_n20_followup_v1_summary.md`.

## Local LOB/MAPPO Pilot

The local LOB/MAPPO stage is a pilot bridge after toy validation. It is not the
full main experiment.

Initial MAPPO pilot:

1. MAPPO gate calibration:
   `python revision/experiments/calibrate_lob_mappo_gate.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --target-activation 0.05,0.15 --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1`
2. MAPPO pilot:
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --methods no_intervention,always_on,threshold,gated_intervention --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_pilot_conservative_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_pilot_conservative_v1/metrics.csv`

The repo-visible summary is
`revision/notes/lob_mappo_pilot_v1_summary.md`.

Gate-drift diagnostic and activation-controlled pilot:

1. Diagnose passive score distributions:
   `python revision/experiments/diagnose_lob_mappo_gate_drift.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --methods no_intervention,gated_intervention,random_gate --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_gate_drift_diagnostic_v1`
2. Run activation-controlled pilot:
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --methods no_intervention,always_on,threshold,gated_intervention_controlled --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_controlled_pilot_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_controlled_pilot_v1/metrics.csv`
3. Diagnose controlled routing split:
   `python revision/experiments/diagnose_lob_mappo_gate_drift.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --methods gated_intervention_controlled --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_controlled_routing_diagnostic_v1`

The repo-visible follow-up summary is
`revision/notes/lob_mappo_gate_drift_diagnostic_v1_summary.md`.

Current decision boundary: do not launch the full LOB/MAPPO main experiment
from this pilot. Activation control works, but controlled gated routing did not
beat matched random on spread/reward in the local LOB/MAPPO pilot.
