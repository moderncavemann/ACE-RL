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

Routing parameter grid:

1. Screen routing parameters:
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 10 --eval-episodes 3 --episode-length 100 --n-agents 10 --seeds 0-2 --routing-rhos 0.25,0.75 --routing-spread-caps 4,8,10 --routing-withhold-rates 0.0,0.5 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_routing_grid_v1`
2. Candidate check, broad-cap low-rho:
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-rhos 0.25 --routing-spread-caps 10 --routing-withhold-rates 0.5 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_routing_grid_candidate_rho025_cap10_withhold05_v1`
3. Candidate check, boost-only strict-cap:
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-rhos 0.75 --routing-spread-caps 4 --routing-withhold-rates 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_routing_grid_candidate_rho075_cap4_withhold00_v1`

The repo-visible routing-grid summary is
`revision/notes/lob_mappo_routing_grid_v1_summary.md`.

Current routing-grid decision: `routing_rho=0.75`,
`routing_spread_cap_ticks=4`, and `routing_withhold_rate=0.0` is the current
MAPPO-local-LOB pilot candidate. It should be rerun with normal baselines before
any full main experiment.

Boost-only validation follow-up:

1. Seeded-random 20/5 normal-baseline pilot:
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --methods no_intervention,always_on,threshold,gated_intervention_controlled --routing-rho 0.75 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_candidate_boost_only_pilot_seeded_random_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_candidate_boost_only_pilot_seeded_random_v1/metrics.csv`
2. Longer 50/10 pilot:
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 50 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 0-4 --methods no_intervention,always_on,threshold,gated_intervention_controlled --routing-rho 0.75 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_candidate_boost_only_longer_pilot_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_candidate_boost_only_longer_pilot_v1/metrics.csv`

The repo-visible boost-only validation summary is
`revision/notes/lob_mappo_boost_only_validation_v1_summary.md`.

Current boost-only validation decision: the 20/5 pilot passes, but the 50/10
pilot does not preserve a timing advantage over matched random. Do not launch
full LOB/MAPPO main experiments yet.

Boost-only drift diagnostic:

1. Compare 20/5 score drift:
   `python revision/experiments/diagnose_lob_mappo_gate_drift.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --methods no_intervention,gated_intervention_controlled,random_gate --random-rate 0.106 --routing-rho 0.75 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_boost_only_drift_20_v1`
2. Compare 50/10 score drift:
   `python revision/experiments/diagnose_lob_mappo_gate_drift.py --episodes 50 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 0-4 --methods no_intervention,gated_intervention_controlled,random_gate --random-rate 0.0624 --routing-rho 0.75 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_boost_only_drift_50_v1`
3. Check oracle timing:
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 50 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 0-4 --methods oracle_low_high_spread --routing-rho 0.75 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_oracle_timing_boost_only_50_trainstats_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_oracle_timing_boost_only_50_trainstats_v1/metrics.csv`
4. Check no-routing ablation:
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 50 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 0-4 --methods gated_intervention_controlled --routing-rho 0.0 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_no_routing_controlled_gate_50_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_no_routing_controlled_gate_50_v1/metrics.csv`

The repo-visible drift diagnostic summary is
`revision/notes/lob_mappo_boost_only_drift_diagnostic_v1_summary.md`.

Current drift diagnostic decision: routing is necessary, but low-rate matched
random can exploit the boost-only routing channel in longer MAPPO training. Do
not force more activations; next work should test whether a sharper conditional
routing channel can make timing matter.

Conditional routing follow-up:

1. Absolute-cap priority grid:
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-modes eligible_priority --routing-rhos 0.75,1.0 --routing-spread-caps 4,6 --routing-withhold-rates 0.0,0.5 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_eligible_priority_grid_20_v1`
2. Relative-priority grids:
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-modes relative_priority --routing-rhos 0.75,1.0 --routing-spread-caps 4 --routing-withhold-rates 0.0,0.5 --routing-relative-improvements 0.5,1.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_relative_priority_grid_20_v2`
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-modes relative_priority --routing-rhos 1.5,2.0 --routing-spread-caps 4 --routing-withhold-rates 0.0 --routing-relative-improvements 0.5,1.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_relative_priority_grid_20_strong_v1`
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-modes relative_priority --routing-rhos 0.75,1.0,1.5,2.0 --routing-spread-caps 4 --routing-withhold-rates 0.0 --routing-relative-improvements 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_relative_priority_grid_20_rel0_v1`
3. Relative-priority oracle upper bound:
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --methods no_intervention,oracle_low_high_spread --routing-mode relative_priority --routing-rho 1.0 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --routing-relative-improvement-ticks 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_relative_priority_oracle_20_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_relative_priority_oracle_20_v1/metrics.csv`
4. Relative-reallocation grid and 50/10 checks:
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-modes relative_reallocate --routing-rhos 0.5,0.75,1.0 --routing-spread-caps 4 --routing-withhold-rates 0.0 --routing-relative-improvements 0.0,0.5 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_relative_reallocate_grid_20_v1`
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 50 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 0-4 --methods no_intervention,gated_intervention_controlled --routing-mode relative_reallocate --routing-rho 0.5 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --routing-relative-improvement-ticks 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_relative_reallocate_rho05_50_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_relative_reallocate_rho05_50_v1/metrics.csv`
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 50 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 5-9 --methods no_intervention,gated_intervention_controlled --routing-mode relative_reallocate --routing-rho 0.5 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --routing-relative-improvement-ticks 0.0 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_relative_reallocate_rho05_50_seeds5_9_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_relative_reallocate_rho05_50_seeds5_9_v1/metrics.csv`

The repo-visible conditional-routing summary is
`revision/notes/lob_mappo_conditional_routing_reallocate_v1_summary.md`.

Current conditional-routing decision: the best relative-reallocation candidate
does not pass the 10-seed 50/10 pilot. Do not launch full LOB/MAPPO main
experiments yet. The next step is assay stabilization: diagnose why
no-intervention spread elevation is absent in many MAPPO seeds before running
more routing grids.

Assay stabilization follow-up:

1. Base no-intervention assays:
   `python revision/experiments/diagnose_lob_mappo_assay.py --episodes 50 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 0-19 --scenarios base --checkpoint-episodes 10,20,50 --checkpoint-eval-episodes 2 --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_no_intervention_assay_base20_v1`
   `python revision/experiments/diagnose_lob_mappo_assay.py --episodes 20 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 0-19 --scenarios base --checkpoint-episodes 10,20 --checkpoint-eval-episodes 2 --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_no_intervention_assay_base20ep_20seeds_v1`
2. Stabilization grid:
   `python revision/experiments/diagnose_lob_mappo_assay.py --episodes 50 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-9 --scenarios low_entropy,entropy_001,lr_1e3,inventory_low,inventory_high,flow_low,flow_high,risk_persistent --checkpoint-episodes 20,50 --checkpoint-eval-episodes 1 --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_no_intervention_assay_stabilization_grid_v1`
3. Inventory-high calibration and comparisons:
   `python revision/experiments/calibrate_lob_mappo_gate.py --episodes 50 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --target-activation 0.05,0.35 --max-fpr 0.25 --high-spread-tau 4.0 --inventory-penalty 0.006 --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1`
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 50 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 0-9 --methods no_intervention,gated_intervention_controlled --routing-mode relative_reallocate --routing-rho 0.5 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --routing-relative-improvement-ticks 0.0 --inventory-penalty 0.006 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_inventory_high_reallocate_rho05_50_10seeds_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_inventory_high_reallocate_rho05_50_10seeds_v1/metrics.csv`
   `python revision/experiments/run_lob_mappo_pilot.py --episodes 50 --eval-episodes 10 --episode-length 100 --n-agents 10 --seeds 0-9 --methods no_intervention,gated_intervention_controlled --routing-mode relative_reallocate --routing-rho 0.75 --routing-spread-cap-ticks 4 --routing-withhold-rate 0.0 --routing-relative-improvement-ticks 0.0 --inventory-penalty 0.006 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output revision/experiments/outputs/lob_mappo_inventory_high_reallocate_rho075_50_10seeds_v1/summary.json --metrics-csv revision/experiments/outputs/lob_mappo_inventory_high_reallocate_rho075_50_10seeds_v1/metrics.csv`

The repo-visible assay-stabilization summary is
`revision/notes/lob_mappo_assay_stabilization_v1_summary.md`.

Current assay-stabilization decision: observation can be stabilized with
`inventory_penalty=0.006`, but risk-aware gated reallocation still does not beat
matched random on enough seeds. Full LOB/MAPPO main experiments remain no-go.

Routing-channel repair follow-up:

1. Oracle-first balanced/continuous reallocation screen:
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-modes continuous_reallocate,balanced_reallocate --routing-rhos 0.25,0.5,0.75 --routing-spread-caps 4 --routing-withhold-rates 0.0 --routing-relative-improvements 0.0 --routing-target-kappas 1.0,4.0 --timing-method oracle_low_high_spread --inventory-penalty 0.006 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_repair_oracle_screen_v1`
2. Threshold-timing diagnostic:
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-modes balanced_reallocate --routing-rhos 0.5,0.75 --routing-spread-caps 4 --routing-withhold-rates 0.0 --routing-relative-improvements 0.0 --routing-target-kappas 1.0,4.0 --timing-method threshold --inventory-penalty 0.006 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_repair_threshold_screen_v1`
3. Gated candidate screen and 10-seed expansion:
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --routing-modes balanced_reallocate --routing-rhos 0.5,0.75 --routing-spread-caps 4 --routing-withhold-rates 0.0 --routing-relative-improvements 0.0 --routing-target-kappas 1.0,4.0 --timing-method gated_intervention_controlled --inventory-penalty 0.006 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_repair_gated_screen_v1`
   `python revision/experiments/run_lob_mappo_routing_grid.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-9 --routing-modes balanced_reallocate --routing-rhos 0.75 --routing-spread-caps 4 --routing-withhold-rates 0.0 --routing-relative-improvements 0.0 --routing-target-kappas 4.0 --timing-method gated_intervention_controlled --inventory-penalty 0.006 --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_repair_gated_candidate_10seeds_v1`

The repo-visible routing-channel repair summary is
`revision/notes/lob_mappo_routing_channel_repair_v1_summary.md`.

Current routing-channel repair decision: `balanced_reallocate` improves
diagnostic visibility and threshold-style high-spread timing can beat matched
random in a 5-seed screen, but the risk-aware gated candidate falls to `4/10`
positive seeds in the 10-seed expansion. Full LOB/MAPPO main experiments remain
no-go; the next issue is label/assay alignment, not another broad routing grid.

Redesigned LOB/MAPPO assay follow-up:

1. Risk-coupled no-intervention screen:
   `python revision/experiments/diagnose_lob_mappo_assay.py --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --scenarios base --risk-coupled-economics --informed-flow-prob 0.3 --spread-elasticity 0.1 --targeted-fill-pricing competitive_benchmark --targeted-competitive-half-spread-ticks 2 --enforce-max-inventory --agent-id-feature --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_redesigned_assay_screen_v1`
2. Candidate A 50-episode acceptance check:
   `python revision/experiments/diagnose_lob_mappo_assay.py --episodes 50 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --scenarios base --risk-coupled-economics --inventory-penalty 0.0005 --informed-flow-prob 0.5 --spread-elasticity 0.05 --targeted-fill-pricing competitive_benchmark --targeted-competitive-half-spread-ticks 2 --enforce-max-inventory --agent-id-feature --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_redesigned_assay_candidate_a_50_v1`
3. Strong-risk-differential acceptance check:
   `python revision/experiments/diagnose_lob_mappo_assay.py --episodes 50 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 --scenarios base --risk-coupled-economics --inventory-penalty 0.0005 --low-sigma 0.005 --high-sigma 0.08 --informed-flow-prob 0.7 --spread-elasticity 0.05 --seed-liquidity-qty 0 --targeted-fill-pricing competitive_benchmark --targeted-competitive-half-spread-ticks 2 --enforce-max-inventory --agent-id-feature --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 --output-dir revision/experiments/outputs/lob_mappo_redesigned_assay_candidate_e_50_v1`

The repo-visible redesigned-assay summary is
`revision/notes/lob_mappo_redesigned_assay_v1_summary.md`.

Current redesigned-assay decision: risk-coupled economics and competitive
targeted-fill pricing fix key conceptual bugs, but no screened candidate passes
the no-intervention acceptance gate. Full LOB/MAPPO main experiments remain
no-go. The next redesign should target the learner/assay structure, especially
shared-actor symmetry and state-contingent risk pricing, rather than more
routing or threshold screens.

Structural redesign v2:

1. Fixed-policy economic acceptance:
   `python revision/experiments/validate_lob_assay_economics.py --seeds 0-99 --inventory-penalty 0.0001 --base-flow-intensity 2 --low-sigma 0.005 --high-sigma 0.08 --informed-flow-prob 0.7 --spread-elasticity 0.14 --seed-liquidity-qty 0 --agent-quote-quantity 1 --routing-mode platform_reallocate --routing-routable-fraction 0.9 --no-routing-platform-balanced --routing-eligibility-mode absolute_cap --routing-target-kappa 4.0 --routing-rho 1.0 --output-dir revision/experiments/outputs/lob_assay_econ_absolute_cap_f09_no_spill_100seed_v2`
2. Frozen seller-response and timing diagnostic:
   `python revision/experiments/run_lob_mappo_frozen_policy_timing_pilot.py --episodes 100 --eval-episodes 5 --episode-length 100 --seeds 0-4 --warm-start-updates 20 --response-train-rate 0.5 --response-train-low-risk-only --risk-stratified-yoke --schedule-from-response-policy --routing-routable-fraction 0.9 --no-routing-platform-balanced --routing-eligibility-mode absolute_cap --routing-target-kappa 4.0 --routing-rho 1.0 --device mps --output-dir revision/experiments/outputs/lob_mappo_frozen_policy_response_schedule_absolute_cap_f09_no_spill_100_5seed_screen_v1`
3. Actual online oracle versus risk-stratified dose-yoked training:
   `python revision/experiments/run_lob_mappo_yoked_timing_pilot.py --episodes 100 --eval-episodes 5 --episode-length 100 --seeds 0-4 --timing-mode oracle_controlled --training-schedule-source timed_online --device mps --output-dir revision/experiments/outputs/lob_mappo_yoked_timed_online_absolute_cap_f09_no_spill_100_5seed_screen_v1`

The economic assay passes and the same-state seller response diagnostic passes,
but timing does not beat the equal-dose low-risk shuffled comparator: the final
gate-off `yoked - timed` spread difference is `-0.3166 +/- 0.2927` with only
`2/5` positive seeds. Do not launch a learned-gate or full-main sweep. See
`revision/notes/lob_mappo_structural_redesign_v2_summary.md`.

State-contingent contract v3 supersedes that no-go for experiment planning:

1. Soft-rank fixed-policy economics:
   `python revision/experiments/validate_lob_assay_economics.py --seeds 0-99 --inventory-penalty 0.0001 --low-sigma 0.005 --high-sigma 0.08 --informed-flow-prob 0.7 --spread-elasticity 0.14 --agent-quote-quantity 1 --no-routing-platform-balanced --routing-contract excess_spread_pool --routing-eligibility-mode soft_rank --routing-routable-fraction 0.9 --routing-rho 1.0 --routing-target-kappa 4.0 --output-dir revision/experiments/outputs/lob_assay_excess_spread_soft_rank_contract_100seed_v3`
2. Scripted oracle timing:
   `python revision/experiments/validate_lob_state_contingent_routing_timing.py --seeds 0-99 --routing-eligibility-mode soft_rank --routing-target-kappa 4.0 --undercut-half-spread 4 --output-dir revision/experiments/outputs/lob_state_contingent_soft_rank_routing_timing_100seed_v2`
3. MAPPO screen:
   `python revision/experiments/run_lob_mappo_contract_response_pilot.py --equilibrium-episodes 100 --response-episodes 100 --response-train-rate 1.0 --calibration-episodes 5 --calibration-quantile 0.75 --calibration-upper-quantile 0.90 --eval-episodes 10 --episode-length 100 --seeds 25-29 --warm-start-steps 200 --warm-start-updates 20 --device mps --output-dir revision/experiments/outputs/lob_mappo_contract_local_advantage_100_5seed_screen_v1`
4. Held-out confirmation: rerun command 3 with `--seeds 30-34` and output
   directory `lob_mappo_contract_local_advantage_100_5seed_heldout_v1`.

Both MAPPO batches pass every joint acceptance check with 5/5 positive timing
seeds in each batch. The mechanism gate is now open for a full-main learned-gate
experiment. Oracle timing is an upper-bound pilot, not the final method claim.
See `revision/notes/lob_mappo_state_contingent_contract_v3_summary.md`.
