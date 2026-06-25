# Toy LOB N-Invariant Gate Diagnostic v1 Summary

This diagnostic follows the N=20 follow-up caveat: the existing v1 full gate is
not a scale-invariant representation. The goal here is feature diagnosis and
gate-v2 plumbing, not a formal N=20 robustness pass.

## Scope

- No-intervention observations:
  - N=5: seeds `70-79`, `800` train episodes, `50` eval episodes.
  - N=10: seeds `70-79`, `800` train episodes, `50` eval episodes.
  - N=20: seeds `70-75`, `800` train episodes, `50` eval episodes.
- The N=20 observation is partial/exploratory. A 10-seed run was started but
  stopped after six completed seeds because the remaining N=20 workers were too
  slow for an interactive diagnostic pass.
- Calibration label: `N10`.
- Calibration records only use no-intervention mixed-risk records.
- Pseudo-positive: `risk_state == low` and `collusion_gap > 0.30`.
- Pseudo-negative: `risk_state == high` and `collusion_gap < 0.20`.

## Added Features

The records now include scale-aware distributional diagnostics:

- `near_min_spread_fraction`
- `sparse_near_min_spread_fraction = 1 - near_min_spread_fraction`
- `spread_entropy`
- `spread_concentration`
- normalized quantile spread dispersion
- normalized flow concentration / HHI
- `v2_entropy_score`
- `v2_fraction_near_min_score`
- `v2_two_stage_score`

## Main Diagnostic Result

The old `alignment` feature is not stable as a positive collusion indicator
under the pseudo-label definition. It is negatively correlated with
risk-adjusted collusion gap in all three N settings.

| Feature | N=5 AUC | N=10 AUC | N=20 AUC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `alignment` | 0.1827 | 0.0837 | 0.0481 | wrong sign / scale-sensitive |
| `near_min_spread_fraction` | 0.2292 | 0.1356 | 0.1271 | wrong sign as a positive feature |
| `sparse_near_min_spread_fraction` | 0.7708 | 0.8644 | 0.8729 | useful inverted near-min signal |
| `spread_entropy` | 0.6866 | 0.8578 | 0.9015 | useful distributional signal |
| `flow_concentration` | 0.8425 | 0.9081 | 0.9036 | useful flow signal |
| `passive_risk_score` | 0.9976 | 0.9978 | 0.9988 | still dominated by spread excess |
| `v2_entropy_score` | 1.0000 | 1.0000 | 1.0000 | calibrated candidate score |
| `v2_fraction_near_min_score` | 0.9999 | 1.0000 | 1.0000 | calibrated candidate score |
| `v2_two_stage_score` | 1.0000 | 1.0000 | 1.0000 | calibrated candidate score |

These AUC values should not be read as intervention results. They are
no-intervention diagnostic separability under the pseudo-label definition.

## Distribution Shift

| Metric | N=5 mean | N=5 p90 | N=10 mean | N=10 p90 | N=20 mean | N=20 p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `collusion_gap` | 0.2883 | 0.5750 | 0.2376 | 0.4519 | 0.1842 | 0.3516 |
| `passive_risk_score` | 0.3002 | 0.4980 | 0.2448 | 0.3791 | 0.1968 | 0.2959 |
| `alignment` | 0.4326 | 0.6667 | 0.3369 | 0.6667 | 0.2924 | 0.6667 |
| `spread_entropy` | 0.5818 | 0.6846 | 0.6682 | 0.8271 | 0.6053 | 0.8311 |
| `flow_concentration` | 0.1748 | 0.3288 | 0.0735 | 0.1312 | 0.0270 | 0.0524 |
| `v2_entropy_score` | 0.3184 | 0.5455 | 0.2850 | 0.4656 | 0.2318 | 0.3810 |
| `v2_two_stage_score` | 0.3006 | 0.5375 | 0.2600 | 0.4453 | 0.2076 | 0.3537 |

The N=20 distributions still shift downward, but the diagnostic supports the
mechanism concern: raw global alignment is the wrong object to rely on for a
larger-N gate.

## Calibrated v2 Thresholds

The v2 gate config was calibrated from N=10 no-intervention records only:

| Score field | Threshold | Activation | TPR | FPR |
| --- | ---: | ---: | ---: | ---: |
| `v2_entropy_score` | 0.311148 | 0.4000 | 0.9979 | 0.0000 |
| `v2_fraction_near_min_score` | 0.278076 | 0.4000 | 0.9912 | 0.0004 |
| `v2_two_stage_score` | 0.292344 | 0.3750 | 0.9997 | 0.0000 |
| `collusion_gap` base threshold | 0.287407 | 0.3162 | 1.0000 | 0.0000 |

The generated config is:

- `revision/experiments/outputs/n_invariant_gate_diagnostic_v1/v2_gate_config.json`

## N=20 Runtime Status

An N=20 v2 runtime smoke was run only to verify that the new gate modes and
generated config execute end-to-end. It used `5` train episodes, `1` eval
episode, and seed `85`. This smoke is not statistically meaningful and should
not be used for paper claims.

The attempted `800/50`, 5-seed N=20 comparison was stopped because the worker
runtime was too high for this interactive pass. A formal v2 intervention
comparison still needs a separately budgeted run.

## Stage Decision

This diagnostic supports a gate-v2 representation update:

- keep the core method as risk-aware gated platform routing;
- stop treating raw global alignment as a positive larger-N collusion feature;
- use spread excess as the base detector;
- use N-aware distributional diagnostics such as entropy, sparse best-quote
  mass, and normalized flow concentration as confidence/veto features;
- do not claim that v2 solves N=20 until a held-out intervention sweep is run.

The formal follow-up protocol is recorded in
`revision/notes/toy_lob_n20_v2_stress_protocol.md`.

## Reproduction

Observation records:

```bash
python revision/experiments/run_toy_lob_observation.py --episodes 800 --eval-episodes 50 --seeds 70,71,72,73,74,75,76,77,78,79 --n-agents 5 --jobs 3 --output-dir revision/experiments/outputs/observation_n5_n_invariant_v1
python revision/experiments/run_toy_lob_observation.py --episodes 800 --eval-episodes 50 --seeds 70,71,72,73,74,75,76,77,78,79 --n-agents 10 --jobs 3 --output-dir revision/experiments/outputs/observation_n10_n_invariant_v1
python revision/experiments/run_toy_lob_observation.py --episodes 800 --eval-episodes 50 --seeds 70,71,72,73,74,75,76,77,78,79 --n-agents 20 --jobs 3 --output-dir revision/experiments/outputs/observation_n20_n_invariant_v1
```

The N=20 command above was stopped after seeds `70-75`; rerun it with a longer
budget for a full 10-seed diagnostic.

Feature diagnostic and v2 calibration:

```bash
python revision/experiments/diagnose_toy_lob_n_invariant_gate.py --inputs N5=revision/experiments/outputs/observation_n5_n_invariant_v1,N10=revision/experiments/outputs/observation_n10_n_invariant_v1,N20=revision/experiments/outputs/observation_n20_n_invariant_v1 --calibration-label N10 --base-gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output-dir revision/experiments/outputs/n_invariant_gate_diagnostic_v1
```

Diagnostic figures:

```bash
python revision/experiments/plot_toy_lob_n_diagnostics.py --inputs N5=revision/experiments/outputs/observation_n5_n_invariant_v1,N10=revision/experiments/outputs/observation_n10_n_invariant_v1,N20=revision/experiments/outputs/observation_n20_n_invariant_v1 --gate-config revision/experiments/outputs/n_invariant_gate_diagnostic_v1/v2_gate_config.json --output-dir revision/experiments/outputs/n_invariant_feature_plots_v1
```

Engineering smoke:

```bash
python revision/experiments/run_toy_lob_robustness.py --episodes 5 --eval-episodes 1 --seeds 85 --scenarios n20 --methods no_intervention,gated_intervention,spread_only_gate,v2_entropy_gate,v2_fraction_near_min_gate,v2_two_stage_gate --gate-config revision/experiments/outputs/n_invariant_gate_diagnostic_v1/v2_gate_config.json --jobs 1 --output revision/experiments/outputs/n20_v2_gate_smoke_v1/summary.json --metrics-csv revision/experiments/outputs/n20_v2_gate_smoke_v1/metrics.csv
```
