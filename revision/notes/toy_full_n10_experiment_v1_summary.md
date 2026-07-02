# N=10 Toy LOB Full Confirmation Package v1

Date: 2026-07-02

This package turns the existing N=10 toy validation into a paper-ready
confirmation bundle. It does not change the method, gate formula, threshold, or
claim scope. It is still a toy LOB-inspired spread-choice environment, not a
full matching-engine LOB or MAPPO experiment.

## Scope

Main claim supported by this package:

> Risk-aware gated routing improves the collusion-reduction /
> false-positive / intervention-cost trade-off in the N=10 toy LOB-inspired
> spread-choice environment.

Do not claim:

> The gated method achieves the lowest raw collusion gap.

Do not claim:

> The method is validated in full LOB or MAPPO.

## Phase 1: Calibration Provenance Check

The main evaluation uses the canonical frozen calibration artifact:

- Canonical artifact:
  `revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json`
- Canonical frozen threshold: `risk_threshold = 0.2393662345171971`
- Calibration records:
  `revision/experiments/outputs/observation_calibration_sweep_v1/`
- Calibration seeds: `0-4`
- Calibration data: no-intervention records only
- Pseudo-positive: `risk_state == low` and `collusion_gap > 0.30`
- Pseudo-negative: `risk_state == high` and `collusion_gap < 0.20`

The provenance check was written to:

- `revision/experiments/outputs/toy_full_n10_calibration_check_v1/gate_config_check.json`
- `revision/experiments/outputs/toy_full_n10_calibration_check_v1/threshold_grid_check.csv`

Result:

- Check threshold: `0.2393662345171971`
- Absolute difference from canonical threshold: `0.0`
- Calibration activation rate: `0.4000`
- Calibration TPR low-collusive: `0.9056`
- Calibration FPR high-normal: `0.0801`

This check did not overwrite the canonical gate config.

## Phase 2: Main N=10 Baselines

Configuration:

- Environment: `ToyLOBPricingEnv`
- N: `10`
- Train episodes: `1500`
- Eval episodes: `100`
- Held-out evaluation seeds: `15-34`
- Gate config: canonical frozen config above
- Routing strength: `rho = 0.75`
- Random baseline: matched to realized gated activation
- Matched random rate: `0.145425`
- Run output:
  `revision/experiments/outputs/toy_full_n10_baselines_v1/summary.json`

| Method | Collusion gap | 95% CI | Gate act. | FPR high-normal | TPR low-collusive | Mean reward | Total demand | Demand suppression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 0.2319 | [0.2175, 0.2465] | 0.0000 | 0.0000 | 0.0000 | 6.7146 | 52.6360 | 0.1496 |
| always_on | -0.0255 | [-0.0480, -0.0017] | 1.0000 | 1.0000 | 1.0000 | 3.7316 | 64.1953 | -0.0284 |
| threshold | 0.1385 | [0.1300, 0.1471] | 0.3235 | 0.8980 | 0.3717 | 5.9259 | 56.1434 | 0.0899 |
| random_gate | 0.2082 | [0.1961, 0.2200] | 0.1459 | 0.1471 | 0.1462 | 6.3170 | 53.6703 | 0.1344 |
| gated_intervention | 0.1334 | [0.1194, 0.1478] | 0.1454 | 0.1289 | 0.8414 | 6.3741 | 56.7083 | 0.0878 |
| oracle_gate | 0.1395 | [0.1273, 0.1517] | 0.1173 | 0.0000 | 1.0000 | 6.2718 | 56.4851 | 0.0917 |

`oracle_gate` is a diagnostic comparator only. It is not deployable and is not a
strict upper bound.

Mean reward is interpreted as an agent-profit preservation / intervention-cost
proxy, not as complete market welfare. Market quality is also reported through
total demand and demand suppression.

Paired matched-random comparison:

- Metric: `random_gate - gated_intervention` collusion-gap delta
- n: `20`
- Mean delta: `0.0748072167`
- SE: `0.0088161811`
- Normal 95% CI: `[0.0575275018, 0.0920869315]`
- Positive held-out seeds: `20 / 20`
- Per-seed output:
  `revision/experiments/outputs/toy_full_n10_aggregated_v1/paired_random_minus_gated_delta.csv`

## Phase 3: Core Ablations

Configuration matches Phase 2:

- Held-out seeds: `15-34`
- Train episodes: `1500`
- Eval episodes: `100`
- Same canonical frozen gate config where applicable
- Run output:
  `revision/experiments/outputs/toy_full_n10_ablation_v1/summary.json`

| Variant | Collusion gap | 95% CI | Gate act. | FPR high-normal | TPR low-collusive | Mean reward | Total demand | Demand suppression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_gate | 0.1334 | [0.1197, 0.1481] | 0.1454 | 0.1289 | 0.8414 | 6.3741 | 56.7083 | 0.0878 |
| spread_only_gate | 0.1236 | [0.1139, 0.1325] | 0.1589 | 0.1131 | 1.0000 | 6.1801 | 56.3020 | 0.0821 |
| no_risk_adjustment | 0.1254 | [0.1153, 0.1353] | 0.3893 | 0.9997 | 0.6772 | 5.7467 | 56.4645 | 0.0818 |
| no_alignment_feature | 0.1475 | [0.1367, 0.1591] | 0.0960 | 0.0000 | 0.8615 | 6.4496 | 55.7909 | 0.0984 |
| no_demand_suppression_feature | 0.1614 | [0.1497, 0.1727] | 0.1270 | 0.0655 | 0.6778 | 6.2060 | 54.7762 | 0.1077 |
| no_routing_intervention | 0.2306 | [0.2209, 0.2404] | 0.3135 | 0.1471 | 0.8332 | 6.5685 | 51.8835 | 0.1503 |

Mechanism interpretation:

- Routing is necessary: `no_routing_intervention` returns close to the
  no-intervention collusion gap.
- Risk adjustment is necessary for false-positive control:
  `no_risk_adjustment` has FPR `0.9997`.
- Removing demand suppression weakens the gap and TPR.
- Removing alignment makes the gate conservative with zero FPR, but increases
  the collusion gap relative to the full gate.
- `spread_only_gate` is a strong Pareto comparator. It has lower raw gap than
  the full gate, but lower mean reward and a higher activation rate; it should
  be reported honestly rather than hidden.

## Generated Artifacts

- Calibration check:
  `revision/experiments/outputs/toy_full_n10_calibration_check_v1/`
- Baseline summary:
  `revision/experiments/outputs/toy_full_n10_baselines_v1/summary.json`
- Ablation summary:
  `revision/experiments/outputs/toy_full_n10_ablation_v1/summary.json`
- Aggregates:
  `revision/experiments/outputs/toy_full_n10_aggregated_v1/`
- Figures:
  `revision/experiments/outputs/toy_full_n10_figures_v1/`

Figures generated:

- `baseline_tradeoff`
- `fpr_gap_tradeoff`
- `reward_gap_tradeoff`
- `gate_fpr_tpr`
- `paired_random_minus_gated_delta`
- `ablation_collusion_gap`
- `ablation_gate_fpr_tpr`
- `observation_collusion_gap`
- `passive_risk_score_distribution`

## Reproduction Commands

Calibration provenance check:

```bash
python revision/experiments/calibrate_toy_lob_gate.py \
  --observation-dir revision/experiments/outputs/observation_calibration_sweep_v1 \
  --output revision/experiments/outputs/toy_full_n10_calibration_check_v1/gate_config_check.json \
  --table-output revision/experiments/outputs/toy_full_n10_calibration_check_v1/threshold_grid_check.csv
```

Main baselines:

```bash
python revision/experiments/run_toy_lob_baselines.py \
  --episodes 1500 \
  --eval-episodes 100 \
  --seeds 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34 \
  --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json \
  --jobs 32 \
  --output revision/experiments/outputs/toy_full_n10_baselines_v1/summary.json
```

Core ablations:

```bash
python revision/experiments/run_toy_lob_ablation.py \
  --episodes 1500 \
  --eval-episodes 100 \
  --seeds 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34 \
  --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json \
  --jobs 32 \
  --output revision/experiments/outputs/toy_full_n10_ablation_v1/summary.json
```

Aggregation and figures:

```bash
python revision/experiments/aggregate_toy_lob_results.py \
  --input revision/experiments/outputs/toy_full_n10_baselines_v1/summary.json \
  --output-csv revision/experiments/outputs/toy_full_n10_aggregated_v1/baseline_aggregate.csv \
  --output-json revision/experiments/outputs/toy_full_n10_aggregated_v1/baseline_aggregate.json

python revision/experiments/aggregate_toy_lob_results.py \
  --input revision/experiments/outputs/toy_full_n10_ablation_v1/summary.json \
  --output-csv revision/experiments/outputs/toy_full_n10_aggregated_v1/ablation_aggregate.csv \
  --output-json revision/experiments/outputs/toy_full_n10_aggregated_v1/ablation_aggregate.json

python revision/experiments/plot_toy_lob_results.py \
  --observation-dir revision/experiments/outputs/observation_calibration_sweep_v1 \
  --baseline-summary revision/experiments/outputs/toy_full_n10_baselines_v1/summary.json \
  --ablation-summary revision/experiments/outputs/toy_full_n10_ablation_v1/summary.json \
  --gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json \
  --output-dir revision/experiments/outputs/toy_full_n10_figures_v1
```

## Stage Decision

This N=10 toy package passes the pre-specified confirmation checks:

- gated intervention beats matched random on paired collusion-gap delta;
- the paired delta mean is positive with a positive 95% CI;
- gated FPR is much lower than threshold FPR;
- gated mean reward is much higher than always-on mean reward;
- no-routing ablation removes the main gap improvement;
- no-risk-adjustment ablation confirms the false-positive risk;
- spread-only is a strong comparator and should be reported as such.

Phase 4 robustness was not rerun. Existing robustness results should remain
secondary evidence, with N=20 treated as a scale stress caveat.
