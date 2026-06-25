# Toy LOB Confirmation Sweep v1 Summary

Date: 2026-06-25

This is the first full toy confirmation sweep after the preliminary held-out
sweep. It uses a fixed gate threshold calibrated only from no-intervention
observation records. No intervention evaluation seeds are used for threshold
tuning.

Raw outputs are under `revision/experiments/outputs/` and are ignored by git.

## Configuration

- Calibration source: `revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json`.
- Frozen gate threshold: `risk_threshold = 0.2393662345171971`.
- Calibration seeds: `0,1,2,3,4`, no intervention only.
- Confirmation evaluation seeds: `15-34`.
- Training episodes: `1500`.
- Evaluation episodes: `100`.
- Baseline runtime: `1086.04s` elapsed with method-seed parallelism.
- Ablation runtime: `923.65s` elapsed with method-seed parallelism.
- Matched random rate: `0.145425`, matched to realized gated activation.

## Baseline Results

| Method | Collusion gap mean | 95% CI | Gate act. | FPR high-normal | TPR low-collusive | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 0.2319 | [0.2175, 0.2465] | 0.0000 | 0.0000 | 0.0000 | 6.7146 |
| always_on | -0.0255 | [-0.0480, -0.0017] | 1.0000 | 1.0000 | 1.0000 | 3.7316 |
| threshold | 0.1385 | [0.1300, 0.1471] | 0.3235 | 0.8980 | 0.3717 | 5.9259 |
| random_gate | 0.1961 | [0.1849, 0.2069] | 0.1416 | 0.1429 | 0.1415 | 6.3246 |
| gated_intervention | 0.1334 | [0.1194, 0.1478] | 0.1454 | 0.1289 | 0.8414 | 6.3741 |
| oracle_gate | 0.1395 | [0.1273, 0.1517] | 0.1173 | 0.0000 | 1.0000 | 6.2718 |

Paired held-out comparison against matched random:

- `random_gate - gated_intervention` collusion-gap delta mean: `0.062695609`.
- Standard error of paired delta: `0.009575607`.
- Positive improvement seeds: `18 / 20`.

Baseline pass/fail:

- Pass: no-intervention has a stable positive collusion gap.
- Pass: gated gap is lower than matched random, with 18 of 20 seed deltas positive.
- Pass: gated FPR is much lower than threshold FPR (`0.1289` vs `0.8980`).
- Pass: gated reward is much higher than always-on reward (`6.3741` vs `3.7316`).
- Pass: gated intervention rate is far below always-on (`0.1454` vs `1.0000`).
- Caveat: always-on still minimizes raw gap, so the correct claim is trade-off
  improvement, not lowest collusion gap.

## Ablation Results

| Variant | Collusion gap mean | 95% CI | Gate act. | FPR high-normal | TPR low-collusive | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_gate | 0.1334 | [0.1197, 0.1481] | 0.1454 | 0.1289 | 0.8414 | 6.3741 |
| spread_only_gate | 0.1236 | [0.1139, 0.1325] | 0.1589 | 0.1131 | 1.0000 | 6.1801 |
| no_risk_adjustment | 0.1254 | [0.1153, 0.1353] | 0.3893 | 0.9997 | 0.6772 | 5.7467 |
| no_alignment_feature | 0.1475 | [0.1367, 0.1591] | 0.0960 | 0.0000 | 0.8615 | 6.4496 |
| no_demand_suppression_feature | 0.1614 | [0.1497, 0.1727] | 0.1270 | 0.0655 | 0.6778 | 6.2060 |
| no_routing_intervention | 0.2306 | [0.2209, 0.2404] | 0.3135 | 0.1471 | 0.8332 | 6.5685 |

Ablation interpretation:

- Routing is necessary: `no_routing_intervention - full_gate` gap delta is
  `0.097173785`, positive in `20 / 20` held-out seeds.
- Risk adjustment is necessary for false-positive control:
  `no_risk_adjustment - full_gate` FPR delta is `0.870735471`, positive in
  `20 / 20` held-out seeds.
- Removing demand suppression weakens the gate on gap and TPR.
- Removing alignment makes the gate more conservative with zero FPR, but
  worsens gap relative to the full gate.
- `spread_only_gate` is a strong simpler baseline. It improves raw gap slightly
  relative to the full gate in this sweep, but has lower mean reward and a
  higher intervention rate. This should be reported as a Pareto/trade-off
  competitor, not hidden.

## Recommended Claim

Use:

> Toy validation confirms that risk-aware gated routing improves the
> collusion-reduction / false-positive / intervention-cost trade-off.

Do not use:

> The gated method achieves the lowest collusion gap.

## Output Artifacts

- Baseline summary: `revision/experiments/outputs/baselines_confirmation_sweep_v1/summary.json`
- Baseline aggregate CSV: `revision/experiments/outputs/aggregated_confirmation_sweep_v1/toy_lob_baseline_aggregate.csv`
- Ablation summary: `revision/experiments/outputs/ablation_confirmation_sweep_v1/summary.json`
- Ablation aggregate CSV: `revision/experiments/outputs/aggregated_confirmation_sweep_v1/toy_lob_ablation_aggregate.csv`
- Figures: `revision/experiments/outputs/figures_confirmation_sweep_v1/`
