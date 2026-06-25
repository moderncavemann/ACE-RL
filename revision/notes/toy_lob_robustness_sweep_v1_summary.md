# Toy LOB Robustness Sweep v1 Summary

Date: 2026-06-25

This is the first controlled scale-up after toy confirmation. It keeps the
frozen gate threshold from `gate_calibration_sweep_v1` and does not tune any
threshold on robustness outcomes.

Raw outputs are under `revision/experiments/outputs/robustness_sweep_v1/` and
are ignored by git.

## Configuration

- Gate config: `revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json`.
- Frozen threshold: `risk_threshold = 0.2393662345171971`.
- Seeds: `35-44`.
- Training episodes: `800`.
- Evaluation episodes: `50`.
- Jobs: `32`.
- Runtime: `1502.99s` elapsed.
- Methods: `no_intervention`, `always_on`, `threshold`, `gated_intervention`,
  plus scenario-specific matched `random_gate`.
- Scenarios: `base_n10`, `n5`, `n20`, `cost_wide`, `risk_fast`,
  `long_horizon`.

## Scenario Results

| Scenario | No-int gap | Random gap | Gated gap | Random - gated delta | Positive seeds | Gated FPR | Threshold FPR | Gated reward | Always reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base_n10 | 0.2325 | 0.2103 | 0.1446 | 0.0657 | 9/10 | 0.1735 | 0.9268 | 6.3493 | 3.8006 |
| n5 | 0.2957 | 0.2567 | 0.1599 | 0.0968 | 10/10 | 0.2284 | 0.9354 | 13.3616 | 9.1375 |
| n20 | 0.1827 | 0.1717 | 0.1690 | 0.0027 | 7/10 | 0.1681 | 0.9804 | 3.0871 | 1.7581 |
| cost_wide | 0.2168 | 0.2015 | 0.1517 | 0.0498 | 9/10 | 0.2224 | 0.9997 | 6.4366 | 5.0693 |
| risk_fast | 0.2767 | 0.2450 | 0.1836 | 0.0614 | 10/10 | 0.1650 | 0.9639 | 5.5792 | 3.7252 |
| long_horizon | 0.2348 | 0.2137 | 0.1380 | 0.0757 | 10/10 | 0.1349 | 0.8947 | 6.1875 | 3.7480 |

## Interpretation

- The trade-off claim is stable in `base_n10`, `n5`, `cost_wide`,
  `risk_fast`, and `long_horizon`: gated intervention beats matched random on
  collusion gap, has much lower FPR than threshold, and preserves reward far
  better than always-on.
- The `n20` setting is the main caveat. Gated still improves slightly over
  matched random and no intervention, but the paired delta is small (`0.0027`)
  and only `7/10` seeds are positive. Larger markets likely need longer
  training, N-specific state/gate calibration checks, or a stronger learner.
- Threshold remains a poor deployable control in every scenario because FPR is
  near-always high (`0.8947-0.9997` across these scenarios).
- Always-on remains an over-intervention baseline: low raw gap but large reward
  loss in all scenarios.

## Stage Decision

The controlled robustness extension mostly supports the toy-scale mechanism,
but it also shows that larger-N behavior should be treated as the next targeted
experiment rather than as solved. The next low-risk step is a dedicated N=20
follow-up with longer training and possibly N-specific gate diagnostics before
moving to full LOB or MAPPO/CRPO.

## Output Artifacts

- Summary JSON: `revision/experiments/outputs/robustness_sweep_v1/summary.json`
- Metrics CSV: `revision/experiments/outputs/robustness_sweep_v1/metrics.csv`
