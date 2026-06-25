# Toy LOB Sweep v1 Summary

Date: 2026-06-25

This is the first held-out toy seed sweep after adding gate calibration,
multi-seed runners, aggregation, and diagnostic oracle support. Raw outputs are
under `revision/experiments/outputs/` and are ignored by git.

## Configuration

- Calibration observation: seeds `0,1,2,3,4`, `1000` training episodes,
  `50` evaluation episodes, no intervention.
- Held-out baseline evaluation: seeds `5,6,7,8,9,10,11,12,13,14`, `1000`
  training episodes, `50` evaluation episodes.
- Gate calibration input: 150,000 no-intervention observation records.
- Frozen gate threshold: `risk_threshold = 0.2393662345171971`.
- Calibration diagnostics: activation `0.4000`, TPR low-collusive `0.9056`,
  FPR high-normal `0.0801`.
- Held-out matched random rate: `0.16148`, matched to the realized activation
  rate of the calibrated gated intervention.

## Held-Out Baselines

| Method | Collusion gap mean | 95% CI | Gate act. | FPR high-normal | TPR low-collusive | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 0.2302 | [0.2151, 0.2436] | 0.0000 | 0.0000 | 0.0000 | 6.6515 |
| always_on | -0.0146 | [-0.0305, 0.0005] | 1.0000 | 1.0000 | 1.0000 | 3.8160 |
| threshold | 0.1269 | [0.1062, 0.1475] | 0.3228 | 0.9584 | 0.3872 | 6.1790 |
| random_gate | 0.1990 | [0.1808, 0.2190] | 0.1650 | 0.1646 | 0.1634 | 6.1949 |
| gated_intervention | 0.1371 | [0.1220, 0.1531] | 0.1615 | 0.1677 | 0.8354 | 6.3802 |
| oracle_gate | 0.1478 | [0.1285, 0.1701] | 0.1438 | 0.0000 | 1.0000 | 6.1184 |

Paired held-out comparison against matched random:

- `random_gate - gated_intervention` collusion-gap delta mean: `0.061888393`.
- Standard error of paired delta: `0.014215244`.
- Positive improvement seeds: `9 / 10`.

Interpretation:

- The no-intervention gap is stable and positive.
- The calibrated gated intervention improves substantially over matched random
  at almost identical activation rate.
- Threshold reduces the gap slightly more, but has a near-always false-positive
  rate in high-risk normal windows.
- Always-on minimizes gap but destroys selectivity and has much lower reward.
- The current `oracle_gate` is a diagnostic true-gap rule, not a strict
  optimized upper bound. Do not describe it as a theoretical upper bound unless
  a rate-matched or post-hoc oracle variant is added.

## Held-Out Ablations

| Variant | Collusion gap mean | 95% CI | Gate act. | FPR high-normal | TPR low-collusive | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_gate | 0.1371 | [0.1219, 0.1532] | 0.1615 | 0.1677 | 0.8354 | 6.3802 |
| spread_only_gate | 0.1397 | [0.1275, 0.1533] | 0.1893 | 0.1486 | 1.0000 | 5.8987 |
| no_risk_adjustment | 0.1105 | [0.0898, 0.1308] | 0.4042 | 0.9997 | 0.7024 | 5.7198 |
| no_alignment_feature | 0.1596 | [0.1451, 0.1731] | 0.1256 | 0.0000 | 0.8319 | 6.2865 |
| no_demand_suppression_feature | 0.1638 | [0.1504, 0.1777] | 0.1608 | 0.0668 | 0.7066 | 6.0791 |
| no_routing_intervention | 0.2483 | [0.2393, 0.2574] | 0.3512 | 0.1376 | 0.8696 | 6.4822 |

Interpretation:

- Removing routing makes the gap worse than no intervention, even though the
  gate fires often. This supports routing as the causal intervention channel.
- Removing risk adjustment produces the lowest raw gap but almost all high-risk
  normal windows become false positives. This supports the risk-aware trade-off
  claim rather than a raw-gap-only claim.
- Removing alignment or demand features weakens the full gate on gap and/or TPR,
  supporting the multi-signal scoring design.

## Next Step

Do not jump to full LOB or MAPPO yet. The immediate paper-safe next step is a
larger toy confirmation sweep with 20 held-out evaluation seeds and 1500-3000
training episodes. That should be run as a long task after adding method-level
parallelism or allocating an overnight run.
