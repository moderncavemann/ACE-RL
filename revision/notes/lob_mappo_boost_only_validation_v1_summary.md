# Local LOB/MAPPO Boost-Only Validation v1 Summary

Date: 2026-07-03

This note follows the routing-grid pilot. It validates the selected
boost-only strict-cap routing candidate using the normal MAPPO pilot baselines.
It is still a pilot validation, not a full LOB/MAPPO main experiment.

## Scope

- Runner: `revision/experiments/run_lob_mappo_pilot.py`
- Routing candidate: `routing_rho=0.75`,
  `routing_spread_cap_ticks=4`, `routing_withhold_rate=0.0`
- Gate: `gated_intervention_controlled`
- Device: `mps` via `--device auto`
- Calibration: frozen MAPPO conservative gate config from
  `revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json`

The runner was updated so `random_gate` uses run-seed-derived RNG seeds. This
matches the routing-grid runner and avoids a fixed random schedule across
held-out seeds.

## 20/5 Normal-Baseline Pilot

Output:

- `revision/experiments/outputs/lob_mappo_candidate_boost_only_pilot_seeded_random_v1/`

Configuration:

- Seeds: `0-4`
- Train episodes per method/seed: `20`
- Eval episodes per method/seed: `5`
- Episode length: `100`

| Method | Avg spread | Best spread | Volume | Mean reward | Gate act. | FPR | TPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 3.4188 | 6.1680 | 3.7752 | -0.007831 | 0.0000 | 0.0000 | 0.0000 |
| always_on | 3.6459 | 6.6804 | 4.5712 | -0.008598 | 1.0000 | 1.0000 | 0.4000 |
| threshold | 3.5886 | 6.6216 | 3.7936 | -0.006771 | 0.2664 | 0.0000 | 0.2000 |
| random_gate | 3.6692 | 6.7180 | 3.8804 | -0.013310 | 0.1000 | 0.0806 | 0.0412 |
| gated_intervention_controlled | 2.9912 | 5.6576 | 3.7828 | -0.010573 | 0.1060 | 0.0000 | 0.0832 |

Paired spread comparison:

- `random_gate - gated_intervention_controlled` mean: `0.6780`
- SE: `0.1501`
- Positive seeds: `5 / 5`

Decision: passed as a short normal-baseline pilot. The boost-only strict-cap
candidate beats matched random on spread, keeps activation in range, and has
lower FPR than random.

## 50/10 Longer Pilot

Output:

- `revision/experiments/outputs/lob_mappo_candidate_boost_only_longer_pilot_v1/`

Configuration:

- Seeds: `0-4`
- Train episodes per method/seed: `50`
- Eval episodes per method/seed: `10`
- Episode length: `100`

| Method | Avg spread | Best spread | Volume | Mean reward | Gate act. | FPR | TPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 3.6052 | 6.6640 | 3.8280 | -0.004746 | 0.0000 | 0.0000 | 0.0000 |
| always_on | 2.3654 | 4.6482 | 6.1116 | -0.009057 | 1.0000 | 1.0000 | 0.2000 |
| threshold | 3.6360 | 6.5180 | 3.9434 | -0.006836 | 0.2474 | 0.0000 | 0.4000 |
| random_gate | 2.4424 | 4.7268 | 4.0712 | -0.011538 | 0.0642 | 0.0701 | 0.0000 |
| gated_intervention_controlled | 2.4983 | 4.7488 | 3.8978 | -0.010634 | 0.0624 | 0.0017 | 0.0000 |

Paired spread comparison:

- `random_gate - gated_intervention_controlled` mean: `-0.0559`
- SE: `0.3553`
- Positive seeds: `2 / 5`

Seed-level paired deltas:

| Seed | Random - gated spread | Gated act. | Random act. |
| ---: | ---: | ---: | ---: |
| 0 | -0.8490 | 0.2590 | 0.0660 |
| 1 | -0.1842 | 0.0000 | 0.0690 |
| 2 | 0.7059 | 0.0000 | 0.0590 |
| 3 | 0.8257 | 0.0050 | 0.0620 |
| 4 | -0.7778 | 0.0480 | 0.0650 |

Decision: not passed as a longer pilot. The longer run still shows that both
random and gated intervention can reduce spread relative to no-intervention,
but the risk-aware timing advantage does not persist against matched random.

## Interpretation

The routing-grid candidate fixed the earlier cap-fail withholding problem and
passed the 20/5 normal-baseline pilot. However, the 50/10 longer pilot exposes a
new stability issue:

- gate activation falls toward the lower bound (`0.0624`);
- several seeds have near-zero gated activation;
- matched random achieves a similar spread reduction with the same low
  activation budget;
- the selection/timing advantage is therefore not robust yet.

Current claim boundary:

> Boost-only routing is a viable short-pilot candidate, but longer MAPPO pilot
> validation is inconclusive and does not yet support full LOB/MAPPO main
> experiments.

Do not write:

> Risk-aware gated routing is validated in MAPPO/full LOB.

## Next Step

Do not launch full main yet. The next targeted diagnostic should explain why
the 50/10 run loses timing advantage:

1. Compare passive risk-aware score distributions after 20-episode and
   50-episode MAPPO training.
2. Check whether low activation comes from true lower-spread behavior or from a
   calibration shift in the gate score.
3. Add a minimum-budget or online-quantile gate variant only if the diagnostic
   shows that the static threshold is too conservative after longer training.
4. Re-run a 50/10 pilot only after the activation/timing issue is understood.

## Reproduction Commands

20/5 seeded-random normal-baseline pilot:

```bash
python revision/experiments/run_lob_mappo_pilot.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods no_intervention,always_on,threshold,gated_intervention_controlled \
  --routing-rho 0.75 \
  --routing-spread-cap-ticks 4 \
  --routing-withhold-rate 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output revision/experiments/outputs/lob_mappo_candidate_boost_only_pilot_seeded_random_v1/summary.json \
  --metrics-csv revision/experiments/outputs/lob_mappo_candidate_boost_only_pilot_seeded_random_v1/metrics.csv
```

50/10 longer pilot:

```bash
python revision/experiments/run_lob_mappo_pilot.py \
  --episodes 50 \
  --eval-episodes 10 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods no_intervention,always_on,threshold,gated_intervention_controlled \
  --routing-rho 0.75 \
  --routing-spread-cap-ticks 4 \
  --routing-withhold-rate 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output revision/experiments/outputs/lob_mappo_candidate_boost_only_longer_pilot_v1/summary.json \
  --metrics-csv revision/experiments/outputs/lob_mappo_candidate_boost_only_longer_pilot_v1/metrics.csv
```
