# Local LOB/MAPPO Pilot v1 Summary

Date: 2026-07-03

This pilot attaches a minimal MAPPO-style shared-policy learner to the local
price-time priority LOB. It is not a full LOB/MAPPO main experiment and should
not be cited as full-method validation.

## Scope

- Backend: `revision/src/envs/local_lob_market_env.py`
- Learner: shared actor with centralized critic observations in
  `revision/src/agents/mappo.py`
- Runner: `revision/experiments/run_lob_mappo_pilot.py`
- Gate calibration runner:
  `revision/experiments/calibrate_lob_mappo_gate.py`
- N: `10`
- Episode length: `100`
- Seeds: `0-4`
- Train episodes per method/seed: `20`
- Eval episodes per method/seed: `5`
- Device: CPU

The goal is to test whether the local LOB, MAPPO-style learner, gate,
rate-matched random baseline, routing channel, and metrics can run together
before any full main experiment is attempted.

## Added MAPPO Pilot Components

The MAPPO implementation is deliberately compact and pilot-oriented:

- shared stochastic actor over discrete half-spread actions;
- centralized critic observations combining agent-local and market-global
  features;
- PPO clipped objective with GAE;
- per-episode on-policy updates;
- deterministic policy evaluation.

This is enough for a pilot plumbing test. It is not yet a tuned MAPPO
implementation.

## Calibration

The previous local LOB gate calibration was produced under tabular Q-learning.
A smoke run showed that directly reusing that calibration can make the MAPPO
gate inactive, so this pilot adds MAPPO-specific no-intervention calibration.

Two no-intervention calibration passes were run:

| Calibration | Target activation | Threshold | Calibration activation | Calibration FPR | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| base | 0.05-0.35 | 0.390052 | 0.1912 | 0.0000 | feasible, but only 1 low-risk high-spread positive |
| conservative | 0.05-0.15 | 0.429374 | 0.1000 | 0.0000 | feasible, but still sparse positives |

Both calibrations used no-intervention MAPPO records only. Neither calibration
uses intervention outcomes, but the sparse low-risk high-spread positives mean
the calibration TPR is not yet very informative.

## Base MAPPO Pilot Result

Output:

- `revision/experiments/outputs/lob_mappo_gate_calibration_v1/`
- `revision/experiments/outputs/lob_mappo_pilot_v1/`

| Method | Avg agent spread | Best spread ticks | Volume | Mean reward | Gate act. | FPR | TPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 2.8477 | 5.4456 | 3.7780 | -0.011668 | 0.0000 | 0.0000 | 0.0000 |
| always_on | 2.5288 | 4.9392 | 4.4168 | -0.013514 | 1.0000 | 1.0000 | 0.2000 |
| threshold | 3.4380 | 6.5116 | 3.7132 | -0.006234 | 0.0540 | 0.0000 | 0.2000 |
| random_gate | 2.8941 | 5.1336 | 4.7308 | -0.014703 | 0.5940 | 0.4285 | 0.4427 |
| gated_intervention | 4.6783 | 8.4680 | 3.0916 | -0.011350 | 0.5680 | 0.0111 | 0.6000 |

Base paired spread comparison:

- `random_gate - gated_intervention` mean: `-1.7842`
- SE: `1.0371`
- Positive seeds: `1 / 5`

## Conservative Calibration Follow-up

Output:

- `revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/`
- `revision/experiments/outputs/lob_mappo_pilot_conservative_v1/`

| Method | Avg agent spread | Best spread ticks | Volume | Mean reward | Gate act. | FPR | TPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 2.8477 | 5.4456 | 3.7780 | -0.011668 | 0.0000 | 0.0000 | 0.0000 |
| always_on | 2.5288 | 4.9392 | 4.4168 | -0.013514 | 1.0000 | 1.0000 | 0.2000 |
| threshold | 3.4380 | 6.5116 | 3.7132 | -0.006234 | 0.0540 | 0.0000 | 0.2000 |
| random_gate | 3.1957 | 5.9852 | 4.4012 | -0.021067 | 0.5040 | 0.3096 | 0.2063 |
| gated_intervention | 4.0259 | 7.1068 | 3.3672 | -0.008723 | 0.4980 | 0.0012 | 0.8000 |

Conservative paired spread comparison:

- `random_gate - gated_intervention` mean: `-0.8302`
- SE: `0.7578`
- Positive seeds: `2 / 5`

## Pilot Decision

Infrastructure status: mostly passed.

- no-intervention produces elevated spread in this local LOB;
- MAPPO training/evaluation runs without crashing;
- the gate is nonzero;
- always-on saturates as expected;
- matched random runs at the realized gated activation rate;
- routing changes order-flow volume;
- reward does not numerically collapse.

Main-experiment entry status: failed.

- Base calibrated gate is overactive in intervention evaluation:
  `0.5680`, above the intended 5%-40% range.
- Conservative calibration is still overactive:
  `0.4980`.
- Gated intervention raises average spread relative to matched random in both
  pilot runs.
- Paired spread deltas are negative, with only `1/5` and `2/5` positive seeds.
- Calibration positives are sparse, so the current MAPPO no-intervention
  threshold is not robust enough.

## Interpretation

This pilot verifies that the MAPPO-local-LOB pipeline is runnable, but it does
not support launching the full LOB/MAPPO main experiment yet.

The immediate blocker is a MAPPO-specific distribution shift:

> A threshold calibrated on MAPPO no-intervention records does not control gate
> activation once agents train under gated intervention.

The current result should be written as:

> The local LOB/MAPPO pilot infrastructure is operational, but full main
> experiments are blocked by gate activation drift and unfavorable matched-random
> spread comparison.

Do not write:

> Risk-aware gated routing is validated in MAPPO/full LOB.

## Recommended Next Step

Before full main, run targeted diagnostics rather than scaling up:

1. Compare passive score distributions under no-intervention-trained MAPPO
   policies and gated-trained MAPPO policies.
2. Add an activation controller or moving-threshold cap so the gate cannot
   drift to 50%-60% activation during intervention training.
3. Revisit the LOB routing incentive: current cap-based routing often withholds
   flow when spreads are wide, but MAPPO policies still learn high spreads under
   the gated condition.
4. Run a longer no-intervention MAPPO calibration to collect enough low-risk
   high-spread positives.
5. Only after a repeat pilot satisfies activation in 5%-40% and a non-negative
   matched-random spread comparison should a full main experiment be launched.

## Reproduction Commands

Base calibration:

```bash
python revision/experiments/calibrate_lob_mappo_gate.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --target-activation 0.05,0.35 \
  --device cpu \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_gate_calibration_v1
```

Base pilot:

```bash
python revision/experiments/run_lob_mappo_pilot.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods no_intervention,always_on,threshold,gated_intervention \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_v1/lob_mappo_gate_config.json \
  --device cpu \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output revision/experiments/outputs/lob_mappo_pilot_v1/summary.json \
  --metrics-csv revision/experiments/outputs/lob_mappo_pilot_v1/metrics.csv
```

Conservative calibration and pilot use the same commands with
`--target-activation 0.05,0.15` and the
`lob_mappo_gate_calibration_conservative_v1` /
`lob_mappo_pilot_conservative_v1` output directories.
