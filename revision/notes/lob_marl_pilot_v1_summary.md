# Local LOB/MARL Pilot v1 Summary

This pilot starts the transition from the toy LOB-inspired spread-choice
environment to a real matching-engine workflow. It is intentionally not a full
MAPPO result and should not be cited as full LOB/MAPPO validation.

## Scope

- Backend: local price-time priority matching engine in
  `revision/src/envs/local_lob_market_env.py`.
- Agents: independent tabular Q learners, used only as a cheap MARL bridge.
- Intervention: platform-level order-flow routing proxy. When the gate is
  active, the platform routes more exogenous market flow only if the best
  spread is competitive; otherwise it withholds part of that flow.
- Gate: risk-aware diagnostic score in
  `revision/src/interventions/lob_gates.py`.
- Purpose: verify that the LOB measurement, gate activation, matched random
  baseline, routing channel, and summary metrics work before a MAPPO adapter is
  attached.

## Run

Calibration used no-intervention records only:

```bash
python revision/experiments/calibrate_lob_gate.py \
  --episodes 30 \
  --eval-episodes 5 \
  --episode-length 200 \
  --n-agents 10 \
  --seeds 0-4 \
  --target-activation 0.05,0.35 \
  --output-dir revision/experiments/outputs/lob_gate_calibration_v1
```

Pilot run:

```bash
python revision/experiments/run_lob_marl_pilot.py \
  --episodes 30 \
  --eval-episodes 5 \
  --episode-length 200 \
  --n-agents 10 \
  --seeds 0-4 \
  --jobs 5 \
  --gate-config revision/experiments/outputs/lob_gate_calibration_v1/lob_gate_config.json \
  --output revision/experiments/outputs/lob_marl_pilot_v1/summary.json \
  --metrics-csv revision/experiments/outputs/lob_marl_pilot_v1/metrics.csv
```

Outputs are under:

- `revision/experiments/outputs/lob_gate_calibration_v1/`
- `revision/experiments/outputs/lob_marl_pilot_v1/`

## Calibration

Selected frozen threshold:

- `risk_threshold`: 0.535316
- passive no-intervention activation rate: 0.3500
- TPR on low-risk high-spread windows: 0.9548
- FPR on high-risk normal-spread windows: 0.0013

The calibration was feasible under the target activation range and did not use
intervention outcomes.

## Pilot Results

All values are means across 5 seeds.

| Method | Avg agent spread | Best spread ticks | Volume | Mean reward | Gate activation | FPR | TPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 3.7632 | 4.2460 | 3.9010 | -0.027944 | 0.0000 | 0.0000 | 0.0000 |
| always_on | 3.7987 | 5.6600 | 7.0348 | -0.043535 | 1.0000 | 1.0000 | 1.0000 |
| threshold | 3.7877 | 4.7436 | 4.7078 | -0.036185 | 0.3292 | 0.0000 | 1.0000 |
| random_gate | 3.8207 | 4.9468 | 5.2000 | -0.042376 | 0.3950 | 0.3826 | 0.4094 |
| gated_intervention | 3.8158 | 4.9294 | 5.0408 | -0.036464 | 0.3910 | 0.0019 | 0.9942 |

Paired random - gated spread delta:

- n = 5 seeds
- mean = 0.0049
- SE = 0.0336
- positive seeds = 3/5

## Diagnostics

The pilot passes the infrastructure gate:

- no-intervention has elevated spread;
- gated activation is nonzero and inside the 5%-40% target range;
- always-on saturates as expected;
- matched random runs at the same realized activation rate;
- routing changes market volume;
- reward does not numerically collapse.

It does not pass an efficacy gate:

- gated intervention does not yet materially reduce spread versus no
  intervention;
- gated is only marginally below matched random on average;
- the paired delta is too small and too noisy for a mechanism claim.

## Interpretation

This run supports moving from toy-only code toward a LOB pilot pipeline, but it
does not support a full LOB/MAPPO effectiveness claim.

The correct current claim is:

> The local LOB/MARL pilot infrastructure is working: the gate activates in a
> controlled range, the matched random baseline is rate-matched, and the
> routing intervention measurably changes order flow.

Do not claim:

> Risk-aware gated routing improves collusion or spread outcomes in full LOB or
> MAPPO.

## Next Step

The next experiment should attach this measurement/intervention layer to a
small MAPPO or PPO seller pilot, or first run a slightly longer local LOB pilot
if the MAPPO adapter cost is high. The entry condition for a full main
experiment remains unchanged: no-intervention spread elevation, controlled gate
activation, measurable intervention effect, stable training, and a fair matched
random baseline.
