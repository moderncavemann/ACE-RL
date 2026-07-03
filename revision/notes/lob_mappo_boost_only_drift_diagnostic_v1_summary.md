# Local LOB/MAPPO Boost-Only Drift Diagnostic v1 Summary

Date: 2026-07-03

This diagnostic explains why the boost-only strict-cap routing candidate passed
the 20/5 pilot but did not preserve a timing advantage in the 50/10 pilot. It is
not full LOB/MAPPO validation.

## Scope

- Diagnostic runner: `revision/experiments/diagnose_lob_mappo_gate_drift.py`
- Routing setting: `routing_rho=0.75`,
  `routing_spread_cap_ticks=4`, `routing_withhold_rate=0.0`
- Gate: `gated_intervention_controlled`
- Calibration: frozen MAPPO conservative gate config
- Device: `mps` via `--device auto`

The diagnostic runner now supports routing overrides and records actual
raw-active rate, control-budget state, and passive risk-aware gate scores.

## Outputs

- 20/5 diagnostic:
  `revision/experiments/outputs/lob_mappo_boost_only_drift_20_v1/`
- 50/10 diagnostic:
  `revision/experiments/outputs/lob_mappo_boost_only_drift_50_v1/`

Methods:

- `no_intervention`
- `gated_intervention_controlled`
- `random_gate`

The random gate uses the matched random rates from the corresponding pilots:

- 20/5: `0.1060`
- 50/10: `0.0624`

## Aggregate Comparison

| Horizon | Method | Spread | Actual act. | Raw act. | Budget open | Passive act. | Passive score | Passive p90 | Action mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20/5 | no_intervention | 3.4188 | 0.0000 | 0.0000 | 1.0000 | 0.3700 | 0.3399 | 0.6489 | 2.4188 |
| 20/5 | gated | 2.9912 | 0.1060 | 0.2024 | 0.8892 | 0.2016 | 0.2712 | 0.5994 | 1.9912 |
| 20/5 | random | 3.6585 | 0.1116 | 0.1116 | 1.0000 | 0.4232 | 0.3763 | 0.7425 | 2.6585 |
| 50/10 | no_intervention | 3.6052 | 0.0000 | 0.0000 | 1.0000 | 0.4392 | 0.3648 | 0.6310 | 2.6052 |
| 50/10 | gated | 2.4983 | 0.0624 | 0.0688 | 0.9822 | 0.0688 | 0.1919 | 0.3832 | 1.4983 |
| 50/10 | random | 2.4393 | 0.0632 | 0.0632 | 1.0000 | 0.0396 | 0.1845 | 0.3626 | 1.4393 |

## Seed-Level Pattern

At 20/5, the gated policy beats matched random on spread in all five seeds.
The random-trained policy still has high passive risk scores and high passive
activation:

- gated passive activation: `0.2016`
- random passive activation: `0.4232`

At 50/10, both gated and random policies move into a low-spread, low-score
regime:

- gated passive activation: `0.0688`
- random passive activation: `0.0396`
- gated spread: `2.4983`
- random spread: `2.4393`

The random policy is not merely winning by false positives. It learns a
similarly low-spread policy under the same low intervention budget.

## Diagnosis

The 50/10 failure is not caused by the rolling activation controller.

Evidence:

- gated raw active rate is `0.0688`;
- gated actual active rate is `0.0624`;
- control budget is open `98.22%` of the time;
- passive score p90 is only `0.3832`, below the frozen threshold `0.4294`.

So the gate is not being artificially suppressed by the budget cap. The policy
itself has shifted toward lower spreads and therefore lower risk-aware scores.

The main issue is:

> In longer MAPPO training, a low-rate matched random intervention can also
> move policies into a low-spread regime, so the risk-aware timing advantage
> fades.

This is different from the earlier cap-fail withholding problem. Boost-only
routing fixed the negative-flow-punishment failure mode, but it did not make the
timing advantage robust to longer training.

## Training-Time Diagnostics

Additional 50/10 runs were used to separate training-time intervention exposure
from evaluation-time gate activation.

Outputs:

- risk-aware training stats:
  `revision/experiments/outputs/lob_mappo_boost_only_gated_trainstats_50_v1/`
- random training stats:
  `revision/experiments/outputs/lob_mappo_boost_only_random_trainstats_50_v1/`
- oracle timing with training stats:
  `revision/experiments/outputs/lob_mappo_oracle_timing_boost_only_50_trainstats_v1/`
- no-routing ablation:
  `revision/experiments/outputs/lob_mappo_no_routing_controlled_gate_50_v1/`

| 50/10 method | Train act. | Train raw act. | Train boost | Eval spread | Eval act. |
| --- | ---: | ---: | ---: | ---: | ---: |
| risk-aware gated | 0.3374 | 0.6730 | 0.2530 | 2.4983 | 0.0624 |
| matched random | 0.0610 | 0.0610 | 0.0458 | 2.4393 | 0.0632 |
| oracle timing | 0.3032 | 0.3032 | 0.2274 | 3.0309 | 0.0000 |
| no-routing gated | - | - | 0.0000 | 3.6052 | 0.2048 |

The training-time results sharpen the diagnosis:

- risk-aware gating intervenes heavily during training, close to the activation
  cap, but this does not beat matched random by 50/10;
- random intervention remains genuinely low-rate during training, yet still
  reaches a low-spread policy;
- oracle timing creates a training-time policy shift, but the 50/10 result is
  only `3/5` seeds positive and not enough to establish a robust upper bound;
- no-routing controlled gate exactly matches the no-intervention spread path,
  so the routing channel is necessary and the gate alone is not causal.

The random comparator is therefore not a weak baseline in longer MAPPO training.
It can exploit the boost-only routing channel with very low intervention
frequency. The open issue is not "make the gate fire more"; it is whether the
intervention channel can make timing matter more than low-rate exposure.

## Decision

Do not launch full LOB/MAPPO main experiments.

Do not immediately add a minimum-budget or online-quantile gate. The diagnostic
does not show that the controller is blocking high-score opportunities. Forcing
extra activations would mostly push the method toward random intervention, which
is exactly the comparator that becomes strong at 50/10.

Current claim boundary:

> Boost-only routing remains a useful pilot candidate, but longer MAPPO
> training shows that matched random low-rate intervention can produce similar
> spread reduction. The MAPPO timing advantage is not robust yet.

## Recommended Next Step

The next diagnostic should test whether the timing advantage is recoverable by
making the intervention conditional effect sharper, not by forcing more
activations:

1. Test a sharper conditional routing channel, such as boost only to agents
   quoting at or inside the spread cap rather than increasing total exogenous
   flow.
2. Add training-time exposure plots for active windows, cap-met windows, and
   post-intervention action changes.
3. Only revisit gate features if a stronger conditional channel shows that
   oracle or near-oracle timing can reliably beat matched random.
4. If matched random still catches up, stop MAPPO tuning and treat the local LOB
   pilot as evidence that this intervention channel is too weak for timing
   claims.

## Reproduction Commands

20/5 drift diagnostic:

```bash
python revision/experiments/diagnose_lob_mappo_gate_drift.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods no_intervention,gated_intervention_controlled,random_gate \
  --random-rate 0.106 \
  --routing-rho 0.75 \
  --routing-spread-cap-ticks 4 \
  --routing-withhold-rate 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_boost_only_drift_20_v1
```

50/10 drift diagnostic:

```bash
python revision/experiments/diagnose_lob_mappo_gate_drift.py \
  --episodes 50 \
  --eval-episodes 10 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods no_intervention,gated_intervention_controlled,random_gate \
  --random-rate 0.0624 \
  --routing-rho 0.75 \
  --routing-spread-cap-ticks 4 \
  --routing-withhold-rate 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_boost_only_drift_50_v1
```

Risk-aware gated training stats:

```bash
python revision/experiments/diagnose_lob_mappo_gate_drift.py \
  --episodes 50 \
  --eval-episodes 10 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods gated_intervention_controlled \
  --routing-rho 0.75 \
  --routing-spread-cap-ticks 4 \
  --routing-withhold-rate 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_boost_only_gated_trainstats_50_v1
```

Random training stats:

```bash
python revision/experiments/diagnose_lob_mappo_gate_drift.py \
  --episodes 50 \
  --eval-episodes 10 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods random_gate \
  --random-rate 0.0624 \
  --routing-rho 0.75 \
  --routing-spread-cap-ticks 4 \
  --routing-withhold-rate 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_boost_only_random_trainstats_50_v1
```

Oracle timing diagnostic:

```bash
python revision/experiments/run_lob_mappo_pilot.py \
  --episodes 50 \
  --eval-episodes 10 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods oracle_low_high_spread \
  --routing-rho 0.75 \
  --routing-spread-cap-ticks 4 \
  --routing-withhold-rate 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output revision/experiments/outputs/lob_mappo_oracle_timing_boost_only_50_trainstats_v1/summary.json \
  --metrics-csv revision/experiments/outputs/lob_mappo_oracle_timing_boost_only_50_trainstats_v1/metrics.csv
```

No-routing ablation:

```bash
python revision/experiments/run_lob_mappo_pilot.py \
  --episodes 50 \
  --eval-episodes 10 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods gated_intervention_controlled \
  --routing-rho 0.0 \
  --routing-spread-cap-ticks 4 \
  --routing-withhold-rate 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output revision/experiments/outputs/lob_mappo_no_routing_controlled_gate_50_v1/summary.json \
  --metrics-csv revision/experiments/outputs/lob_mappo_no_routing_controlled_gate_50_v1/metrics.csv
```
