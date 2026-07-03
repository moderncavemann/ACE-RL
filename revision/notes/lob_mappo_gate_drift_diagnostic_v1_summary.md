# Local LOB/MAPPO Gate Drift Diagnostic v1 Summary

Date: 2026-07-03

This follow-up extends the first local LOB/MAPPO pilot. It is still a pilot
diagnostic, not a full LOB/MAPPO main experiment and not method validation.

## Scope

- Backend: `revision/src/envs/local_lob_market_env.py`
- Learner: `revision/src/agents/mappo.py`
- Gate: `revision/src/interventions/lob_gates.py`
- Diagnostic runner:
  `revision/experiments/diagnose_lob_mappo_gate_drift.py`
- Pilot runner: `revision/experiments/run_lob_mappo_pilot.py`
- N: `10`
- Episode length: `100`
- Seeds: `0-4`
- Train episodes per method/seed: `20`
- Eval episodes per method/seed: `5`
- Device: `mps` via `--device auto`

The goal is to test whether the previous MAPPO blocker can be narrowed down:
a no-intervention-calibrated static threshold did not reliably control
activation after agents trained under intervention.

## Code Changes

The gate now supports a controlled mode:

- `risk_aware`: original static risk-aware threshold.
- `risk_aware_controlled`: same risk-aware score, but with a rolling activation
  budget controlled by `activation_cap`, `activation_window`, and
  `activation_cooldown`.

The MAPPO runner now supports:

- method name `gated_intervention_controlled`;
- automatic matched-random rate based on the primary gated method;
- aggregate fields for raw active rate and control-budget state.

## Gate Drift Diagnostic

Output:

- `revision/experiments/outputs/lob_mappo_gate_drift_diagnostic_v1/`

Methods:

- `no_intervention`
- `gated_intervention`
- `random_gate`

The random gate used the previous conservative pilot matched rate,
`0.4980`, as a diagnostic comparator.

| Method | Avg spread | Passive gate act. | Actual gate act. | Passive score mean | Passive score p90 | Volume | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 3.4188 | 0.3700 | 0.0000 | 0.3399 | 0.6489 | 3.7752 | -0.007831 |
| gated_intervention | 3.4072 | 0.3608 | 0.3608 | 0.3212 | 0.6978 | 3.4780 | -0.010767 |
| random_gate | 3.4504 | 0.3328 | 0.5180 | 0.3184 | 0.7628 | 4.1556 | -0.011817 |

Interpretation:

- Under the MPS run, the old static gated method activates at `0.3608`, not the
  `0.4980` seen in the earlier CPU conservative pilot. MAPPO policy
  distributions are therefore sensitive to runtime/training path.
- Passive score distributions remain close to the calibrated threshold. The
  no-intervention policy would trigger passively on `37.0%` of evaluation
  windows, near the top of the intended range.
- The diagnostic random gate is overactive relative to the current MPS policy
  distribution because it reuses the previous conservative pilot rate.

This diagnostic supports adding an explicit activation budget, but it does not
show that static risk-aware routing is effective in MAPPO.

## Controlled MAPPO Pilot

Output:

- `revision/experiments/outputs/lob_mappo_controlled_pilot_v1/`

Methods:

- `no_intervention`
- `always_on`
- `threshold`
- `gated_intervention_controlled`
- `random_gate`, automatically matched to controlled gate activation

| Method | Avg spread | Best spread | Volume | Mean reward | Gate act. | Raw act. | FPR | TPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 3.4188 | 6.1680 | 3.7752 | -0.007831 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| always_on | 3.8667 | 7.1744 | 3.0136 | -0.009587 | 1.0000 | 1.0000 | 1.0000 | 0.2000 |
| threshold | 3.5759 | 6.5624 | 3.7096 | -0.010372 | 0.0728 | 0.0728 | 0.0000 | 0.2000 |
| random_gate | 3.4216 | 6.4988 | 3.7652 | -0.012087 | 0.1740 | 0.1740 | 0.1370 | 0.0343 |
| gated_intervention_controlled | 3.8780 | 7.1764 | 3.6088 | -0.015574 | 0.1776 | 0.3420 | 0.0000 | 0.2154 |

Paired spread comparison:

- `random_gate - gated_intervention_controlled` mean: `-0.4564`
- SE: `1.0696`
- Positive seeds: `2 / 5`

Pilot diagnostics:

- no-intervention spread elevation: passed;
- gate nonzero: passed;
- gate activation in target range: passed, `0.1776`;
- matched random runs: passed, `0.1740`;
- always-on saturation: passed;
- routing changes volume: passed;
- reward does not numerically collapse: passed.

Main-experiment efficacy gate:

- failed. Controlled gated routing has higher spread than matched random and
  worse mean reward in this pilot.

## Controlled Routing Split

Output:

- `revision/experiments/outputs/lob_mappo_controlled_routing_diagnostic_v1/`

This diagnostic reruns only `gated_intervention_controlled` and records the
per-step routing state.

| Window | Share of all steps | Share of active steps | Avg spread | Best spread | Volume | Routing multiplier | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| active, cap met | 0.0168 | 0.0946 | 6.0000 | 3.6190 | 6.5238 | 1.9375 | -0.000076 |
| active, cap not met | 0.1608 | 0.9054 | 5.0095 | 8.6318 | 2.3234 | 0.6250 | -0.011821 |
| inactive | 0.8224 | - | 3.6134 | 6.9645 | 3.8006 | 1.0000 | -0.016624 |

Only `9.5%` of active windows satisfy the routing spread cap and receive flow
boost. Most active windows are high-spread cap failures, so the intervention is
mostly withholding flow rather than rewarding competitive quotes. This is a
specific routing-design blocker, not just an activation-control blocker.

## Decision

Activation control works as an engineering fix, but it does not justify moving
to a full LOB/MAPPO main experiment.

The current MAPPO-local-LOB status is:

> Infrastructure is runnable and activation can be budgeted, but the
> intervention effect is not favorable relative to matched random.

Do not write:

> Risk-aware gated routing is validated in MAPPO/full LOB.

Do write:

> The MAPPO pilot identified a remaining intervention-design blocker: after
> activation is controlled, gated routing still does not improve spread/reward
> relative to matched random in the local LOB pilot.

## Recommended Next Step

Do not scale to full main yet. The next targeted step should adjust and test the
routing incentive itself:

1. Compare action distributions before and after active intervention windows.
2. Test a small routing-parameter grid, not a full experiment:
   `routing_rho`, `routing_spread_cap_ticks`, and `routing_withhold_rate`.
3. Only repeat the MAPPO pilot after finding a routing setting where active
   intervention does not systematically widen spreads.

## Reproduction Commands

Gate-drift diagnostic:

```bash
python revision/experiments/diagnose_lob_mappo_gate_drift.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods no_intervention,gated_intervention,random_gate \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_gate_drift_diagnostic_v1
```

Controlled pilot:

```bash
python revision/experiments/run_lob_mappo_pilot.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods no_intervention,always_on,threshold,gated_intervention_controlled \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output revision/experiments/outputs/lob_mappo_controlled_pilot_v1/summary.json \
  --metrics-csv revision/experiments/outputs/lob_mappo_controlled_pilot_v1/metrics.csv
```

Controlled routing split:

```bash
python revision/experiments/diagnose_lob_mappo_gate_drift.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods gated_intervention_controlled \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_controlled_routing_diagnostic_v1
```
