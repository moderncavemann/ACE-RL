# Local LOB/MAPPO Routing-Channel Repair v1 Summary

Date: 2026-07-10

This note records a targeted attempt to repair the local LOB/MAPPO pilot enough
to justify a later full-main experiment. It should not be cited as full
LOB/MAPPO validation.

## Goal

Previous diagnostics showed that the local LOB/MAPPO stage had two blockers:

1. the no-intervention high-spread assay was unstable unless using the
   `inventory_high` stabilization setting; and
2. risk-aware gated routing did not reliably beat same-rate matched random, and
   the low-risk high-spread oracle did not provide a clean upper bound.

This repair targeted the routing channel, not the risk gate threshold. The
hypothesis was that `relative_reallocate` had a dead zone: in homogeneous
high-spread regimes there may be no undercutting agent, so the platform has no
positive target for reallocation.

## Code Changes

`LocalLOBMarketEnv` now supports two additional routing modes:

| Mode | Mechanism | Purpose |
| --- | --- | --- |
| `continuous_reallocate` | During active gates, reallocate a fixed share of normal order flow across all agents with softmax weights favoring lower spreads. | Removes the no-undercutter dead zone. |
| `balanced_reallocate` | Same continuous target selection, but targeted flow is paired buy/sell flow for the selected agent. | Reduces inventory-noise penalties from targeted one-sided flow. |

`run_lob_mappo_routing_grid.py` was extended to:

- run timing methods `gated_intervention_controlled`, `oracle_low_high_spread`,
  and `threshold` against same-rate matched random;
- accept assay-environment overrides such as `--inventory-penalty`; and
- record the full base config in the output summary.

## Commands

Smoke:

```bash
python revision/experiments/run_lob_mappo_routing_grid.py \
  --episodes 1 --eval-episodes 1 --episode-length 20 --n-agents 10 --seeds 0 \
  --routing-modes continuous_reallocate,balanced_reallocate \
  --routing-rhos 0.5 --routing-spread-caps 4 --routing-withhold-rates 0.0 \
  --routing-relative-improvements 0.0 --routing-target-kappas 2.0 \
  --timing-method oracle_low_high_spread --inventory-penalty 0.006 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json \
  --device auto --hidden-dim 32 --update-epochs 1 --minibatch-size 128 \
  --output-dir revision/experiments/outputs/lob_mappo_routing_channel_repair_smoke
```

Oracle-first repair screen:

```bash
python revision/experiments/run_lob_mappo_routing_grid.py \
  --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 \
  --routing-modes continuous_reallocate,balanced_reallocate \
  --routing-rhos 0.25,0.5,0.75 --routing-spread-caps 4 --routing-withhold-rates 0.0 \
  --routing-relative-improvements 0.0 --routing-target-kappas 1.0,4.0 \
  --timing-method oracle_low_high_spread --inventory-penalty 0.006 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json \
  --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_repair_oracle_screen_v1
```

Threshold-timing diagnostic:

```bash
python revision/experiments/run_lob_mappo_routing_grid.py \
  --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 \
  --routing-modes balanced_reallocate --routing-rhos 0.5,0.75 \
  --routing-spread-caps 4 --routing-withhold-rates 0.0 \
  --routing-relative-improvements 0.0 --routing-target-kappas 1.0,4.0 \
  --timing-method threshold --inventory-penalty 0.006 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json \
  --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_repair_threshold_screen_v1
```

Gated repair screen and 10-seed candidate expansion:

```bash
python revision/experiments/run_lob_mappo_routing_grid.py \
  --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 \
  --routing-modes balanced_reallocate --routing-rhos 0.5,0.75 \
  --routing-spread-caps 4 --routing-withhold-rates 0.0 \
  --routing-relative-improvements 0.0 --routing-target-kappas 1.0,4.0 \
  --timing-method gated_intervention_controlled --inventory-penalty 0.006 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json \
  --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_repair_gated_screen_v1

python revision/experiments/run_lob_mappo_routing_grid.py \
  --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-9 \
  --routing-modes balanced_reallocate --routing-rhos 0.75 \
  --routing-spread-caps 4 --routing-withhold-rates 0.0 \
  --routing-relative-improvements 0.0 --routing-target-kappas 4.0 \
  --timing-method gated_intervention_controlled --inventory-penalty 0.006 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/lob_mappo_gate_config.json \
  --device auto --hidden-dim 64 --update-epochs 4 --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_repair_gated_candidate_10seeds_v1
```

## Results

### Low-risk Oracle Timing

Output:

- `revision/experiments/outputs/lob_mappo_repair_oracle_screen_v1/`

No cell reached the minimum screen requirement of at least `4/5` positive seeds.
The best mean-delta rows were unstable:

| Timing | Routing | rho | kappa | Activation | Oracle spread | Random spread | Random - oracle | Positive seeds | Reward delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low-risk oracle | balanced | 0.50 | 4.0 | 0.3364 | 4.4967 | 4.6021 | 0.1054 | 2/5 | 0.0172 |
| low-risk oracle | balanced | 0.75 | 1.0 | 0.2472 | 4.1370 | 4.1726 | 0.0356 | 3/5 | 0.0118 |

Interpretation: the low-risk high-spread oracle still does not provide a usable
upper bound. This blocks a direct risk-aware full-main claim.

### High-spread Threshold Timing

Output:

- `revision/experiments/outputs/lob_mappo_repair_threshold_screen_v1/`

Threshold timing did produce a stronger screen result:

| Timing | Routing | rho | kappa | Activation | Timed spread | Random spread | Random - timed | Positive seeds | Reward delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold | balanced | 0.75 | 4.0 | 0.2456 | 3.6546 | 5.2428 | 1.5882 | 4/5 | 0.0004 |
| threshold | balanced | 0.75 | 1.0 | 0.2564 | 3.6172 | 4.6369 | 1.0197 | 4/5 | -0.0017 |

Interpretation: balanced reallocation can make high-spread timing matter in the
short screen. This is a partial channel repair, but it is not yet evidence for
the risk-aware method because threshold timing is not risk adjusted.

### Risk-aware Gated Timing

Output:

- `revision/experiments/outputs/lob_mappo_repair_gated_screen_v1/`
- `revision/experiments/outputs/lob_mappo_repair_gated_candidate_10seeds_v1/`

The 5-seed gated screen found one candidate:

| Timing | Routing | rho | kappa | Activation | Gated spread | Random spread | Random - gated | Positive seeds | Reward delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| risk-aware gated | balanced | 0.75 | 4.0 | 0.2136 | 4.2928 | 4.7114 | 0.4186 | 4/5 | 0.0117 |

However, the same candidate did not survive the 10-seed expansion:

| Timing | Routing | rho | kappa | Activation | Gated spread | Random spread | Random - gated | Positive seeds | Reward delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| risk-aware gated | balanced | 0.75 | 4.0 | 0.2064 | 4.1058 | 4.1185 | 0.0127 | 4/10 | 0.0119 |

Per-seed deltas remained mixed:

| Seed | Random spread | Gated spread | Random - gated |
| ---: | ---: | ---: | ---: |
| 0 | 4.5364 | 6.1576 | -1.6212 |
| 1 | 3.4046 | 3.6148 | -0.2102 |
| 2 | 3.0000 | 1.7274 | 1.2726 |
| 3 | 6.0000 | 5.5698 | 0.4302 |
| 4 | 2.6912 | 4.3942 | -1.7030 |
| 5 | 5.0000 | 2.2828 | 2.7172 |
| 6 | 4.0000 | 1.2012 | 2.7988 |
| 7 | 1.2822 | 3.6074 | -2.3252 |
| 8 | 4.5692 | 5.6058 | -1.0366 |
| 9 | 6.7012 | 6.8968 | -0.1956 |

## Decision

Do not launch full LOB/MAPPO main experiments from this repair.

The repair improved mechanism visibility and found a partial channel signal:
balanced reallocation can make threshold-style high-spread timing beat same-rate
random in a short 5-seed screen. But the risk-aware gated candidate collapsed to
`4/10` positive seeds and near-zero mean spread advantage when expanded to 10
seeds. The low-risk oracle also failed to produce a stable upper bound.

Current status:

1. toy-scale validation remains passed;
2. local LOB/MAPPO observation can be stabilized with `inventory_penalty=0.006`;
3. balanced reallocation is a better diagnostic routing channel than sparse
   relative reallocation;
4. full LOB/MAPPO effectiveness remains no-go because risk-aware timing is not
   stable against same-rate matched random.

## Next Gate

The next repair should not be another broad routing grid. The evidence now
points to label/assay alignment:

1. check whether the inventory-high failure is actually a low-risk collusion
   regime or mostly an inventory-aversion high-spread regime;
2. if it is not low-risk collusion, redesign the local LOB assay before trying
   to make risk-aware gating pass;
3. if high-spread threshold timing is the only robust signal, report it as a
   diagnostic upper-bound result, not as the proposed risk-aware method.

Full-main promotion still requires at least:

1. no-intervention failure rate >= 70% on at least 20 held-out seeds;
2. risk-aware gated spread below matched random with at least 70% positive
   paired seeds;
3. reward not materially worse;
4. no-routing ablation near no-intervention;
5. shuffled gate weaker than real gated; and
6. oracle timing clearly stronger than real gated and matched random.
