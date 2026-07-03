# Local LOB/MAPPO Assay Stabilization v1 Summary

Date: 2026-07-03

This note reframes the local LOB/MAPPO work as a claim-aligned experiment
pipeline. It separates observation evidence, assay stabilization, main
comparison, and upper-bound diagnostics. It should not be cited as full
LOB/MAPPO validation.

## Experiment Alignment

| Claim / question | Experiment type | Evidence required | Current status |
| --- | --- | --- | --- |
| Does the MAPPO-local-LOB assay naturally produce a high-spread failure mode? | Observation / preliminary | No-intervention failure rate across many seeds | Base 50-episode assay fails; inventory-high assay passes observation only |
| Does risk-aware timing beat same-rate random intervention? | Main comparison | Paired random-minus-gated spread delta positive on most seeds | Not passed |
| Is the routing channel itself enough to make timing matter? | Oracle / upper bound | Oracle timing should clearly beat random | Not passed in base short-horizon reallocation |
| Is the result robust to key assay parameters? | Sensitivity | Stable failure mode and stable comparison under controlled changes | Not passed yet |
| Can this support full LOB/MAPPO main? | Gate to full main | Observation and main comparison both pass | No-go |

## Observation: Base No-Intervention Assay

Outputs:

- 50-episode base assay:
  `revision/experiments/outputs/lob_mappo_no_intervention_assay_base20_v1/`
- 20-episode base assay:
  `revision/experiments/outputs/lob_mappo_no_intervention_assay_base20ep_20seeds_v1/`

The base no-intervention assay is horizon-sensitive:

| Setting | Seeds | Episodes | Failure rate, spread >= 3 | Strong failure, spread >= 4 | Mean spread | Action alignment | Action entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 20 | 20 | 0.70 | 0.45 | 3.6569 | 0.9703 | 0.0351 |
| base | 20 | 50 | 0.45 | 0.15 | 2.8358 | 0.9910 | 0.0103 |

The 20-episode setting barely passes the observation gate, but the same assay
loses the failure mode by 50 episodes. This means the base MAPPO-local-LOB
failure mode is not stable enough for full main experiments.

## Stabilization Grid

Output:

- `revision/experiments/outputs/lob_mappo_no_intervention_assay_stabilization_grid_v1/`

The stabilization grid only used no-intervention MAPPO. It did not tune the
intervention method.

| Scenario | Seeds | Episodes | Failure rate | Strong failure | Mean spread | Action alignment | Action entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inventory_high | 10 | 50 | 1.00 | 0.80 | 4.8476 | 0.9925 | 0.0094 |
| low_entropy | 10 | 50 | 0.70 | 0.30 | 3.5418 | 0.9945 | 0.0057 |
| flow_low | 10 | 50 | 0.70 | 0.40 | 3.4146 | 0.9968 | 0.0038 |
| lr_1e3 | 10 | 50 | 0.60 | 0.10 | 3.1982 | 0.9737 | 0.0304 |
| entropy_001 | 10 | 50 | 0.50 | 0.30 | 2.7758 | 0.9777 | 0.0256 |
| risk_persistent | 10 | 50 | 0.40 | 0.20 | 2.7943 | 0.9931 | 0.0086 |
| flow_high | 10 | 50 | 0.30 | 0.00 | 2.4942 | 0.9815 | 0.0200 |
| inventory_low | 10 | 50 | 0.00 | 0.00 | 1.2947 | 0.9992 | 0.0009 |

`inventory_high` is the first stabilized no-intervention assay candidate. It
passes the observation gate, but passing observation is not enough for full main
experiments.

## Inventory-High Gate Calibration

Output:

- `revision/experiments/outputs/lob_mappo_gate_calibration_inventory_high_v1/`

Calibration used no-intervention seeds 0-4 under `inventory_penalty=0.006`.
The evaluation seeds used for later comparison were not used to tune the
threshold.

Selected threshold:

- risk threshold: `0.5760`
- calibration activation: `0.3492`
- low-risk high-spread TPR: `0.7461`
- high-risk normal FPR: `0.0000`

## Main Comparison Under Stabilized Assay

Outputs:

- `revision/experiments/outputs/lob_mappo_inventory_high_reallocate_rho05_50_10seeds_v1/`
- `revision/experiments/outputs/lob_mappo_inventory_high_reallocate_rho075_50_10seeds_v1/`

The comparison used `relative_reallocate`, frozen inventory-high gate
calibration, and matched random with the same realized activation rate.

| Setting | Seeds | No-intervention spread | Gated spread | Random spread | Delta random - gated | Positive seeds | Gated reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rho=0.5 | 10 | 4.8523 | 4.3684 | 4.9780 | 0.6096 | 5/10 | -0.0330 |
| rho=0.75 | 10 | 4.8523 | 5.0305 | 5.2640 | 0.2335 | 5/10 | -0.0325 |

The inventory-high assay creates a stable high-spread regime, but risk-aware
gated routing still does not beat matched random on enough seeds. Both tested
reallocation strengths have only `5/10` positive paired seeds.

## Short-Horizon Base Main Comparison

Output:

- `revision/experiments/outputs/lob_mappo_relative_reallocate_rho075_20_20seeds_v1/`

The base 20-episode assay passed the observation gate, but the main comparison
failed:

- no-intervention spread: `3.6569`
- gated spread: `3.7858`
- random spread: `3.7823`
- paired random-minus-gated delta: `-0.0035`
- positive seeds: `10/20`

## Oracle Upper Bound

Output:

- `revision/experiments/outputs/lob_mappo_relative_reallocate_oracle_20_20seeds_v1/`

Oracle timing did not produce a usable upper bound in the base short-horizon
assay:

- paired random-minus-oracle spread delta: `-0.5527`
- positive seeds: `9/20`

This means the issue is not simply the learned risk-aware score threshold. The
current MAPPO-local-LOB routing channel and assay do not yet make intervention
timing reliably identifiable.

## Code / Reproducibility Updates

New diagnostic:

- `revision/experiments/diagnose_lob_mappo_assay.py`

Runner updates:

- `run_lob_mappo_pilot.py` and `calibrate_lob_mappo_gate.py` now expose
  environment overrides such as `--inventory-penalty`,
  `--base-flow-intensity`, `--p-low-to-high`, and `--p-high-to-low`.
- `relative_reallocate` now supports `routing_target_kappa`. The default
  `kappa=0` reproduces the uniform target distribution used in the runs above;
  positive kappa implements a softmax target distribution favoring lower
  spreads among eligible undercutters.

## Decision

Do not launch full LOB/MAPPO main experiments yet.

The local LOB/MAPPO work now has a clearer diagnosis:

1. Base no-intervention is not a stable long-horizon failure mode.
2. Inventory-high stabilizes the observation failure mode.
3. Even under inventory-high, risk-aware gated reallocation does not beat
   matched random on enough seeds.
4. Oracle timing does not currently provide a strong upper bound.

The correct paper framing remains:

> Toy-scale validation supports the risk-aware gated-routing mechanism. The
> local LOB/MAPPO extension remains diagnostic and does not yet support a full
> LOB/MAPPO effectiveness claim.

## Next Full-Main Gate

A future full LOB/MAPPO main experiment should only start after a candidate
passes all of the following:

1. no-intervention failure rate >= 70% on at least 20 held-out seeds;
2. gated activation between 5% and 35%;
3. gated spread below matched random with at least 70% positive paired seeds;
4. reward not materially worse than random or no-intervention;
5. no-routing ablation near no-intervention;
6. shuffled gate weaker than real gated;
7. oracle timing clearly stronger than real gated and matched random.

Current status: observation can be stabilized, but main comparison and oracle
upper-bound gates are not passed.
