# ABIDES/MAPPO Adaptive v5 Supplemental-Control Protocol

Date frozen: 2026-07-13

## Purpose

Adaptive v5 remains the completed primary experiment. This post-main study
does not alter its method, status, endpoint, policy seeds, or claim. It answers
two narrower reviewer-facing questions:

1. Does risk-aware timing still beat a schedule that preserves the exact
   signal sequence and temporal clustering but circularly shifts its alignment?
2. Is the two-feature public risk score materially better than volatility-only
   or lagged-return-only logistic proxies trained on the same passive labels?

The study also records an economic decomposition and a cost/frontier table on
fresh evaluation paths. It is a supplemental control package, not a new v6.

## Frozen Source

- Passed adaptive-v5 policies: seeds `530000-530019`.
- V5 response fingerprint:
  `8636827cb45cdb774a1351183858aaa0edc058b90f3d4a269c51c74ca467674c`.
- Base policies and all five v5 response adapters are loaded from their atomic
  checkpoints and verified against the recorded final model digests.
- New controls start from the same frozen base policy and train for the same
  `96` response episodes, in batches of eight, on the same response paths and
  action uniforms used by v5.
- Final evaluation paths are fresh seeds `533000-533009`; they are disjoint
  from v5 evaluation seeds `532000-532009` and base convergence seeds.

Reusing the v5 policy seeds is intentional: this is a paired missing-control
study. It is not an independent policy-population robustness experiment.

## New Controls

### Circular Shift

For each training or evaluation episode, the complete risk-aware continuous
signal sequence is generated first from lagged public state. A seeded nonzero
circular offset is then applied. The comparator therefore preserves:

- the exact multiset of signal strengths;
- the number of active decisions;
- the complete cyclic autocorrelation and run structure;
- the exogenous path and action uniforms.

It changes only the alignment between that sequence and the market path, apart
from the unavoidable linear episode boundary after a cyclic shift.

### Public One-Feature Proxies

Two balanced logistic models are fit once on the existing passive gate records:

- `volatility_only`: public trailing volatility;
- `return_only`: absolute lagged public return.

Both models use `simulator_low_risk` as their training label. This is explicit
simulator supervision, not unsupervised latent-collusion discovery. Their
probability threshold is fixed at `0.5`; pacing rates are selected on passive
validation records only to match the two-feature model's paced activation.
No intervention, v5 response, or supplemental outcome enters calibration.

The proxy comparisons are diagnostic. The package is permitted to pass if a
simple proxy matches or beats the two-feature model; such a result would require
simplifying the method claim rather than hiding the comparator.

## Learner And Timing Contract

The learner is a compact parameter-sharing/multihead PPO implementation with a
centralized critic and a separately trained intervention adapter. The base
actor trunk and actor heads remain frozen; the intervention adapter and critic
are optimized per arm. It must be described as `MAPPO-style adapter training`,
not as an unqualified standard MAPPO implementation or complete policy retrain.

The signal is computed before the current quote action. It uses only the state
available at the start of the decision, including lagged quotes, trailing
public volatility, and lagged public return. Soft-rank routing weights are then
computed from the current quotes after agents respond to the announced signal.

Episodes are defined as finite-horizon 100-decision economic tasks. Inventory
is marked to the contemporaneous fundamental in the final reward; the task
terminates at that horizon. The zero terminal bootstrap follows this finite-
horizon objective and must not be described as continuing-market truncation.

## Primary Supplemental Endpoint

The primary contrast is paired
`circular_shifted - risk_aware` low-risk customer execution half-spread.
Positive values favor risk-aware alignment.

The structured-timing control passes only if:

- all 20 policy seeds complete;
- the mean paired contrast is positive;
- at least 14/20 paired seed differences are positive;
- the t-based 95% confidence interval is strictly above zero;
- circular signal dose and circular structure are exact in training/evaluation;
- CRN paths, gate-off actions, response convergence, and volume invariants pass.

The public proxy arms must remain within `0.05` absolute activation rate of the
risk-aware arm for their spread contrasts to be interpreted as approximately
budget matched. Their superiority or inferiority is reported, not used as an
overall pass veto.

## Economic And Market Diagnostics

For every arm, report:

- low- and high-risk execution half-spread;
- activation, high-risk activation, and mean signal strength;
- customer execution-cost tick-units per episode;
- aggregate dealer reward units per episode;
- spread revenue, adverse-selection component, and inventory penalty;
- targeted events, flow HHI, and inventory dispersion;
- total customer flow, which is fixed at 20 by construction.

Volume conservation is a mechanism invariant and sanity check, not an empirical
market-activity benefit. No total-welfare claim is allowed without an explicit
customer/dealer accounting identity.

## Stop Rule

The formal run is single-shot. If circular timing fails, report that temporal
structure explains part or all of the v5 random-control advantage. If a simple
proxy matches the full risk score, narrow the method to that proxy. Do not tune
offsets, proxy thresholds, pacing rates, response horizon, or seed inclusion on
the formal outcomes.

Reproduction command:

```bash
python3 revision/experiments/run_abides_mappo_adaptive_v5_supplement.py \
  --config revision/configs/abides_mappo_adaptive_v5_supplement_v1.json
```
