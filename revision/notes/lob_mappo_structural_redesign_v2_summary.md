# Local LOB/MAPPO Structural Redesign v2 Summary

Date: 2026-07-10

This note records the structural follow-up to the v1 redesigned assay. It is a
mechanism and identifiability study, not a full LOB/MAPPO method validation.

## Decision

Do not launch the full LOB/MAPPO main experiment.

The redesign now passes fixed-policy economic acceptance and shows that a
MAPPO seller policy can learn to respond to an announced low-risk routing
signal. It still does not show that high-spread timing within low-risk states
reliably beats an equal-dose, risk-stratified shuffled schedule.

The evidence boundary is:

> Risk-coupled platform routing is economically well-defined and learnable in
> the redesigned local LOB. The additional value of high-spread timing over
> low-risk equal-dose random timing is not robust in MAPPO training.

## Structural Problems Fixed

| Layer | Old failure | v2 repair |
| --- | --- | --- |
| Risk economics | Fundamental volatility did not enter seller PnL | Informed order direction, fundamental settlement, adverse-selection accounting, and spread-elastic demand |
| Actor symmetry | Shared actor produced nearly identical quotes | Shared-trunk multihead and independent actor modes |
| Quote priority | Fixed agent order created a permanent low-ID advantage | Rotating quote submission priority |
| Routing subsidy | Targeted fills could pay the seller's own collusive quote | Resting-quote fills with real resting capacity |
| Demand conservation | Earlier routing could add or remove flow | Platform reallocation conserves expected total demand |
| Order spillover | Platform routing controlled only the first unit and leaked the remainder to price-time priority | The full external order is allocated across platform candidates before any book fallback |
| Risk preservation | Artificial bid/ask pairs removed directional adverse selection | Unbalanced platform flow preserves the informed external order side |
| Random paths | Arms could see different exogenous paths | Separate CRN streams for risk, shocks, and flow |
| Evaluation seeds | `seed + offset + episode` overlapped across training seeds | A non-overlapping training-seed/eval-episode mapping |
| Dose matching | Random timing did not match training dose | Per-episode exact dose yoking |
| Comparator support | Global shuffling moved interventions into high-risk states | Shuffling only within low-risk windows |
| Endpoint feedback | Online activation was outcome-conditioned | Gate-off post-training spread is the primary persistence endpoint |

The final routing candidate controls 90% of external flow. Gate-off platform
allocation is uniform. When the gate is active, the platform reallocates that
flow to resting quotes at or below a half-spread cap of 3. The remaining 10%
uses ordinary price-time priority. This is a platform-mediated order-flow
assay, not a claim about a fully decentralized LOB.

## Fixed-Policy Economic Acceptance

Final configuration:

| Parameter | Value |
| --- | ---: |
| Agents | 10 |
| Half-spread actions | 1 through 7 ticks |
| Inventory penalty | 0.0001 |
| Base flow intensity | 2.0 |
| Low/high sigma | 0.005 / 0.08 |
| Informed-flow probability | 0.7 |
| Spread elasticity | 0.14 |
| Platform-routable fraction | 0.90 |
| Platform flow direction | Unbalanced, preserves informed side |
| Routing eligibility | Absolute half-spread cap 3 |
| Routing intensity | 1.0 |
| Quote quantity | 1 |
| Quote priority | Rotating |
| Economic validation paths | 100 |

Result:

| Acceptance quantity | Result |
| --- | ---: |
| Low-risk common-spread optimum | 4 |
| High-risk common-spread optimum | 7 |
| Gate-off unilateral undercut gain | -0.00144 |
| Gate-on unilateral undercut gain | +0.00753 |
| Gate-off best response against spread 5 | 7 |
| Gate-on best response against spread 5 | 3 |
| Best-response shift | 4 ticks downward |
| Symmetric high-spread routing subsidy | None |
| All fixed-policy checks | Pass |

Command:

```bash
python revision/experiments/validate_lob_assay_economics.py \
  --seeds 0-99 --inventory-penalty 0.0001 --base-flow-intensity 2 \
  --low-sigma 0.005 --high-sigma 0.08 --informed-flow-prob 0.7 \
  --spread-elasticity 0.14 --seed-liquidity-qty 0 \
  --agent-quote-quantity 1 --routing-mode platform_reallocate \
  --routing-routable-fraction 0.9 --no-routing-platform-balanced \
  --routing-eligibility-mode absolute_cap --routing-target-kappa 4.0 \
  --routing-rho 1.0 \
  --output-dir revision/experiments/outputs/lob_assay_econ_absolute_cap_f09_no_spill_100seed_v2
```

This is the first local-LOB configuration in this sequence that gives risk a
real economic role, avoids a common-spread subsidy, and makes intervention
change the seller's best response.

## MAPPO Assay And Seller Response

The MAPPO screen uses a shared trunk with per-agent actor heads, a shared
centralized critic, 100 training episodes, five evaluation episodes, balanced
risk transitions (`0.08/0.08`), 20 warm-start updates, CRN, and MPS.

No-intervention low-risk mean half-spreads across seeds 0-4 were:

```text
5.6951, 4.8249, 4.8435, 5.7175, 4.1409
```

All five exceeded or met the low-risk economic benchmark of 4, so the
100-episode no-intervention assay passed the observation gate.

For a frozen response policy trained with randomized intervention exposure
only in low-risk states, the same-state counterfactual `signal=0` versus
`signal=1` result was:

| Metric | Result |
| --- | ---: |
| Mean low-risk high-spread action reduction | 0.4799 half-spread tick |
| Standard error | 0.2726 |
| Positive seeds | 4/5 |
| Mean cap-compliance gain | 0.1337 |

This passes the predeclared response gate of a 0.3-tick mean reduction and at
least 70% positive seeds. The routing signal is therefore learnable. It does
not by itself establish a timing advantage.

## Timing Tests

### Same frozen policy, low-risk shuffled schedule

The oracle schedule was generated from the same response policy's gate-off
potential-outcome path. The comparator kept the same low-risk support and exact
dose but shuffled activation times.

```text
yoked - timed low-risk spread = 0.0206 +/- 0.0159
positive seeds = 3/5
```

This fails the 70% positive-seed gate. Most response policies had already
learned a low-spread gate-off attractor, leaving little online timing headroom.

### Reference-scheduled timed versus shuffled training

Timed and shuffled policies started from identical weights, used aligned
action RNG, saw identical exogenous paths, and received exactly equal
per-episode dose. Training schedules came from a no-intervention reference
policy. The primary endpoint was post-training gate-off spread.

```text
yoked - timed low-risk spread = 0.3448 +/- 0.3175
positive seeds = 3/5
```

The mean is positive but unstable; two seeds favor shuffled timing by about
0.35 tick. This does not pass the screen.

### Timed-online oracle versus shuffled training

The strongest actual-method test let the oracle gate observe the timed arm's
own pre-action state. The shuffled arm replayed a within-low-risk permutation
of the exact timed dose. Gate-off spread remained the primary endpoint.

```text
yoked - timed low-risk spread = -0.3166 +/- 0.2927
positive seeds = 2/5
```

This fails clearly. Aligning the gate to the timed policy's current state does
not produce a stable learning advantage over equal-dose low-risk shuffling.

Final screen command:

```bash
python revision/experiments/run_lob_mappo_yoked_timing_pilot.py \
  --episodes 100 --eval-episodes 5 --episode-length 100 --seeds 0-4 \
  --timing-mode oracle_controlled --training-schedule-source timed_online \
  --device mps \
  --output-dir revision/experiments/outputs/lob_mappo_yoked_timed_online_absolute_cap_f09_no_spill_100_5seed_screen_v1
```

## Interpretation By Module

| Module | Status | Evidence |
| --- | --- | --- |
| Risk-coupled LOB economics | Pass | Low/high optima separate and adverse-selection cost rises with risk |
| Platform routing channel | Pass | Gate changes best response from 7 to 3 without changing total expected demand |
| Seller ability to respond to signal | Pass as a diagnostic | Same-state counterfactual response passes 0.3-tick/70% criteria |
| No-intervention 100-episode assay | Pass on screen seeds | 5/5 low-risk means are at or above benchmark 4 |
| Low-risk high-spread timing | Fail | Frozen-policy and training comparisons do not reach 70% positive seeds |
| Learned risk-aware gate | Not run | Oracle upper bound did not pass, so learned-gate scaling is not justified |
| Full LOB/MAPPO main | No-go | Core timing claim remains unsupported |

The key scientific result is not that routing is ineffective. Routing changes
incentives and policies. The unresolved claim is narrower:

> Conditional high-spread timing has not been shown to improve on an equal-dose
> intervention already restricted to low-risk states.

This means a risk-only selective intervention remains a strong comparator. If
future work cannot beat it, the full-LOB claim must be reduced from
"collusion-risk timing matters" to "avoiding intervention in legitimate
high-risk states matters."

## Statistical Boundary

- Raw outputs are ignored and remain under `revision/experiments/outputs/`.
- The reported screens use seeds 0-4 and are not held-out confirmation.
- No held-out oracle or learned-gate sweep should be launched from these
  results because the screen gate failed.
- Older MAPPO outputs created before the non-overlapping evaluation-seed fix
  are diagnostic only.
- The sequence intentionally stopped after the predeclared actual-method
  screen failed; no further `kappa`, routing-intensity, or seed-subset search was
  used to select a positive cell.

## Next Research Step

Do not run a broader parameter grid. The next useful question is whether the
paper contribution should be narrowed or the intervention should be redesigned
to create a genuinely state-dependent marginal effect.

Two defensible options remain:

1. Treat low-risk-only randomized routing as the full-LOB method and present
   high-spread timing as a negative translation result. This is the honest
   option supported by the current evidence.
2. Design a routing contract whose marginal benefit depends on the diagnosed
   high-spread state, then rerun fixed-policy best-response curves before any
   MAPPO experiment. The contract must beat a risk-stratified equal-dose
   comparator under the gate-off persistence endpoint.

Do not calibrate or run the learned gate until an oracle timing protocol passes
on both screen and held-out seeds.
