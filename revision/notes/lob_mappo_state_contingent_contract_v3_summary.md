# Local LOB/MAPPO State-Contingent Contract v3 Summary

Date: 2026-07-11

This note records the mechanism gate for the redesigned local-LOB routing
contract. It supersedes the v2 timing no-go for experiment planning. It is an
oracle-timing pilot, not a completed learned-gate or full-main result.

## Decision

The fixed-policy, scripted-oracle, MAPPO screen, and independent held-out gates
all pass. The local LOB/MAPPO program may proceed to a pre-registered full-main
comparison with a learned risk-aware gate.

The supported claim is:

> In the redesigned risk-coupled local LOB, a demand-conserving routing
> contract with a high-spread-dependent pool gives oracle state-contingent
> timing a reproducible advantage over a risk-stratified equal-dose shuffled
> schedule.

This result does not yet establish that the learned gate beats matched random,
or that the full-main experiment passes.

## Contract

Ninety percent of external flow is platform-routable. On an announced window,
the platform mixes uniform allocation with a spread-ranked target:

```text
severity_t = clip((avg_half_spread_t - reference) / scale, 0, 1)
alpha_t = gate_t * routing_rho * severity_t
target_i,t = softmax(-kappa * half_spread_i,t)
weight_i,t = (1 - alpha_t) / N + alpha_t * target_i,t
```

The remaining flow uses the ordinary book. Total expected demand is unchanged.
When all sellers post the same spread, the target is uniform and routing is
exactly the baseline allocation. A unilateral undercut immediately increases
flow share, so learning does not require first crossing an absolute cap.

For MAPPO deployment, `reference` is the 75th percentile and `scale` is
`max(q90 - q75, 1/N)`, computed only from separate no-intervention calibration
episodes. The announced seller signal is the continuous contract strength, not
a binary gate bit.

## Learner Protocol

1. Train the ordinary multihead MAPPO actor for 100 no-intervention episodes.
2. Calibrate `q75/q90` from five disjoint no-intervention stochastic rollouts.
3. Copy and freeze the base actor.
4. Add a zero-initialized per-agent linear response adapter that is active only
   in low-risk windows with positive announced contract strength.
5. Train the adapter for 100 episodes with full low-risk coverage. Its actor
   advantage is the contract-local difference reward
   `r_i,t - mean_j(r_j,t)`; the centralized critic still learns full GAE
   returns.
6. Evaluate one frozen response policy under oracle timing and an exact-dose
   within-low-risk shuffled schedule. Both arms share risk paths and per-step
   action uniforms.

Gate-off response behavior is checked against the frozen base and must match
exactly. This prevents low-risk intervention gradients from changing the
legitimate high-risk policy.

## Fixed-Policy Acceptance

The 100-seed soft-rank contract check passes all 16 economic criteria.

| Quantity | Result |
| --- | ---: |
| Low/high-risk common-spread optimum | 4 / 7 |
| Gate-off unilateral undercut gain | -0.00144 |
| Routed unilateral undercut gain | +0.00752 |
| Routed gain increase | +0.00897 |
| Gate-off/routed best response | 7 / 4 |
| Routed best-response gain | +0.01158 |
| Normal-state severity | 0 |
| Normal-state routing gain change | 0 |
| Symmetric common-spread subsidy | None |
| Checks | 16/16 pass |

```bash
python revision/experiments/validate_lob_assay_economics.py \
  --seeds 0-99 --inventory-penalty 0.0001 \
  --low-sigma 0.005 --high-sigma 0.08 --informed-flow-prob 0.7 \
  --spread-elasticity 0.14 --agent-quote-quantity 1 \
  --no-routing-platform-balanced --routing-contract excess_spread_pool \
  --routing-eligibility-mode soft_rank --routing-routable-fraction 0.9 \
  --routing-rho 1.0 --routing-target-kappa 4.0 \
  --output-dir revision/experiments/outputs/lob_assay_excess_spread_soft_rank_contract_100seed_v3
```

## Scripted Oracle Acceptance

The fixed responder uses the minimum one-tick `5 -> 4` deviation. The material
effect threshold is scaled to 75% of the analytically reachable timing delta.

| Quantity | Result |
| --- | ---: |
| High-spread `shuffled - timed` reduction | 0.02558 +/- 0.00038 |
| Positive seeds | 100/100 |
| Normal-state spread change | 0 |
| Timed/shuffled effective alpha | 0.2500 / 0.1221 |
| Timed responder reward delta | +0.00144 |
| Checks | 6/6 pass |

```bash
python revision/experiments/validate_lob_state_contingent_routing_timing.py \
  --seeds 0-99 --routing-eligibility-mode soft_rank \
  --routing-target-kappa 4.0 --undercut-half-spread 4 \
  --output-dir revision/experiments/outputs/lob_state_contingent_soft_rank_routing_timing_100seed_v2
```

## MAPPO Results

The formal screen used previously unused local-MAPPO seeds 25-29. The held-out
confirmation used seeds 30-34 with no protocol changes.

| Metric | Screen 25-29 | Held-out 30-34 |
| --- | ---: | ---: |
| `yoked - timed` low-risk spread | 0.0527 +/- 0.0241 | 0.1257 +/- 0.0198 |
| Positive timing seeds | 5/5 | 5/5 |
| Expected counterfactual response | 0.1992 +/- 0.1147 | 0.5118 +/- 0.0903 |
| Positive response seeds | 5/5 | 5/5 |
| Timed effective alpha | 0.0429 | 0.0730 |
| Yoked effective alpha | 0.0098 | 0.0195 |
| Nominal activation rate | 0.0554 | 0.1058 |
| Joint acceptance checks | 12/12 pass | 12/12 pass |

Combined descriptive statistics, not a replacement for the independent
screen/held-out decisions:

| Metric | Combined result |
| --- | ---: |
| `yoked - timed` low-risk spread | 0.0892 +/- 0.0191 |
| Positive timing seeds | 10/10 |
| Expected counterfactual response | 0.3555 +/- 0.0863 |
| Positive response seeds | 10/10 |
| `base - timed` low-risk spread | 0.1211 +/- 0.0287 |
| Positive base comparisons | 10/10 |
| `yoked - timed` high-risk spread | 0.00024 +/- 0.00015 |
| No-intervention failure seeds | 10/10 |

Screen command:

```bash
python revision/experiments/run_lob_mappo_contract_response_pilot.py \
  --equilibrium-episodes 100 --response-episodes 100 \
  --response-train-rate 1.0 --calibration-episodes 5 \
  --calibration-quantile 0.75 --calibration-upper-quantile 0.90 \
  --eval-episodes 10 --episode-length 100 --seeds 25-29 \
  --warm-start-steps 200 --warm-start-updates 20 --device mps \
  --output-dir revision/experiments/outputs/lob_mappo_contract_local_advantage_100_5seed_screen_v1
```

The held-out command is identical except for `--seeds 30-34` and output
directory `lob_mappo_contract_local_advantage_100_5seed_heldout_v1`.

## Acceptance Boundary

The pilot now passes the prerequisites for a full-main experiment:

- the no-intervention failure mode appears in at least 70% of seeds;
- nominal timed and shuffled doses match exactly;
- calibrated activation remains between 5% and 40%;
- timed routing receives more state-contingent effective pool than shuffled;
- response capacity is material and positive in at least 70% of seeds;
- timed low-risk spread beats shuffled with at least 70% positive seeds;
- high-risk spread is preserved;
- reward does not collapse;
- gate-off adapter behavior exactly matches the base actor.

The next experiment must freeze this contract and calibration protocol. It
should compare no intervention, equal-dose shuffled/random timing, learned
risk-aware timing, oracle timing, and aggressive controls, with no-routing and
no-state-dependent-pool ablations. The learned gate, not oracle timing, remains
the main unresolved method claim.

Raw outputs remain ignored under `revision/experiments/outputs/`.
