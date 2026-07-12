# ABIDES/MAPPO Adaptive Two-Stage Extension v1 Summary

Status: failed; adaptive ABIDES/MAPPO paper main remains no-go  
Date: 2026-07-12

## Decision

The frozen checkpoint extension does not pass the adaptive timing acceptance
gate. The two-stage learned gate retains a positive mean advantage over the
exact-dose matched-random control, but that advantage is not stable across
policy seeds and its 95% confidence interval includes zero. Training also does
not satisfy the frozen convergence rule.

This result closes the automatic extension path. Do not continue the same
policies to another episode count, add seeds to the observed comparison, or
search routing/gate parameters on these outcomes. The reward-governance
baseline package, second-environment pilot, and adaptive paper-main run are not
permitted by this protocol.

## Frozen Extension

| Item | Value |
| --- | --- |
| Parent checkpoints | episode 100, protocol fingerprint `6644c7eb...80e` |
| Training seeds | `440000-440004` |
| Final training episode | 150 |
| Fresh evaluation paths | `451000-451009` |
| Methods trained separately | no intervention, learned gate, matched random, spread threshold, always-on |
| Agents | 10 controlled ABIDES market makers |
| Episode length | 100 decisions |
| Device/workers | CPU, five process workers |
| Extension runtime | about 21 minutes per worker, concurrently |
| Protocol fingerprint | `f784a622...da14` |

Every worker restored the exact episode-100 model, optimizer, minibatch RNG,
and training history before continuing. All five seeds completed, and the
atomic checkpoints remain resumable. The evaluation paths are disjoint from
the episode-100 evaluation paths.

## Integrity

All registered integrity checks pass:

- identical initial models;
- common training and evaluation paths;
- exact within-risk training and evaluation dose yoking;
- five required independent policy seeds;
- demand/volume conservation.

The failure is therefore not explained by unequal intervention dose, unequal
exogenous paths, missing workers, or lost checkpoints.

## Primary Results

Positive values of `matched random - learned` favor risk-aware timing.

| Endpoint | Mean delta | SE | 95% CI | Positive seeds | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| Intervention-on low-risk execution half-spread | 0.10224 | 0.08369 | [-0.13011, 0.33460] | 3/5 | No |
| Gate-off policy low-risk execution half-spread | 0.12911 | 0.07485 | [-0.07872, 0.33693] | 4/5 | No |

The frozen rule requires at least 4/5 positive seeds for the primary online
endpoint as well as all other checks. The online result fails that rule, and
both intervals include zero.

| Method | Low-risk half-spread | Mean reward | Activation | High-risk activation |
| --- | ---: | ---: | ---: | ---: |
| No intervention | 6.97051 | 11.77530 | 0.0000 | 0.0000 |
| Learned two-stage gate | 5.68000 | 9.46880 | 0.3124 | 0.1066 |
| Exact-dose matched random | 5.78224 | 9.50570 | 0.3124 | 0.1066 |
| Spread-only threshold | 5.83218 | 9.37236 | 0.3500 | 0.3624 |
| Always-on | 4.12178 | 5.95792 | 1.0000 | 1.0000 |

The risk-separation repair remains meaningful: the learned gate keeps
high-risk activation near 10.7%, versus 36.2% for the spread-only threshold.
It also lowers mean spread relative to no intervention without the always-on
reward loss. Those trade-off observations do not establish that learned timing
is better than matched random after adaptive retraining.

## Seed Stability

| Policy seed | Online delta | Gate-off delta | All methods converged |
| ---: | ---: | ---: | --- |
| 440000 | 0.16031 | 0.19189 | No |
| 440001 | 0.26479 | 0.31706 | Yes |
| 440002 | -0.08424 | 0.04014 | No |
| 440003 | 0.27873 | 0.20914 | Yes |
| 440004 | -0.10837 | -0.11271 | No |

Only two of five seeds pass the all-method convergence check. Always-on reward
drift remains the most frequent failure, but it is not the whole explanation:
learned and matched policies also drift on seeds 440000 and 440004, and seed
440004 shows broader reward nonstationarity. The core timing failure remains
even if always-on is ignored because the primary endpoint itself is only 3/5
positive.

## Interpretation And Claim Boundary

At episode 100, the same policy seeds produced 5/5 positive timing deltas and
a confidence interval above zero. The preregistered episode-150 check on fresh
evaluation paths falls to 3/5 and crosses zero. Because both training duration
and evaluation paths differ, the change cannot be assigned to either factor in
isolation; it does show that the earlier advantage is not stable enough for a
paper-main claim.

The supported statement is:

> Two-stage risk separation controls high-risk intervention and shows a
> positive average timing signal, but learned risk-aware timing does not
> robustly outperform an exact-dose matched-random schedule after separate
> adaptive MAPPO training in the controlled ABIDES dealer market.

The completed frozen-policy ABIDES timing experiment and the toy-scale result
remain valid within their existing claim boundaries. They must not be relabeled
as evidence for adaptive-policy superiority.

## Next Allowed Work

The scientifically valid next step is to freeze this as a negative adaptive
result. A future revival must be a new preregistered study that first resolves
MAPPO optimization stability on independent development seeds, without using
timing efficacy to choose the training horizon, and then evaluates timing on
new policy and market seeds. It is not a continuation of this experiment.

Raw outputs are intentionally ignored by Git. The durable result is this note;
the local machine retains `summary.json`, `rows.json`, the logbook, and atomic
checkpoints under
`revision/experiments/outputs/abides_mappo_adaptive_two_stage_extension_v1/`.
