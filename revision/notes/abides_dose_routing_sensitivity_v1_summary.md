# ABIDES Dose And Routing Sensitivity v1 Summary

Date: 2026-07-14  
Status: formal secondary sensitivity complete

## Decision

The frozen three-point dose curve and five-cell routing-intensity study
completed all 80 policy-seed tasks. The implementation and audit checks pass.
The routing result is classified as `broadly_robust`: all five predeclared
settings have a favorable mean risk-aware-minus-random direction and none has
a significantly adverse confidence interval.

This is secondary sensitivity evidence only. It does not replace the v5
operating point, and a nonsignificant contrast is not evidence of equivalence.

## Frozen Execution

- source policy seeds: `530000-530009`
- passive calibration paths: `537000-537009`
- fresh evaluation paths: `538000-538009`
- adapter response budget: 96 episodes per policy seed and method
- evaluation paths: 10 per policy seed
- dose targets: 0.10, 0.20, 0.30
- routing settings: `(kappa, routable fraction)` equal to `(2, 0.50)`,
  `(4, 0.50)`, `(8, 0.50)`, `(4, 0.25)`, and `(4, 0.75)`
- protocol fingerprint:
  `7498d1547fdfe3057a0b5faae809d21e6e3f185bf1802ede7922da505826cccf`

The run produced 2,700 per-path rows and 270,000 compressed per-step records.
A completed rerun started zero containers and reproduced the aggregate hash
`0bd6828acf1f9b156253d7234364b7f992697d9aa6eb9b2b4e12509f9d426193`.

## Dose Curve

The paired endpoint is matched-random minus risk-aware low-risk execution
half-spread, so a positive value favors risk-aware timing.

| Target dose | Paired mean | 95% t-CI | Positive seeds | Direction |
| ---: | ---: | ---: | ---: | --- |
| 0.10 | -0.00463 | [-0.00757, -0.00169] | 1/10 | matched random favored |
| 0.20 | 0.00947 | [0.00606, 0.01287] | 10/10 | risk-aware favored |
| 0.30 | 0.00469 | [0.00001, 0.00937] | 9/10 | risk-aware favored |

Risk-aware timing also has lower low-risk execution spread than the genuinely
activation-and-routing-TV-matched spread-only baseline at all three doses:
paired spread-only-minus-risk-aware means are 0.00843, 0.02342, and 0.05995,
with 10/10 positive policy seeds at each dose.

The low-dose reversal is retained as a negative result. It rules out a claim
that risk-aware timing dominates matched random at every intervention dose.

## Routing Sensitivity

| `kappa` | Routable fraction | Random minus risk-aware mean | 95% t-CI | Positive seeds |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.50 | 0.00433 | [0.00069, 0.00798] | 7/10 |
| 4 | 0.25 | 0.00170 | [-0.00034, 0.00374] | 5/10 |
| 4 | 0.50 | 0.00947 | [0.00606, 0.01287] | 10/10 |
| 4 | 0.75 | 0.01954 | [0.01310, 0.02597] | 10/10 |
| 8 | 0.50 | 0.01043 | [0.00721, 0.01365] | 10/10 |

The weak `(4, 0.25)` cell is reported rather than reclassified or retuned. The
predeclared robustness rule still yields `broadly_robust` because the mean
direction is favorable in all five cells and no cell is significantly adverse.

## Integrity

All runner checks pass, including exact fixed response budgets, common paths,
common action uniforms, exact random-dose matching, valid routing weights,
volume conservation, and complete step records. The independent verifier
recomputes the per-seed statistics and routing classification from the
committed per-path and per-step artifacts. The final archival-commit field is
filled only after the first artifact commit, then the bundle is re-exported and
verified again.

## Claim Boundary

The supported statement is:

> At the frozen 0.20 operating region, the risk-aware timing advantage over
> matched random is broadly robust across the predeclared routing-intensity
> perturbations, and risk-aware timing outperforms a genuinely budget-matched
> spread-only controller across the three tested doses.

The results do not support universal dominance across dose: matched random is
better at the 0.10 dose. They also do not complete the separate temporary
commitment, official-learner, or ecology holdout requirements.
