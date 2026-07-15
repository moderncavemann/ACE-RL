# Current Frozen Results

This page reports the latest retained ABIDES experiment package. Positive
contrasts mean that the comparator has worse low-risk customer execution than
risk-aware routing. Confidence intervals are policy-seed 95% t intervals unless
otherwise stated.

## Adaptive-v5 Fresh Evaluation

All methods use the same fresh 20-policy evaluation panel. Reward loss is
relative to no intervention.

| Method | Low-risk half-spread | Activation | High-risk activation | Mean dealer reward | Reward loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| no intervention | 6.53379 | 0.00% | 0.00% | 10.47111 | 0.00% |
| risk-aware | 6.27344 | 20.30% | 5.13% | 10.03070 | 4.21% |
| matched random | 6.33510 | 20.30% | 17.95% | 10.05837 | 3.94% |
| circular shifted | 6.32778 | 20.30% | 16.48% | 10.04855 | 4.04% |
| volatility only | 6.30942 | 20.70% | 5.13% | 10.08908 | 3.65% |
| return only | 6.34124 | 21.80% | 15.02% | 10.08514 | 3.69% |
| spread threshold | 6.09278 | 35.00% | 35.90% | 9.44237 | 9.82% |
| always on | 5.14798 | 100.00% | 100.00% | 7.42190 | 29.12% |
| no routing | 6.53105 | 20.30% | 5.13% | 10.46703 | 0.04% |

| Comparator minus risk-aware | Mean | 95% CI | Positive policies |
| --- | ---: | ---: | ---: |
| matched random | 0.06166 | [0.05786, 0.06547] | 20/20 |
| circular shifted | 0.05434 | [0.05107, 0.05762] | 20/20 |
| volatility only | 0.03598 | [0.03269, 0.03927] | 20/20 |
| return only | 0.06781 | [0.06397, 0.07164] | 20/20 |
| no intervention | 0.26036 | [0.25357, 0.26714] | 20/20 |
| no routing | 0.25761 | [0.25106, 0.26416] | 20/20 |

Risk-aware routing saves 89.66 low-risk customer tick-units per episode relative
to matched random. Aggregate dealer reward is 27.67 units lower per episode;
every method routes exactly 20 customer units per decision.

## Learner And Ecology Holdouts

Groups are reported separately and use the path-balanced audit estimand.

| Setting | Policies | Random minus risk-aware | 95% CI | Positive | Exact sign p | Two-way bootstrap 95% CI | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| official full-policy MAPPO | 10 | 0.18182 | [0.04935, 0.31428] | 8/10 | 0.10938 | [0.07947, 0.30310] | Supported |
| official IPPO | 10 | 0.09397 | [-0.13722, 0.32516] | 7/10 | 0.34375 | [-0.10179, 0.28988] | Inconclusive |
| E1: faster risk transitions | 10 | 0.03328 | [0.02776, 0.03880] | 10/10 | 0.00195 | [0.02238, 0.04400] | Supported |
| E2: weaker informed flow | 10 | 0.03984 | [0.03358, 0.04610] | 10/10 | 0.00195 | [0.02939, 0.05044] | Supported |

Official MAPPO is supported by both interval estimators, while its exact sign
test alone is not significant. IPPO remains inconclusive under both interval
estimators; the seed reversals are retained in the committed per-seed data.

## Dose Sensitivity

| Target dose | Random minus risk-aware [95% CI] | Positive | Spread-only minus risk-aware [95% CI] | Positive | Realized spread-only TV match |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.10 | -0.00463 [-0.00757, -0.00169] | 1/10 | 0.00843 [0.00619, 0.01067] | 10/10 | Yes; TV ratio 0.9941 |
| 0.20 | 0.00947 [0.00606, 0.01287] | 10/10 | 0.02342 [0.02023, 0.02661] | 10/10 | Yes; TV ratio 0.9338 |
| 0.30 | 0.00469 [0.00001, 0.00937] | 9/10 | 0.05995 [0.05629, 0.06361] | 10/10 | No; TV ratio 0.8859 |

The low-dose reversal is a retained negative result and rules out uniform
dominance across intervention doses. The 0.10 two-way bootstrap interval is
[-0.01081, 0.00142], so that reversal is not supported by every inference
method. The spread-only controller was passively activation/TV calibrated at
all doses but achieved the frozen realized-TV criterion only at 0.10 and 0.20.

## Routing Sensitivity

| Kappa | Routable fraction | Random minus risk-aware | 95% CI | Positive policies |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.50 | 0.00433 | [0.00069, 0.00798] | 7/10 |
| 4 | 0.25 | 0.00170 | [-0.00034, 0.00374] | 5/10 |
| 4 | 0.50 | 0.00947 | [0.00606, 0.01287] | 10/10 |
| 4 | 0.75 | 0.01954 | [0.01310, 0.02597] | 10/10 |
| 8 | 0.50 | 0.01043 | [0.00721, 0.01365] | 10/10 |

These cells support local, not uniform, robustness around the tested routing
region. The committed statistics retain the wider two-way bootstrap intervals.

## Claim Boundary And External Adaptations

The forced-undercut diagnostic contains 400 events from 20 policies. Rival
spread compression, continuation flow loss, and continuation reward loss are
not supported; high-spread recovery by the fixed horizon is 96.25%, 95% CI
[94.41%, 98.09%]. The required scientific label is therefore
`high_spread_coordination_only`, not punishment-supported collusion or learned
retaliation.

The Brero-style threshold buy-box and Banerjee-style temporary buyer commitment
are executable ABIDES adaptations. Their frozen convergence and/or budget
matching gates did not pass, so their efficacy values are descriptive only.
This package does not claim superiority over the original literature methods.

## Audit Sources

- Fresh adaptive-v5 panel:
  `revision/paper/data/abides_mappo_adaptive_v5_supplement_v1_per_seed.csv`
- Learner and ecology holdouts:
  `revision/paper/artifacts/abides_holdout_audit_bundle_v1/`
- Dose and routing sensitivity:
  `revision/paper/artifacts/abides_dose_routing_sensitivity_v1/`
- Forced-undercut diagnostic:
  `revision/paper/artifacts/abides_collusion_diagnostic_v1/`
- Brero-style adaptation:
  `revision/paper/artifacts/abides_audit_bundle_v1/`
- Banerjee-style adaptation:
  `revision/paper/artifacts/abides_temporary_commitment_benchmark_v1/`

## Supported Claim

> In the tested controlled N=10 ABIDES dealer markets, simulator-supervised
> public-risk timing with pre-quote-announced, post-quote-applied,
> demand-conserving soft-rank routing improves low-risk customer execution
> relative to matched announced-signal random timing under official full-policy
> MAPPO and in two pre-frozen compact-adapter ecology perturbations. The effect
> is local to the tested operating region and includes a retained low-dose
> reversal.

This evidence does not establish punishment-supported collusion, universal
learner robustness, literature-baseline superiority, larger-N robustness,
cross-engine validity, or real-market effectiveness.
