# ABIDES Dose And Routing Sensitivity v1 Protocol

Status: frozen before calibration or outcomes  
Date: 2026-07-14

## Dose Curve

The dose study uses source policies `530000-530009` as a predeclared
secondary-sensitivity panel. Realized activation targets are `0.10`, `0.20`,
and `0.30`. Controller settings are calibrated once on passive paths
`537000-537009`, then frozen. Final dose evaluation paths are
`538000-538009`.

At each dose, train independent 96-episode adapters for `risk_aware`,
`matched_random`, and `spread_only_budget_matched`, from the same base policy
and initial adapter digest with common response paths and action uniforms.
Routing remains soft rank with `kappa=4` and routable fraction `0.50`.

Report customer cost against realized activation and mean routing TV, dealer
reward loss against customer cost, high-risk activation against customer
cost, and paired random-minus-risk CI at every dose. This package, unlike the
v5 operating-point plot, may be called a dose frontier.

## Routing Intensity

At the approximately 20% frozen dose, use five unique one-factor settings:

| `kappa` | routable fraction |
| ---: | ---: |
| 2 | 0.50 |
| 4 | 0.50 |
| 8 | 0.50 |
| 4 | 0.25 |
| 4 | 0.75 |

At minimum, separately adapt and evaluate `risk_aware` and `matched_random`
with identical paths, uniforms, budgets, and base policies. Sensitivity
results never replace the v5 setting.

The classification is `broadly_robust` when a majority of settings have the
frozen favorable mean direction and none has a significantly adverse CI;
`locally_robust` when support is confined near the center; and
`parameter_specific` when multiple settings reverse or are unstable.
Nonsignificance is not equivalence.
