# ABIDES/MAPPO Adaptive v3 Risk-Pair Development Summary

Status: pass; candidate frozen for one fresh holdout  
Date: 2026-07-12

The simplified risk-only candidate passed the frozen development protocol on
policy seeds `470000-470004` and evaluation seeds `472000-472009`. All five
response policies converged at 96 episodes.

| Method | Low-risk execution half-spread | Activation | High-risk activation | Mean reward | Volume |
| --- | ---: | ---: | ---: | ---: | ---: |
| Risk-aware | 5.96515 | 0.22200 | 0.05976 | 10.50221 | 20.0 |
| Exact-dose matched random | 6.01332 | 0.22200 | 0.17530 | 10.47909 | 20.0 |

The paired `random - risk-aware` spread difference was `0.04817` tick
(`SE=0.00235`, 95% CI `[0.04165, 0.05469]`) with `5/5` positive policy seeds.
Training and evaluation used identical exogenous paths and exact signal-dose
matching. Gate-off actions exactly matched the frozen base. Reward, volume,
activation, convergence, and high-risk activation guardrails all passed.

This is development evidence, not the final claim test. No parameter or module
may change before the reserved `480xxx/481xxx` holdout. The supported candidate
is observable low-risk routing; the failed lagged spread-severity module is not
part of the frozen holdout method.
