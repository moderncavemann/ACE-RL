# ABIDES/MAPPO Adaptive v3 Learned Development Protocol

Status: frozen after oracle development passed and before learned efficacy  
Date: 2026-07-12

The announced oracle response gate passed all registered checks with `4/5`
positive policy seeds. Its block-shuffled advantage was small (`0.00628` tick,
95% CI crossing zero), so within-risk timing remains a secondary diagnostic.

Four independently trained response adapters are cloned from each converged
base policy:

| Method | Announced signal |
| --- | --- |
| `learned_full` | observable low-risk probability times lagged excess-spread severity |
| `overall_random` | exact full-signal strengths globally permuted |
| `risk_block_shuffled` | exact strengths permuted within risk and ten-step blocks |
| `risk_only` | observable low-risk probability without spread severity |

The primary claim test is `overall_random - learned_full` low-risk execution
half-spread. The risk-block comparator tests only within-risk timing. Risk-only
tests whether severity reduces unnecessary activation without materially
worsening spread.

Oracle response policies were already stable by the final three checkpoints.
The learned development budget is therefore frozen at 96 episodes, with fixed
panels at episodes 48, 72, and 96. This choice was made before any learned
timing outcome existed.

Development passes only if all integrity, convergence, volume, reward, and
risk-control checks pass and learned beats overall random on at least four of
five policy seeds with positive mean. Severity is retained only if it reduces
activation by at least `0.02` while worsening spread by no more than `0.10`
tick, or strictly improves spread without increasing activation.

The reserved holdout remains unopened. Development success permits a new
frozen holdout protocol; it is not itself confirmatory evidence.
