# ABIDES/MAPPO Adaptive v3 Frozen Holdout Protocol

Status: frozen before opening holdout policy or evaluation seeds  
Date: 2026-07-12

## Purpose

This is the single confirmatory test of the simplified adaptive-v3 candidate.
It asks whether an observable risk-aware announced routing signal remains
better than an exact-dose matched-random signal after both arms adapt through
MAPPO in the controlled ABIDES dealer market.

The claim is deliberately narrow. This protocol does not test the rejected
lagged spread-severity module and does not establish a complete paper-scale
baseline suite or cross-market generalization.

## Fresh Split

| Role | Seeds |
| --- | --- |
| Base policy training | `480000-480004` |
| Base convergence panel | `482000-482004` |
| Confirmatory response evaluation | `481000-481009` |

None of these seeds appeared in adaptive-v3 development. The base policies are
trained from random initialization under no intervention for at most 288
episodes. They must independently pass the unchanged three-panel convergence
and high-spread failure-mode gate. There is no post-hoc extension if a base
seed fails by the frozen maximum.

## Frozen Comparison

Each converged base is cloned into two zero-initialized response adapters:

- `risk_aware`: paced observable low-risk probability announced before quotes;
- `matched_random`: the exact per-episode signal-strength multiset globally
  permuted, preserving dose while destroying risk contingency.

Both arms use common exogenous paths, common action uniforms, the same 96
response episodes, GAE updates every eight episodes, and the same demand-
conserving soft-rank routing contract. Fixed convergence panels occur at
episodes 48, 72, and 96. Evaluation is deterministic and does not update the
policies.

## Pass Rule

All integrity, convergence, gate-off identity, volume, reward, activation, and
high-risk activation checks must pass. In addition:

- mean `matched_random - risk_aware` low-risk execution half-spread is positive;
- at least `4/5` policy seeds have a positive paired difference;
- the paired 95% confidence interval lower bound is greater than zero.

The holdout is run once without parameter or module changes. A pass supports
moving to governance baselines and a second market ecology. A fail freezes a
no-go result for this candidate; it must not be repaired on these holdout seeds.

The JSON key `development_evaluation_seeds` is retained only for runner
compatibility; in the holdout config it contains the reserved confirmatory
evaluation panel above.
