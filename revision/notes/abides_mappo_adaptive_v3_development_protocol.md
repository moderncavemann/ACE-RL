# ABIDES/MAPPO Adaptive v3 Development Protocol

Status: frozen before adaptive-v3 timing efficacy runs  
Date: 2026-07-12

## Purpose

Adaptive v2 is retired as a negative diagnostic. Its policy seeds
`440000-440004` and evaluation paths `450000-451009` may not be reused for
selection or inference. Adaptive v3 repairs four identified structural issues:

1. interventions are announced before quotes and depend only on lagged public
   state;
2. demand is continuously reallocated by soft spread rank rather than a hard
   already-existing undercutter condition;
3. a paced token controller replaces the first-come episode cap;
4. all response arms clone a common converged no-intervention MAPPO base.

## Claim Decomposition

The primary full-method comparator is an overall rate-and-strength matched
random schedule. It tests the complete value of risk-aware allocation. A
risk-stratified block shuffle is a harder secondary mechanism comparator: it
removes risk allocation and tests only within-risk timing. A risk-only gate is
the ablation for lagged spread severity.

Failure against the risk-stratified comparator does not invalidate risk-aware
allocation, but it forbids a within-risk opportunity-timing claim. Failure to
beat risk-only removes the spread-severity module from the final method.

## Frozen Splits

| Role | Seeds |
| --- | --- |
| Fixed-policy economics | `470100-470109` |
| Base-policy development | `470000-470004` |
| Fixed convergence panel | `471000-471004` |
| Development timing evaluation | `472000-472009` |
| Reserved policy holdout | `480000-480004` |
| Reserved market-path holdout | `481000-481009` |

The holdout split cannot be opened until the development implementation,
method set, metrics, and acceptance thresholds are frozen in a separate
holdout config.

## Allowed Development

MAPPO optimization changes may be selected only from no-intervention fixed-
panel stability, entropy, KL, value fit, and failure-mode formation. Timing
outcomes cannot select learning rate, episode count, batch size, or network
shape. The frozen starting candidate uses GAE, eight complete episodes per PPO
update, learning rate `3e-4`, entropy coefficient `0.02`, reward scale `100`,
four update epochs, and 512-sample minibatches. Reward scale was selected from
the first no-intervention smoke because scale `10` produced value losses of
`48-93` and near-zero explained variance; no timing outcome was observed.

The base is evaluated every 24 episodes on the same five fixed paths. It is
accepted only after three consecutive evaluations have spread range at most
`0.15`, reward range at most 10%, and mean half-spread above `4.0`. Training is
capped at 192 episodes.

## Mechanism Gates

1. Fixed-policy soft-rank economics must conserve all 20 customer orders,
   produce no systematic symmetric-state subsidy, and give a unilateral
   undercutter positive marginal reward and better customer execution on all
   ten economics seeds.
2. An announced oracle response must beat its block-shuffled control on at
   least four of five development policy seeds before a learned gate is tested.
3. Learned timing must beat overall matched random on at least four of five
   development policy seeds with positive paired mean, high-risk activation at
   most `0.15`, and no more than 10% reward degradation.
4. The learned-versus-risk-only result decides whether the spread-severity
   stage remains in the method.

Only after all applicable development gates pass may the reserved holdout be
run once. A passing holdout permits governance baselines and the second market
ecology; it is still not the 10-20-seed paper main.
