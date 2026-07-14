# ABIDES Standard Learner Holdout v1 Protocol

Status: source and evaluation design frozen; wrapper implementation pending  
Date: 2026-07-14

## Source

The preferred learner is the official `marlbenchmark/on-policy` MAPPO
repository at commit
`de66d7a4b23fac2513f56f96f73b3f5cb96695ac`, MIT license. The wrapper must be
stored as a reviewable diff and preserve the official actor, centralized
critic, PPO update, recurrent masks, and buffer semantics. The repository's
own compact trainer remains named `compact MAPPO-style intervention-adapter`.

The required first holdout is `official_full_policy_MAPPO`. IPPO is included
only if the official implementation can be configured by disabling
centralized value inputs without replacing core update code. A repository
implementation that updates its own actor is instead named
`repository_full_policy_PPO_variant` and cannot close the standard-learner
requirement.

## Frozen Split And Budget

| Role | Seeds |
| --- | --- |
| Policy seeds | `550000-550009` |
| Base evaluation | `551000-551004` |
| Response evaluation | `552000-552009` |
| Hyperparameter validation | `553000-553002` |

Each method receives 38,400 base environment decisions and 9,600 response
decisions. Methods are no intervention, risk-aware, matched random, and the
budget-matched Brero adaptation if its P0 implementation is stable. The
primary contrast is matched-random minus risk-aware low-risk execution
half-spread.

Observation/action spaces, environment paths, final paths, total decisions,
and search budget are identical across methods. The exact hyperparameter grid
and selection rule must be added to the config in a protocol-freeze commit
before validation starts. Selection may read only the three validation seeds;
all tried cells are retained. Final evaluation runs once.

Report direction, policy-seed CI, convergence, sample efficiency, wall time,
CPU/GPU utilization, and all failures. A failed or reversed result narrows the
claim to the existing intervention-adapter learner and does not trigger more
search.
