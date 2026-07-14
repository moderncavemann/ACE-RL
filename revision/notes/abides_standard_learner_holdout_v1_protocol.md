# ABIDES Standard Learner Holdout v1 Protocol

Status: source, wrapper contract, grid, and evaluation design frozen before validation
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

## Frozen Validation Grid

The base-only validation grid crosses actor/critic learning rate in
`{3e-4, 5e-4}` with entropy coefficient in `{0.01, 0.02}`. All cells use the
official recurrent policy, hidden size 64, one recurrent layer, chunk length
10, 10 PPO epochs, one mini-batch, gamma 0.99, GAE lambda 0.95, clip 0.2, and
value normalization. Each cell trains three no-intervention policies for 96
episodes using policy seeds `553000-553002` and is evaluated only on paths
`554000-554002`.

Selection maximizes mean no-intervention dealer reward subject to all updates
and metrics remaining finite. Ties prefer lower learning rate and then lower
entropy. Customer spread, intervention outcomes, final policy seeds, and final
evaluation paths cannot enter selection. All four cells are retained. The
selected values and calibration hash must be committed before formal training.

Both centralized-value MAPPO and decentralized-value IPPO use the same selected
actor hyperparameters and the same official actor, PPO trainer, recurrent
masks, and shared replay buffer implementation. They differ only in whether
the critic receives the frozen centralized observation or the local actor
observation.
