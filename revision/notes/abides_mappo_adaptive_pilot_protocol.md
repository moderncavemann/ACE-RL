# ABIDES/MAPPO Adaptive Pilot Protocol

Status: frozen before adaptive efficacy runs  
Date: 2026-07-12

## Purpose

The completed ABIDES timing confirmation holds one passive MAPPO policy fixed
across intervention arms. This pilot tests the missing question: whether the
timing advantage survives when each method starts from the same policy
initialization but trains its own MAPPO policy under its deployed intervention.

This is an adaptive-routing acceptance gate, not yet the paper main experiment.
Reward-governance baselines and a second market ecology are required after this
gate passes.

## Phase A Methods

Each training seed initializes five otherwise identical MAPPO trainers:

| Method | Training intervention |
| --- | --- |
| `no_intervention` | routing never active |
| `learned_gate` | frozen observable gate with a 35% episode budget |
| `matched_random` | learned arm's exact episode dose shuffled within risk |
| `spread_only_threshold` | non-risk-adjusted spread/undercut threshold |
| `always_on` | routing active at every decision |

The learned arm runs first in each training episode. Its realized activation
vector is shuffled only within the same exogenous risk regime for the matched
arm. Thus learned and matched training receive identical total and
risk-stratified intervention dose while destroying state timing. All arms use
the same environment path and action uniforms under common random numbers.

## Frozen Pilot Scale

| Item | Value |
| --- | ---: |
| Independent MAPPO initialization seeds | 5 (`410000-410004`) |
| Training episodes per method and seed | 100 |
| Decisions per episode | 100 |
| Shared held-out evaluation paths | 10 (`420000-420009`) |
| Agents | 10 |
| Device | CPU |
| Parallel seed workers | 5 |

The independent statistical unit is the MAPPO training seed. Evaluation paths
are averaged within each trained policy and are not counted as additional
independent training replicates.

## Recovery Contract

After every complete training episode, the runner atomically saves all five
trainer model states, optimizer states, minibatch RNG states, training history,
and the completed episode index. A restart resumes only unfinished seeds from
their last complete episode. An interruption during an episode reruns that
episode from the prior atomic checkpoint.

The run manifest freezes the config, git commit, gate hash, Docker image ID,
and protocol fingerprint. Pending work fails closed if any of these change.

## Evaluation

For every trained policy seed, evaluation reports:

- deployed low-risk customer execution half-spread;
- mean seller reward and exact customer-flow volume;
- activation and high-risk activation rates;
- gate-off outcomes for learned and matched policies;
- paired learned-versus-matched effects averaged over the ten common paths;
- training curves, last-window slopes, and previous-window differences.

The primary contrast is:

`matched_random - learned_gate` low-risk customer execution half-spread.

Positive values favor the complete learned adaptive method.

## Pilot Pass Gate

Phase A passes only if all conditions hold:

1. all five independent training seeds finish without non-finite updates;
2. no-intervention produces mean low-risk half-spread above 4.0 on at least
   four of five trained-policy seeds;
3. learned activation remains in `[0.05, 0.35]` on every seed;
4. aggregate learned activation in high-risk windows is at most `0.15`;
5. learned and matched training/evaluation doses are exact within risk;
6. learned beats matched random on at least four of five training seeds and the
   paired mean is positive;
7. learned reward is no more than 10% below matched random;
8. customer-flow volume is exactly conserved;
9. the final 20-episode training diagnostics do not show a material unresolved
   trend; otherwise the same checkpoints are extended before inference.

Failure does not permit routing, threshold, or seed search. A non-converged run
is extended from checkpoint; a stable negative timing result stops the adaptive
main.

## Remaining Paper-Main Gates

Passing Phase A permits, but does not complete, the paper main. The next frozen
protocol must add fairly tuned end-to-end governance baselines using the same
MAPPO backbone and training budget:

- fixed reward penalty;
- PPO-Lagrangian or CRPO;
- action shielding;
- no-risk-adjustment and no-routing ablations.

The external-validity stage must also add a second environment, preferably an
ABIDES RMSC04 hybrid with background noise/value/momentum traders and the same
demand-conserving platform contract. The current controlled informed-flow
dealer market remains the causal mechanism environment.

Only after adaptive Phase A, governance baselines, and the second-environment
pilot pass may a 10-20 training-seed paper main be launched.
