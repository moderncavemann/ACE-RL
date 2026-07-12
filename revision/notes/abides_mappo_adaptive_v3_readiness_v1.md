# ABIDES/MAPPO Adaptive v3 Readiness v1

Status: method repair successful; full-scale experiment entry gate open  
Date: 2026-07-13

## Readiness Decision

Adaptive-v3 is ready to enter a frozen full-scale experiment package. The
entry decision is supported by three independent layers:

| Layer | Result |
| --- | --- |
| Fixed-policy contract economics | 10/10 seeds pass demand conservation, symmetry, undercutter gain, and customer execution checks |
| Development adaptive response | 5/5 positive; random minus risk-aware `0.04817`, 95% CI `[0.04165, 0.05469]` |
| Fresh confirmatory holdout | 5/5 positive; random minus risk-aware `0.08786`, 95% CI `[0.07852, 0.09720]` |

The old failure has therefore been repaired at the assay, contract, learner,
and statistical-comparator levels. Continuing to tune the validated two-arm
cell is prohibited.

## Meaning Of Full Scale

The next run should be the paper-facing adaptive package, not another efficacy
screen. It must add:

- at least 20 fresh policy seeds with disjoint base-convergence and market-
  evaluation panels;
- `no_intervention`, exact-dose matched random, and risk-aware routing as the
  core comparison;
- threshold and always-on governance controls;
- a no-routing ablation using the same risk-aware announcements;
- one second predeclared ABIDES market ecology as a generalization test;
- atomic per-seed checkpoints, manifest hashes, run lock, JSONL logbook, and
  paired seed-level inference.

The primary endpoint remains low-risk customer execution half-spread. The
primary contrast remains exact-dose matched random minus risk-aware. Raw spread
controls may outperform risk-aware by intervening more often; the intended
claim remains the execution/risk-control/intervention-cost trade-off.

## Frozen Components

The full-scale package may increase seeds and add comparator arms, but it must
not change the following based on future efficacy outcomes:

- passive risk artifact `revision/configs/abides_two_stage_gate_v1.json`;
- pre-action risk-only signal and paced controller;
- soft-rank routing with `kappa=4.0` and routable fraction `0.50`;
- 100 decisions per episode, 96 response episodes, and eight-episode GAE
  updates;
- learning rate `0.0003`, entropy coefficient `0.02`, reward scale `100`;
- exact-dose global permutation comparator and common-random-number design;
- convergence, reward, activation, high-risk activation, volume, sign-rate,
  and positive-CI guardrails.

Passing this readiness note permits protocol construction and launch. It is
not itself the full-scale result.
