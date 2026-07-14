# ABIDES Standard Learner Holdout v1 Summary

Date: 2026-07-14  
Status: formal execution complete; implementation and independent audit pass

## Decision

The official full-policy MAPPO holdout supports the risk-aware timing effect.
The official IPPO holdout has the same positive mean direction, but its
policy-seed confidence interval crosses zero. This closes the requirement to
test a standard public learner implementation, but it does not support a
claim that the effect is robust across all tested learners.

The supported learner statement is:

> Risk-aware timing improves low-risk customer execution relative to
> equal-dose matched-random timing under the tested official full-policy MAPPO
> implementation. IPPO is directionally consistent but inconclusive.

## Frozen Execution

- official source: `marlbenchmark/on-policy` at commit
  `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`, MIT license;
- selected validation cell: actor/critic learning rate `5e-4`, entropy
  coefficient `0.01`;
- formal policy seeds: `550000-550009` for each learner;
- base training: 384 episodes per seed;
- response training: 96 episodes per seed and method;
- final evaluation: 10 paired paths, 100 decisions per path;
- methods: no intervention, risk-aware, matched random, and the frozen
  Brero-style threshold buy-box ABIDES adaptation;
- device: CPU, 20 parallel workers; MPS was not used;
- formal wall time: approximately 58.5 minutes;
- protocol fingerprint:
  `5d8390d9b0b099a55db7b5c726a10c1ce1fe088ba2ce075972d87cad25e53768`.

All 20 learner/seed tasks completed. A completed rerun started zero containers
and reproduced summary SHA-256
`a7e04748d39a468e6a81223bc780c89466eeee59877363f2ea90348592ee1287`.

## Primary Results

The endpoint is policy-seed paired matched-random minus risk-aware low-risk
execution half-spread. Positive values favor risk-aware timing.

| Learner | Paired mean | 95% t-CI | Positive seeds | Exact sign p | Two-way bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| official full-policy MAPPO | 0.18182 | [0.04935, 0.31428] | 8/10 | 0.10938 | [0.07947, 0.30310] |
| official IPPO | 0.09397 | [-0.13722, 0.32516] | 7/10 | 0.34375 | [-0.10179, 0.28988] |

Selected operating-point means are:

| Learner/method | Low-risk half-spread | Activation | High-risk activation | Dealer reward / episode |
| --- | ---: | ---: | ---: | ---: |
| MAPPO no intervention | 6.92097 | 0.0000 | 0.0000 | 10177.29 |
| MAPPO matched random | 6.38189 | 0.2081 | 0.2428 | 9100.00 |
| MAPPO risk-aware | 6.20007 | 0.2199 | 0.1369 | 8962.24 |
| IPPO no intervention | 6.93360 | 0.0000 | 0.0000 | 10194.79 |
| IPPO matched random | 6.24076 | 0.2152 | 0.2604 | 8819.91 |
| IPPO risk-aware | 6.14679 | 0.2191 | 0.1379 | 8893.96 |

The MAPPO confidence interval excludes zero even though two individual policy
seeds reverse. Those reversals remain in the committed table. IPPO has three
reversals and substantially wider uncertainty, so its positive mean is not
promoted to a confirmatory success.

## External Baseline Boundary

The Brero-style arm is an ABIDES adaptation, not the original paper's learned
platform policy. It produces lower raw spread, but is not budget matched in
this holdout. Under MAPPO it activates 0.7878 of decisions with routing TV
0.2151 and dealer reward 4911.79, versus risk-aware activation 0.2199,
routing TV 0.0633, and reward 8962.24. IPPO has the same aggressive-dose
pattern. It is therefore an operating-point comparison, not evidence for or
against same-budget method superiority.

## Integrity And Claim Boundary

All 14 implementation checks pass: exact base/response budgets, pinned
official source, identical initial response models, common training and
evaluation paths/action uniforms, exact random dose, conserved volume,
complete per-path records, and step-record hashes. The combined audit bundle
recomputes t intervals, exact sign tests, the 10,000-replicate paired two-way
bootstrap, model digests, paths, and all 80,000 official-learner step records.

The result supports transfer to the tested official MAPPO learner. It does not
support universal learner robustness, because IPPO remains inconclusive. No
additional hyperparameter search or training extension follows that outcome.
