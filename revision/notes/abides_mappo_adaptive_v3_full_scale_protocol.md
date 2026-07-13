# ABIDES/MAPPO Adaptive v3 Full-Scale Protocol

Status: frozen before primary or second-ecology policy seeds are opened  
Date: 2026-07-13

## Purpose

The adaptive-v3 development and fresh holdout established the two-arm
mechanism. This package tests the paper-facing claim with more independent
policy seeds, governance controls, a routing ablation, and one predeclared
market-dynamics shift. It is a confirmation package, not another parameter
search.

## Claim Map

| Claim | Method or module | Evidence |
| --- | --- | --- |
| Risk-contingent timing matters | passive risk model plus pre-action announcement | risk-aware versus exact-dose globally shuffled routing |
| Routing is the causal incentive channel | demand-conserving soft-rank routing | risk-aware versus no-routing with the same signal sequence |
| Selectivity matters | paced risk-aware signal | threshold and always-on governance controls |
| The result is not one seed or one regime duration | independent MAPPO policies and fast-risk ecology | 20-seed primary plus 10-seed secondary package |

## Frozen Environments

The primary environment retains the validated N=10 controlled ABIDES dealer
market: informed-flow probability `0.80`, low/high shock sigma `0.75/4.00`,
risk transitions `0.04/0.10`, and inventory penalty `0.002`.

The second ecology changes only temporal persistence. Both risk transition
probabilities double to `0.08/0.20`. The stationary high-risk share remains
`0.08 / (0.08 + 0.20) = 0.2857`, equal to the primary ecology. This tests
generalization to shorter regimes without changing long-run class balance,
spread actions, flow economics, routing strength, or reward scale.

| Role | Primary seeds | Fast-risk seeds |
| --- | --- | --- |
| No-intervention base policies | `490020-490039` | `500000-500009` |
| Base convergence panel | `491000-491004` | `501000-501004` |
| Final response evaluation | `492000-492009` | `502000-502009` |

All six seed sets are disjoint from adaptive-v3 development and holdout.
Seed `490000` was consumed only by the pre-freeze base-runner engineering
smoke and is excluded from every formal cell.

## Methods

| Method | Response training and contract |
| --- | --- |
| `no_intervention` | converged frozen base, zero signal and no routing |
| `risk_aware` | paced passive low-risk probability announced before quotes |
| `matched_random` | exact risk-aware signal-strength multiset globally permuted per episode |
| `spread_only_threshold` | paced lagged-spread severity, without risk adjustment |
| `always_on` | signal strength one at every decision |
| `no_routing` | exact risk-aware signal sequence visible to the adapter, routing disabled |

Every adaptive arm starts from a zero-initialized adapter on the same frozen
base trunk and trains for 96 episodes. Complete episodes are updated in batches
of eight with GAE. The response budget, optimizer, entropy, routing `kappa=4`,
routable fraction `0.50`, and 100-decision horizon are unchanged from holdout.

The threshold control is explicitly pre-action. It uses the previous quote
profile encoded in the current observation; no comparator may inspect current
actions before announcing its contract.

## Primary Endpoint And Inference

The primary endpoint is low-risk customer execution half-spread. The primary
contrast is `matched_random - risk_aware`, paired by independently trained
policy seed. Positive values favor risk-aware routing. Report the mean, SE,
t-based 95% CI, and positive/tie/negative policy-seed counts.

The routing contrast is `no_routing - risk_aware`. Governance controls are not
required to have worse raw spread: threshold and always-on may buy lower spread
with more high-risk intervention or reward loss.

## Acceptance Rules

The primary package passes only if every integrity and guardrail check passes:

- 20/20 required policies complete and all adaptive arms converge on the three
  fixed response panels;
- common exogenous paths and action uniforms hold for every method;
- matched random and no-routing receive the exact risk-aware signal dose;
- all zero-signal adapter actions exactly match their frozen base;
- at least 14/20 policy seeds favor risk-aware over matched random, the paired
  mean is positive, and the 95% CI lower bound is positive;
- at least 14/20 seeds favor risk-aware over no-routing, with a positive mean
  and positive 95% CI lower bound;
- no-intervention has execution half-spread above 4.0 on at least 14/20 seeds;
- risk-aware activation is between 5% and 35%, and mean high-risk activation is
  at most 15%;
- threshold high-risk activation exceeds risk-aware by at least 5 percentage
  points, and risk-aware activates less than always-on;
- risk-aware reward is within 10% of both matched random and no intervention;
- every method conserves the 20-unit customer-flow budget.

The fast-risk package applies the same rules with 7/10 sign thresholds. It is
reported separately and is not pooled with the primary ecology.

## Recovery And Stop Rules

Base policies checkpoint every eight episodes. Response checkpoints contain
all five adapters, optimizers, RNG states, histories, convergence panels, base
digest, and dose/path integrity flags. Global run locks prevent duplicate
launches; `manifest.json`, `run_state.json`, `LOGBOOK.md`, and
`logbook.jsonl` record provenance and progress.

An interrupted run resumes only under the same protocol fingerprint, artifact
hashes, Docker image, and Git commit. A failed formal cell is frozen as a
negative result. No routing, gate, threshold, episode, seed, or ecology change
may be selected from formal outcomes.
