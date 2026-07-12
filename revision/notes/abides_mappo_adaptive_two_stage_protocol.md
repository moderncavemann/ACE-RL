# ABIDES/MAPPO Adaptive Two-Stage Pilot Protocol

Status: frozen after representation diagnosis and before v2 efficacy runs  
Date: 2026-07-12

## Allowed Structural Change

Adaptive pilot v1 showed that the original joint gate entangled two different
questions:

1. is the current market state economically low risk; and
2. is there a high-spread routing opportunity with an eligible undercutter?

The original logistic classifier used both exogenous risk proxies and current
quote-distribution features to predict their conjunction. Adaptive policies
made routing opportunities nearly ubiquitous, causing raw proposals and
high-risk activation to saturate.

The v2 gate is the only method change:

- risk inference uses only lagged public volatility and absolute lagged return;
- the risk threshold is selected on the original passive validation split with
  the risk classifier's own high-risk FPR constrained to at most `0.15`;
- current excess spread and modal undercut eligibility are applied afterward as
  a deterministic routing-opportunity condition;
- the existing 35% episode controller remains a final budget, not a substitute
  for risk calibration.

No routing, MAPPO, reward, market, training-budget, metric, or acceptance
parameter changes from adaptive pilot v1.

## Evidence Permitting v2

The representation diagnostic uses v1-trained policies only as a development
distribution and fresh environment seeds `421000-421004`. It is not an efficacy
test. On learned adaptive policies, the two-stage candidate achieved:

| Metric | Joint v1 | Two-stage v2 |
| --- | ---: | ---: |
| raw/risk proposal rate | 0.8404 | 0.3936 |
| accepted activation | 0.3500 | 0.3200 |
| risk-classifier high-risk FPR | 0.7136 | 0.1061 |
| accepted high-risk activation | 0.4515 | 0.0773 |
| precision | 0.6594 | 0.9363 |
| target recall | 0.3172 | 0.4118 |

This passes a representation guardrail only. It does not establish learned
timing efficacy after policies retrain under the v2 gate.

## Fresh Efficacy Split

| Item | Value |
| --- | ---: |
| Independent MAPPO training seeds | `440000-440004` |
| Held-out evaluation paths | `450000-450009` |
| Training episodes per method/seed | 100 |
| Decisions per episode | 100 |
| Methods | no intervention, learned, matched random, threshold, always-on |

The v1 training/evaluation seeds and all diagnostic seeds are excluded. The
runner, checkpoint contract, CRN action uniforms, within-risk exact-dose yoke,
convergence rules, and pass conditions are unchanged from
`abides_mappo_adaptive_pilot_protocol.md`.

## Decision Rule

Passing v2 requires all integrity and efficacy checks in the frozen runner,
including at least 4/5 positive training seeds, positive paired mean, aggregate
high-risk activation at most 0.15, reward/volume guardrails, and convergence.

If v2 is stable negative, adaptive timing is no-go. If only convergence fails,
training may be extended from frozen checkpoints under a separately recorded
extension protocol. Only a complete v2 pass permits implementation of the
reward-governance baselines and second-environment pilot required for the paper
main.
