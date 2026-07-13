# ABIDES/MAPPO Adaptive v5 Formal-Primary Protocol

Status: frozen before v5 seeds are opened  
Date: 2026-07-13

## Motivation

Adaptive-v4 failed its all-method response-convergence veto even though every
primary and mechanism arm was stable and all twenty matched-random minus
risk-aware deltas were positive. The sole source was the deliberately extreme
always-on control, which continued reducing spread under the fixed 96-episode
budget in 10/20 seeds.

V5 makes one pre-registered estimand-alignment correction. It does not alter
the method, routing contract, gate, learner, horizon, evaluation endpoint,
baselines, effect thresholds, or guardrails.

## Fresh Split

| Role | Frozen seeds |
| --- | --- |
| Formal base policies | `530000-530019` |
| Base fixed evaluation paths | `531000-531004` |
| Response evaluation paths | `532000-532009` |

No v4 model, optimizer state, response trajectory, or evaluation path is
reused.

## Base And Response

The base protocol is identical to v4: all policies train for exactly 384
episodes and must pass the fixed population gate over episodes 336/360/384.

If the base passes, all six response arms receive exactly 96 episodes:
`no_intervention`, `risk_aware`, `matched_random`, `spread_only_threshold`,
`always_on`, and `no_routing`. All paths, doses, final evaluation procedures,
and efficacy checks remain identical to v4.

## Estimand-Aligned Convergence

The 0.15 terminal response-range veto applies to:

- `no_intervention`;
- `risk_aware`;
- `matched_random`;
- `spread_only_threshold`;
- `no_routing`.

Always-on receives the same budget and its terminal range is always reported,
but it is a fixed-budget aggressive control rather than a component of the
primary timing estimand. Its lack of asymptotic stability cannot veto a stable
risk-aware-versus-random comparison. No arm may be extended, replaced, or
removed after results are opened.

## Pass Rule

All v4 checks remain mandatory except for the explicitly narrowed convergence
method set above. In particular, the paired matched-random minus risk-aware
95% CI must exclude zero, at least 70% of seed deltas must be positive, the
routing ablation must have a positive CI, activation and reward guardrails
must pass, and all CRN/dose/volume/provenance checks must pass.

This is the only v5 primary cell. Failure is a formal no-go; no further seed,
horizon, threshold, or convergence-rule revision is permitted for this claim.
