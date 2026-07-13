# ABIDES/MAPPO Adaptive v4 Formal-Primary Protocol

Status: frozen before formal-primary seeds are opened  
Date: 2026-07-13

## Purpose

Adaptive-v4 changes only the no-intervention base training protocol. It uses
the fixed-horizon rule that passed the independent `510xxx` development study.
The risk artifact, announced signal, routing contract, MAPPO response learner,
six comparison methods, endpoint, and response acceptance rules remain frozen
from the validated adaptive-v3 holdout.

## Fresh Data Split

| Role | Frozen seeds |
| --- | --- |
| Formal base policies | `520000-520019` |
| Base fixed evaluation paths | `521000-521004` |
| Response evaluation paths | `522000-522009` |

These sets are disjoint from v3 development, v3 holdout, v3 full-scale,
fast-risk ecology, and v4 base-development seeds.

## Stage A: Formal Base

All twenty no-intervention policies train for exactly 384 episodes. The base
passes only if every policy reaches the fixed horizon and retains the failure
mode, at least 18/20 terminal spread ranges are at most 0.20 tick, all terminal
reward range fractions are at most 0.10, and the range of the cross-policy
mean spread over episodes 336/360/384 is at most 0.10 tick.

Failure stops the primary experiment before any response arm is opened.

## Stage B: Six-Arm Response

If Stage A passes, each frozen base policy is copied into these arms:

1. `no_intervention`
2. `risk_aware`
3. `matched_random`
4. `spread_only_threshold`
5. `always_on`
6. `no_routing`

Each arm uses 96 response-training episodes and the same ten held-out response
paths. Matched random uses the exact learned-gate dose. Gate-off evaluation is
the primary endpoint. Common random numbers, optimizer-state equality at arm
start, volume conservation, and checkpoint/provenance checks are mandatory.

The primary timing claim passes only if matched-random minus risk-aware spread
is positive for at least 70% of policy seeds and the paired 95% confidence
interval excludes zero. Risk-aware activation must remain at most 0.35,
high-risk activation at most 0.15, reward degradation at most 10%, and the
no-routing arm must return near the no-intervention outcome. Threshold and
always-on remain aggressive trade-off controls, not targets to beat on raw
spread alone.

## Stop Rule And Claim Boundary

There is one formal primary cell. No routing grid, threshold adjustment, seed
replacement, horizon extension, or method selection is permitted after its
outcomes are opened. A failed base or response gate is a formal no-go.

Passing supports the claim that risk-aware gated routing improves the
collusion-reduction, false-positive, and intervention-cost trade-off in this
controlled ABIDES/MAPPO market. It does not establish robustness to every LOB
ecology, market size, learner, or real-market dataset.
