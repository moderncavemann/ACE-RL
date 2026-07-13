# ABIDES/MAPPO Adaptive v4 Base Development Protocol

Status: frozen before v4 development seeds are opened  
Date: 2026-07-13

## Motivation

Adaptive-v3 repaired the economic contract and passed a fresh holdout plus the
ten-policy fast-risk full-scale cell. Its primary full-scale cell was blocked
before efficacy because 18/20 no-intervention bases passed an all-seed rolling
range rule and two continued slowly drifting upward at episode 288.

That rule creates policy selection: only faster-converging bases may enter the
response comparison. Adaptive-v4 changes only the base training protocol. It
does not change the risk artifact, announced signal, routing contract, MAPPO
response learner, methods, endpoint, or efficacy acceptance rules.

## Fixed-Horizon Design

Twenty new no-intervention policies use development seeds `510000-510019` and
fixed evaluation paths `511000-511004`. Every policy trains for exactly 384
episodes. No policy stops early and no policy is excluded based on its own
learning curve.

The 384-episode horizon and population criteria are frozen before these seeds
are opened. The horizon is longer than v3's 288 maximum because the v3 formal
run established a slow-convergence tail; it is not selected from any v4
efficacy outcome. There is no intervention arm in this development study.

## Acceptance Rule

The fixed-horizon base protocol passes only if:

- all 20 policies reach episode 384 and retain the high-spread failure mode;
- at least 18/20 policies have a spread range at most `0.20` tick over the
  fixed panels at episodes 336, 360, and 384;
- 20/20 policies have a reward range fraction at most `0.10` over those panels;
- the range of the cross-policy mean spread over those three panels is at most
  `0.10` tick;
- optimizer outputs remain finite and all checkpoint/provenance checks pass.

The older per-seed `0.15` diagnostic is still reported but does not select or
drop policies. Passing this development gate permits a new, separately frozen
formal primary run with fresh policy and evaluation seeds. Failing it freezes
adaptive-v4 as no-go; the thresholds and horizon may not be changed using the
development outcomes.

## Recovery And Boundary

Each eight-episode batch atomically saves the model, optimizer, RNG, training
history, and evaluation panels. Run lock, manifest, JSONL logbook, and Git
commit checks are unchanged.

V3 seeds `490020-490039`, including `490022/490029`, are not continued or used
as v4 policies. V4 development results may determine only whether the already
frozen fixed-horizon protocol is viable. They provide no evidence that the
risk-aware intervention is effective.
