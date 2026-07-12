# ABIDES/MAPPO Passive Assay v1

Status: passed  
Date: 2026-07-12

## Purpose

This stage asks whether no-intervention MAPPO produces a stable high-spread
failure mode in the repaired ABIDES dealer market. It does not test the learned
gate or routing efficacy.

## Protocol

| Item | Value |
| --- | ---: |
| MAPPO train seed | 900 |
| training episodes | 80 |
| steps per episode | 40 |
| held-out evaluation seeds | 200000-200009 |
| device | Apple MPS |
| policy | ten-head actor, centralized critic |
| policy advantage | relative immediate |
| routing during training/evaluation | off |
| pass rule | mean half-spread > 4.0 in at least 7/10 episodes |

The actor consumes only public state and each maker's private inventory/flow;
it does not receive the simulator risk label. The checkpoint is saved as a CPU
state dict under ignored experiment outputs so it can be loaded on CPU or MPS.

## Result

| Metric | Result |
| --- | ---: |
| held-out mean half-spread | 6.9455 |
| high-spread held-out episodes | 10 / 10 |
| held-out action counts for spreads 1-7 | 0, 4, 3, 3, 12, 153, 3825 |
| wall time | 117.4 s |
| checkpoint SHA-256 | `12b6f7fc29e00aad385bea318205581b7c73fc07903595908c0cef095a53ce93` |

Training moved from a mean half-spread of 4.47 in episodes 1-10 to 6.93 in
episodes 71-80 while mean reward increased. The high-spread regime is therefore
a learned no-intervention attractor rather than a fixed-policy fixture.

## Boundary

The assay passes the failure-mode prerequisite for G5. It does not establish
that the learned gate can infer low risk, that routing changes MAPPO outcomes,
or that learned timing beats an equal-dose comparator.
