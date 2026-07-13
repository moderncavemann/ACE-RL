# ABIDES/MAPPO Adaptive v3 Full-Scale v1 Summary

Status: overall no-go; primary base gate failed, fast-risk ecology passed  
Date: 2026-07-13

## Decision

The frozen two-ecology full-scale package does not pass as a paper-facing
adaptive main result. The primary ecology was correctly stopped before any
intervention comparison because only 18 of 20 no-intervention base policies
met the preregistered convergence rule by episode 288.

The separately preregistered fast-risk ecology completed and passed every
adaptive comparison, mechanism, governance, convergence, and integrity check.
That result is valid positive generalization evidence, but it cannot replace
the blocked primary cell.

The supported statements are:

> In the fast-risk N=10 controlled ABIDES dealer ecology, risk-aware announced
> soft-rank routing outperforms exact-dose random timing after separate MAPPO
> adaptation on all ten policy seeds, while preserving reward and demand.

> The complete adaptive-v3 full-scale package remains unconfirmed because the
> primary-ecology base-policy convergence prerequisite failed on 2 of 20
> policy seeds; no primary efficacy comparison was run.

## Frozen Provenance

| Item | Value |
| --- | --- |
| Protocol | `revision/notes/abides_mappo_adaptive_v3_full_scale_protocol.md` |
| Frozen Git commit | `b5262f4` |
| Docker image | `ace-rl-abides-ace:adaptive-v3`, `sha256:38ea9d4d...` |
| ABIDES source | `f9cbe51342b7dedd9587e4e069040d68a5c6477f` |
| Risk artifact | `revision/configs/abides_two_stage_gate_v1.json`, `sha256:e27b36f8...` |
| Primary base fingerprint | `cbbac34f1c60217d4c01ce23decaec4a0bfcf6e906a2d6c082da406c8e4b915d` |
| Fast-risk base fingerprint | `51c1f179a66643f3ec284a80b5a0f98737832583ad41ae5109abe15c7d08fef0` |
| Fast-risk response fingerprint | `b657526d4d9000d3d997ee9dcf0fb25169873e1d6ab41ecd47f03b5c64c38815` |

The protocol, seed splits, market ecologies, methods, response budget, and
acceptance rules were committed before any formal seed was opened. Seed
`490000`, consumed by a pre-freeze engineering smoke, was excluded; the formal
primary policies used untouched seeds `490020-490039`.

## Primary Ecology: Stopped At Base Gate

The 20-policy primary base run started at 10:33:54 and ended at 10:58:28 AEST,
about 24.6 minutes with 20 CPU workers. All 20 policies reproduced the
high-spread failure mode, and 18 passed convergence. Two reached the frozen
288-episode maximum without satisfying the three-panel spread-range threshold
of `0.15` tick:

| Seed | Episodes | Panel spreads at 240 / 264 / 288 | Spread range | Reward range fraction | Result |
| ---: | ---: | --- | ---: | ---: | --- |
| 490022 | 288 | 6.2620 / 6.4066 / 6.4548 | 0.1928 | 0.0401 | fail |
| 490029 | 288 | 6.3400 / 6.4416 / 6.4958 | 0.1558 | 0.0304 | fail |

Both failures are slow upward spread drift, not numerical collapse: their
reward ranges pass the 10% guardrail and their spreads remain above the 4.0
failure-mode floor. Nevertheless, the frozen rule required all 20 bases. No
seed was extended, replaced, or dropped, and the primary response output
directory was never created.

## Fast-Risk Ecology

The second ecology doubled both transition probabilities from `0.04/0.10` to
`0.08/0.20`, preserving the stationary high-risk share at 0.2857. A 1000-path
pre-run check measured high-risk shares `0.2803` and `0.2845`, while transitions
per 100-step episode increased from `5.66` to `11.39`.

All ten no-intervention base policies converged in 168-288 episodes. Base
training ran from 10:58:57 to 11:22:10 AEST, about 23.2 minutes. The six-arm
adaptive run then trained five zero-initialized response adapters per base for
96 episodes and completed from 11:22:26 to 12:01:51 AEST, about 39.4 minutes.

### Method Results

| Method | Low-risk execution half-spread | Activation | High-risk activation | Mean reward | Volume |
| --- | ---: | ---: | ---: | ---: | ---: |
| no intervention | 6.27653 | 0.0000 | 0.0000 | 10.08634 | 20.0 |
| risk-aware | 6.01706 | 0.1870 | 0.0719 | 9.63583 | 20.0 |
| exact-dose matched random | 6.05348 | 0.1870 | 0.1439 | 9.62488 | 20.0 |
| lagged-spread threshold | 5.80143 | 0.3500 | 0.3633 | 9.02556 | 20.0 |
| always-on | 4.76870 | 1.0000 | 1.0000 | 6.77269 | 20.0 |
| no routing | 6.27567 | 0.1870 | 0.0719 | 10.08526 | 20.0 |

### Paired Contrasts

Positive comparator-minus-risk-aware values favor risk-aware routing.

| Comparator minus risk-aware | Mean | SE | 95% CI | Positive / negative |
| --- | ---: | ---: | --- | ---: |
| matched random | 0.03642 | 0.00183 | [0.03228, 0.04056] | 10 / 0 |
| no intervention | 0.25947 | 0.00599 | [0.24593, 0.27302] | 10 / 0 |
| no routing | 0.25861 | 0.00475 | [0.24787, 0.26936] | 10 / 0 |
| lagged-spread threshold | -0.21562 | 0.00519 | [-0.22737, -0.20387] | 0 / 10 |
| always-on | -1.24835 | 0.03089 | [-1.31824, -1.17846] | 0 / 10 |

Risk-aware routing improves execution over matched random on every independent
policy seed at the same total activation rate. It halves high-risk activation
relative to random and has slightly higher mean reward. No-routing nearly
equals no intervention, while full risk-aware routing improves spread by
`0.25861` tick, identifying routing as the causal incentive channel.

Threshold and always-on achieve lower raw spread by intervening more often in
high-risk windows and with larger reward loss. The supported contribution is
therefore the execution/risk-control/intervention-cost trade-off, not lowest
raw spread.

## Integrity And Recovery

Every fast-risk check passes:

- 10/10 required base and response policy seeds complete;
- all six methods share exogenous paths and action uniforms;
- matched random and no-routing use the exact risk-aware signal dose;
- all response methods pass three fixed convergence panels;
- zero-signal actions exactly match each frozen base;
- all methods conserve 20 units of customer demand;
- activation, high-risk activation, reward, routing, sign-rate, and positive-CI
  guardrails pass.

All base and response checkpoints include protocol fingerprints and atomic
optimizer/RNG state. Re-running the completed fast-risk command loaded all ten
results without starting a worker and finished in 1.4 seconds.

## Claim Boundary And Next Work

Do not report this package as a completed adaptive ABIDES/MAPPO main result.
The valid evidence now consists of the five-seed fresh holdout and the ten-seed
fast-risk full-scale result; the primary 20-seed efficacy cell is missing by
design because its prerequisite failed.

No automatic continuation is allowed. A future adaptive-v4 study must choose
its base-policy stopping rule and maximum horizon on independent development
seeds before opening new formal policies. It must not extend `490022/490029`,
replace them, relax the `0.15` rule, or pool the fast-risk result into the
primary cell.

Raw outputs remain local under
`revision/experiments/outputs/abides_mappo_adaptive_v3_full_main_*`; this note
is the durable repo-visible decision.
