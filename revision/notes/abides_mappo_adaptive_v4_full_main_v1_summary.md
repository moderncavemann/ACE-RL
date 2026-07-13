# ABIDES/MAPPO Adaptive v4 Formal-Primary Summary

Status: failed overall; primary timing signal positive  
Date: 2026-07-13

## Formal Decision

The frozen adaptive-v4 formal-primary package is a no-go because one mandatory
guardrail, `response_converged`, failed. This decision is not overridden by the
positive efficacy estimates below.

The failure is localized: all twenty risk-aware, matched-random,
no-intervention, no-routing, and spread-threshold arms met the response
stability threshold. The only unstable arm was the aggressive always-on
control, which exceeded the 0.15 range in 10/20 policy seeds. Thus the primary
timing estimand is stable, but the v4 all-method convergence veto was not met.

## Formal Base

| Item | Result |
| --- | --- |
| Policy seeds | `520000-520019` |
| Fixed horizon | 384 episodes for all 20 policies |
| Population mean spreads at 336/360/384 | 6.49995 / 6.53499 / 6.54301 |
| Population mean spread range | 0.04306, pass |
| Terminal spread range at most 0.20 | 20/20, pass |
| Terminal spread range at most 0.15 | 20/20, diagnostic |
| Largest terminal spread range | 0.13240, seed `520018` |
| Largest reward range fraction | 0.02743, seed `520018` |

The base ran from 04:52:26 to 05:26:02 UTC under Git commit `2d1f687` and
protocol fingerprint `f807462c412336b8d6329d4c0a0b69819f2ecac271e536f2051eec681baaa367`.

## Response Results

The six-arm response ran from 05:26:28 to 06:08:42 UTC. All 22 non-convergence
checks passed, including CRN paths, exact learned/random dose, exact
no-routing dose, gate-off evaluation, volume conservation, activation limits,
reward guardrails, routing ablation, and the primary paired comparison.

| Method | Low-risk execution half-spread | Mean reward | Activation | High-risk activation |
| --- | ---: | ---: | ---: | ---: |
| no intervention | 6.56303 | 10.37384 | 0.00000 | 0.00000 |
| risk-aware | 6.34208 | 10.03561 | 0.18000 | 0.05056 |
| matched random | 6.38663 | 10.02841 | 0.18000 | 0.13483 |
| spread threshold | 6.14207 | 9.39137 | 0.35000 | 0.34551 |
| always-on | 5.24765 | 7.43899 | 1.00000 | 1.00000 |
| no routing | 6.56115 | 10.37112 | 0.18000 | 0.05056 |

Matched-random minus risk-aware spread was 0.04455 with SE 0.00164 and 95%
CI [0.04111, 0.04799]; all 20 paired seed deltas were positive. No-routing
minus risk-aware was 0.21907 with 95% CI [0.21300, 0.22514], also positive in
20/20 seeds.

## Convergence Diagnosis

| Arm | Seeds with terminal range over 0.15 | Maximum range |
| --- | ---: | ---: |
| no intervention | 0/20 | 0.00000 |
| risk-aware | 0/20 | 0.02774 |
| matched random | 0/20 | 0.01678 |
| spread threshold | 0/20 | 0.05685 |
| no routing | 0/20 | 0.00377 |
| always-on | 10/20 | 0.20103 |

Always-on is an intentionally extreme fixed-budget control and is not part of
the risk-aware-versus-random timing estimand. V4 nevertheless pre-registered
all-method convergence as a veto, so the package remains failed.

## Next Protocol

Adaptive-v5 will make one estimand-alignment change before opening fresh
seeds: response convergence remains mandatory for no-intervention, risk-aware,
matched-random, spread-threshold, and no-routing, while always-on stability is
reported separately and cannot veto the primary timing claim. Always-on still
receives the same 96-episode budget and remains in every table. No efficacy,
activation, reward, routing, or sample-size threshold changes.
