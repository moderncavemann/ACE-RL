# ABIDES/MAPPO Adaptive v5 Full-Main Summary

Status: passed  
Date: 2026-07-13

## Decision

The fresh adaptive-v5 formal-primary package passes all 23 pre-registered
checks. It is the completed adaptive full-main result for the controlled N=10
ABIDES dealer market.

The supported claim is:

> In the controlled N=10 ABIDES dealer market, a simulator-supervised proxy
> over public lagged risk signals used to announce demand-conserving soft-rank
> routing before quotes improves low-risk customer execution quality over
> exact-dose random routing after separate MAPPO-style intervention-adapter
> training, while conserving volume and limiting high-risk intervention.

This is a timing and intervention trade-off result. Risk-aware routing does
not achieve the lowest raw execution spread: threshold and always-on controls
intervene more aggressively and reduce spread further, with substantially
worse high-risk intervention and/or reward.

## Frozen Protocol

| Item | Value |
| --- | --- |
| Base policy seeds | `530000-530019` |
| Base evaluation paths | `531000-531004` |
| Response evaluation paths | `532000-532009` |
| Base training | 384 episodes for every policy |
| Response training | 96 episodes per arm |
| Agents and horizon | 10 market makers, 100 decisions per episode |
| Methods | no intervention, risk-aware, exact-dose random, spread threshold, always-on, no routing |
| Routing | announced, demand-conserving soft rank; `kappa=4`, routable fraction `0.5` |
| Risk model | passive logistic model over public volatility and absolute lagged return |
| Docker image | `sha256:38ea9d4ddae027e040ea0386f31d2f49bb14946ddc6ae0332ad384efde981a68` |
| ABIDES commit | `f9cbe51342b7dedd9587e4e069040d68a5c6477f` |
| Git commit | `2a8b7c453ffdb919f1dcce7141160ab5fa989568` |
| Base fingerprint | `8a409007a978ab74a7a9ddb94757e2c7b681ac57e2e0d3449c684b1b779afb51` |
| Response fingerprint | `8636827cb45cdb774a1351183858aaa0edc058b90f3d4a269c51c74ca467674c` |

V5 was committed before any `530xxx` output existed. It differs from the
recorded failed v4 package only in fresh seeds/namespaces and an explicitly
estimand-aligned response-convergence method set. No method, response budget,
effect threshold, activation guardrail, reward guardrail, or endpoint changed.

## Fixed-Horizon Base Gate

All twenty fresh no-intervention policies trained for exactly 384 episodes and
retained the high-spread failure mode. No policy was stopped or selected by an
intermediate learning curve.

| Frozen base check | Result |
| --- | --- |
| Reached episode 384 | 20/20, pass |
| Failure mode retained | 20/20, pass |
| Terminal spread range at most 0.20 | 20/20, pass |
| Terminal spread range at most 0.15 | 20/20, diagnostic |
| Terminal reward range fraction at most 0.10 | 20/20, pass |
| Population mean spread range at most 0.10 | 0.03924, pass |

Population mean spreads at episodes 336/360/384 were 6.50577, 6.52454, and
6.54501. The largest per-policy spread range was 0.10900 and the largest
reward range fraction was 0.02263, both for seed `530008`.

## Full-Main Results

| Method | Low-risk execution half-spread | Mean reward | Activation | High-risk activation | Mean volume |
| --- | ---: | ---: | ---: | ---: | ---: |
| no intervention | 6.53908 | 11.20118 | 0.00000 | 0.00000 | 20.0 |
| risk-aware | 6.29318 | 10.81131 | 0.20400 | 0.04984 | 20.0 |
| matched random | 6.37741 | 10.84218 | 0.20400 | 0.19938 | 20.0 |
| spread threshold | 6.09680 | 10.25004 | 0.35000 | 0.34268 | 20.0 |
| always-on | 5.25223 | 8.39000 | 1.00000 | 1.00000 | 20.0 |
| no routing | 6.53613 | 11.19676 | 0.20400 | 0.04984 | 20.0 |

The primary paired `matched random - risk-aware` spread difference is 0.08423
tick (`SE=0.00166`, 95% CI `[0.08075, 0.08771]`). All 20 policy-seed
differences are positive. Relative to matched random, risk-aware routing uses
the same 20.4% total activation but reduces high-risk activation from 19.94%
to 4.98%.

Risk-aware spread is 0.24590 below no intervention, a 3.76% reduction, while
reward is 3.48% below no intervention. Its reward is only 0.03087 below matched
random. The routing ablation is decisive: `no routing - risk-aware` is 0.24295
tick with 95% CI `[0.23651, 0.24938]`, positive in 20/20 seeds, while no
routing returns close to the no-intervention outcome.

Threshold reduces spread more than risk-aware but raises high-risk activation
to 34.27% and lowers reward by 5.48% relative to risk-aware. Always-on obtains
the lowest raw spread but activates in every state and loses 25.10% reward
relative to no intervention.

## Convergence And Integrity

All five pre-registered estimand-relevant arms pass the 0.15 terminal range
check on 20/20 policy seeds:

| Arm | Seeds over 0.15 | Maximum range |
| --- | ---: | ---: |
| no intervention | 0/20 | 0.00000 |
| risk-aware | 0/20 | 0.03019 |
| matched random | 0/20 | 0.01085 |
| spread threshold | 0/20 | 0.07877 |
| no routing | 0/20 | 0.00849 |
| always-on, reported control | 8/20 | 0.20896 |

The always-on fixed-budget control remains non-asymptotic on eight seeds. This
does not enter the v5 convergence veto and must remain visible whenever its
endpoint is reported.

Every other integrity condition passes:

- training and final-evaluation exogenous paths are identical across arms;
- risk-aware and random intervention doses are exact;
- risk-aware and no-routing doses are exact;
- gate-off actions exactly match the frozen base policy;
- every arm conserves mean customer volume at 20 units;
- all 20 base digests are distinct, and every trained arm has 20 distinct
  final model digests;
- every result has one protocol fingerprint and the frozen Docker digest.

The response rerun after completion took about 13 ms, launched no Docker
workers, and reproduced the same pass, confirming exact checkpoint/resume
behavior. The original base run took 33.8 minutes. The response wall time was
87.3 minutes because two foreground macOS Game Mode sessions throttled the
x86 Docker VM; this changed runtime, not protocol or results.

## Relation To V4

V4 remains a formal failure. It passed the primary paired effect and every
non-convergence check, but an all-six-arm convergence veto failed solely due
to always-on drift in 10/20 seeds. V5 did not reinterpret that result. It
froze an estimand-aligned convergence rule and repeated the full package with
fresh policies and evaluation paths. The fresh confirmation passes.

## Claim Boundary

This experiment validates risk-aware adaptive routing in one controlled
ABIDES matching-engine ecology, with supporting fast-risk evidence from v3.
It does not establish real-market effectiveness, generic latent-collusion
detection, robustness to other matching engines, larger dealer populations,
or arbitrary MARL learners. The passive risk model uses observable simulator
market signals; it is not evidence that collusion can be identified from real
order-book data without labels.

Raw outputs remain ignored under
`revision/experiments/outputs/abides_mappo_adaptive_v5_*`. The frozen configs,
protocol, v4 failure summary, and this final summary are repo-visible.

## Post-Main Controls

The frozen v5 supplement subsequently passed on all 20 source policies. A
continuous-signal circular-shift control, volatility-only proxy, and
return-only proxy were each weaker than the full risk-aware method in 20/20
paired seeds. The supplement also adds customer-cost, dealer-reward, inventory,
flow-concentration, and frontier reporting. See
`revision/notes/abides_mappo_adaptive_v5_supplement_v1_summary.md`.
