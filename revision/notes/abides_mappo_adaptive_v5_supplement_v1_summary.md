# ABIDES/MAPPO Adaptive v5 Supplement Summary

Status: passed
Date: 2026-07-13

## Decision

The post-main supplement passes all 16 frozen checks on all 20 adaptive-v5
policy seeds. It closes the two most important comparator gaps left by the v5
main result:

1. risk-aware routing beats a continuous-signal circular-shift control that
   preserves dose and cyclic temporal structure but breaks risk alignment;
2. the full two-feature public-signal proxy beats volatility-only and
   lagged-return-only proxies at comparable activation rates.

The supported claim is:

> In the controlled N=10 ABIDES dealer market, a simulator-supervised proxy
> over public lagged risk signals, announced before quotes through a
> demand-conserving soft-rank routing contract, improves low-risk customer
> execution over exact-dose random and temporal-structure-preserving routing
> controls after separate MAPPO-style intervention-adapter training.

This remains a controlled mechanism result. It is not evidence of unlabeled
real-market collusion detection, a generic MARL algorithm, standard full-policy
MAPPO performance, or robustness across dealer counts and market ecologies.

## Frozen Protocol

| Item | Value |
| --- | --- |
| Source policies | Passed adaptive-v5 seeds `530000-530019` |
| New response training | 96 episodes for each of three new adapters |
| Final evaluation | Fresh paths `533000-533009` |
| New controls | circular-shifted signal, volatility-only, return-only |
| Existing controls re-evaluated | no intervention, exact-dose random, spread threshold, always-on, no routing |
| Agents and horizon | 10 dealers, 100 decisions per episode |
| Routing | pre-quote announced, demand-conserving soft rank; `kappa=4`, routable fraction `0.5` |
| Formal commit | `c8a3468cef4045fa1f1b831ede08ec49f60b3b04` |
| Protocol fingerprint | `caafc92da2a62bdd2f8fd14189d61a3444f6019372f280b530d6109203ec3c1b` |
| Docker image | `sha256:38ea9d4ddae027e040ea0386f31d2f49bb14946ddc6ae0332ad384efde981a68` |
| ABIDES commit | `f9cbe51342b7dedd9587e4e069040d68a5c6477f` |

The circular control applies a deterministic nonzero circular shift to each
episode's exact continuous risk-signal sequence. It therefore preserves the
signal multiset, dose, burst lengths under a cyclic interpretation, and local
cyclic transitions while removing alignment with the contemporaneous public
risk state. Exact structure and dose were checked during both adaptation and
evaluation.

The simple proxies were fit on the same disjoint passive calibration records
as the full risk model: 1,000 training and 1,000 validation observations. Their
pacing rates were fixed to the full model's passive validation activation, not
to final outcomes. All three models use the simulator's low-risk regime as the
training label. Runtime inference uses only public volatility and/or absolute
lagged return, but this is still simulator-supervised training.

## Main Results

| Method | Low-risk spread | Activation | High-risk activation | Mean reward | Reward loss vs no intervention |
| --- | ---: | ---: | ---: | ---: | ---: |
| no intervention | 6.53379 | 0.00% | 0.00% | 10.47111 | 0.00% |
| risk-aware | 6.27344 | 20.30% | 5.13% | 10.03070 | 4.21% |
| matched random | 6.33510 | 20.30% | 17.95% | 10.05837 | 3.94% |
| circular shifted | 6.32778 | 20.30% | 16.48% | 10.04855 | 4.04% |
| volatility only | 6.30942 | 20.70% | 5.13% | 10.08908 | 3.65% |
| return only | 6.34124 | 21.80% | 15.02% | 10.08514 | 3.69% |
| spread threshold | 6.09278 | 35.00% | 35.90% | 9.44237 | 9.82% |
| always on | 5.14798 | 100.00% | 100.00% | 7.42190 | 29.12% |
| no routing | 6.53105 | 20.30% | 5.13% | 10.46703 | 0.04% |

Positive paired differences mean that the comparator has worse low-risk
execution than risk-aware routing.

| Comparator minus risk-aware | Mean | 95% CI | Positive seeds |
| --- | ---: | ---: | ---: |
| matched random | 0.06166 | [0.05786, 0.06547] | 20/20 |
| circular shifted | 0.05434 | [0.05107, 0.05762] | 20/20 |
| volatility only | 0.03598 | [0.03269, 0.03927] | 20/20 |
| return only | 0.06781 | [0.06397, 0.07164] | 20/20 |
| no intervention | 0.26036 | [0.25357, 0.26714] | 20/20 |
| no routing | 0.25761 | [0.25106, 0.26416] | 20/20 |

The temporal result is not attributable only to total dose or generic signal
clustering: risk-aware and circular shifted both activate 20.30% of the time
with the same mean signal strength, but circular shifting raises high-risk
activation from 5.13% to 16.48% and worsens low-risk spread on every seed.

The simple-proxy result shows that the second public feature contributes. The
volatility-only proxy nearly matches both total and high-risk activation, yet
is 0.03598 tick worse on all 20 seeds. Return-only has more high-risk
activation and is 0.06781 tick worse. The full model pays a small reward cost
relative to the simpler proxies, so this remains a customer-execution/risk
control/reward trade-off rather than strict Pareto dominance.

## Economic Interpretation

- Relative to no intervention, risk-aware routing reduces low-risk execution
  half-spread by 3.98% and mean dealer reward by 4.21%.
- Relative to exact-dose random, it reduces low-risk spread by 0.97%, saves
  89.66 customer tick-units per episode, and changes aggregate dealer reward
  by -27.67 reward units per episode (-0.275% in mean dealer reward).
- Threshold and always-on reach lower raw spreads at much larger activation,
  high-risk intervention, and dealer reward cost. They are aggressive
  operating points, not failed methods.
- Mean routed volume is exactly 20 for every method by construction. This is a
  demand-conservation sanity check, not an empirical liquidity result.
- No routing returns close to no intervention, confirming that routing is the
  operative causal channel.

The paper-facing frontier is a comparison of method operating points. It does
not replace a formal within-method routing-dose sensitivity study.

## Integrity And Reproducibility

All frozen checks pass:

- 20/20 required policy seeds completed;
- all three new arms converged on all seeds;
- common exogenous paths were exact in adaptation and final evaluation;
- circular signal dose and cyclic structure were exact in both phases;
- gate-off actions exactly reproduced the frozen base policy;
- proxy activation differences stayed within the frozen 0.05 tolerance;
- every method conserved 20 units of routed demand;
- v5 source model digests matched the frozen source checkpoints.

The original run completed in 8,473.53 seconds (2 h 21 min 13.53 s) with eight
concurrent CPU/Docker workers. A completed rerun took 1.47 seconds, launched no
containers, and reproduced the same PASS from checkpoints and result files.

The first 20-worker engineering launch timed out before producing any outcome
because the emulated x86 Docker runtime was over-subscribed. That failed launch
is preserved locally under
`revision/experiments/outputs/abides_mappo_adaptive_v5_supplement_v1_failed_20workers_20260713T1020Z`.
It did not alter seeds, episodes, models, thresholds, endpoints, or pass rules.

Artifacts:

- per-seed table:
  `revision/paper/data/abides_mappo_adaptive_v5_supplement_v1_per_seed.csv`;
- activation/reward/high-risk frontier:
  `revision/paper/figures/abides_mappo_adaptive_v5_supplement_frontiers.{png,pdf}`;
- reproducibility manifest:
  `revision/paper/data/abides_mappo_adaptive_v5_supplement_v1_reproducibility.json`;
- local raw-output checksums:
  `revision/experiments/outputs/abides_mappo_adaptive_v5_supplement_v1/SHA256SUMS`.

Raw checkpoints and logs remain ignored. The compact per-seed CSV, figures,
protocol, config, code, and this summary are repo-visible.

## Implementation Audit Boundary

- The signal is computed from lagged state and inserted into actor observations
  before current actions are sampled. Current quotes affect soft-rank
  allocation after the pre-quote contract is announced.
- The latent risk field is not included in actor or centralized-critic inputs;
  it is used for simulator-supervised calibration and post hoc stratification.
- Checkpoints include model state, optimizer state, NumPy RNG state, and action
  call count.
- The learner is the repository's compact MAPPO-style implementation. The base
  actor trunk and actor heads are frozen during intervention adaptation; the
  intervention adapter and critic are optimized. It must not be described as
  standard full-policy MAPPO retraining.
- The 100-step task is treated as a finite-horizon terminal episode, and wealth
  is marked to the fundamental in the environment reward. The result should not
  be described as continuing-market truncation.

## Remaining Limits

Adaptive v5 plus this supplement is sufficient for the narrow controlled
ABIDES claim and for paper drafting. It does not close external validity. The
remaining high-value extension is one protocol-frozen environment or learner
holdout, such as a final-v5 fast-risk ecology or a standard MAPPO/IPPO
implementation. It is not necessary to rerun every N or to tune until
risk-aware obtains the lowest raw spread.
