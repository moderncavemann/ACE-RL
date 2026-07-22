# ICAIF Experiment Result Inventory v2

Date: 2026-07-21

## Frozen Assessment

The scientific ABCE primary study is complete. The exact risk-only official
MAPPO/IPPO holdout is also complete and now has an independent audit bundle.
No additional formal training is required for the current controlled N=10
ABIDES claim. The paper-facing package should now be rewritten around ABCE;
adaptive-v5 remains an earlier independent replication.

## Completion Matrix

| Requirement | Evidence | Status |
| --- | --- | --- |
| A: fixed-dose state alignment | exact random, circular, and periodic controls on one 20-policy common panel | Complete; all 20/20 positive |
| B: current pre-quote revelation | one-step-lagged revelation control | Complete; 20/20 positive |
| C1: quote-sensitive allocation | identity-permuted weight multiset | Complete; 20/20 positive |
| C2: second allocation family | hard-best aligned and exact random | Complete; 20/20 positive |
| Primary audit | 1,800 paths, 180,000 steps, checks, hashes, strict and bundle verifiers | Complete |
| Exact official learner holdout | 800 paths, 80,000 steps, MAPPO/IPPO statistics and verifier | Complete |
| Paper-facing assets | two main tables, framework, representative trace, learner seed supplement | Complete in LaTeX snippets |
| Anonymous release migration | ABCE-first results/status/claim plus LFS instructions | Complete in release template |
| Frozen mechanism profile | 100,000-call module timings, parameters, RAM, trace storage, proxy export, verifier | Complete; non-outcome supplement |
| Same-panel untreated reference | optional frozen-policy descriptive run | Not run; nonblocking; no-routing is labeled as a mechanism reference |

## Primary ABCE Contrasts

The estimand is comparator minus target low-risk execution half-spread. A
positive value favors the target.

| Claim | Contrast | Mean | 95% policy-seed t-CI | Two-way bootstrap 95% CI | Positive |
| --- | --- | ---: | ---: | ---: | ---: |
| A | exact random minus aligned | 0.0399567 | [0.0363245, 0.0435890] | [0.0328650, 0.0471069] | 20/20 |
| A | circular minus aligned | 0.0385498 | [0.0355733, 0.0415263] | [0.0262077, 0.0506275] | 20/20 |
| A | periodic minus aligned | 0.0600150 | [0.0568603, 0.0631696] | [0.0497439, 0.0706538] | 20/20 |
| B | lagged minus current | 0.0338907 | [0.0307272, 0.0370541] | [0.0301741, 0.0374948] | 20/20 |
| C1 | identity-permuted minus quote-aligned | 0.2242591 | [0.2181268, 0.2303913] | [0.2157784, 0.2329250] | 20/20 |
| C2 | hard-best random minus aligned | 0.0407691 | [0.0371179, 0.0444204] | [0.0337658, 0.0478613] | 20/20 |

All exact sign tests have two-sided p=0.0000019073. Prespecified B/C Holm
adjustments reject. The policy seed is the primary inference unit.

## Primary Arm Summary

| Axis | Arm | Low-risk spread | Announced activation | High-risk announced activation | Mean TV | Aggregate reward | Inventory SD | Max inventory | Weight-quote rank correlation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | aligned soft-rank | 6.307137 | 0.201000 | 0.094174 | 0.062217 | 9378.235940 | 8.897142 | 39.730 | 0.893259 |
| A | exact random soft-rank | 6.347094 | 0.201000 | 0.194409 | 0.062935 | 9339.188000 | 9.422313 | 40.805 | 0.897902 |
| A | circular shifted | 6.345687 | 0.201000 | 0.215359 | 0.062429 | 9323.501440 | 9.715450 | 40.865 | 0.892449 |
| A | periodic exact multiset | 6.367152 | 0.201000 | 0.216122 | 0.062613 | 9349.004000 | 9.330288 | 41.590 | 0.895775 |
| B | lagged revelation | 6.341028 | 0.201000 | 0.094174 | 0.060154 | 9434.241920 | 8.878872 | 39.795 | 0.914892 |
| C1 | identity permuted | 6.531396 | 0.201000 | 0.094174 | 0.060306 | 9728.213020 | 8.901575 | 39.605 | -0.027767 |
| C2 | hard-best aligned | 6.303365 | 0.201000 | 0.094174 | 0.064544 | 9374.230800 | 8.855230 | 39.430 | 0.888119 |
| C2 | hard-best exact random | 6.344135 | 0.201000 | 0.194409 | 0.065145 | 9334.279820 | 9.401042 | 40.940 | 0.893148 |
| reference | no routing | 6.524371 | 0.201000 | 0.094174 | 0.000000 | 9727.157200 | 8.666710 | 39.700 | 0.000000 |

No routing is a same-panel mechanism reference. It still announces the signal
and must not be relabeled as no intervention.

## Economic Effect Sizes And Selectivity

Relative to exact-multiset random soft-rank, aligned routing improves the
preregistered low-risk execution endpoint by 0.0399567 ticks, or 0.6295% of
the random-arm level. This corresponds to 47.92 fewer low-risk customer
tick-units per episode. High-risk announced exposure falls by 51.56%, aggregate
dealer reward is 39.05 units higher, and inventory SD is 5.57% lower.

These component effects do not imply all-state or total-welfare dominance.
Aligned all-state execution half-spread is 6.35105 versus 6.34198 for exact
random, a 0.00907-tick deterioration. The result is selective: state alignment
improves the prespecified low-risk endpoint while limiting high-risk exposure.

## Public-Risk Proxy Evidence

The final two-feature proxy was fit and evaluated on disjoint passive records:
1,000 training and 1,000 validation observations. Its simulator-supervised
low-risk AUC is 0.8284. Volatility-only and absolute-lagged-return-only AUCs are
0.7573 and 0.7976. Runtime M1 reads only `public_volatility` and
`absolute_lagged_return`; latent simulator risk is used only for offline labels
and post-hoc stratification.

Spread-only is retained separately as an outcome comparator, not called a
risk classifier. At target doses 0.10 and 0.20 its realized activation/TV
budget passes the frozen matching tolerances and its low-risk spread is worse
than risk-aware by 0.00843 and 0.02342 ticks. At dose 0.30 its realized TV
ratio is 0.8859 and fails the lower matching bound, so that cell is not called
strictly matched.

The compact export and 2,000 compressed source rows are in
`revision/paper/artifacts/abides_abce_mechanism_profile_v1/`.

## Frozen Efficiency Profile

The non-outcome microbenchmark uses the formal `linux/amd64` ABCE Docker image,
one CPU/thread, fixed `N=10`, 10,000 warm-up calls, and 100,000 measured calls
per module. On the recorded Apple Silicon host under Docker architecture
emulation, median / p95 / p99 latencies are:

| Control path | Median us | p95 us | p99 us |
| --- | ---: | ---: | ---: |
| M1 logistic core | 5.21 | 5.98 | 7.06 |
| M1 public state to proposal | 149.35 | 159.80 | 173.91 |
| M2 threshold and pacing | 9.39 | 10.53 | 11.31 |
| M3 soft-rank allocation | 56.09 | 58.63 | 63.87 |
| M3 hard-best allocation | 41.09 | 43.79 | 46.89 |
| Contract end-to-end | 230.45 | 244.77 | 280.50 |
| Announced no-routing end-to-end | 161.28 | 176.69 | 207.55 |

The routing increment over announced no-routing is 69.18 us at the median
(1.43x). Against the minimal 1.56-us uniform no-contract allocation branch,
the full contract adds 228.89 us (147.49x); this large ratio reflects the
near-empty reference and should be reported with the absolute time. These are
control-path microbenchmarks, not complete ABIDES exchange-step or production
latencies.

The compact response architecture has 14,919 base actor-critic parameters, a
560-parameter actor adapter, and 6,257 response-trainable parameters when the
critic is included. Peak profiler process RSS is 67.97 MB; peak container
cgroup memory is 185.44 MB. The primary compressed trace uses 125,216,735 bytes
(695.65 bytes/step), the exact-learner trace uses 12,125,886 bytes (151.57
bytes/step), and the two managed artifacts total 139,963,734 bytes.

This supports transparent cost accounting only. The paper must not upgrade it
to "negligible overhead" or "deployment efficient."

## Exact Risk-Only Learner Holdout

The frozen learner configuration uses official MAPPO and IPPO, 10 policy seeds
per learner, 384 base episodes, 96 response episodes, 10 evaluation paths, 100
steps per path, and `include_lagged_spread_severity=false`.

| Learner | Mean random minus risk-aware | 95% t-CI | Positive | Sign p | Two-way bootstrap 95% CI | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| official full-policy MAPPO | 0.1138583 | [0.0335149, 0.1942018] | 9/10 | 0.0214844 | [0.0490896, 0.1854086] | Supported |
| official IPPO | 0.1235244 | [-0.0165793, 0.2636281] | 8/10 | 0.1093750 | [0.0119015, 0.2476144] | Inconclusive under primary t-CI |

The Brero-style arm inside this holdout is unmatched and descriptive only.

## Earlier Evidence And Boundaries

- Adaptive-v5 fresh-panel random minus risk-aware is 0.06166
  [0.05786, 0.06547], 20/20 positive. It is retained as an earlier independent
  replication, not the headline effect and not pooled with ABCE.
- Dose 0.10 reverses under the policy-seed t interval. This prevents a uniform
  dominance claim across doses.
- The forced-undercut diagnostic supports `high_spread_coordination_only`; it
  does not support punishment or learned retaliation.
- Brero-style and Banerjee-style ABIDES adaptations remain descriptive because
  their frozen admissibility gates failed. No superiority over the original
  literature mechanisms is claimed.

## Audit Inventory

### Primary ABCE

- artifact: `revision/paper/artifacts/abides_abce_primary_followup_v1/`
- 1,800 per-path rows;
- 20 policy populations x 9 arms x 10 paths;
- 180,000 compressed per-step rows;
- per-policy rows, statistics, checks, model/path digests, manifest, SHA sums;
- deterministic paper tables and preregistered representative trace;
- strict-raw and bundle-only verifier.

### Exact learners

- artifact:
  `revision/paper/artifacts/abides_abce_exact_risk_only_learner_holdout_v1/`
- 800 per-path rows;
- 2 learners x 10 policy seeds x 4 methods x 10 paths;
- 80,000 compressed per-step rows;
- per-policy rows, statistics, checks, model/path digests, manifest, SHA sums;
- strict-raw and bundle-only verifier.

## Paper Asset Map

| Main-text item | File | Purpose |
| --- | --- | --- |
| Figure 1 | `revision/paper/latex/icaif26_assets/figure_framework.tex` | method and information order |
| Table 1 | `revision/paper/latex/icaif26_assets/table_main_results.tex` | A/B/C arm outcomes and confirmatory contrasts |
| Table 2 | `revision/paper/latex/icaif26_assets/table_transfer_results.tex` | exact MAPPO/IPPO transfer and earlier replication |
| Figure 2 | `revision/paper/latex/icaif26_assets/figure_abce_trace.tex` | preregistered representative path |
| Supplement | `figure_paired_seeds.tex`, `figure_sensitivity.tex`, `table_behavioral_boundary.tex` | learner heterogeneity, operating boundary, behavioral claim |
| Supplement | `table_efficiency_profile.tex` | frozen control-path latency, parameters, and resource boundary |

The manuscript rewrite contract and replacement prose are in
`revision/paper/abce_manuscript_rewrite_v1.md`.

## Final Claim Boundary

Supported: controlled N=10 ABIDES, exact fixed-multiset state alignment,
timely current revelation, quote-sensitive allocation, transfer across the two
tested allocation families, and positive exact risk-only official MAPPO mean
effect under the frozen primary rule.

Not supported: all-state spread dominance, total welfare, universal
transparency, allocation-mechanism independence, learner independence,
larger-N robustness, second-engine validity, real-market effectiveness, or
superiority over the original Brero/Banerjee mechanisms. The profile does not
support native deployment latency or negligible-overhead claims.
