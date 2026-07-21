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
superiority over the original Brero/Banerjee mechanisms.
