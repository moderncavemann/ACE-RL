# ICAIF Experiment Result Inventory v1

Date: 2026-07-15

## Assessment

All numerical results proposed for the two main-text tables and the
sensitivity figure can be recovered from the current experiment outputs. The
main numerical claims were spot-checked against committed per-seed CSVs or
sealed statistics JSON files.

The package is not yet fully paper-ready in three respects:

1. the method/information-timing figure has not been rendered;
2. the two sensitivity PNGs have not been assembled into one two-panel figure;
3. adaptive-v5 main/supplement primitive per-path and per-step records remain
   local/ignored, while only the 20-row supplement per-seed table is committed.
   That committed table contains spread, reward, activation, and high-risk
   activation, but not the full customer-cost, inventory, flow-concentration,
   reward-component, and targeted-event metrics needed for the proposed S1.

The last item does not invalidate the reported v5 result, but it means that
the committed holdout audit bundle is more independently reproducible than
the main v5 table.

## Availability Matrix

| Proposed paper item | Numerical data | Paper-ready asset | Audit status |
| --- | --- | --- | --- |
| Table 1: adaptive-v5 fresh panel | Yes | Table must still be typeset | Four core metrics in a 20-row committed per-seed CSV; full economic metrics and primitive rows remain local |
| Table 2: official learners and ecologies | Yes | Table must still be typeset | Sealed per-path, per-seed, and per-step bundle |
| Figure 1: method and information timing | Method semantics/configs exist | Missing | Not applicable as a result artifact |
| Figure 2A: dose sensitivity | Yes | Existing standalone PNG; combined panel missing | Sealed 2,700-row per-path and 270,000-step bundle |
| Figure 2B: routing sensitivity | Yes | Existing standalone PNG; combined panel missing | Same sealed sensitivity bundle |
| Forced-undercut boundary | Yes | Inline paragraph is sufficient | 400 events, 20 policy seeds, 13,200 trajectory rows |
| Brero/Banerjee adaptations | Yes, descriptive only | Supplement tables needed | Separate sealed bundles; formal integrity gates failed |
| Reproducibility paragraph | Yes | Text not yet integrated into paper | Holdout bundle has 2,000 paths, 40 seeds, 200,000 steps |

## Adaptive v5 Fresh Evaluation Panel

All methods below use the same fresh supplement evaluation panel. Reward loss
is relative to no intervention on this panel.

| Method | Low-risk half-spread | Activation | High-risk activation | Mean dealer reward | Reward loss |
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

Positive contrasts mean that the comparator has worse low-risk execution than
risk-aware routing.

| Comparator minus risk-aware | Mean | Policy-seed 95% t-CI | Positive seeds |
| --- | ---: | ---: | ---: |
| matched random | 0.06166 | [0.05786, 0.06547] | 20/20 |
| circular shifted | 0.05434 | [0.05107, 0.05762] | 20/20 |
| volatility only | 0.03598 | [0.03269, 0.03927] | 20/20 |
| return only | 0.06781 | [0.06397, 0.07164] | 20/20 |
| no intervention | 0.26036 | [0.25357, 0.26714] | 20/20 |
| no routing | 0.25761 | [0.25106, 0.26416] | 20/20 |

The original v5 primary panel gives matched-random minus risk-aware equal to
0.08423, 95% CI [0.08075, 0.08771], with 20/20 positive policy seeds. The
fresh supplement panel is the clean common-panel source for the main table.

Economic interpretation on the fresh panel:

- low-risk customer tick-units saved relative to matched random: 89.66 per
  episode;
- aggregate dealer reward change relative to matched random: -27.67 units per
  episode;
- every method routes exactly 20 customer units per decision.

## Learner And Ecology Holdouts

The ecology values below are the path-balanced audit estimand. Groups are not
pooled.

| Setting | Policies | Random minus risk-aware | 95% t-CI | Positive | Exact sign p | Two-way bootstrap 95% CI | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| official full-policy MAPPO | 10 | 0.18182 | [0.04935, 0.31428] | 8/10 | 0.10938 | [0.07947, 0.30310] | Supported |
| official IPPO | 10 | 0.09397 | [-0.13722, 0.32516] | 7/10 | 0.34375 | [-0.10179, 0.28988] | Inconclusive |
| E1: faster risk transitions | 10 | 0.03328 | [0.02776, 0.03880] | 10/10 | 0.00195 | [0.02238, 0.04400] | Supported |
| E2: weaker informed flow | 10 | 0.03984 | [0.03358, 0.04610] | 10/10 | 0.00195 | [0.02939, 0.05044] | Supported |

The original compact-runner pooled-step ecology estimates are 0.03419 for E1
and 0.03471 for E2. They agree in direction but are a different estimand.

## Dose Sensitivity

| Target dose | Random minus risk-aware [95% t-CI] | Positive | Spread-only minus risk-aware [95% t-CI] | Positive | Formal realized spread-only TV match |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.10 | -0.00463 [-0.00757, -0.00169] | 1/10 | 0.00843 [0.00619, 0.01067] | 10/10 | Yes; TV ratio 0.9941 |
| 0.20 | 0.00947 [0.00606, 0.01287] | 10/10 | 0.02342 [0.02023, 0.02661] | 10/10 | Yes; TV ratio 0.9338 |
| 0.30 | 0.00469 [0.00001, 0.00937] | 9/10 | 0.05995 [0.05629, 0.06361] | 10/10 | No; TV ratio 0.8859 |

The 0.30 spread-only controller passed passive calibration matching (calibration
TV ratio 0.9932), but its realized evaluation TV ratio fell below the frozen
0.90 lower bound. Therefore the paper should say that spread-only was
passively activation/TV calibrated at all three doses and realized-matched at
0.10 and 0.20. It should not call all three realized cells strictly matched.

The 0.10 policy-seed t-interval reversal is a retained negative result and
rules out uniform dominance across intervention doses. Its two-way bootstrap
interval is [-0.01081, 0.00142], so the reversal is not supported under every
inference method. The 0.20 and 0.30 two-way bootstrap intervals also touch or
cross zero even though their policy-seed t-intervals are positive.

## Routing Sensitivity

| Kappa | Routable fraction | Random minus risk-aware | 95% t-CI | Positive seeds |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.50 | 0.00433 | [0.00069, 0.00798] | 7/10 |
| 4 | 0.25 | 0.00170 | [-0.00034, 0.00374] | 5/10 |
| 4 | 0.50 | 0.00947 | [0.00606, 0.01287] | 10/10 |
| 4 | 0.75 | 0.01954 | [0.01310, 0.02597] | 10/10 |
| 8 | 0.50 | 0.01043 | [0.00721, 0.01365] | 10/10 |

The supported wording is local robustness under the preregistered
policy-seed t-interval/directional classification, not uniform robustness
under every inference method or routing intensity. Two-way bootstrap intervals
exclude zero only for `(4, 0.75)` and `(8, 0.50)`; the center `(4, 0.50)`
interval is approximately [-0.0000004, 0.01856]. Any figure using t-intervals
must label them explicitly, with the bootstrap table retained in the
supplement.

## Forced-Deviation Claim Boundary

The diagnostic contains 400 forced-undercut events from 20 policies and 20
paths per policy.

| Requirement | Result | Decision |
| --- | --- | --- |
| rival spread compression | 0.000083; 95% CI [-0.000012, 0.000179] | Not supported |
| continuation flow loss | 0; 95% CI [0, 0] | Not supported |
| continuation reward loss | 0.02750; 95% CI [-0.001385, 0.056385] | Not supported |
| high-spread recovery by H | 96.25%; 95% CI [94.41%, 98.09%] | Supported |

The required scientific label is `high_spread_coordination_only`. The paper
must not claim punishment-supported collusion or learned retaliation.

## External Mechanism Adaptations

- Brero-style threshold buy-box: executable ABIDES adaptation, but the frozen
  response-convergence check failed and the selected candidate was not budget
  matched. Its efficacy results are descriptive only.
- Banerjee-style temporary buyer commitment: executable ABIDES adaptation,
  but the literature-form arm failed convergence in 2/20 policy seeds and the
  calibrated operating point was unmatched. Its efficacy results are
  descriptive only.
- Their lower raw spreads occur at more aggressive activation, high-risk
  intervention, and/or dealer reward cost. This is not evidence that ACE-RL
  beats the original literature methods, and it is not evidence that those
  methods fail.

## Additional Evidence Worth Presenting

1. **Problem existence.** State that all 20 fresh no-intervention base policies
   retained the preregistered high-spread failure mode after 384 episodes. This
   establishes that the intervention is addressing a reproducible simulator
   outcome rather than a selected seed.
2. **Proxy validity and separation.** Report the frozen passive-validation
   risk AUC of 0.8284 for the final two-feature risk model, along with the
   disjoint 1,000-record train and 1,000-record validation split. Make clear
   that this is simulator-supervised proxy validity, not real-market collusion
   detection. The AUC is currently in a local ignored diagnostic summary, so a
   compact committed calibration export is needed before the paper cites it.
3. **Dose definition.** Distinguish announced signal dose/multiset from
   realized nonzero routing activation and routing total variation. In the
   official MAPPO holdout the announced signal is matched, but realized
   activation is 0.2199 for risk-aware and 0.2081 for random.
4. **Seed heterogeneity.** Preserve the two official-MAPPO reversals and three
   IPPO reversals in a supplementary paired-seed plot. The t and bootstrap
   intervals support MAPPO, while the exact sign test alone does not.
5. **Main-result audit parity.** Export a sealed adaptive-v5 main/supplement
   per-path and per-step bundle before artifact submission. At present the
   learner/ecology holdout is independently auditable at a finer grain than
   the primary v5 table.
6. **Inference sensitivity.** State that the dose/routing figure uses
   policy-seed t-intervals. Preserve the paired two-way bootstrap intervals in
   the supplement because they are materially wider for several sensitivity
   cells.

## Source Map

- adaptive-v5 fresh-panel results:
  `revision/paper/data/abides_mappo_adaptive_v5_supplement_v1_per_seed.csv`
- adaptive-v5 detailed local aggregate:
  `revision/experiments/outputs/abides_mappo_adaptive_v5_supplement_v1/summary.json`
- final two-feature proxy AUC (currently local/ignored):
  `revision/experiments/outputs/abides_adaptive_gate_shift_diagnostic_v1/summary.json`
- learner/ecology audit statistics:
  `revision/paper/artifacts/abides_holdout_audit_bundle_v1/statistics.json`
- learner/ecology primitive data:
  `revision/paper/artifacts/abides_holdout_audit_bundle_v1/`
- dose/routing statistics and primitive data:
  `revision/paper/artifacts/abides_dose_routing_sensitivity_v1/`
- forced-deviation diagnostic:
  `revision/paper/artifacts/abides_collusion_diagnostic_v1/`
- Brero-style domain benchmark:
  `revision/paper/artifacts/abides_audit_bundle_v1/`
- Banerjee-style temporary commitment:
  `revision/paper/artifacts/abides_temporary_commitment_benchmark_v1/`

## Supported Claim

> In the tested controlled N=10 ABIDES dealer markets, simulator-supervised
> public-risk timing with pre-quote-announced, post-quote-applied,
> demand-conserving soft-rank routing improves low-risk customer execution
> relative to matched announced-signal random timing under official full-policy
> MAPPO and in two pre-frozen compact-adapter ecology perturbations. The effect
> is locally robust around the tested operating region, with a retained
> low-dose reversal.

Required restrictions: use high-spread coordination rather than
punishment-supported collusion; do not claim universal learner robustness,
literature-baseline superiority, larger-N robustness, cross-engine validity,
or real-market effectiveness.
