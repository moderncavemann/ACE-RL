# ACE-RL Experiment Program Overview v1

Date: 2026-07-13

This note consolidates the experiment program from the first toy assay through
the completed adaptive ABIDES/MAPPO full main. It separates confirmatory
evidence from engineering smokes, parameter screens, and recorded negative
results.

## Current Decision

The strongest completed result is the adaptive-v5 full main in the controlled
N=10 ABIDES dealer market. It passes all 23 preregistered checks on 20 fresh
policy seeds after each intervention arm receives separate MAPPO adaptation.

The supported claim is:

> In the controlled N=10 ABIDES dealer market, a passively trained risk model
> used to announce demand-conserving soft-rank routing before quotes improves
> low-risk customer execution quality over exact-dose random routing after
> separate MAPPO adaptation, while conserving volume and limiting high-risk
> intervention.

This is a selective-intervention trade-off result. It is not a lowest-spread,
real-market, generic collusion-detection, or all-scales claim.

## Experiment Inventory

| Stage | Environment and data | Main scale | Status | Scientific role |
| --- | --- | --- | --- | --- |
| Toy observation and calibration | Synthetic LOB-inspired spread-choice game; simulator records only | Calibration seeds 0-4 | Passed | Established a measurable low-risk high-spread failure mode and froze the gate from no-intervention data only. |
| Toy held-out sweep | Same toy environment | 10 held-out seeds, 1000 train / 50 eval episodes | Passed | First evidence that gated routing beats matched random and that routing and risk adjustment matter. |
| Toy N=10 full confirmation | Same toy environment | Seeds 15-34, 1500 train / 100 eval episodes | Passed | Paper-ready toy baselines, paired inference, core ablations, and trade-off figures. |
| Toy robustness | N=5, N=10, N=20, wider costs, faster risk, longer horizon | Seeds 35-44, 800 train / 50 eval episodes | Mostly passed | Supported controlled robustness except for a weak N=20 timing margin. |
| Toy N=20 follow-up | N=20 stress test with frozen N=10 gate | Seeds 45-64, 1500 train / 100 eval episodes | Partial | Longer training improved consistency, but the effect remained smaller and spread-only remained a strong comparator. |
| Early local LOB/MAPPO pilots | Repository-local synthetic matching engine | Mostly 5-10 seeds | No-go | Exposed assay/risk misalignment, routing-subsidy, dose-matching, and policy-symmetry problems. Parameter grids are diagnostic, not paper evidence. |
| Local LOB structural redesign v2 | Risk-coupled synthetic LOB with demand-conserving routing | 100 fixed-policy seeds and 5 MAPPO timing seeds | No-go | Fixed economics and seller response passed, but online oracle timing lost to the equal-dose comparator on seed stability. |
| Local LOB state-contingent contract v3 | Redesigned synthetic LOB/MAPPO response assay | Two independent 5-seed oracle batches | Passed mechanism gate | Established that high-spread-dependent routing can make state timing identifiable. |
| Local LOB learned-gate full main | Redesigned synthetic LOB/MAPPO; true simulator risk used for oracle-distillation labels | Seeds 60-79; 100 base + 100 response episodes; 10 eval episodes | Passed | Demonstrated learned timing over exact-dose random and circular controls with routing and pool ablations. |
| ABIDES dependency and adapter gates | Official modular ABIDES engine at commit `f9cbe51`; synthetic informed-flow market paths | RMSC04 smoke plus controlled ten-dealer adapter | Passed | Verified the real matching engine, ten-agent synchronization, official order-book fills, and reproducible Docker runtime. |
| ABIDES fixed-policy economics and oracle | Controlled N=10 ABIDES dealer market | 10 seeds per gate | Passed | Verified risk-coupled adverse selection, demand conservation, routing causality, and identifiable oracle timing before learned training variance. |
| ABIDES observable learned-gate G5 | Frozen passive MAPPO seller policy | 5 screen + 5 fresh holdout seeds | Passed | Observable-only classifier and exact-dose timing controls passed before full-main launch. |
| ABIDES frozen-policy full main | Controlled N=10 ABIDES market; frozen seller policy | 20 seeds, 100 decisions | Passed | Learned timing beat matched random and circular timing, but policies did not adapt separately by arm. |
| Adaptive ABIDES v1/v2 | Same controlled ABIDES market; each arm retrained | 5 policy seeds, extended to 150 episodes | Failed | Two-stage post-action/severity gate was unstable: 3/5 online seed wins and a CI crossing zero. |
| Adaptive ABIDES v3 holdout | Pre-action announced soft-rank contract, complete-episode GAE, passive risk-only gate | 5 fresh policy seeds | Passed | Reopened the adaptive mechanism gate with a structural redesign rather than parameter tuning. |
| Adaptive ABIDES v3 full scale | Primary and fast-risk ecologies | Primary 20 seeds; fast-risk 10 seeds | Overall no-go | Primary stopped at 18/20 base convergence; the preregistered fast-risk ecology passed 10/10 and remains secondary generalization evidence. |
| Adaptive ABIDES v4 | Fixed 384-episode base and six response arms | 20 fresh policy seeds | Formally failed | Primary efficacy passed 20/20, but the preregistered all-arm convergence veto was tripped only by always-on. |
| Adaptive ABIDES v5 full main | Same v4 method with fresh seeds and an estimand-aligned convergence set | 20 base policies; 384 base + 96 response episodes | Passed | Final adaptive full-main result; all 23 checks passed without changing the method or efficacy endpoint. |

## Key Confirmatory Results

### Toy N=10

- No intervention collusion gap: `0.2319`, 95% CI `[0.2175, 0.2465]`.
- Gated collusion gap: `0.1334`; matched random: `0.2082`.
- Paired random minus gated delta: `0.07481`, positive in `20/20` seeds.
- Gated FPR: `0.1289`; threshold FPR: `0.8980`.
- Gated reward: `6.3741`; always-on reward: `3.7316`.
- No-routing returned near no intervention; no-risk-adjustment reached FPR
  `0.9997`.

The toy result validates the original mechanism and its trade-off logic. It
does not validate a matching-engine or adaptive-policy claim.

### Local Synthetic LOB/MAPPO Full Main

- Exact-dose matched random minus learned low-risk spread: `0.09998`, 95% CI
  `[0.07478, 0.12519]`; `19` positive, `1` tie, `0` negative seeds.
- Circular minus learned: `0.10299`, 95% CI `[0.07714, 0.12884]`.
- Activation: `0.1030`; high-risk FPR: `0.0000`.
- Routing training and the state-dependent routing pool both passed their
  trained ablations.

This was an important mechanism confirmation, but its learned classifier was
oracle distillation from the simulator's true risk regime and the environment
was not ABIDES.

### ABIDES Frozen-Policy Full Main

- Matched random minus learned low-risk execution half-spread: `0.10893`, 95%
  CI `[0.09175, 0.12610]`, positive in `20/20` seeds.
- Circular minus learned: `0.11137`, 95% CI `[0.09541, 0.12732]`, positive in
  `20/20` seeds.
- Every arm conserved exactly 20 units of customer demand.
- No-routing exactly matched no intervention.

This validates timing on the ABIDES matching engine with a frozen MAPPO seller
policy. It does not establish adaptation under each intervention arm.

### ABIDES Adaptive v5 Full Main

- Risk-aware low-risk execution half-spread: `6.29318`; matched random:
  `6.37741`; no intervention: `6.53908`.
- Primary matched-random minus risk-aware delta: `0.08423` tick, SE `0.00166`,
  95% CI `[0.08075, 0.08771]`, positive in `20/20` policy seeds.
- Risk-aware and random both activated `20.4%` of the time, but high-risk
  activation was `4.98%` versus `19.94%`.
- Risk-aware reduced spread by `3.76%` versus no intervention with a `3.48%`
  reward reduction; volume remained exactly 20 in every arm.
- No-routing minus risk-aware was `0.24295` tick, 95% CI
  `[0.23651, 0.24938]`, positive in `20/20` seeds.
- Threshold and always-on achieved lower raw spread by intervening more often;
  always-on lost `25.10%` reward and remained non-asymptotic in `8/20` seeds.

## How The Method Evolved

The original idea remained the same: intervene selectively when high spread is
less plausibly explained by legitimate risk, and change seller incentives
through platform routing rather than a direct reward penalty.

The implementation became more defensible in three steps:

1. The toy environment established the risk-adjusted diagnosis and selective
   routing trade-off with held-out calibration and evaluation.
2. Local LOB failures forced risk into the actual trading economics, removed
   synthetic targeted-fill subsidies, conserved demand, and introduced
   exact-dose temporal controls.
3. ABIDES adaptive v5 uses an observable passive risk model, announces the
   routing contract before quotes, applies demand-conserving soft-rank routing,
   and trains each MAPPO arm separately from a common frozen base.

## Data, Scale, And Remaining Boundaries

- No historical order-book dataset is used. All evidence comes from synthetic
  simulator paths.
- The paper-facing matching-engine environment is a controlled ABIDES dealer
  ecology, not the canonical RMSC04 background-agent composition.
- N=10 is the only completed matching-engine main scale. Toy experiments cover
  N=5 and N=20 stress tests; there is no exhaustive N=2 through N=20 sweep.
- The final risk model uses public volatility and absolute lagged return in the
  simulator. It does not prove real-world latent collusion detection.
- Larger dealer populations, another matching engine, another MARL learner,
  and real-data external validity remain untested.

## Paper-Facing Evidence Order

1. Use adaptive ABIDES v5 as the main result.
2. Use the ABIDES fixed-policy experiment as timing identification and the
   no-routing arm as the causal mechanism check.
3. Use the N=10 toy package for transparent risk-adjustment, FPR, and component
   ablations.
4. Report the fast-risk adaptive-v3 result as secondary ecology evidence.
5. Keep N=20, adaptive-v2, adaptive-v3 primary, and v4 as explicit limitations
   or protocol-history evidence; do not pool them into the positive main table.

Primary source notes:

- `revision/notes/toy_full_n10_experiment_v1_summary.md`
- `revision/notes/toy_lob_robustness_sweep_v1_summary.md`
- `revision/notes/toy_lob_n20_followup_v1_summary.md`
- `revision/notes/lob_mappo_structural_redesign_v2_summary.md`
- `revision/notes/lob_mappo_learned_gate_full_main_v1_summary.md`
- `revision/notes/abides_mappo_full_main_v1_summary.md`
- `revision/notes/abides_mappo_adaptive_two_stage_extension_v1_summary.md`
- `revision/notes/abides_mappo_adaptive_v3_full_scale_v1_summary.md`
- `revision/notes/abides_mappo_adaptive_v4_full_main_v1_summary.md`
- `revision/notes/abides_mappo_adaptive_v5_full_main_v1_summary.md`
