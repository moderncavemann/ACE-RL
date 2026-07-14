# ACE-RL Domain Benchmark And Audit Completion v1

Date: 2026-07-14  
Status: P0 and P1 execution complete; final claim boundary frozen

## Overall Decision

The seven-workflow domain benchmark completion program is complete. This is a
completed experiment package, not a smoke test or a parameter screen. The
outcomes are mixed and are retained as such: the official MAPPO and both
pre-frozen ecology holdouts are favorable; IPPO is positive but inconclusive;
the forced-deviation diagnostic rejects strong punishment-supported collusion
wording; and the Brero/Banerjee formal packages fail their frozen convergence
integrity gates, so their efficacy values remain descriptive.

No result-triggered parameter search, seed replacement, episode extension, or
selective ecology/learner reporting was used.

## Workflow Decisions

| Workflow | Completion | Scientific decision |
| --- | --- | --- |
| Brero-style threshold buy-box | mapping, tests, passive calibration, 20-policy adaptive formal run, audit complete | only `ABIDES adaptation`; frozen convergence failed and budget-matched candidate remained unmatched, so no confirmatory superiority claim |
| Banerjee-style temporary buyer commitment | mapping, tests, literature-form and calibrated points, 20-policy adaptive formal run, audit complete | executable mechanism adaptation; literature-form convergence failed in 2/20 seeds, so efficacy is descriptive only |
| Truly budget-matched spread-only | activation and routing-TV calibration from passive data, formal and dose comparisons complete | risk-aware is better at all three tested doses; domain confirmatory use remains blocked by the shared formal convergence gate |
| Forced undercut diagnostic | 20 policies, 20 paths, 400 events complete | `high_spread_coordination_only`; no supported punishment asymmetry |
| Dose/routing sensitivity | 0.10/0.20/0.30 dose curve and five frozen routing cells complete | `broadly_robust` around the frozen region; matched random wins at dose 0.10 and that reversal is retained |
| Standard learner holdout | official MAPPO and IPPO, 10 seeds each, complete | MAPPO favorable; IPPO positive but inconclusive; no universal cross-learner claim |
| Ecology holdout and independent audit | E1 and E2, 10 fresh seeds each, audit-grade replay complete | both favorable, supporting robustness only within the two tested N=10 ecologies |

## Final Supported Claim

The strongest evidence-aligned statement is:

> In the tested controlled N=10 ABIDES dealer markets, simulator-supervised
> public-risk timing with pre-quote-announced, post-quote-applied,
> demand-conserving soft-rank routing improves low-risk customer execution
> relative to equal-dose matched-random timing under official full-policy
> MAPPO and in two pre-frozen compact-adapter ecology perturbations. The timing
> effect is locally robust across the tested dose/routing region, with a
> retained low-dose reversal.

Required restrictions:

- use `high-spread coordination`, not punishment-supported or learned
  collusion;
- call Brero and Banerjee mechanisms `ABIDES adaptations`;
- do not claim literature-baseline superiority because their formal integrity
  or budget-matching requirements did not pass;
- do not claim robustness across all learners: IPPO is inconclusive;
- do not claim large-N, cross-matching-engine, or real-market validity;
- keep the existing v5/supplement results unchanged.

## Definition Of Done

- [x] At least one literature-derived external intervention mechanism was
  mapped, tested, and run adaptively; both Brero and Banerjee adaptations were
  completed, with their negative integrity outcomes retained.
- [x] `spread_only_budget_matched` was calibrated and evaluated.
- [x] Budget matching used passive data without formal efficacy access.
- [x] Domain benchmark used 20 source policies and 10 fresh evaluation paths.
- [x] Deviation-punishment diagnostic fixed the wording boundary.
- [x] Three-point within-method dose curve completed.
- [x] `kappa` and routable-fraction sensitivity completed.
- [x] Official full-policy MAPPO and IPPO holdouts completed.
- [x] Two pre-frozen fresh ecology holdouts completed and both reported.
- [x] Per-path and compressed per-step audit data saved.
- [x] Policy-seed t-CI, exact sign test, paired two-way bootstrap, and Holm
  secondary inference are independently reproducible.
- [x] Hashes, fingerprints, model/path digests, execution/freeze commits,
  official learner commit, ABIDES commit, and Docker image IDs are recorded.
- [x] GitHub Actions runs tests, all independent verifiers, and deterministic
  export checks on committed artifacts.
- [x] Failed launches, failed convergence gates, negative cells, and seed
  reversals are retained.
- [x] Claims are narrowed automatically from the final decision tree.

## Unified Audit Package

The combined learner/ecology package is
`revision/paper/artifacts/abides_holdout_audit_bundle_v1/`. It contains 2,000
per-path rows, 40 derived per-seed rows, and 200,000 compressed per-step
records across four separately reported groups. The independent verifier
checks every source result and step row, exact path/action/risk pairing,
activation, high-risk activation, targeted events, routing/flow vectors,
volume, model digests, t/sign/bootstrap/Holm statistics, checksums, source
commits, Docker/ABIDES provenance, paper CSVs, and completed resumes.

The accepted ecology replay is v3. v1 failed closed, and v2's zero-activation
schema defect is documented rather than overwritten. The final bundle is
sealed with an archival commit and verified in strict mode before publication.
