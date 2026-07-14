# ABIDES Domain Benchmark v1 Summary

Status: formal execution complete; audit verification pass; directive integrity fail
Date: 2026-07-14

## Execution

- Protocol fingerprint: `16f527416f09846baab3147008c301821c06ba1cd04e601065531645df5fef1f`.
- Execution and protocol-freeze commit: `4d3b660f10f6a5c1d06491bd336f543e5a69e993`.
- Source policy seeds: `530000-530019`; all `20/20` completed.
- Response budget: 96 episodes per new adapter; all completed.
- Final evaluation: ten paired paths and nine methods per policy seed.
- Formal wall time: approximately 1 hour 53 minutes; no execution failures.
- A completed rerun took 1.76 seconds, started no containers, and reproduced
  aggregate SHA-256
  `fe340794ba97159ca297d0bffa39ea594c6f0563d1a58eb1c3c2ea7e23649b96`.

The independent bundle verifier recomputed 1,800 per-path rows, 20 per-seed
rows, 180,000 per-step records, paired statistics, the 10,000-replicate
two-way bootstrap, hashes, paired paths and action uniforms, routing weights,
demand conservation, and paper exports.

## Frozen Integrity Result

Nineteen of the twenty directive checks passed. Check 14, the frozen response
convergence rule, failed. The all-new-arm terminal-range criterion passed for
12/20 policy seeds. Per arm, `spread_only_budget_matched` passed for 20/20,
`brero_threshold_buybox_budget_matched` for 19/20, and the aggressive
literature-form `brero_threshold_buybox` for 13/20. Seven of the eight
seed-level failures were caused only by the aggressive Brero arm; one also
included the budget-matched Brero arm.

The threshold remains `0.15`. No seed was removed, no episode was appended,
and no convergence tolerance was changed after outcomes were opened.

Therefore:

```text
implementation_status = fail
scientific_result = not_evaluated_integrity_fail
```

This formal cell cannot support a confirmatory efficacy or superiority claim.

## Descriptive Results Only

The following values are retained for auditability and diagnosis, not as
confirmatory paper evidence:

| Contrast or method | Descriptive result |
| --- | ---: |
| `matched_random - risk_aware` low-risk half-spread | `0.05489`, t-based 95% CI `[0.05171, 0.05808]`, 20/20 positive |
| `spread_only_budget_matched - risk_aware` | `0.06837`, t-based 95% CI `[0.06474, 0.07200]`, 20/20 positive |
| `brero_threshold_buybox_budget_matched - risk_aware` | `-0.04935`, t-based 95% CI `[-0.08771, -0.01099]`, 8/20 positive |
| Risk-aware activation / routing TV / dealer reward | `0.19545` / `0.06222` / `9.37824` |
| Spread-only matched activation / routing TV / dealer reward | `0.18195` / `0.05565` / `9.35862` |
| Brero selected activation / routing TV / dealer reward | `0.39350` / `0.10613` / `9.14712` |

The Brero candidate was already calibration-unmatched and remains a trade-off
operating point. Its lower raw spread cannot be read as same-budget
superiority. The risk-aware timing signal is descriptively present against
both matched random and the matched spread-only rule, but the preregistered
integrity failure blocks confirmatory use.

## Claim Boundary And Next Step

The existing v5 and supplement results remain unchanged. This domain benchmark
does not upgrade their claim and does not justify parameter search or extra
training. The next preregistered workflow is the fixed-policy forced-undercut
diagnostic. It is independent of this adaptive-baseline convergence failure
and determines whether the paper may retain strong collusion language or must
use `high-spread coordination`.

The committed audit bundle under
`revision/paper/artifacts/abides_audit_bundle_v1/` is the sole source for this
cell's values and integrity status.
