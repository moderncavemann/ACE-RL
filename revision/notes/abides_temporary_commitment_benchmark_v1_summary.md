# ABIDES Temporary Commitment Benchmark v1 Summary

Date: 2026-07-14
Status: formal run complete; implementation integrity failed at frozen convergence check

## Decision

The ABIDES adaptation of Banerjee-style temporary buyer commitment completed
all 20 source policies, 96 response episodes per new adapter, and 10 fresh
evaluation paths per policy. Contract execution and reproducibility checks
passed. The formal package nevertheless has `implementation_status=fail`
because the predeclared response-convergence requirement failed for the
literature-form adapter in 2 of 20 policy seeds.

Consequently, all efficacy values below are descriptive audit outputs, not
confirmatory evidence. The convergence tolerance is not relaxed, the response
budget is not extended, and no parameter search follows this result.

## Frozen Design

- source policy seeds: `530000-530019`
- response episodes: 96 per new adapter
- evaluation paths: `539000-539009`
- literature-form point: window 10, three-grid-step undercut, protected
  fraction 0.50, duration 5
- passive-calibration operating point: window 10, three-grid-step undercut,
  protected fraction 0.25, duration 1
- calibration status: `unmatched`
- completed-resume aggregate:
  `7bd3a7add930909352ad75c98446f727059be72b53ac15761837af5ca8c1e2e7`
- completed resume: zero containers started and aggregate unchanged

## Integrity Result

Seventeen of the eighteen runner checks pass. The passing checks include:

- 20/20 policy seeds and exact 96-episode response budgets;
- identical initial adapter digest and unchanged frozen v5 source digests;
- exact common training/evaluation paths and action uniforms;
- no latent-risk access by the external mechanism;
- exact protected commitment duration and tagged resting-order consumption;
- frozen-price protected execution with no lit-book fallback;
- routing-weight validity and demand conservation at every step;
- mechanism-off action equivalence, complete per-path grid, and step hashes.

The sole failed check is `response_converged`. The budget-matched operating
point passes the 0.15 terminal spread-range limit in all 20 seeds. The
literature-form point fails in seed `530008` with range 0.1854 and seed
`530014` with range 0.2359. All other literature-form seeds pass.

## Descriptive Results

These values are retained for auditability only.

| Method | Low-risk execution spread | Dealer reward / episode | Activation | High-risk activation | Routing TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| no intervention | 6.5308 | 8442.70 | 0.0000 | 0.0000 | 0.0000 |
| matched random | 6.3664 | 8060.03 | 0.1933 | 0.1314 | 0.0614 |
| risk-aware | 6.2830 | 8027.57 | 0.1955 | 0.1051 | 0.0625 |
| temporary commitment, literature form | 5.2435 | 5557.40 | 0.6062 | 0.6686 | 0.2728 |
| temporary commitment, calibrated operating point | 6.2571 | 7871.01 | 0.2497 | 0.2546 | 0.0562 |

Paired policy-seed low-risk spread contrasts, comparator minus risk-aware:

| Contrast | Mean | 95% t-CI | Positive seeds |
| --- | ---: | ---: | ---: |
| matched random minus risk-aware | 0.0834 | [0.0796, 0.0873] | 20/20 |
| commitment operating point minus risk-aware | -0.0259 | [-0.0392, -0.0126] | 4/20 |
| literature form minus risk-aware | -1.0395 | [-1.0768, -1.0023] | 0/20 |

The negative commitment contrasts mean the commitment arms have lower raw
customer spread. They do not establish dominance. The calibrated point was
not within the frozen activation tolerance, activates more often in high-risk
states, and has lower dealer reward than risk-aware. The literature-form point
is much more aggressive and is the arm that failed convergence.

## Claim Boundary

This run supports only the implementation statement that the protected-order
commitment contract is executable and auditable in ABIDES. It does not support
a confirmatory comparison against ACE-RL because the frozen convergence gate
failed. Even absent that failure, the calibrated point is explicitly
`unmatched` and could only be discussed as an operating-point trade-off, never
as a same-budget superiority test.
