# ABIDES Temporary Commitment Benchmark v1 Protocol

Status: frozen before passive calibration and formal outcomes  
Date: 2026-07-14

## Scope

This secondary external-mechanism study implements the mapping documented in
`literature_baseline_mapping_v1.md`. It is always described as an **ABIDES
adaptation of Banerjee-style temporary buyer commitment**, never an exact
reproduction of the two-seller differentiated-products experiment.

The standing rule uses only current and lagged public quotes. After ten public
market-mean half-spreads are available, a dealer qualifies by quoting at least
three action-grid steps below that lagged reference. The lowest qualifying
quote takes up the contract, with exogenous-uniform tie breaking. No second
contract starts while one is active.

The literature-form point protects exactly 50% of the 20 customer events for
five decisions including take-up. Protected customer orders execute only
against a separately tagged, real ABIDES resting order at the frozen take-up
bid or ask. Each execution consumes resting quantity, cannot fall back to the
lit book, and cannot be reached by residual customer flow. Total demand remains
20 every decision.

## Passive Calibration

The literature-form parameters remain `(window=10, qualification=3,
fraction=0.50, duration=5)`. A separate budget-matched point freezes the same
window and qualification rule and searches only:

- protected fraction in `{0.10, 0.25, 0.50, 0.75}`;
- duration in `{1, 3, 5, 10, 20}`.

Selection uses the already frozen 20,000 passive public-quote records and only
active-routing rate plus mean routing total variation. It cannot read customer
spread, dealer reward, or method rank. The active-rate tolerance is `0.02` and
the routing-TV ratio range is `[0.90, 1.10]`. If no candidate passes both,
the nearest candidate is retained and labeled `unmatched`.

## Adaptive Comparison

The source panel is the 20 frozen v5 base policies `530000-530019`. The two
new commitment adapters start from the same zero adapter digest and train for
96 episodes on the exact v5 response environment paths and action uniforms.
The no-intervention base and frozen v5 risk-aware/matched-random policies are
the paired comparators. Final evaluation uses fresh paths `539000-539009` and
action namespace `591117` exactly once.

All arms retain per-path and compressed per-step records. Contract take-up,
duration, tagged resting-order consumption, frozen-price execution, no
fallback, total demand, paths, uniforms, model digests, and checkpoint resume
are integrity checks independent from efficacy.

The primary contrast remains matched-random minus risk-aware low-risk customer
execution half-spread. Commitment-minus-risk-aware contrasts are secondary and
Holm adjusted. A superior commitment baseline narrows the ACE-RL superiority
claim; it does not trigger more parameter search. An unmatched commitment is
reported only as an operating-point trade-off.
