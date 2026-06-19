# Toy LOB Observation Plan

The first toy stage is not designed to prove ACE-RL works. It is designed to
answer whether the simplified market has the right failure mode.

## Step 0: Calibration

Canonical parameters:

- `N = 10`
- spread levels: `1..7`
- `Q0 = 100`
- demand alpha: `0.20`
- allocation eta: `0.80`
- `c_low = 0.50`
- `c_high = 2.50`
- persistent Markov risk for the mixed setting

Analytic benchmarks:

- low-risk competitive spread should land near grid level `2`
- low-risk collusive spread should land near grid level `5` or `6`
- high-risk competitive spread should land near grid level `4`

This gives the needed confounder: high spread can be normal under high risk.

## Step 1: No-Intervention Observation

Run:

- fixed low-risk
- fixed high-risk
- mixed Markov risk
- optional low-risk N sweep
- low-risk deviation probe

Observation passes only if low-risk no-intervention produces a meaningful
risk-adjusted collusion gap, spread alignment, demand suppression, and some
evidence of punishment/recovery after a forced one-step spread cut.

The tabular Q state is:

- risk bin
- average spread bin
- own spread bin
- market minimum spread bin
- spread range bin

The minimum spread and spread range are required because a one-agent undercut in
an N=10 market may barely move the average spread.

No-intervention records also include `passive_risk_score`, a non-intervening
risk-aware gate score used later to calibrate thresholds from observed score
distributions.

## Step 2: Baselines

Only after Step 1 passes, compare:

- no intervention
- always-on routing intervention
- spread threshold rule
- matched-rate random gate
- risk-aware gated routing intervention

Baseline runs should use a method x seed grid. The random gate should be
matched to the realized risk-aware gate activation rate, not fixed at an
arbitrary rate.

## Step 3: Gate Ablations

Only after Step 2 passes, compare full gate against:

- spread-only gate
- no risk adjustment
- no alignment feature
- no demand suppression feature
- no routing intervention

The current intervention is a gated routing-sensitivity intervention: when the
gate fires, the platform increases the order-flow advantage for lower spreads.
It is not yet a temporal defector-protection mechanism.
