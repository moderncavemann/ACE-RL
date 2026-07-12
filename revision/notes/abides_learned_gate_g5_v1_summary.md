# ABIDES Learned Gate G5 v1

Status: passed screen and fresh holdout  
Date: 2026-07-12

## Frozen Inputs

- Passive MAPPO checkpoint SHA-256:
  `12b6f7fc29e00aad385bea318205581b7c73fc07903595908c0cef095a53ce93`.
- Learned gate artifact SHA-256:
  `f05d4c9f2ca16b8219f9786659ad076da81d4452e3d1cdd4a56ccae040c572cc`.
- Probability threshold: 0.842.
- Runtime activation ceiling: 35% per episode.
- Episode length: 40 decisions.
- Primary endpoint: comparator minus learned low-risk customer execution
  half-spread. Positive values favor learned timing.

Matched random independently permutes the learned arm's actual activation
vector within each realized risk stratum. Circular shuffled rotates the same
risk-stratified vector. Both preserve total and risk-specific dose and use the
same exogenous path and action uniforms.

## Initial Screen Failure and Repair

The first screen already had positive timing effects against both comparators
on 5/5 seeds, but raw proposal rates reached 0.375 and 0.475 on two seeds. It
therefore failed only the frozen 35% activation ceiling.

The classifier and threshold were not changed. A deterministic token budget was
added to reject proposals after `floor(0.35 * episode_steps)` acceptances. It
cannot create activations or change proposal probabilities. The complete screen
was then rerun from the beginning.

## Screen: Seeds 320000-320004

| Metric | Result |
| --- | ---: |
| actual activation range | 0.175-0.350 |
| raw proposal range | 0.175-0.475 |
| matched random minus learned | 0.10209 |
| matched 95% CI | [0.05642, 0.14777] |
| matched positive/tie/negative | 5 / 0 / 0 |
| circular minus learned | 0.11094 |
| circular 95% CI | [0.04713, 0.17474] |
| circular positive/tie/negative | 5 / 0 / 0 |
| learned mean reward | 12.1959 |
| matched mean reward | 12.4482 |

All activation, reward, and volume checks passed.

## Fresh Holdout: Seeds 330000-330004

| Metric | Result |
| --- | ---: |
| actual activation range | 0.175-0.350 |
| raw proposal range | 0.175-0.400 |
| matched random minus learned | 0.09466 |
| matched 95% CI | [0.03627, 0.15304] |
| matched positive/tie/negative | 5 / 0 / 0 |
| circular minus learned | 0.10390 |
| circular 95% CI | [0.08706, 0.12075] |
| circular positive/tie/negative | 5 / 0 / 0 |
| learned mean reward | 11.5644 |
| matched mean reward | 11.7706 |

All activation, reward, and volume checks passed without modifying the frozen
gate, policy, routing contract, or runtime controller after the screen.

## Decision

G0-G5 are passed. The N=10 ABIDES/MAPPO experiment may enter the 20-seed full
main under a newly frozen seed namespace. This is evidence for a frozen
MAPPO-policy timing comparison; adaptive retraining under each intervention arm
must be reported separately if added and is not implied by this gate.
