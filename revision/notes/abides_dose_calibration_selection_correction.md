# ABIDES Dose Calibration Selection Correction

Date: 2026-07-14  
Status: corrected before any sensitivity response training or evaluation

The first passive calibration execution used the preregistered candidate grid
but ranked candidates by exact nominal activation error before risk
selectivity. It therefore selected risk-probability cutoff `0.05` at the
`0.10` and `0.20` targets. Passive high-risk activation was respectively
`0.0906` and `0.1959`, nearly equal to total activation. This was a semantic
calibration failure: the controller had become near-unconditional pacing and
was no longer a meaningful risk-aware dose arm.

No response adapter had been trained and no efficacy, reward, spread, or
formal evaluation outcome had been observed. The correction changes only the
passive calibration selection order:

1. retain candidates within the frozen `0.02` nominal activation tolerance;
2. select the highest observable risk-probability cutoff within that band;
3. use activation error only as a tie-breaker;
4. continue matching the spread-only comparator on activation and routing TV;
5. additionally require pairwise activation difference at most `0.02`.

The candidate grids, dose targets, passive paths, formal evaluation paths,
routing settings, source policies, and all scientific outcome definitions are
unchanged. The original numerical selections are recorded here rather than
silently discarded. The corrected artifact must be committed and hashed
before formal sensitivity execution.
