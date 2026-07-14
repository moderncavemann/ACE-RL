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

At the selection-order correction stage, the candidate grids, dose targets,
passive paths, formal evaluation paths, routing settings, source policies, and
all scientific outcome definitions remained unchanged. The original numerical
selections are recorded here rather than silently discarded. The corrected
artifact must be committed and hashed before formal sensitivity execution.

## Comparator Feasibility Correction

The corrected risk selection produced a valid `0.10` risk arm at realized
activation `0.081`, but no spread-only candidate in the original scale grid
could simultaneously satisfy the unchanged nominal activation, pairwise
activation, and `[0.9, 1.1]` routing-TV ratio tolerances. The nearest selected
scale `0.50` had TV ratio `0.833`. Before any response training, the comparator
calibration grid was therefore interpolated between `0.50` and `0.75` with
scales `0.55`, `0.60`, `0.65`, and `0.70`. This is a budget-feasibility repair,
not an efficacy search: no spread, reward, or formal evaluation outcome was
available, and all matching tolerances remain unchanged.

## Realized-activation Definition Correction

While wiring the formal runner, the passive calibration implementation was
found to count `signal_strength > 0` as activation. That is not the directive's
realized routing dose: when all dealers quote identically, a positive signal
can still produce zero routing total variation. Before any response training,
the calibration endpoint was corrected to
`routing_total_variation > 1e-12`; signal activation remains reported as a
separate diagnostic. All candidate grids, matching tolerances, paths, and
scientific outcomes remain unchanged.
