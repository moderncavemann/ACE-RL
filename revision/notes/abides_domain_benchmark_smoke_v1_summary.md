# ABIDES Domain Benchmark v1 Engineering Smoke

Status: plumbing passed; no scientific result  
Date: 2026-07-14

The preregistered engineering smoke completed for source policies `530000`
and `530001`, four response episodes per new adapter, and evaluation paths
`539900-539901`. It exercised all nine domain-benchmark methods, checkpoint
and resume, real ABIDES routing, per-path export, and compressed per-step audit
records.

All 13 smoke integrity checks passed: common training/evaluation paths and
action uniforms, identical new-adapter initialization, unchanged v5 source
digests, exact gate-off actions, completed response budgets, demand
conservation, convergence marked not applicable for smoke, exact per-path row
count, and present step-record hashes. The output contains 36 per-path rows
and 3,600 step records. Each seed took about 138 seconds while running in
parallel. A completed rerun took 1.47 seconds and launched no seed work.

The derived image was
`sha256:2b72c2a8f0215d59e8cbf6b6e992ef2e7d3745d5cb5e8e764f2c249cd51a8c23`.
The smoke protocol fingerprint was
`1d3c36ea2883aed41e305c87f27d515ed9e032488eaf56ef91c74b68502cf6b2`.

The smoke's Brero budget-matched placeholder used an arbitrary fixed `k=2`
only to exercise the code path; its realized activation was not matched. No
spread, reward, activation, or paired-effect value from this smoke is eligible
for a paper table or parameter choice. Passive calibration remains the next
stage.
