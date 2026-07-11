# Local LOB/MAPPO Learned-Gate Full-Main Protocol

Date frozen: 2026-07-11

This protocol is frozen before inspecting any local-LOB outcomes for seeds
60-79. It follows the passed learned-gate screen and held-out confirmation in
`lob_mappo_learned_gate_v1_summary.md`.

## Scope

This is the full-main experiment for the repository's redesigned local
synthetic LOB and two-stage MAPPO response assay. It is not an ABIDES, real
order-book, latent-risk-inference, or scale-generalization experiment.

The primary claim under test is:

> Frozen learned oracle-distillation timing reduces low-risk spread relative
> to a within-episode, risk-stratified, exact-dose random schedule.

The gate directly observes the simulator risk regime. Results cannot be used
to claim a generic learned collusion detector.

## Hardware

- Apple M3 Ultra, 512 GiB unified memory.
- PyTorch neural-network updates use `mps`.
- Matching-engine transitions, trajectory assembly, and statistics use CPU.
- Seeds run sequentially through one MPS process to avoid multi-process GPU
  contention and nondeterministic resource scheduling.

## Frozen Configuration

| Layer | Value |
| --- | --- |
| Full-main seeds | 60-79 |
| Agents | 10 |
| Episode length | 100 |
| Base equilibrium training | 100 episodes |
| Response training | 100 episodes per response variant |
| Evaluation | 10 stochastic CRN episodes per seed |
| Calibration | 5 no-intervention episodes per seed |
| Calibration quantiles | q75 reference, q90 scale |
| Routable flow | 0.90 |
| Main contract | excess-spread pool, soft-rank target, rho=1.0 |
| MAPPO | multihead base plus frozen-base intervention adapter |
| Gate artifact | `revision/configs/lob_mappo_learned_gate_v1.json` |

The gate artifact was trained on seeds 35-39, threshold-validated on 40-44,
screened on 45-49, and confirmed on 50-54. No coefficient, threshold,
calibration rule, routing parameter, or acceptance criterion may change after
the full-main launch.

## Method Arms

### Main Comparison

- `base_no_intervention`: frozen base actor, no routing.
- `learned_gate`: frozen learned schedule and state-dependent pool.
- `matched_random`: learned schedule permuted within each episode and risk
  regime; total and regime-specific dose are exact.
- `circular_shuffled`: learned schedule circularly shifted within each risk
  regime; dose and within-regime activation-run structure are retained while
  state alignment changes.
- `oracle_gate`: diagnostic upper bound only.

All intervention arms use the same frozen response policy, external CRN path,
and per-step action uniforms.

### Aggressive Controls

- `threshold_fixed_pool`: fixed-share response policy; activate whenever the
  passive calibrated excess-spread condition is positive, regardless of risk.
- `always_on_fixed_pool`: fixed-share response policy; activate every step.

These controls use a separately trained fixed-pool response adapter so their
policy is matched to their contract. They are evaluated on spread, high-risk
impact, activation, volume, inventory, and seller reward; they are not allowed
to redefine the primary endpoint.

### Trained Ablations

- `no_routing_trained`: adapter trained with the same announced schedules but
  routing rho fixed to zero, then evaluated with no routing.
- `no_state_pool_learned` and `no_state_pool_matched`: adapter trained under a
  fixed-share rather than excess-spread-dependent routing pool.

Gate-off behavior for every adapter must exactly match the frozen base actor.

## Primary Endpoint

For each seed:

```text
delta_seed = matched_random low-risk mean half-spread
             - learned_gate low-risk mean half-spread
```

Positive values favor learned timing. Screen, held-out, and full-main batches
remain separate decisions; earlier results are not pooled into the full-main
pass/fail test.

## Frozen Acceptance Rules

The strict full-main package passes only if all of the following hold on seeds
60-79:

1. Gate train, validation, and full-main seeds are disjoint.
2. Artifact and complete runtime environment configurations match.
3. Training schedules replay exactly.
4. Random and circular comparators match total and low/high-risk dose exactly.
5. Every response adapter is exactly base-equivalent with the gate off.
6. No-intervention failure appears in at least 70% of seeds.
7. Learned activation remains between 5% and 40%.
8. Learned high-risk FPR is at most 5%, and target recall is at least 70%.
9. Mean `matched_random - learned_gate` delta is positive.
10. At least 70% of paired seed deltas are positive.
11. The normal-approximation 95% CI lower bound for the primary delta is above
    zero; the final report will also recompute the exact t interval.
12. Learned timing allocates more effective state-dependent pool than matched
    random.
13. Learned improves low-risk spread relative to no intervention.
14. High-risk spread is preserved within the registered half-tick guardrail.
15. Learned reward is no more than 0.005 below matched random in absolute mean.
16. Learned beats circular-shuffled timing with at least 70% positive seeds.
17. The state-dependent pool has a larger timing advantage than the fixed-pool
    ablation.
18. Routing-trained intervention has a larger base-relative improvement than
    the no-routing-trained ablation.

Aggressive controls may attain lower raw spread. The paper-facing claim is a
spread / high-risk false-positive / intervention / reward trade-off, not the
lowest raw spread.

## Checkpointing

Each completed seed is atomically saved under the output directory's
`seed_results/` folder. `run_manifest.json` stores a SHA-256 fingerprint over
the gate artifact, seed set, environment, MAPPO configuration, and experiment
parameters. Resume rejects any mismatched fingerprint.

## Frozen Command

```bash
python revision/experiments/run_lob_mappo_learned_gate_pilot.py \
  --stage full_main \
  --learned-gate-artifact revision/configs/lob_mappo_learned_gate_v1.json \
  --seeds 60-79 \
  --equilibrium-episodes 100 --response-episodes 100 \
  --response-train-rate 1.0 \
  --calibration-episodes 5 --calibration-quantile 0.75 \
  --calibration-upper-quantile 0.90 \
  --eval-episodes 10 --episode-length 100 \
  --warm-start-steps 200 --warm-start-updates 20 \
  --device mps \
  --output-dir revision/experiments/outputs/lob_mappo_learned_gate_full_main_20seed_v1
```

Raw outputs remain ignored. The final aggregate, exact recomputation, failure
analysis, and claim boundary must be saved in a repo-visible note.
