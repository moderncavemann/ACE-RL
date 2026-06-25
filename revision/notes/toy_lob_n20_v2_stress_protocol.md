# Toy LOB N=20 Gate-v2 Stress-Test Protocol

N=20 is a scale stress test, not the primary toy market size. The main toy
setting remains N=10. N=20 is included because the paper claim concerns
large-N / multi-agent RL trading, so the gate should be tested for immediate
failure when the number of agents doubles.

## Why N=20

N=20 answers three reviewer-facing questions:

- Does an N=10-calibrated gate overfit N=10 feature distributions?
- Does the intervention still beat matched random timing when agent count
  changes?
- Does larger N expose a representation problem in the gate?

The current evidence supports the third point. The N=20 follow-up did not show
complete method failure, but the effect size got smaller and the old global
`alignment` feature became a poor positive collusion signal. The N-invariant
diagnostic therefore motivates gate-v2 distributional features rather than a
jump to full LOB/MAPPO.

## Claim Boundary

Allowed current claim:

> N=20 acts as a larger-N stress test. It reveals a scale-specific gate
> representation issue: global alignment is not a stable positive collusion
> feature, motivating N-invariant distributional features.

Not allowed yet:

> Gate-v2 solves N=20.

The repo currently has feature diagnostics and runtime smoke for gate-v2, but
not a completed held-out N=20 gate-v2 intervention sweep.

## Formal Sweep Design

- Scenario: N=20 mixed-risk toy LOB-inspired environment.
- Calibration: frozen gate-v2 thresholds from N=10 no-intervention records.
- Held-out seeds: 20 seeds, default `90-109`.
- Train episodes: `800` first; increase to `1500` only if runtime budget allows.
- Eval episodes: `50` first; increase to `100` only if runtime budget allows.
- Output: ignored raw outputs under
  `revision/experiments/outputs/n20_v2_formal_sweep_v1/`.

Default methods:

- `no_intervention`
- `threshold`
- `always_on`
- `gated_intervention`
- `spread_only_gate`
- `v2_entropy_gate`
- `v2_fraction_near_min_gate`
- `v2_two_stage_gate`

Matched random baselines are generated separately for:

- `gated_intervention`
- `v2_entropy_gate`
- `v2_fraction_near_min_gate`
- `v2_two_stage_gate`

Each matched random rate is the realized activation rate of the corresponding
gate method.

Optional controls:

- `no_risk_adjustment`
- `no_routing_intervention`

## Runner

Use the resumable runner:

```bash
python revision/experiments/run_toy_lob_n20_v2_sweep.py \
  --episodes 800 \
  --eval-episodes 50 \
  --seeds 90-109 \
  --jobs 6 \
  --gate-config revision/experiments/outputs/n_invariant_gate_diagnostic_v1/v2_gate_config.json \
  --output-dir revision/experiments/outputs/n20_v2_formal_sweep_v1
```

For controls:

```bash
python revision/experiments/run_toy_lob_n20_v2_sweep.py \
  --episodes 800 \
  --eval-episodes 50 \
  --seeds 90-109 \
  --jobs 6 \
  --include-controls \
  --gate-config revision/experiments/outputs/n_invariant_gate_diagnostic_v1/v2_gate_config.json \
  --output-dir revision/experiments/outputs/n20_v2_formal_sweep_v1
```

If interrupted, rerun the same command. Completed
`job_results/{method}_seed{seed}.json` files are skipped.

To aggregate existing partial results without launching new jobs:

```bash
python revision/experiments/run_toy_lob_n20_v2_sweep.py \
  --aggregate-only \
  --episodes 800 \
  --eval-episodes 50 \
  --seeds 90-109 \
  --output-dir revision/experiments/outputs/n20_v2_formal_sweep_v1
```

## Pass/Fail Questions

The formal N=20 stress test should answer:

- Does each v2 gate beat its own matched random baseline?
- Does any v2 gate improve over `gated_intervention` on N=20?
- Does a v2 gate approach or beat `spread_only_gate` raw gap while keeping a
  better reward / FPR trade-off?
- Does the result reduce dependence on raw global alignment?
- Do optional controls confirm that low raw gap is not caused by high-FPR
  no-risk-adjustment behavior?

If these pass, the supported claim is:

> N-invariant gate features improve larger-N generalization in the toy stress
> test.

If they do not pass, the correct limitation is:

> Toy validation supports N=5/N=10 and shows N=20 is a scale-specific
> representation challenge requiring stronger learned or N-invariant gates.
