# ABIDES/MAPPO Full-Main Readiness v1

Status: ready to launch the frozen 20-seed main  
Date: 2026-07-12

## Decision

All preregistered readiness gates G0-G5 pass. The N=10 full-main configuration
is frozen at `revision/configs/abides_mappo_full_main_v1.json`, and its runner
has passed both single-worker and parallel-worker end-to-end smokes.

The launch command is:

```bash
python3 revision/experiments/run_abides_mappo_full_main.py
```

This command has not been run on seeds 340000-340019 yet. Those seeds remain
untouched held-out full-main seeds.

## Passed Gates

| Gate | Evidence |
| --- | --- |
| G0 dependencies | frozen ABIDES commit, RMSC04 and Gym smoke, clean `pip check` |
| G1 multi-agent integrity | 10 real traders, real resting orders/fills, cash and inventory conservation, deterministic seed replay |
| G2 routing integrity | fixed demand, dealer-directed real-order fills, gate-off equivalence, no synthetic cash/inventory mutation |
| G3 economics | high-risk widening +6.0 and wide-state routing gain +8.5085, both 10/10; competitive subsidy 0 |
| G4 oracle timing | shuffled-minus-oracle 0.06642, 95% CI [0.06378, 0.06905], 10/10 positive |
| passive MAPPO assay | mean half-spread 6.9455, high spread on 10/10 held-out episodes |
| gate calibration | validation F1 0.8623, activation 0.275, high-risk FPR 0.1486 |
| G5 screen | matched-minus-learned 0.10209 and circular-minus-learned 0.11094, both 5/5 positive |
| G5 holdout | matched-minus-learned 0.09466 and circular-minus-learned 0.10390, both 5/5 positive |

## Frozen Main

- Seeds: 340000-340019.
- Horizon: 100 decisions per seed and method.
- Methods: no intervention, learned gate, risk-stratified matched random,
  circular shuffled, spread-only threshold, always-on, no-routing, and oracle
  diagnostic.
- Primary endpoint: low-risk customer execution half-spread.
- Primary contrast: matched random minus learned gate.
- Inference: paired mean, standard error, t-based 95% interval, and
  positive/tie/negative seed count.
- Guardrails: activation, high-risk activation, reward, exact volume, and
  no-routing equivalence.

## Reproducible Artifacts

- ABIDES source:
  `f9cbe51342b7dedd9587e4e069040d68a5c6477f`.
- Passive MAPPO checkpoint:
  `revision/configs/abides_mappo_passive_v1.pt`.
- Checkpoint SHA-256:
  `12b6f7fc29e00aad385bea318205581b7c73fc07903595908c0cef095a53ce93`.
- Learned gate: `revision/configs/abides_learned_gate_v1.json`.
- Gate SHA-256:
  `f05d4c9f2ca16b8219f9786659ad076da81d4452e3d1cdd4a56ccae040c572cc`.
- Runtime controller: `revision/configs/abides_gate_runtime_v1.json`.

## Resource Plan

Full-main inference uses CPU, not MPS. The actor is small and CPU inference
avoids contention with the existing MPS training task. The runner starts 16
processes; each owns one persistent ABIDES Docker worker and processes all arms
for one seed before taking another seed. Docker exposes 24 CPUs and about 7.65
GiB, which is sufficient for the direct-kernel workers validated here. A
two-worker, eight-method smoke completed successfully; the earlier 16-worker
ABIDES resource benchmark also passed. Increasing Docker memory is not required
before launch.

## Claim Boundary

This is an actual ABIDES event kernel, `ExchangeAgent`, and price-time order
book with ten distinct MAPPO-controlled market makers and real execution
bookkeeping. The economic assay uses a controlled informed-customer-flow dealer
market; it is not the canonical full RMSC04 background-agent ecology.

The primary main is a frozen-policy timing experiment: the seller policy was
trained by MAPPO under no intervention, then held fixed across timing arms to
isolate the causal routing effect. It supports a learned timing claim in an
ABIDES/MAPPO environment. It does not by itself claim that separately retrained
MAPPO policies under every intervention converge to the same ordering; that is
a distinct adaptive-response extension.
