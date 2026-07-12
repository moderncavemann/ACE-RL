# ABIDES/MAPPO Checkpoint and Runtime v1

Status: verified before full-main launch  
Date: 2026-07-12

## Recovery Contract

The full-main runner writes one atomic checkpoint after all eight methods for a
seed complete. Each checkpoint contains the protocol fingerprint and complete
seed row. The same command resumes by loading valid checkpoints and scheduling
only missing seeds.

If the process, terminal, Docker worker, or machine stops:

- all completed seed checkpoints remain valid;
- the in-progress seed is rerun from its beginning;
- maximum recomputation is one seed, approximately 35 seconds under the frozen
  100-step configuration;
- a checkpoint from any different seed/method/horizon/artifact configuration is
  rejected by fingerprint validation;
- a live lock prevents two processes writing the same run directory.

The output directory contains `manifest.json`, `run_state.json`, `LOGBOOK.md`,
append-only `logbook.jsonl`, per-seed checkpoints, final `rows.json`, and final
`summary.json`.

## Recovery Smoke

Seed 330000, 40 steps, eight methods:

| Run | Wall time | Behavior |
| --- | ---: | --- |
| initial | 11.83 s | ran seed and wrote checkpoint |
| identical rerun | 1.37 s | loaded checkpoint; no seed worker started |

The recovered summary exactly matched the initial summary.

## Runtime Benchmarks

All benchmarks use already-consumed calibration/screen seeds, never full-main
seeds 340000-340019.

| Workload | Workers | Wall time | Mean seed runtime |
| --- | ---: | ---: | ---: |
| 1 seed x 100 steps x 8 methods | 1 | 34.06 s | 32.54 s |
| 4 seeds x 100 steps x 8 methods | 4 | 37.29 s | 33.76 s |
| 16 seeds x 100 steps x 8 methods | 16 | 41.55 s | 35.92 s |

The frozen 20-seed main will execute as 16 immediately scheduled workers plus
four queued seeds. Expected wall time is 75-90 seconds. Operational planning
should use two minutes normally and a conservative three-to-five-minute upper
bound if the host or Docker runtime is concurrently loaded.

## Resource Decision

Keep the full-main runner on CPU with 16 workers. The 16-worker benchmark passed
without memory pressure or integrity failures under Docker's current 24-CPU,
approximately 8-GB allocation. No Docker memory increase or MPS allocation is
needed for this frozen-policy inference run.
