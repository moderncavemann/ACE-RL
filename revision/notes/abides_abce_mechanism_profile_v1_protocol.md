# ABCE Frozen Mechanism Profile v1

Status: frozen, non-outcome profiling protocol.

## Purpose And Boundary

This profile supplements the completed ABCE evidence with implementation-cost
measurements. It does not train a policy, execute an outcome comparison, tune a
threshold, or add a post-protocol arm. The frozen ABCE primary artifact and its
inference remain unchanged.

The reported latency is a microbenchmark of the online control path, not a
complete ABIDES exchange-step latency and not a production deployment
benchmark. Absolute values are specific to the recorded Docker image, CPU
allocation, host architecture, and software versions. They support only a
bounded implementation-cost statement; they do not justify "negligible
overhead" or "deployment efficient."

## Frozen Setup

- Formal ABCE image: `ace-rl-abides-ace:abce-v1`, immutable image ID from the
  profile config.
- One Docker CPU, fixed `N=10`, and all declared BLAS/OpenMP thread limits set
  to one.
- The image is `linux/amd64`. On the recorded Apple Silicon workstation it is
  executed through Docker architecture emulation; this fact is retained in the
  output metadata.
- Each callable receives 10,000 warm-up calls and 100,000 measured calls.
  Timing samples are batches of 10 calls; median, p95, and p99 are computed over
  per-call batch times.
- Inputs are fixed public-state and quote arrays. No latent risk state is read.

## Measured Modules

| Label | Frozen callable | Complexity at fixed feature dimension |
| --- | --- | --- |
| M1 core | two-feature logistic `probability` | `O(d)`, `d=2` |
| M1 full | public state to announced proposal | `O(N + d)` |
| M2 | thresholded token-bucket pacing decision | `O(1)` |
| M3 soft | demand-conserving soft-rank allocation | `O(N)` |
| M3 hard | minimum quote plus eligible-best allocation | `O(N)` |
| End-to-end on | M1 full, M2, and soft-rank M3 | `O(N + d)` |
| Announced no-routing | M1 full and M2, followed by uniform allocation | `O(N + d)` |
| No-contract reference | uniform allocation construction | `O(N)` |

The end-to-end overhead is reported against both references. Announced
no-routing isolates the incremental allocation cost while retaining M1/M2 and
matches the control-path semantics of the paper's mechanism reference. The
minimal no-contract branch measures the full online contract cost against
uniform allocation only. Neither timing comparison is an outcome arm.

## Model And Resource Accounting

Parameter counts follow the exact compact ABCE response architecture:

- frozen base actor trunk and ten actor heads;
- centralized critic;
- ten linear intervention-adapter heads;
- response-trainable total, which includes both the actor adapter and critic.

Peak resident memory is the profiling process/container peak, not total host
RAM. Storage accounting uses actual managed-artifact bytes and compressed trace
bytes per recorded step for the primary and exact-learner audit bundles.

## Proxy And Economic Exports

The run also performs deterministic, non-training exports:

- recompute final and one-feature proxy AUCs from the previously frozen,
  disjoint 1,000-record training and 1,000-record validation sources;
- export the validation rows needed to recompute those AUCs;
- retain spread-only as an outcome comparator, not as a risk classifier;
- derive the aligned-versus-exact-random economic effect sizes directly from
  the frozen ABCE `statistics.json`.

Simulator labels are present only in the offline validation export. Runtime
M1 features are `public_volatility` and `absolute_lagged_return`; latent risk is
not a runtime input.

## Execution

Run from the repository root:

```bash
docker run --rm --platform linux/amd64 --cpus=1 \
  --entrypoint python \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e VECLIB_MAXIMUM_THREADS=1 \
  -e ACE_RL_PROFILE_IMAGE_ID=sha256:351dd23c02b950cd986004ec2695901fe379e561375cc16f2548b041a3bcbff3 \
  -e ACE_RL_PROFILE_WORKTREE_DIRTY=false \
  -v "$PWD:/workspace" -w /workspace \
  -e PYTHONPATH=/workspace/revision:/opt/ace-abides \
  ace-rl-abides-ace:abce-v1 \
  revision/experiments/profile_abides_abce_mechanism_v1.py \
  --config revision/configs/abides_abce_mechanism_profile_v1.json
```

The repository mount is writable only so the profiler can create its managed
artifact. It does not modify the frozen primary or learner artifacts.
