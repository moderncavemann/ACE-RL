# ABIDES Ecology Holdout v1 Protocol

Status: frozen before training  
Date: 2026-07-14

The holdout tests whether the v5 timing effect depends on one risk-persistence
or informed-flow setting. Each cell retrains base and response policies from
scratch. Existing v5 checkpoints are not zero-shot robustness evidence.

## Cells

| Cell | Policy seeds | Change from v5 |
| --- | --- | --- |
| E1 fast transition | `560000-560009` | `p_low_to_high=0.08`, `p_high_to_low=0.20`; stationary high-risk prevalence remains unchanged. |
| E2 weaker informed flow | `570000-570009` | `informed_flow_probability=0.60`. |

Each cell has disjoint five-path base evaluation and ten-path response
evaluation namespaces defined in the JSON config. Base training is 384
episodes and response training is 96 episodes, both with 100 decisions per
episode. Minimum methods are no intervention, risk-aware, matched random, and
no routing. Budget-matched spread-only and Brero arms are included only if
their P0 implementations have passed before ecology execution begins.

Both cells are always reported. Two favorable cells support robustness within
the tested ecologies; one favorable cell defines a boundary; two failures
restrict the claim to the original v5 ecology. Results cannot be pooled to
hide a cell-level reversal.

## Audit Replay

The formal response runner's seed result stores the pre-registered aggregate
and convergence checks. Audit-grade per-path and per-step data are generated
afterward by loading the frozen final checkpoints and replaying exactly the
same ten evaluation paths without training. The audit evaluator must reproduce
each seed's formal aggregate before accepting its records. It uses all six
methods, all ten policy seeds in both cells, deterministic gzip output, and a
source fingerprint that includes the frozen config, formal manifest, summary,
and every seed-result hash. This replay cannot change policies, paths, methods,
or the formal efficacy result.

The compact response runner exposes realized `flow_shares` rather than the
domain-runner `routing_weights` field. The audit schema preserves those shares
and derives total variation relative to uniform flow, marking every affected
step with `routing_weights_source=realized_flow_shares`. It does not represent
the derived value as a latent platform probability. The initial v1 audit replay
failed closed on this schema difference before writing any accepted task; v2
adds the explicit adapter and retains that failed launch in the logbook.
