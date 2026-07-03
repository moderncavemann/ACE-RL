# Local LOB/MAPPO Conditional Routing Reallocation v1 Summary

Date: 2026-07-03

This note records the follow-up after the boost-only MAPPO pilot failed to
preserve a timing advantage at 50/10. The goal was to test whether a sharper
order-flow routing channel can make risk-aware timing matter before launching a
full LOB/MAPPO main experiment.

This is not full LOB/MAPPO validation.

## Code Changes

The local LOB environment now supports conditional routing modes:

- `eligible_priority`: extra routed flow goes to agents quoting within the
  absolute spread cap.
- `relative_priority`: extra routed flow goes to agents quoting below the
  current average half-spread.
- `relative_reallocate`: existing flow is reallocated toward agents quoting
  below the current average half-spread, without increasing expected total
  flow when eligible agents exist.

The MAPPO pilot, routing grid, and drift diagnostic now expose:

- `--routing-mode`
- `--routing-relative-improvement-ticks`
- `routing_targeted_volume`
- `routing_targeted_orders`
- `routing_eligible_agents`

## Absolute Eligible Priority

Output:

- `revision/experiments/outputs/lob_mappo_eligible_priority_grid_20_v1/`

The 20/5 absolute-cap priority grid did not produce a usable candidate. Some
cells showed gated below matched random, but the best apparent wins were not
driven by meaningful targeted exposure. When the cap was widened enough to
produce targeted flow, matched random usually caught up or beat gated timing.

Decision: do not use `eligible_priority` for the full pilot.

## Relative Priority

Outputs:

- `revision/experiments/outputs/lob_mappo_relative_priority_grid_20_v2/`
- `revision/experiments/outputs/lob_mappo_relative_priority_grid_20_strong_v1/`
- `revision/experiments/outputs/lob_mappo_relative_priority_grid_20_rel0_v1/`
- `revision/experiments/outputs/lob_mappo_relative_priority_oracle_20_v1/`

The first relative-priority implementation initially fell through to the old
book-multiplier path. That branch bug was fixed before interpreting the
relative-priority results.

After the fix, relative-priority routed real targeted flow, but it did not meet
the go criteria:

- best 20/5 cells still had gated spread around `4.0`;
- stronger `rho=1.5` and `rho=2.0` did not solve the issue;
- `rel=0.0` produced targeted flow but unstable seed outcomes;
- oracle timing under relative-priority was weak at 20/5: paired
  random-minus-oracle spread delta `0.2054`, only `2/5` seeds positive, and
  oracle reward worse than no-intervention.

Decision: do not use `relative_priority` for the full pilot.

## Relative Reallocation

Outputs:

- 20/5 grid:
  `revision/experiments/outputs/lob_mappo_relative_reallocate_grid_20_v1/`
- 50/10 candidate, `rho=0.75`:
  `revision/experiments/outputs/lob_mappo_relative_reallocate_candidate_50_v1/`
- 50/10 candidate, `rho=0.5`, seeds 0-4:
  `revision/experiments/outputs/lob_mappo_relative_reallocate_rho05_50_v1/`
- 50/10 candidate, `rho=0.5`, seeds 5-9:
  `revision/experiments/outputs/lob_mappo_relative_reallocate_rho05_50_seeds5_9_v1/`

The 20/5 reallocation grid found the strongest short-horizon candidate:

| Setting | Gated spread | Random spread | Delta random - gated | Positive seeds | Reward delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rho=0.75`, `rel=0.0` | 2.5245 | 4.8949 | 2.3704 | 4/5 | -0.0105 |
| `rho=0.5`, `rel=0.0` | 3.7854 | 4.4668 | 0.6814 | 4/5 | 0.0003 |

The `rho=0.75` candidate did not preserve robustness at 50/10:

| Setting | Gated spread | Random spread | Delta random - gated | Positive seeds |
| --- | ---: | ---: | ---: | ---: |
| `rho=0.75`, seeds 0-4 | 3.8563 | 4.4038 | 0.5475 | 3/5 |

The more conservative `rho=0.5` candidate was better on seeds 0-4 but failed on
held-out seeds 5-9:

| Seed batch | No-intervention spread | Gated spread | Random spread | Delta random - gated | Positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-4 | 3.6052 | 2.9686 | 3.7460 | 0.7775 | 4/5 |
| 5-9 | 2.1124 | 3.0766 | 2.5527 | -0.5239 | 1/5 |
| 0-9 combined | 2.8588 | 3.2431 | 3.3699 | 0.1268 | 5/10 |

The combined 10-seed result is not enough to proceed to full LOB/MAPPO main
experiments:

- paired random-minus-gated delta mean is only `0.1268`;
- SE is `0.4677`;
- only `5/10` seeds are positive;
- no-intervention has clear spread elevation in only `4/10` seeds;
- gated mean spread is above no-intervention in the combined sample.

## Decision

Do not launch full LOB/MAPPO main experiments from the current local LOB/MAPPO
pilot.

The conditional routing work found one promising short-horizon reallocation
candidate, but it did not survive a 10-seed 50/10 pilot. The main blocker is no
longer just the routing parameterization. The local LOB/MAPPO setup itself does
not yet provide a stable enough no-intervention failure mode across held-out
seeds, and matched random remains competitive once the seed set is expanded.

Current stage status:

> Toy-scale validation passed. Local LOB/MAPPO pilot remains blocked. Do not
> claim full LOB/MAPPO effectiveness.

## Recommended Next Work

Stop small routing-parameter grids for now. The next useful step is to stabilize
the local LOB/MAPPO assay before another full-main gate:

1. diagnose why no-intervention spread elevation is absent in many seeds;
2. test whether longer training, lower exploration noise, or environment
   parameters produce a stable failure mode without intervention;
3. only after the no-intervention failure mode is stable, rerun
   `relative_reallocate rho=0.5` against matched random on 10-20 seeds;
4. proceed to full main only if gated beats matched random on most seeds and
   no-intervention remains elevated.

