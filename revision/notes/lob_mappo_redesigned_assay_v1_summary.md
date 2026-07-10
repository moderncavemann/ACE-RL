# Local LOB/MAPPO Redesigned Assay v1 Summary

Date: 2026-07-10

This note records the first redesign attempt after diagnosing that the old
local LOB/MAPPO assay did not give risk state a real economic role. It is an
assay-redesign result, not a full LOB/MAPPO method validation.

## Redesign

The local LOB environment now has optional risk-coupled economics. Old defaults
remain unchanged for reproducibility; the redesigned assay is enabled by flags.

Key additions:

| Component | Change | Purpose |
| --- | --- | --- |
| Risk economics | `risk_coupled_economics`, `mark_to_fundamental`, `informed_flow_prob` | Make high-risk states create real adverse-selection cost instead of only an external volatility label. |
| Demand | `spread_elasticity` | Penalize high spreads through lower external order flow. |
| Targeted routing hygiene | `targeted_fill_pricing=competitive_benchmark` | Prevent targeted fills from subsidizing agents at their own high spread. |
| Inventory hygiene | `enforce_max_inventory` | Make `max_inventory` operational rather than only an observation scale. |
| Learner symmetry | `agent_id_feature` | Give the shared actor a minimal way to condition on agent identity. |
| Diagnostics | spread revenue, adverse-selection loss, inventory penalty, fundamental shock | Separate rent extraction from defensive inventory loss. |

The redesigned timing in `risk_coupled_economics` is:

1. agents quote around the current fundamental;
2. external flow arrives, with informed order side correlated with the upcoming
   fundamental shock;
3. fills are processed;
4. fundamental moves;
5. wealth is marked to fundamental.

This restores the intended link: high-risk states create larger adverse-selection
cost, while low-risk high spreads can represent markup rather than risk
compensation.

## Commands

Smoke:

```bash
python revision/experiments/diagnose_lob_mappo_assay.py \
  --episodes 2 --eval-episodes 1 --episode-length 20 --n-agents 10 --seeds 0 \
  --scenarios base --risk-coupled-economics --informed-flow-prob 0.3 \
  --spread-elasticity 0.1 --targeted-fill-pricing competitive_benchmark \
  --targeted-competitive-half-spread-ticks 2 --enforce-max-inventory \
  --agent-id-feature --device auto --hidden-dim 32 --update-epochs 1 \
  --minibatch-size 128 \
  --output-dir revision/experiments/outputs/lob_mappo_redesigned_assay_smoke
```

Main assay screens:

```bash
python revision/experiments/diagnose_lob_mappo_assay.py \
  --episodes 20 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 \
  --scenarios base --risk-coupled-economics --informed-flow-prob 0.3 \
  --spread-elasticity 0.1 --targeted-fill-pricing competitive_benchmark \
  --targeted-competitive-half-spread-ticks 2 --enforce-max-inventory \
  --agent-id-feature --device auto --hidden-dim 64 --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_redesigned_assay_screen_v1

python revision/experiments/diagnose_lob_mappo_assay.py \
  --episodes 50 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 \
  --scenarios base --risk-coupled-economics --inventory-penalty 0.0005 \
  --informed-flow-prob 0.5 --spread-elasticity 0.05 \
  --targeted-fill-pricing competitive_benchmark \
  --targeted-competitive-half-spread-ticks 2 --enforce-max-inventory \
  --agent-id-feature --device auto --hidden-dim 64 --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_redesigned_assay_candidate_a_50_v1

python revision/experiments/diagnose_lob_mappo_assay.py \
  --episodes 50 --eval-episodes 5 --episode-length 100 --n-agents 10 --seeds 0-4 \
  --scenarios base --risk-coupled-economics --inventory-penalty 0.0005 \
  --low-sigma 0.005 --high-sigma 0.08 --informed-flow-prob 0.7 \
  --spread-elasticity 0.05 --seed-liquidity-qty 0 \
  --targeted-fill-pricing competitive_benchmark \
  --targeted-competitive-half-spread-ticks 2 --enforce-max-inventory \
  --agent-id-feature --device auto --hidden-dim 64 --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_redesigned_assay_candidate_e_50_v1
```

Other screened outputs:

- `revision/experiments/outputs/lob_mappo_redesigned_assay_param_screen_v1/`
- `revision/experiments/outputs/lob_mappo_redesigned_assay_candidate_b_v1/`
- `revision/experiments/outputs/lob_mappo_redesigned_assay_candidate_c_50_v1/`
- `revision/experiments/outputs/lob_mappo_redesigned_assay_candidate_d_50_v1/`
- `revision/experiments/outputs/lob_mappo_redesigned_assay_candidate_f_50_v1/`

## Acceptance Criteria

This stage should pass before any oracle/gated intervention comparison:

1. no-intervention failure rate >= 0.70 on the screen seeds;
2. high-risk spread exceeds low-risk spread by roughly >= 0.30 tick;
3. high-spread regime has non-negative mean reward / spread rent, not only
   inventory-penalty avoidance;
4. action alignment is not so complete that timing and undercutting are
   mechanically unidentifiable;
5. the same candidate should not collapse when moving from 20 to 50 training
   episodes.

No screened candidate passed all criteria.

## Results

| Candidate | Episodes | Main change | Failure rate | Avg spread | Mean reward | High - low spread | Action alignment | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| screen base | 20 | p=0.3, elasticity=0.1, default inventory | 0.80 | 4.1139 | -0.0070 | 0.2428 | 0.9283 | Fails reward/risk split |
| inventory_low | 20 | lower inventory penalty | 0.40 | 2.7190 | -0.0009 | 0.2800 | 0.9384 | Fails failure rate |
| flow_low | 20 | lower base flow | 1.00 | 4.4753 | -0.0019 | 0.0709 | 0.9317 | Fails risk split/reward |
| candidate A | 20 | inv=0.0005, p=0.5, elasticity=0.05 | 0.60 | 3.2071 | 0.0018 | 0.5625 | 0.9174 | Promising but fails failure rate |
| candidate A50 | 50 | candidate A longer training | 0.60 | 2.7671 | 0.0009 | 0.4504 | 0.9726 | Fails failure rate/alignment |
| candidate C50 | 50 | candidate A + higher flow | 0.40 | 2.4001 | 0.0036 | 0.0004 | 0.9999 | Fails risk split/alignment |
| candidate D50 | 50 | candidate A + no seed liquidity | 0.60 | 3.0000 | 0.0044 | 0.0000 | 1.0000 | Fails risk split/alignment |
| candidate E50 | 50 | stronger risk differential | 0.40 | 3.0639 | -0.0041 | 0.7417 | 0.9265 | Fails failure rate/reward |
| candidate F50 | 50 | candidate E + lower inventory penalty | 0.20 | 2.4906 | 0.0007 | 0.6578 | 0.9514 | Fails failure rate |

## Decision

Do not launch full LOB/MAPPO main experiments.

The redesign fixed an important conceptual bug: risk can now enter trading
economics through informed flow and fundamental mark-to-market. It also fixed
the most obvious targeted-routing subsidy path by allowing competitive
benchmark pricing. However, the local MAPPO assay still fails acceptance:

1. candidates with stable high-spread failures tend to have weak risk-state
   alignment or negative reward;
2. candidates with clean high-risk-versus-low-risk spread differences do not
   produce a stable no-intervention high-spread failure mode;
3. action alignment remains very high, often near 1.0, so the shared actor still
   collapses into seed-level attractors.

The current bottleneck is no longer just the environment risk label. It is the
combined assay/learner setup: the shared MAPPO actor with mostly symmetric
agents still tends to learn seed-level constant-spread attractors rather than
state-contingent risk pricing and asymmetric undercutting.

## Next Step

The next redesign should be structural, not another parameter screen:

1. add one-hot agent identity or independent/per-agent actors, not only scalar
   agent ID;
2. make risk-state conditioning a supervised acceptance target in a passive
   market-maker sanity test before MAPPO training;
3. add a clean common-random-number/yoked random protocol only after the
   no-intervention assay passes;
4. only then calibrate a gate and run oracle/gated intervention comparisons.

Paper framing remains unchanged:

> Toy-scale validation supports the risk-aware gated-routing mechanism. The
> LOB/MAPPO extension is not yet a valid full-main experiment because the
> redesigned assay has not passed no-intervention acceptance.
