# Local LOB/MAPPO Routing Grid v1 Summary

Date: 2026-07-03

This is a targeted routing-design diagnostic after the MAPPO activation-control
pilot. It is not a full LOB/MAPPO main experiment and should not be cited as
final method validation.

## Scope

- Runner: `revision/experiments/run_lob_mappo_routing_grid.py`
- Backend: `revision/src/envs/local_lob_market_env.py`
- Learner: `revision/src/agents/mappo.py`
- Gate: `gated_intervention_controlled`
- Device: `mps` via `--device auto`

The grid compares activation-controlled gated routing against a matched random
gate under different routing parameters.

## Grid Screen

Output:

- `revision/experiments/outputs/lob_mappo_routing_grid_v1/`

Configuration:

- Seeds: `0-2`
- Train episodes per method/seed: `10`
- Eval episodes per method/seed: `3`
- Episode length: `100`
- `routing_rho`: `0.25, 0.75`
- `routing_spread_cap_ticks`: `4, 8, 10`
- `routing_withhold_rate`: `0.0, 0.5`

Top screen rows:

| rho | cap | withhold | Gate act. | Gated spread | Random spread | Random - gated | Reward delta | Active cap-met | Candidate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | 10 | 0.50 | 0.1189 | 3.0191 | 3.6418 | 0.6227 | -0.0020 | 0.8411 | yes |
| 0.75 | 4 | 0.00 | 0.1344 | 3.1422 | 3.7622 | 0.6200 | -0.0036 | 0.1074 | yes |
| 0.25 | 4 | 0.50 | 0.2411 | 3.5772 | 4.5386 | 0.9613 | -0.0143 | 0.1889 | no, reward cost |
| 0.25 | 8 | 0.00 | 0.1267 | 3.0512 | 3.8188 | 0.7676 | -0.0088 | 0.5263 | no, reward cost |
| 0.25 | 10 | 0.00 | 0.1211 | 3.0329 | 3.6463 | 0.6134 | -0.0065 | 0.8165 | no, reward cost |

Screen interpretation:

- The broad-cap, low-rho settings reduce spread in the 10/3 screen, but several
  do so with nontrivial reward cost.
- The `rho=0.75, cap=4, withhold=0` row is different: it preserves the original
  competitive spread cap but removes cap-fail withholding. This tests whether
  the earlier failure came from negative flow punishment rather than the gate
  timing.

Because this screen uses only 3 seeds and short training, both candidate rows
were rerun at the same size as the previous controlled pilot.

## Candidate Checks

### Candidate A: `rho=0.25, cap=10, withhold=0.5`

Output:

- `revision/experiments/outputs/lob_mappo_routing_grid_candidate_rho025_cap10_withhold05_v1/`

Configuration:

- Seeds: `0-4`
- Train episodes per method/seed: `20`
- Eval episodes per method/seed: `5`

| Metric | Value |
| --- | ---: |
| gated activation | 0.1508 |
| gated spread | 4.1532 |
| matched random spread | 2.4544 |
| random - gated spread | -1.6988 |
| paired positive seeds | 2 / 5 |
| gated reward | -0.012003 |
| matched random reward | -0.017081 |
| gated - random reward | 0.005078 |
| active cap-met share | 0.5252 |

Decision: failed. The 10/3 screen did not survive the 20/5 candidate check.
This setting improves reward relative to random but badly worsens spread.

### Candidate B: `rho=0.75, cap=4, withhold=0.0`

Output:

- `revision/experiments/outputs/lob_mappo_routing_grid_candidate_rho075_cap4_withhold00_v1/`

Configuration:

- Seeds: `0-4`
- Train episodes per method/seed: `20`
- Eval episodes per method/seed: `5`

| Metric | Value |
| --- | ---: |
| gated activation | 0.1060 |
| gated spread | 2.9912 |
| matched random spread | 3.6692 |
| random - gated spread | 0.6780 |
| paired positive seeds | 5 / 5 |
| paired SE | 0.1501 |
| gated reward | -0.010573 |
| matched random reward | -0.013310 |
| gated - random reward | 0.002737 |
| gated volume | 3.7828 |
| matched random volume | 3.8804 |
| active cap-met share | 0.1245 |
| active cap-not-met share | 0.8755 |
| active routing multiplier | 1.1167 |

Decision: passed as a pilot routing candidate.

This setting keeps the strict competitive spread cap (`4`) but removes cap-fail
withholding. Cap-met windows still receive flow boost; cap-not-met active
windows no longer receive a flow penalty. That converts the intervention from
mostly negative flow punishment into asymmetric boost-only routing.

## Interpretation

The previous controlled pilot failed partly because active windows were mostly
cap failures and therefore received reduced order flow. In this grid, simply
removing cap-fail withholding while keeping the competitive boost channel
produced a favorable matched-random comparison in the 20/5 candidate check.

Current claim boundary:

> A small MAPPO-local-LOB routing grid identified boost-only routing
> (`rho=0.75, cap=4, withhold=0`) as a viable pilot candidate.

Do not write:

> The full LOB/MAPPO method is validated.

## Next Step

Do not launch the full main experiment yet. First run a controlled MAPPO pilot
with this routing setting and the normal baselines:

- `no_intervention`
- `threshold`
- `always_on`
- `random_gate`
- `gated_intervention_controlled`

Use at least the previous pilot size (`20` train episodes, `5` eval episodes,
seeds `0-4`). If that repeats the favorable matched-random comparison and keeps
activation in range, then move to a longer pilot (`50` train episodes, `10` eval
episodes) before deciding on full main.

`run_lob_mappo_pilot.py` now accepts `--routing-rho`,
`--routing-spread-cap-ticks`, and `--routing-withhold-rate` for this follow-up.

## Reproduction Commands

Grid screen:

```bash
python revision/experiments/run_lob_mappo_routing_grid.py \
  --episodes 10 \
  --eval-episodes 3 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-2 \
  --routing-rhos 0.25,0.75 \
  --routing-spread-caps 4,8,10 \
  --routing-withhold-rates 0.0,0.5 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_routing_grid_v1
```

Candidate A check:

```bash
python revision/experiments/run_lob_mappo_routing_grid.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --routing-rhos 0.25 \
  --routing-spread-caps 10 \
  --routing-withhold-rates 0.5 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_routing_grid_candidate_rho025_cap10_withhold05_v1
```

Candidate B check:

```bash
python revision/experiments/run_lob_mappo_routing_grid.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --routing-rhos 0.75 \
  --routing-spread-caps 4 \
  --routing-withhold-rates 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output-dir revision/experiments/outputs/lob_mappo_routing_grid_candidate_rho075_cap4_withhold00_v1
```

Candidate normal-baseline pilot:

```bash
python revision/experiments/run_lob_mappo_pilot.py \
  --episodes 20 \
  --eval-episodes 5 \
  --episode-length 100 \
  --n-agents 10 \
  --seeds 0-4 \
  --methods no_intervention,always_on,threshold,gated_intervention_controlled \
  --routing-rho 0.75 \
  --routing-spread-cap-ticks 4 \
  --routing-withhold-rate 0.0 \
  --gate-config revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json \
  --device auto \
  --hidden-dim 64 \
  --update-epochs 4 \
  --minibatch-size 512 \
  --output revision/experiments/outputs/lob_mappo_candidate_boost_only_pilot_v1/summary.json \
  --metrics-csv revision/experiments/outputs/lob_mappo_candidate_boost_only_pilot_v1/metrics.csv
```
