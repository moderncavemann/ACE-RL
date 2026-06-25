# Toy LOB N=20 Follow-up v1 Summary

Date: 2026-06-25

This targeted follow-up investigates the N=20 caveat from the robustness sweep.
It keeps the frozen N=10-calibrated gate threshold and does not retune the gate
on N=20 intervention outcomes.

Raw outputs are under `revision/experiments/outputs/` and are ignored by git.

## Longer Training Check

Configuration:

- Scenario: `n20`
- Frozen threshold: `risk_threshold = 0.2393662345171971`
- Seeds: `45-64`
- Training episodes: `1500`
- Evaluation episodes: `100`
- Jobs: `32`
- Runtime: `1911.71s` elapsed
- Methods: `no_intervention`, `always_on`, `threshold`,
  `gated_intervention`, `spread_only_gate`, scenario-matched `random_gate`

| Method | Collusion gap | Gate act. | FPR high-normal | TPR low-collusive | Mean reward | Total demand |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_intervention | 0.1869 | 0.0000 | 0.0000 | 0.0000 | 3.4184 | 54.5200 |
| always_on | -0.0368 | 1.0000 | 1.0000 | 1.0000 | 1.7282 | 65.1620 |
| threshold | 0.1267 | 0.3396 | 0.9075 | 0.5647 | 3.0220 | 56.7565 |
| random_gate | 0.1799 | 0.1335 | 0.1341 | 0.1342 | 3.1763 | 54.7852 |
| gated_intervention | 0.1604 | 0.1331 | 0.1304 | 0.8907 | 3.1479 | 55.6249 |
| spread_only_gate | 0.1380 | 0.1471 | 0.1321 | 1.0000 | 3.0698 | 56.5488 |

Paired matched-random comparison:

- `random_gate - gated_intervention` collusion-gap delta mean: `0.019470632`.
- Standard error: `0.006007418`.
- Positive seeds: `17 / 20`.

Interpretation:

- Longer training partially resolves the N=20 caveat. Gated now beats matched
  random by a small but consistent margin (`17/20` seeds positive), unlike the
  earlier short robustness run where the margin was near zero.
- The effect size remains much smaller than N=10/N=5. This is not yet a robust
  large-N success.
- `spread_only_gate` is the strongest raw-gap comparator in this N=20 follow-up.
  It has lower gap than full gated intervention, similar FPR, higher TPR, and
  slightly lower reward. This suggests the current multi-signal weights are not
  optimal for N=20.
- Threshold still has an unacceptable false-positive rate (`0.9075`), and
  always-on still over-intervenes with large reward loss.

## N=10 vs N=20 Distribution Diagnostics

Configuration:

- No-intervention diagnostic observations.
- Seeds: `45-54`.
- Training episodes: `800`.
- Evaluation episodes: `50`.
- Compared N=10 and N=20 under the same frozen gate threshold.

| Metric | N=10 mean | N=10 p90 | N=20 mean | N=20 p90 |
| --- | ---: | ---: | ---: | ---: |
| passive_risk_score | 0.2203 | 0.3260 | 0.1903 | 0.2769 |
| collusion_gap | 0.1867 | 0.3696 | 0.1719 | 0.3111 |
| alignment | 0.4030 | 0.6667 | 0.3058 | 0.6667 |
| demand_suppression | 0.1220 | 0.2609 | 0.1158 | 0.2317 |

Gated activation by risk state in the N=20 follow-up:

- Low risk: `0.1046`
- High risk: `0.1918`

Interpretation:

- N=20 has lower no-intervention collusion gap and passive risk score than N=10,
  so the available improvement margin is smaller.
- N=20 also has lower average alignment. The current gate score uses alignment
  as a positive signal, so this distribution shift can weaken the full
  multi-signal gate relative to spread-only.
- High-risk activation is higher than low-risk activation in aggregate. The
  FPR high-normal remains controlled, but this indicates that risk-state and
  feature distributions shift materially at N=20.

## Stage Decision

The N=20 follow-up should not be treated as a failure, but it remains a clear
large-N caveat. The next experiment should not be full LOB/MAPPO. It should be
a targeted N=20 representation/gate diagnostic:

- Compare full gate against `spread_only_gate` under longer horizons.
- Add N-invariant features such as fraction of agents near minimum spread,
  spread-distribution entropy, flow concentration, and top-k spread dispersion.
- Run N=20 post-hoc threshold diagnostics for analysis only, not as deployable
  evidence.
- Treat any 3000-episode N=20 run as a long follow-up, not as the next default
  interactive run.

## Output Artifacts

- Follow-up summary: `revision/experiments/outputs/n20_followup_1500_v1/summary.json`
- Follow-up metrics: `revision/experiments/outputs/n20_followup_1500_v1/metrics.csv`
- N=10 observations: `revision/experiments/outputs/observation_n10_diagnostic_v1/`
- N=20 observations: `revision/experiments/outputs/observation_n20_diagnostic_v1/`
- Diagnostic figures: `revision/experiments/outputs/n20_diagnostics_v1/`
