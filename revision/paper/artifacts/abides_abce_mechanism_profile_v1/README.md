# ABCE Frozen Mechanism Profile v1

This is a non-outcome, single-thread microbenchmark of the ABCE online control
path in the frozen formal Docker image. It is not full ABIDES step latency.

## Latency

| Module | Median us | p95 us | p99 us |
| --- | ---: | ---: | ---: |
| M1 logistic core | 5.3625 | 6.5791 | 7.4793 |
| M1 state to proposal | 154.5354 | 177.2627 | 209.8043 |
| M2 pacing | 9.6875 | 10.7713 | 12.0294 |
| M3 soft-rank | 59.1042 | 63.6557 | 71.1047 |
| M3 hard-best | 43.8417 | 50.4000 | 63.7211 |
| Contract end-to-end | 247.8979 | 294.1260 | 333.7869 |
| Announced no-routing end-to-end | 169.8000 | 196.7385 | 226.4569 |
| No-contract allocation | 1.5792 | 1.7625 | 2.2335 |


Contract-on minus announced no-routing median overhead is
`78.0979` microseconds
(`1.46x`). Contract-on minus the minimal
no-contract allocation branch is
`246.3187` microseconds
(`156.98x`). These values are specific to the recorded
emulated `linux/amd64` container and do not justify a production-efficiency
claim.

## Parameters And Memory

- base actor-critic: `14919` parameters;
- actor adapter: `560` parameters;
- response-trainable including critic: `6257` parameters;
- profiler process peak RSS: `68091904` bytes.

## Existing Evidence Export

- low-risk improvement over exact random: `0.03996` ticks
  (`0.630%`);
- low-risk customer cost saved: `47.92` tick-units/episode;
- high-risk announced exposure reduction: `51.6%`;
- aggregate dealer reward difference: `39.05`;
- inventory SD reduction: `5.6%`;
- all-state spread difference (aligned minus random): `0.00907` ticks, so the result remains selective.

The frozen two-feature public-risk proxy has validation AUC
`0.8284` on a disjoint 1,000/1,000
passive split. Runtime uses only public volatility and absolute lagged return;
the simulator risk label is offline-only.
