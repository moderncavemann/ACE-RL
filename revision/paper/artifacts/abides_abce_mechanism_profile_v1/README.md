# ABCE Frozen Mechanism Profile v1

This is a non-outcome, single-thread microbenchmark of the ABCE online control
path in the frozen formal Docker image. It is not full ABIDES step latency.

## Latency

| Module | Median us | p95 us | p99 us |
| --- | ---: | ---: | ---: |
| M1 logistic core | 5.2084 | 5.9752 | 7.0626 |
| M1 state to proposal | 149.3458 | 159.7964 | 173.9089 |
| M2 pacing | 9.3875 | 10.5333 | 11.3125 |
| M3 soft-rank | 56.0917 | 58.6252 | 63.8668 |
| M3 hard-best | 41.0917 | 43.7919 | 46.8885 |
| Contract end-to-end | 230.4542 | 244.7670 | 280.5028 |
| Announced no-routing end-to-end | 161.2791 | 176.6875 | 207.5480 |
| No-contract allocation | 1.5625 | 1.5961 | 2.0960 |


Contract-on minus announced no-routing median overhead is
`69.1751` microseconds
(`1.43x`). Contract-on minus the minimal
no-contract allocation branch is
`228.8917` microseconds
(`147.49x`). These values are specific to the recorded
emulated `linux/amd64` container and do not justify a production-efficiency
claim.

## Parameters And Memory

- base actor-critic: `14919` parameters;
- actor adapter: `560` parameters;
- response-trainable including critic: `6257` parameters;
- profiler process peak RSS: `67973120` bytes.

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
