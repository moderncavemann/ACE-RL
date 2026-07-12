# ABIDES Dependency Smoke v1 Summary

Date: 2026-07-12

## Decision

The isolated ABIDES dependency and single-agent Gym smoke gate passes.

This establishes that the official modular ABIDES event kernel, RMSC04 market,
ExchangeAgent, legacy pomegranate order-size model, and ABIDES-Gym pause/step
path can run on this M3 Ultra host through a pinned `linux/amd64` container.

It does not establish an ABIDES/MAPPO result. The ten-agent controller,
platform routing contract, ACE-RL observations/rewards, and learned gate have
not yet been connected to ABIDES.

## Frozen Runtime

| Item | Value |
| --- | --- |
| ABIDES repository | `jpmorganchase/abides-jpmc-public` |
| Commit | `f9cbe51342b7dedd9587e4e069040d68a5c6477f` |
| Base image | `python:3.8.20-slim-bullseye` |
| Platform | `linux/amd64` under Docker Desktop |
| RMSC04 image size | about 806 MB |
| Gym image size | about 1.11 GB |
| Dependency validation | `pip check`: no broken requirements |

Critical legacy pins include:

- `numpy==1.22.0`;
- `pandas==1.2.4`;
- `scipy==1.10.0`;
- `pomegranate==0.14.5`;
- `gym==0.18.0`;
- `ray==1.7.0`;
- `protobuf==3.20.3`.

## RMSC04 Result

The official-scale smoke used a five-minute 09:30-09:35 market window, seed 7,
1,000 noise agents, 102 value agents, 12 momentum agents, two adaptive market
makers, and one exchange.

| Metric | Result |
| --- | ---: |
| Backend | `abides` |
| Total agents | 1,117 |
| Order-history events | 582 |
| Final bid levels | 19 |
| Final ask levels | 28 |
| Final last trade, cents | 99,986 |
| Wall time under amd64 emulation | 1.18 seconds |

A smaller 30-agent RMSC04 run was repeated twice with the same seed. Both runs
produced exactly 279 order-history events, 10 bid levels, 12 ask levels, and a
last trade of 99,974, confirming deterministic smoke behavior.

## Gym Result

`markets-daily_investor-v0` successfully:

- imported ABIDES-Gym and Ray;
- initialized an RMSC04 background market;
- reset to a `(7, 1)` observation;
- executed one action;
- returned the next `(7, 1)` observation without terminating.

The backend path contains no fallback to the repository's local LOB.

## Diagnosed Compatibility Failures

Three failures were isolated during construction:

1. RMSC03 imports `POVExecutionAgent`, which is absent from the frozen source
   tree. RMSC03 remains an upstream source/config mismatch and is not used.
2. Gym 0.18 metadata fails under modern setuptools. The Gym image pins
   `setuptools==57.5.0` and `wheel==0.37.1`.
3. Ray 1.7 fails with protobuf 5 generated-descriptor checks. The image pins
   `protobuf==3.20.3`.

The prior `GeneralMixtureModel` failure is resolved by the isolated
`pomegranate==0.14.5` wheel; import and RMSC04 execution both pass.

## Remaining Gate

The dependency gate is open, but the multi-agent environment gate is not.
The frozen ABIDES main branch explicitly permits at most one Gym agent, and its
provided Gym environments are single-agent investor/execution tasks.

The next implementation must add a fail-closed ten-market-maker controller
that:

1. maps all ten MAPPO actions to distinct ABIDES agents;
2. returns per-agent observations, inventories, fills, cash, and rewards;
3. pauses the event kernel at synchronized decision times;
4. implements demand-conserving platform routing without synthetic fills;
5. preserves seed and external-flow alignment across intervention arms.

Only after fixed-policy economics and oracle timing pass on that adapter should
the learned gate or a full ABIDES seed sweep begin.

## Reproduction

```bash
revision/infra/abides/run_smoke.sh
```

See `revision/infra/abides/README.md` for target-level commands and boundaries.
