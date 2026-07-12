# ABIDES Isolated Runtime

This directory builds the archived modular J.P. Morgan ABIDES distribution in
an isolated Linux/Python 3.8 runtime. It does not modify the host Python 3.12
environment used by ACE-RL MAPPO and Apple MPS.

## Frozen Source

- Repository: `https://github.com/jpmorganchase/abides-jpmc-public.git`
- Commit: `f9cbe51342b7dedd9587e4e069040d68a5c6477f`
- Container platform: `linux/amd64`
- Python: `3.8.20`

The amd64 target is intentional. The archived `pomegranate==0.14.5` release
provides a compatible Linux x86-64 wheel, while a native Apple Silicon build is
not part of this protocol.

## Reproduction

Start Docker Desktop, then run:

```bash
revision/infra/abides/run_smoke.sh
```

The script:

1. builds the ABIDES Core/Markets/RMSC04 target;
2. runs a five-minute RMSC04 market with the official-scale 1,000 noise,
   102 value, 12 momentum, two market-maker, and one exchange agents;
3. builds the archived ABIDES-Gym dependency target;
4. resets and steps `markets-daily_investor-v0` once;
5. requires `pip check` to report no broken dependencies.

Every runtime check is fail-closed. The smoke scripts require the frozen source
commit and an actual `ExchangeAgent`; there is no local-LOB or synthetic
fallback path.

## Build Targets

| Target | Purpose |
| --- | --- |
| `abides-markets` | Core and Markets without pomegranate, Gym, or Ray |
| `abides-rmsc04` | Adds frozen pomegranate for the RMSC04 order-size model |
| `abides-gym` | Adds frozen Gym/Ray dependencies and ABIDES-Gym |

`requirements-gym.lock` records the fully resolved dependency set that passed
on 2026-07-12. In particular, `setuptools==57.5.0` is needed to build
`gym==0.18.0`, and `protobuf==3.20.3` is needed by Ray 1.7 generated protocol
descriptors.

## Known Upstream Boundary

The frozen main-branch `rmsc03.py` imports a `POVExecutionAgent` class that is
not present in the same source tree. RMSC03 is therefore excluded from the
passing smoke protocol; RMSC04 is the validated base configuration.

ABIDES-Gym main supports one Gym agent only. Passing this smoke proves the
official event kernel, exchange, market configuration, and Gym pause/step path
work. It does not provide the ten controlled market makers required by the
ACE-RL MAPPO experiment. That requires a separate multi-agent controller and
routing adapter.
