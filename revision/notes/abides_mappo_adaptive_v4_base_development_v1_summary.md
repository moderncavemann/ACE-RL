# ABIDES/MAPPO Adaptive v4 Base Development Summary

Status: passed  
Date: 2026-07-13

## Decision

The independently frozen fixed-horizon base protocol passed. All twenty
no-intervention policies trained for exactly 384 episodes; no policy was
stopped, retained, or removed based on its intermediate learning curve.

This result permits a fresh formal-primary base run. It does not test any
intervention method and provides no learned-gate efficacy evidence.

## Frozen Study

| Item | Value |
| --- | --- |
| Policy seeds | `510000-510019` |
| Fixed evaluation paths | `511000-511004` |
| Training horizon | 384 episodes for every policy |
| Terminal panels | episodes 336, 360, 384 |
| Runtime | 2026-07-13 03:38:14 to 04:49:48 UTC |
| Git commit | `e6383974ed15e0a103837140853d2419a870346e` |
| Protocol fingerprint | `b8158b63f2fc5a119451c47a260b34876562906026fded5a37e5c08c6396fb40` |
| Docker image | `sha256:38ea9d4ddae027e040ea0386f31d2f49bb14946ddc6ae0332ad384efde981a68` |

## Acceptance Results

| Frozen check | Result |
| --- | --- |
| All policies reached episode 384 | 20/20, pass |
| High-spread failure mode retained | 20/20, pass |
| Spread range at most 0.20 | 20/20, pass; required at least 18/20 |
| Legacy spread range at most 0.15 | 20/20, diagnostic only |
| Reward range fraction at most 0.10 | 20/20, pass |
| Population mean spread range at most 0.10 | 0.03581, pass |

The population mean spreads at episodes 336, 360, and 384 were 6.48791,
6.50508, and 6.52372. The largest per-policy terminal spread range was
0.09460 (seed `510009`), and the largest terminal reward range fraction was
0.01721 (also seed `510009`). All twenty model digests were distinct.

## Interpretation

At the old 288-episode maximum, only 17/20 fresh policies satisfied the old
per-seed rolling-range diagnostic. By episode 336 all twenty did. Earlier in
training, individual seeds also switched repeatedly between stable and
unstable. This confirms that the old early-stop rule could select policies by
convergence speed. Fixed-horizon training removes that selection channel while
preserving the assay's high-spread failure mode.

The next stage must use new policy and evaluation seeds. Development policies
cannot enter the response comparison, and the terminal criteria cannot be
retuned from these outcomes.
