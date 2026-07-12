# ABIDES/MAPPO Adaptive v3 Holdout v1

Status: passed; adaptive mechanism gate open  
Date: 2026-07-13

## Decision

The redesigned adaptive method passes its single frozen fresh-seed holdout.
After separate MAPPO adaptation, observable risk-aware announced routing lowers
low-risk customer execution half-spread relative to an exact-dose globally
shuffled schedule on all five policy seeds. The paired 95% confidence interval
is strictly above zero.

The supported statement is:

> In the controlled N=10 ABIDES dealer market, a passively trained risk model
> used to announce demand-conserving soft-rank routing before quotes improves
> low-risk execution quality over equal-dose random routing after separate
> MAPPO policy adaptation, without materially degrading reward or volume.

This supersedes the adaptive-v2 no-go for experiment planning. It does not
erase that negative result: the old post-action, severity-weighted contract
failed under its own frozen protocol.

## Frozen Holdout

| Item | Value |
| --- | --- |
| Base policy seeds | `480000-480004` |
| Base convergence seeds | `482000-482004` |
| Final evaluation seeds | `481000-481009` |
| Base training | no intervention, 216-264 episodes to convergence |
| Response training | 96 episodes per arm, GAE, updates every 8 episodes |
| Agents and horizon | 10 market makers, 100 decisions per episode |
| Routing | pre-action announced, demand-conserving soft rank, `kappa=4`, routable fraction `0.5` |
| Risk signal | passive logistic model using public volatility and absolute lagged return |
| Comparator | exact per-episode signal-strength multiset, globally permuted |
| Docker image | `ace-rl-abides-ace:adaptive-v3` (`sha256:38ea9d4d...`) |
| ABIDES commit | `f9cbe51342b7dedd9587e4e069040d68a5c6477f` |
| Holdout protocol fingerprint | `b86e8e55433a3b2b579d8bbf92ab382fc42517f3d17c2fdb0d0aeccf20c0fdf7` |

The protocol and its stricter positive-CI acceptance rule were committed as
`7385324` before the holdout output directory was created.

## Fresh Base Gate

All five no-intervention policies independently reproduced the high-spread
failure mode and passed the unchanged three-panel convergence rule.

| Policy seed | Episodes | Final half-spread | Spread range | Reward range fraction |
| ---: | ---: | ---: | ---: | ---: |
| 480000 | 240 | 6.2484 | 0.0870 | 0.0169 |
| 480001 | 264 | 6.4136 | 0.0960 | 0.0176 |
| 480002 | 240 | 6.5020 | 0.1336 | 0.0231 |
| 480003 | 216 | 5.9626 | 0.1306 | 0.0243 |
| 480004 | 240 | 6.2642 | 0.1380 | 0.0251 |

The base run started at 23:29:20 and completed at 23:51:14 AEST, about 21.9
minutes with five CPU workers. Every eight-episode batch has an atomic trainer,
optimizer, RNG, and history checkpoint.

## Holdout Results

| Method | Low-risk execution half-spread | Activation | High-risk activation | Mean reward | Volume |
| --- | ---: | ---: | ---: | ---: | ---: |
| Risk-aware | 5.98838 | 0.1990 | 0.0427 | 9.58452 | 20.0 |
| Exact-dose matched random | 6.07624 | 0.1990 | 0.1890 | 9.58673 | 20.0 |

The paired `matched random - risk-aware` difference is `0.08786` tick
(`SE=0.00336`, 95% CI `[0.07852, 0.09720]`). All five policy-seed differences
are positive: `0.08698`, `0.08936`, `0.08668`, `0.09866`, and `0.07760`.

Risk-aware reward is only `0.00221` below matched random, or about `0.02%`.
Both arms route exactly 20 units of customer demand. The risk-aware schedule
also cuts high-risk activation from `0.1890` to `0.0427` at the same total
activation rate, which confirms that the comparison changes timing rather than
dose.

## Integrity

Every frozen check passes:

- training and evaluation exogenous paths are identical across arms;
- action uniforms are common and signal-strength dose is exact per episode;
- all ten response policies start from the matching fresh base digest;
- all five policy pairs pass three fixed response-convergence panels;
- zero-signal actions exactly match each frozen base policy;
- activation, high-risk activation, reward, and volume guardrails pass;
- paired mean, `4/5` sign rule, and positive 95% CI rule all pass.

The response run started at 23:51:32 and completed at 00:07:07 AEST, about
15.6 minutes with five CPU workers. Checkpoints permit exact continuation after
an interruption; no continuation was needed in this holdout.

## What Changed

The successful result is structural, not a routing-parameter search:

1. The intervention is announced before seller quotes, so the policy can
   respond to the contract instead of being labeled after its action.
2. Soft-rank reallocation conserves demand and rewards relative price
   competitiveness; it does not pay a common high spread as a targeted-fill
   subsidy.
3. MAPPO uses complete-episode GAE updates and a frozen base trunk with a
   zero-initialized intervention adapter.
4. The post-action spread-severity module was removed after its preregistered
   development ablation failed. The final gate uses only exogenous risk
   inference and pacing.
5. The comparator preserves the exact intervention strengths and common
   market path, changing only their risk contingency.

## Claim Boundary And Next Stage

This result validates adaptive risk-aware timing against matched random in one
controlled ABIDES ecology. It is not yet a completed adaptive paper main: the
current holdout has two methods and five independent policy seeds. The next
allowed stage is a frozen larger-seed package with governance controls
(`no_intervention`, threshold, always-on), a no-routing causal ablation, and a
second market ecology. Those runs must not change the validated risk model,
routing contract, response budget, or primary endpoint.

Raw outputs remain ignored under
`revision/experiments/outputs/abides_mappo_adaptive_v3_*`. The frozen configs,
protocol, development summary, and this holdout summary are repo-visible.
