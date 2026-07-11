# Local LOB/MAPPO Learned-Gate Full-Main v1 Summary

Date: 2026-07-11

This note records the frozen 20-seed full-main run specified in
`lob_mappo_learned_gate_full_main_protocol.md`. No gate, routing, learner,
calibration, comparator, or acceptance parameter changed after launch.

## Decision

The full-main experiment passes every registered protocol, primary-effect,
market-quality, comparator, and trained-ablation check.

The supported claim is:

> In the redesigned risk-coupled local synthetic LOB/MAPPO response assay,
> frozen learned oracle-distillation timing reduces low-risk spread relative
> to exact-dose matched-random and circular-shifted timing. Routing training
> and the high-spread-dependent pool both materially strengthen the effect.

The claim remains bounded to this simulator and its observable true risk
regime. It is not evidence for a general latent-collusion detector, ABIDES,
real order books, or scale generalization.

## Execution

| Item | Frozen value |
| --- | --- |
| Seeds | 60-79 |
| Agents | 10 |
| Base training | 100 episodes per seed |
| Response training | 100 episodes for each of three adapters per seed |
| Evaluation | 10 stochastic CRN episodes per seed |
| Gate calibration | 5 no-intervention episodes, q75/q90 |
| Neural backend | Apple MPS |
| Environment/statistics | CPU |
| Protocol commit | `4f3e966` |
| Protocol fingerprint | `8c5ce9cef8a6c5098bf6f7d43b4d239cf4c4d4089962da6e1774abecd595151f` |
| Gate artifact SHA-256 | `b0aca84ae1c204bc6562ac5b37e6f3b5b8c5f96f5bbf01391f7315b99e783851` |
| Wall time | 2.77 hours, about 8.31 minutes per seed |

All 20 seed checkpoints, the final summary, the run manifest, and the frozen
artifact hashes agree. Resume replay loaded all 20 checkpoints without
retraining and reproduced the same aggregate.

## Primary Result

The primary endpoint is paired `matched_random - learned_gate` low-risk mean
half-spread. Positive values favor learned timing.

| Statistic | Result |
| --- | ---: |
| Mean delta | 0.09998 |
| Standard error | 0.01204 |
| 95% t interval | [0.07478, 0.12519] |
| Positive / tie / negative seeds | 19 / 1 / 0 |
| Exact two-sided sign-test p, excluding tie | 0.00000381 |

The smallest nonzero paired gains are about `0.00295` and `0.00331`; the
single tie is seed 73. No seed favors matched random.

The learned gate also improves over no intervention:

| Statistic | Result |
| --- | ---: |
| `base - learned` low-risk spread | 0.14792 +/- 0.01915 |
| 95% t interval | [0.10784, 0.18800] |
| Positive / tie / negative seeds | 19 / 1 / 0 |

No-intervention low-risk spread exceeds 4.0 in `20/20` seeds, so the target
high-spread failure mode is present throughout the full-main sample.

## Gate Quality

| Metric | Result |
| --- | ---: |
| Activation rate | 0.1030 |
| Target prevalence | 0.10395 |
| Precision | 1.0000 |
| Recall | 0.99086 |
| Negative-state FPR | 0.0000 |
| High-risk FPR | 0.0000 |
| Effective alpha, learned | 0.06987 |
| Effective alpha, matched random | 0.02243 |
| Counterfactual expected response | 0.47227 +/- 0.07205 |
| Positive response seeds | 20/20 |

Learned and oracle results remain nearly identical. Their low-risk spread
difference is `-0.00002 +/- 0.00025`, consistent with the gate's explicit
oracle-distillation target.

## Market Outcomes

| Method | Low-risk spread | High-risk spread | Reward | Volume | Activation | Effective alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no intervention | 5.2275 | 5.6392 | 0.002571 | 1.1652 | 0.000 | 0.0000 |
| learned gate | 5.0796 | 5.6388 | 0.002414 | 1.1984 | 0.103 | 0.0699 |
| matched random | 5.1796 | 5.6389 | 0.002459 | 1.1739 | 0.103 | 0.0224 |
| circular shuffled | 5.1826 | 5.6390 | 0.002574 | 1.1750 | 0.103 | 0.0199 |
| oracle diagnostic | 5.0797 | 5.6388 | 0.002402 | 1.1989 | 0.104 | 0.0698 |
| fixed-pool threshold | 4.8391 | 5.6388 | 0.001996 | 1.2606 | 0.334 | 0.3340 |
| fixed-pool always-on | 3.4507 | 5.6333 | 0.001462 | 1.5914 | 1.000 | 1.0000 |
| no-routing trained | 5.2702 | 5.6392 | 0.002530 | 1.1558 | 0.103 | 0.0000 |
| fixed-pool learned | 4.8416 | 5.6382 | 0.002206 | 1.2598 | 0.103 | 0.1030 |
| fixed-pool matched | 4.8546 | 5.6381 | 0.002325 | 1.2528 | 0.103 | 0.1030 |

Relative to matched random, learned timing increases volume by `0.02455`
with 95% t interval `[0.01698, 0.03212]`. Its reward is lower by only
`0.0000446` on average; the paired 95% interval
`[-0.000150, 0.000239]` includes zero. The matched-minus-learned high-risk
spread difference is `0.00013`, with interval `[-0.00029, 0.00055]`.

The correct interpretation remains a market-quality trade-off, not a claim of
costless intervention.

## Timing Comparator

The risk-stratified circular comparator preserves exact dose and cyclic
within-risk activation structure while shifting state alignment.

| Statistic | Circular - learned |
| --- | ---: |
| Mean delta | 0.10299 |
| Standard error | 0.01235 |
| 95% t interval | [0.07714, 0.12884] |
| Positive / tie / negative seeds | 19 / 1 / 0 |
| Exact sign-test p, excluding tie | 0.00000381 |

The learned advantage is therefore not explained by intervention count, risk
exposure, or the matched-random arm's complete loss of temporal clustering.

## Trained Ablations

### State-Dependent Pool

Under a fixed-share pool, the exact-dose timing advantage falls to:

| Statistic | Fixed-pool matched - learned |
| --- | ---: |
| Mean delta | 0.01308 |
| Standard error | 0.00522 |
| 95% t interval | [0.00215, 0.02400] |
| Positive / negative seeds | 16 / 4 |

The per-seed difference between full-contract and fixed-pool timing advantage
is `0.08691 +/- 0.01191`, with 95% interval `[0.06197, 0.11184]` and `20/20`
positive seeds. The state-dependent pool is therefore the main source of the
larger timing advantage.

The fixed pool lowers raw spread more aggressively, but it allocates the full
pool whenever active and has lower reward. This is an intervention-strength
trade-off, not evidence that the fixed pool is the better selective mechanism.

### Routing Training

The no-routing-trained adapter does not improve the baseline:

| Statistic | Base - no-routing trained |
| --- | ---: |
| Mean delta | -0.04264 |
| Standard error | 0.01238 |
| 95% t interval | [-0.06855, -0.01673] |
| Positive / tie / negative seeds | 3 / 1 / 16 |

The per-seed difference between routing-trained and no-routing-trained
base-relative improvement is `0.19056 +/- 0.02544`, with 95% interval
`[0.13731, 0.24381]` and `19/20` positive seeds. The announced signal alone is
not the effective causal channel; routing incentives during response learning
are necessary.

## Aggressive Controls

Threshold and always-on fixed-pool controls attain lower raw low-risk spread,
but at substantially larger intervention exposure:

- threshold activation is `0.3340` overall and `0.4922` in high-risk states;
- always-on activation is `1.0000` in both risk regimes;
- learned activation is `0.1030` overall and zero in high risk;
- learned reward exceeds threshold by `0.000419`, 95% interval
  `[0.000164, 0.000673]`;
- learned reward exceeds always-on by `0.000953`, 95% interval
  `[0.000098, 0.001807]`.

Accordingly, the learned method does not win on minimum raw spread. It wins on
the registered spread-reduction / high-risk false-positive / intervention /
reward trade-off.

## Protocol Integrity

All checks pass:

- gate train/validation/full-main seeds are disjoint;
- the artifact and complete runtime environment match;
- all three response-training schedules replay exactly;
- random and circular schedules match total and risk-stratified dose for every
  episode;
- every trained adapter exactly matches the frozen base with its gate off;
- all arms share CRN external paths and per-step action uniforms;
- 20/20 per-seed fingerprints match the run manifest;
- no parameter or acceptance rule changed after launch.

## Claim Boundary

The experiment supports a completed full-main result for learned
oracle-distillation timing in the redesigned N=10 local synthetic LOB/MAPPO
assay.

It does not support:

- latent risk inference: the true simulator risk regime is observed;
- discovery of tacit collusion from unlabeled order-book data;
- ABIDES, another external matching engine, or real-market validity;
- robustness to agent counts other than N=10;
- fully end-to-end retraining of an entire seller policy under each evaluation
  gate arm.

No further tuning or seed extension in this same assay is justified. The next
methodological step, if needed for the paper claim, is an observable-only risk
gate and an external matching-engine transfer experiment.

## Reproduction

```bash
python revision/experiments/run_lob_mappo_learned_gate_pilot.py \
  --stage full_main \
  --learned-gate-artifact revision/configs/lob_mappo_learned_gate_v1.json \
  --seeds 60-79 \
  --equilibrium-episodes 100 --response-episodes 100 \
  --response-train-rate 1.0 \
  --calibration-episodes 5 --calibration-quantile 0.75 \
  --calibration-upper-quantile 0.90 \
  --eval-episodes 10 --episode-length 100 \
  --warm-start-steps 200 --warm-start-updates 20 \
  --device mps \
  --output-dir revision/experiments/outputs/lob_mappo_learned_gate_full_main_20seed_v1
```

Raw outputs remain ignored under `revision/experiments/outputs/`.
