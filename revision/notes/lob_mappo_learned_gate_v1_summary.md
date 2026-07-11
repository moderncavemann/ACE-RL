# Local LOB/MAPPO Learned-Gate v1 Summary

Date: 2026-07-11

This note records the first frozen learned-gate comparison after the
state-contingent contract v3 mechanism gate. It evaluates a supervised
oracle-distillation gate against a within-episode, risk-stratified, exact-dose
matched-random schedule in the redesigned local synthetic LOB/MAPPO assay.

## Decision

The independent screen and held-out batches both pass every registered
protocol and mechanism check. The local LOB/MAPPO program may proceed to a
pre-registered, larger-seed full-main comparison.

The supported claim is:

> In the redesigned risk-coupled local LOB/MAPPO response assay, a frozen gate
> learned from passive states to reproduce the calibrated low-risk condition
> for a positive excess-spread pool assigns that pool more effectively
> and reduces low-risk spread relative to risk-stratified exact-dose matched
> random timing.

This is not evidence that a general learned collusion detector has been
validated. The gate target is an explicit oracle-distillation label, and the
environment is the repository's local synthetic LOB rather than ABIDES or a
real-market benchmark.

## Frozen Protocol

| Layer | Setting |
| --- | --- |
| Market | N=10 local price-time-priority LOB, 100 steps per episode |
| Economics | risk-coupled informed flow, fundamental marking, spread-elastic demand |
| Contract | 90% routable flow, demand-conserving excess-spread pool, soft-rank target |
| Base learner | multihead MAPPO, 100 no-intervention equilibrium episodes |
| Calibration | five separate no-intervention episodes, per-seed q75/q90 |
| Response learner | frozen base actor plus intervention adapter, 100 episodes |
| Evaluation | 10 stochastic CRN episodes per seed, identical action uniforms across arms |
| Gate train | seeds 35-39, 10 passive record episodes per seed |
| Gate validation | seeds 40-44, threshold selection only |
| Screen | seeds 45-49 |
| Held-out | seeds 50-54 |
| Device | Apple MPS |

The seed groups are disjoint. The learned gate never consumes intervention
outcomes. Screen and held-out outcomes do not affect model coefficients,
feature normalization, threshold selection, routing parameters, or acceptance
thresholds.

## Learned Gate

The frozen model is standardized logistic regression with the following
pre-action features:

- low-risk indicator;
- calibrated contract severity;
- spread entropy;
- flow concentration;
- volume suppression;
- low-risk by severity interaction;
- low-risk by volume-suppression interaction.

The low-risk indicator is the simulator's true regime state, which is also
provided to MAPPO agents in this assay. The gate does not infer latent risk
from observable price history.

The supervised target is:

```text
low_risk AND calibrated_contract_severity > 0
```

The selected probability threshold is `0.5`. The tracked compact artifact is
`revision/configs/lob_mappo_learned_gate_v1.json`, with SHA-256:

```text
b0aca84ae1c204bc6562ac5b37e6f3b5b8c5f96f5bbf01391f7315b99e783851
```

| Passive split | Activation | Precision | Recall | High-risk FPR |
| --- | ---: | ---: | ---: | ---: |
| Train 35-39 | 0.0544 | 1.0000 | 1.0000 | 0.0000 |
| Validation 40-44 | 0.0784 | 1.0000 | 1.0000 | 0.0000 |

The perfect classification is expected because the model receives the
variables that define its oracle-distillation label. It verifies serialization
and transfer across policies/seeds; it is not an independent result about
latent collusion discovery.

## Matched-Random Control

For each evaluation episode:

1. Run the frozen base policy on a passive CRN reference trajectory.
2. Apply the frozen learned gate to pre-action passive states.
3. Permute the learned schedule separately within low-risk and high-risk
   indices in that same episode.
4. Evaluate learned, matched-random, and oracle schedules with the same frozen
   response policy, risk path, external randomness, and action uniforms.

This preserves the exact total activation count and exact low/high-risk dose
for every episode. Fine-grained within-regime timing and temporal clustering
are destroyed; burst autocorrelation is not separately matched.

## Primary Results

The primary endpoint is paired `matched_random - learned_gate` low-risk mean
half-spread. Positive values favor the learned gate.

| Result | Screen 45-49 | Held-out 50-54 |
| --- | ---: | ---: |
| Paired delta mean | 0.0876 | 0.1040 |
| Standard error | 0.0215 | 0.0157 |
| 95% t interval | [0.0277, 0.1474] | [0.0605, 0.1475] |
| Positive seeds | 5/5 | 5/5 |
| Learned activation | 0.0988 | 0.0980 |
| Learned high-risk FPR | 0.0000 | 0.0000 |
| Learned effective alpha | 0.0615 | 0.0607 |
| Matched effective alpha | 0.0168 | 0.0197 |
| Base - learned low-risk spread | 0.1291 | 0.1573 |
| Matched - learned high-risk spread | 0.00042 | -0.00004 |
| Expected response diagnostic | 0.3377 | 0.5542 |

The screen and held-out decisions are independent. Combining them is
descriptive only:

- paired delta `0.0958 +/- 0.0129`;
- 95% t interval `[0.0667, 0.1249]`;
- `10/10` positive seeds;
- two-sided exact sign-test `p=0.001953`;
- pooled activation `0.0984` across 10,000 evaluation steps;
- pooled high-risk FPR `0.0000`;
- all exact-dose, gate-off-equivalence, and seed-provenance checks pass.

## Market Outcomes

| Batch | Method | Low-risk spread | Mean reward | Total volume |
| --- | --- | ---: | ---: | ---: |
| Screen | no intervention | 5.3355 | 0.002811 | 1.1406 |
| Screen | learned gate | 5.2064 | 0.002674 | 1.1608 |
| Screen | matched random | 5.2940 | 0.002836 | 1.1472 |
| Screen | oracle diagnostic | 5.2064 | 0.002674 | 1.1608 |
| Held-out | no intervention | 5.2803 | 0.004412 | 1.1822 |
| Held-out | learned gate | 5.1230 | 0.004143 | 1.2106 |
| Held-out | matched random | 5.2270 | 0.004425 | 1.1922 |
| Held-out | oracle diagnostic | 5.1230 | 0.004143 | 1.2106 |

The learned gate increases volume and lowers spread, but mean reward is
slightly below matched random. The paired matched-minus-learned reward
difference is `0.000163` in the screen and `0.000282` held out; both 95%
intervals include zero. The result should be described as a market-quality /
seller-reward trade-off, not as costless improvement.

Learned and oracle market outcomes are identical in both batches. This is
consistent with perfect oracle-label distillation and reinforces the claim
boundary: the experiment validates deployment of known state-contingent
timing, not difficult label learning.

## Acceptance Checks

Both screen and held-out pass all of the following independently:

- gate train, validation, and evaluation seeds are disjoint;
- the tracked gate artifact and complete environment configuration match the
  runtime protocol;
- response-training schedules replay exactly;
- learned and matched schedules have exact total and risk-stratified dose;
- the gate-off adapter exactly matches the frozen base actor;
- no-intervention high-spread failure appears in at least 70% of seeds;
- learned activation remains in the registered 5%-40% range;
- high-risk FPR is at most 5%, and target recall is at least 70%;
- learned timing receives more effective contract pool than matched timing;
- seller response capacity is material and seed-stable;
- learned spread beats matched random with at least 70% positive seeds;
- learned spread improves over no intervention;
- high-risk spread is preserved;
- reward stays within the registered absolute tolerance.

## Claim Boundary

The experiment supports:

- learned oracle-distillation timing beats exact-dose matched random in this
  redesigned local LOB/MAPPO assay;
- the v3 state-dependent routing pool makes timing economically identifiable;
- the result transfers to two disjoint five-seed evaluation batches.

It does not support:

- discovery of true tacit collusion from unlabeled market data;
- inference of a latent risk regime from observable LOB/price-flow history;
- superiority in ABIDES, a production matching engine, or real order books;
- scale generalization beyond N=10;
- a completed full-main baseline/ablation package;
- costless seller outcomes;
- robustness to a circular or block-shuffled timing comparator that preserves
  temporal clustering;
- end-to-end policy retraining separately under every gate arm.

## Full-Main Readiness

The mechanism gate is open, but this 10-seed screen plus confirmation is still
a pilot package. The next full-main run should use untouched local-LOB seeds
60-79 and freeze every setting above. It should include:

- no intervention;
- learned gate;
- within-risk exact-dose matched random;
- within-risk circular or block-shuffled timing;
- oracle timing as diagnostic only;
- threshold and always-on aggressive controls;
- no-routing and no-state-dependent-pool ablations.

The primary comparison remains learned versus matched random. The full-main
claim must fail if the paired mean is not positive, fewer than 70% of seeds are
positive, protocol integrity fails, or the reward/market-quality trade-off is
not reported.

## Reproduction

Train and freeze the gate:

```bash
python revision/experiments/train_lob_mappo_learned_gate.py \
  --train-seeds 35-39 --validation-seeds 40-44 \
  --equilibrium-episodes 100 --calibration-episodes 5 \
  --calibration-quantile 0.75 --calibration-upper-quantile 0.90 \
  --record-episodes 10 --episode-length 100 \
  --warm-start-steps 200 --warm-start-updates 20 --device mps \
  --frozen-artifact revision/configs/lob_mappo_learned_gate_v1.json \
  --output-dir revision/experiments/outputs/lob_mappo_learned_gate_v1
```

Run the screen:

```bash
python revision/experiments/run_lob_mappo_learned_gate_pilot.py \
  --stage screen \
  --learned-gate-artifact revision/configs/lob_mappo_learned_gate_v1.json \
  --seeds 45-49 --equilibrium-episodes 100 --response-episodes 100 \
  --response-train-rate 1.0 --calibration-episodes 5 \
  --calibration-quantile 0.75 --calibration-upper-quantile 0.90 \
  --eval-episodes 10 --episode-length 100 \
  --warm-start-steps 200 --warm-start-updates 20 --device mps \
  --output-dir revision/experiments/outputs/lob_mappo_learned_gate_100_5seed_screen_v1
```

The held-out command is identical except for `--stage heldout`,
`--seeds 50-54`, and output directory
`lob_mappo_learned_gate_100_5seed_heldout_v1`.

Raw outputs remain ignored under `revision/experiments/outputs/`.
