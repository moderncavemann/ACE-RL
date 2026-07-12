# ABIDES/MAPPO Adaptive Pilot v1 Summary

Status: failed; joint gate is no-go for adaptive main  
Date: 2026-07-12

## Scope

Five independently initialized MAPPO policies were trained for each of five
methods in the controlled N=10 ABIDES dealer market. Each method received 100
training episodes of 100 decisions. Learned and matched-random training used
the same exogenous path, action uniforms, and exact within-risk intervention
dose in every episode.

This is the first ABIDES experiment in the revision where each deployed method
trains its own seller policy. It is not the earlier frozen-policy timing test.

## Integrity

All engineering and causal-comparison checks passed:

- five independent policy seeds completed;
- initial model parameters were identical within each seed;
- training and evaluation paths matched under CRN;
- learned and random doses matched exactly within risk;
- customer flow remained exactly 20 in every arm;
- no non-finite update or worker failure occurred;
- every training episode produced an atomic resumable checkpoint.

Mean per-seed runtime was 2,262 seconds (37.7 minutes), with all five seeds run
in parallel.

## Main Result

| Method | Low-risk execution half-spread | Reward | Activation | High-risk activation |
| --- | ---: | ---: | ---: | ---: |
| no intervention | 6.94364 | 11.82342 | 0.0000 | 0.0000 |
| learned joint gate | 5.87648 | 9.48790 | 0.3500 | 0.4128 |
| matched random | 5.83960 | 9.43884 | 0.3500 | 0.4128 |
| spread threshold | 5.78940 | 9.30196 | 0.3500 | 0.4603 |
| always-on | 4.27909 | 6.34367 | 1.0000 | 1.0000 |

The primary `matched_random - learned_gate` low-risk spread contrast was
`-0.03688`, SE `0.08344`, 95% CI `[-0.26854, 0.19478]`, with 2 positive and 3
negative policy seeds. The learned method therefore did not beat equal-dose
random timing after separate adaptive training.

The gate-off matched-minus-learned policy contrast was also inconclusive:
`-0.04244`, 95% CI `[-0.26977, 0.18489]`, with 2/5 positive seeds.

## Failed Gates

- learned timing mean was not positive;
- only 2/5 independent policy seeds favored learned timing;
- learned high-risk activation was 0.4128 versus the 0.15 ceiling;
- only 2/5 seeds had every method pass the frozen convergence checks.

The no-intervention failure mode, reward guardrail, and exact volume all
passed. Extending the v1 policies is not justified before fixing risk
representation because high-risk saturation is already a decisive structural
failure.

## Root Cause Diagnostic

The learned gate's mean raw proposal rate was 0.8634. The 35% controller was
therefore clipping a nearly always-positive classifier rather than preserving
selective risk timing. On fresh gate-off diagnostic paths, adaptive learned
policies had a routing-opportunity rate of 0.9896. The original joint
classifier used action-dependent entropy, dispersion, and undercut features to
predict a label that combined low risk with opportunity, so adaptive quote
distribution shift destroyed its risk calibration.

A two-stage diagnostic separated exogenous risk inference from deterministic
routing opportunity. Without using adaptive efficacy outcomes to select its
threshold, it reduced risk-classifier high-risk FPR from 0.7136 to 0.1061 and
accepted high-risk activation from 0.4515 to 0.0773 while retaining activation
0.3200 and target recall 0.4118.

This permits one fresh-seed v2 adaptive pilot under
`abides_mappo_adaptive_two_stage_protocol.md`. It does not permit a paper main,
reward-governance baseline campaign, or RMSC04-hybrid claim yet.
