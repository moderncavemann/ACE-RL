# ABIDES Domain Benchmark v1 Protocol

Status: protocol frozen; formal execution blocked pending passive calibration  
Date: 2026-07-14

## Question And Claim Boundary

The benchmark asks whether the existing risk-aware routing mechanism improves
low-risk customer execution relative to a literature-derived public-quote
threshold rule and a genuinely budget-matched spread-only rule. It does not
test real-market detection, exact reproduction of either cited paper, total
welfare, arbitrary learners, or arbitrary market ecologies.

Existing v5 and supplement configs, outputs, fingerprints, summaries, tables,
and figures are immutable sources. New arms load the same frozen base policies
but write only to new output directories.

## Frozen Design

| Item | Value |
| --- | --- |
| Source policies | `530000-530019` |
| Source base fingerprint | `8a409007a978ab74a7a9ddb94757e2c7b681ac57e2e0d3449c684b1b779afb51` |
| Source v5 fingerprint | `8636827cb45cdb774a1351183858aaa0edc058b90f3d4a269c51c74ca467674c` |
| Passive calibration paths | `535000-535009` |
| Final evaluation paths | `534000-534009` |
| Training path namespace | existing v5 response namespace `620000000` |
| Training action namespace | existing v5 response namespace `511117` |
| Final action namespace | `541117` |
| Response budget | 96 episodes per new adapter, 100 decisions per episode |
| Main routing | soft rank, `kappa=4`, routable fraction `0.50` |
| Budget tolerances | active-rate difference at most `0.02`; TV ratio in `[0.90,1.10]` |

Formal methods are `no_intervention`, `risk_aware`, `matched_random`,
`spread_only_budget_matched`, `brero_threshold_buybox`,
`brero_threshold_buybox_budget_matched`, `spread_only_threshold`, `always_on`,
and `no_routing`. Banerjee arms may be included only if their implementation
status is frozen before the formal-cell commit. The minimum list cannot be
reduced after outcomes are opened.

The three new adapters start from the corresponding frozen base digest and an
identical adapter initialization digest. Existing v5 arms are loaded from
their frozen response checkpoints and evaluated, not retrained. All methods
use paired exogenous paths and action uniforms.

## Budget Calibration

Calibration reads passive quote/state records and mechanism-budget fields
only. The calibrator rejects any input containing efficacy fields such as
execution spread, customer cost, reward, or ranking. It computes planned
routing weights and

`TV_t = 0.5 * sum_i(abs(w_i,t - 1/N))`.

The Brero candidate set is the complete Cartesian product `k=1..10` and
`m in {0,1}`. The spread-only candidate set is fixed in the calibration JSON.
Candidate ranking is lexicographic: budget distance, then the lower-complexity
parameter tuple. No candidate is added after calibration. The selected
artifact records every candidate, both target metrics, selected parameters,
match status, input hashes, and code commit. Formal execution remains blocked
until that artifact hash is inserted into the formal config in a dedicated
commit.

## Endpoints And Inference

Primary contrast:

`brero_threshold_buybox_budget_matched - risk_aware`

on low-risk customer execution half-spread, aggregated first over the ten
evaluation paths and then inferred over 20 policy seeds.

Secondary confirmatory contrasts are spread-only budget matched, Banerjee
budget matched if present at freeze, and fresh matched random, each minus
risk-aware. Secondary p-values receive Holm correction; raw p-values remain
visible. Aggressive Brero, spread threshold, always-on, no routing, and no
intervention are diagnostic operating points.

Report policy-seed paired t 95% CI, exact two-sided sign test, and a paired
two-way percentile bootstrap with 10,000 replicates and RNG seed `20260714`.
The bootstrap resamples policy seeds and, within each sampled seed, paired
evaluation paths. The policy seed remains the primary unit of inference.

Risk-aware can be called better than the budget-matched Brero adaptation only
if the baseline is matched, the mean and t-CI favor risk-aware, at least 14/20
seed differences are positive, sign and bootstrap results do not conflict,
and reward, high-risk activation, routing TV, and volume are fully reported.
Otherwise the result is classified as `external_better`, `tradeoff`, or
`inconclusive`; no new search follows.

## Integrity And Audit

The runner must automatically check all twenty directive checks: complete
source seeds; common base and initial adapter digests; unchanged v5 digests;
common training paths; common evaluation paths; common action uniforms;
disjoint calibration/evaluation seeds; no latent-risk access by external
rules; per-step and per-episode demand conservation; valid routing weights;
budget match status; exact gate-off base actions; 96 completed episodes;
frozen convergence rule; complete provenance; no-container completed resume;
retained failures; method volume 20; exact per-path row count; and paper
exports rebuilt only from committed audit data.

Per-path, per-seed, and compressed per-step records follow the schemas in the
formal config. `implementation_status` is independent of `scientific_result`.
Smoke output is permanently marked `engineering_smoke=true` and is excluded
from all paper exports.

## Stop Rules

- A calibration mismatch does not trigger a wider search. The arm is labeled
  unmatched and becomes a trade-off point only.
- A scientific loss does not trigger new seeds, episodes, candidates, or
  thresholds.
- A scientific-parameter bug requires a new protocol version and fresh seed
  namespace. Execution-only recovery preserves the failed launch and the
  existing fingerprint.
- Formal execution is prohibited while `formal_execution_allowed` is false.
