# Literature Baseline Mapping v1

Status: frozen before implementation or outcome generation  
Access date: 2026-07-14  
Directive SHA-256: `dbba9e2fc9cc2b9132af7ea789173aa1db3c1aa9a41321b709a5460a95b7fa4b`

This document fixes the source-to-ABIDES mapping before either literature
baseline is implemented. Each implementation is an adaptation, not an exact reproduction.
Paper-facing names must include "ABIDES adaptation of".

## Brero Threshold Buy-Box

Source: Gianluca Brero, Nicolas Lepore, Eric Mibuari, and David C. Parkes,
"Learning to Mitigate AI Collusion on Economic Platforms," arXiv:2202.07106v2,
last revised 2022-06-11. Source URL:
<https://arxiv.org/abs/2202.07106v2>. The downloaded v2 PDF used for this
mapping has SHA-256
`4a61b2594c46bf2c28c5c3c528fca64da12f18ae3a887f6ee530cbba6a6e4f60`.

### Source mechanism

| Object | Brero et al. |
| --- | --- |
| Sellers | Repeated differentiated-product price setters. |
| Public state | Current seller price vector; the offline critic may additionally observe seller learning state, while the in-the-wild actor observes prices. |
| Platform action | A buy-box/display rule chooses the sellers visible to demand. A threshold rule displays seller `i` iff `p_i <= tau(p)`. |
| Information timing | The platform commits to a rule; current prices are submitted; the rule determines the displayed set. |
| Seller response | Q-learning sellers adapt to the standing platform rule. |
| Platform objective | Discounted consumer surplus in a behavioral Stackelberg problem. |
| Learned component | A Stackelberg-POMDP platform policy can learn state-dependent thresholds. |

### ABIDES mapping

| Object | ACE-RL ABIDES adaptation |
| --- | --- |
| Sellers | Ten dealer policies choose integer half-spreads in `1..7` around the current fundamental. |
| Public state | Current submitted half-spread vector only for eligibility. No latent risk, post-hoc risk label, reward, inventory, or future outcome is available to the rule. |
| Platform action | For candidate `(k,m)`, set `tau=min(7, kth_order_statistic(s,k)+m)` and retain all ties with `s_i <= tau`. |
| Allocation | The non-routable share keeps uniform/base allocation. The routable share is allocated only among eligible dealers. Total customer flow remains 20. |
| Information timing | A standing-rule indicator is announced before quotes; the eligible set and routing weights are computed after current quotes. |
| Seller response | A fresh intervention adapter is trained for 96 response episodes from each frozen v5 base policy. |
| Evaluation objective | Low-risk customer execution half-spread is the primary endpoint; reward, high-risk activation, flow HHI, volume, and routing TV are mandatory trade-off metrics. |

Preserved features are platform commitment, quote-conditioned threshold
eligibility, exclusion from customer visibility/allocation, and seller
adaptation. Changes required by the domain are half-spreads in place of retail
prices, demand-conserving routing in place of a literal web display, ten
dealers in place of two sellers, and a finite frozen threshold family in place
of a learned Stackelberg-POMDP leader.

The adaptation is intentionally not enhanced with the ACE-RL risk proxy. It
cannot read latent or public risk features. The order-statistic family is
auditable and scale-appropriate, but it is not the paper's learned leader.
Accordingly, valid language is "ABIDES adaptation of the Brero-style threshold
buy-box family." Invalid language is "reproduction of Brero et al." or a
comparison between the papers' reported consumer-surplus numbers and the
ABIDES endpoint.

Two operating points are frozen:

- `brero_threshold_buybox`: the selected literature-form threshold rule is
  applied every step and is an aggressive, budget-unmatched operating point.
- `brero_threshold_buybox_budget_matched`: `(k,m)` is selected using passive
  records only to minimize distance to the risk-aware active-routing rate and
  mean routing TV. If no candidate meets both tolerances, it remains visible
  but is labeled `unmatched` and cannot support same-budget superiority.

## Banerjee Temporary Buyer Commitment

Source: Soumen Banerjee, "Interpreting and Countering Collusion in
Deep-Learning Pricing Algorithms," arXiv:2303.02576v3, last revised
2026-05-17. Source URL: <https://arxiv.org/abs/2303.02576v3>. The downloaded v3
PDF used for this mapping has SHA-256
`3da03d0c6d850b45cd2702ea5dee8205d3efdeffaa499d0c267ec9ed4a89b63c`.

### Source mechanism

| Object | Banerjee v3 |
| --- | --- |
| Sellers | Two strategic DQN price setters in a differentiated-products Bertrand market. |
| Public state | Price levels plus rival movement direction and persistence. The learner is blind to order-book state. |
| Qualifying event | A candidate supply price is at least three price-grid steps below a reference computed from the preceding ten spot-price means. |
| Commitment | The lowest qualifying seller, with random tie-breaking, receives a protected block equal to 50% of demand for five periods including take-up. The contract price is frozen at take-up. |
| Residual market | The other 50% of demand remains in the ordinary spot market. No second book starts while one is active. |
| Mechanism channel | Protection raises the deviator's continuation payoff during punishment, weakening the loss that deters a deep undercut. |
| Punishment diagnostic | Same-state forced qualifying deviations decompose immediate gain, continuation loss, and net deviation value. |

### ABIDES mapping decision

The direct discretization is three half-spread action levels below a trailing
ten-decision mean reference, 50% protected customer flow, and a five-decision
contract. Qualification uses only lagged/current public quotes. The standing
rule is known before training, but the mechanism state remains outside dealer
observations. A qualifying dealer must execute against real resting orders;
the protected block must conserve total demand, consume resting quantity, and
use the frozen take-up quote rather than an artificial transfer.

The economic object differs from the paper because lower ABIDES half-spread is
not itself a posted transaction price and routing is side-specific. Therefore
implementation must first pass deterministic tests for take-up timing,
contract duration, tie handling, volume conservation, and real-order
execution. If the frozen quote cannot be honored without bypassing the
matching engine or creating demand, the implementation is renamed
`undercutter_protection_ablation` and removed from the external-baseline
superiority table.

The literature-form point freezes `(reference_window=10,
qualification_steps=3, protected_fraction=0.50, duration=5)`. A separate
routing-TV budget-matched point may change only protected fraction and/or
duration over a protocol-frozen passive-calibration grid; it may not use
customer spread or dealer reward for selection. This is an ABIDES adaptation,
not a claim that the original Banerjee result has been replicated.

## Learner Source

The standard-learner holdout pins the official MAPPO repository
<https://github.com/marlbenchmark/on-policy> at commit
`de66d7a4b23fac2513f56f96f73b3f5cb96695ac` under its MIT license. The
associated paper is Chao Yu et al., "The Surprising Effectiveness of PPO in
Cooperative, Multi-Agent Games," arXiv:2103.01955. The holdout is a wrapper of
that implementation over this competitive dealer environment; it is not a
claim that ABIDES is one of the original paper's cooperative benchmarks.

Evaluation follows the reproducibility principles in Rihab Gorsane et al.,
"Towards a Standardised Performance Evaluation Protocol for Cooperative
MARL," arXiv:2209.10485: fixed search budgets, disjoint validation/final
evaluation, multiple seeds, uncertainty reporting, and retention of all tried
configurations.
