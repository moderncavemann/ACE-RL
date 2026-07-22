# ABCE Manuscript Rewrite Pack v1

Date: 2026-07-21

Status: frozen paper-facing interpretation. No additional formal training is
required for this claim set.

## Research Contract

| Gap / promise | Method element | Identifying comparison | Primary evidence | Supported conclusion |
| --- | --- | --- | --- | --- |
| A. An intervention schedule may work only because of its total dose | State-aligned announced contract | exact-multiset random, circular shift, periodic exact multiset | low-risk execution half-spread, policy-seed paired t-CI, sign test, two-way bootstrap | alignment with lagged public state adds value beyond the fixed announced-strength multiset |
| B. Dealers may need actionable current contract information | pre-quote current-contract revelation | one-step-lagged revelation | same primary endpoint and inference | timely current revelation is better than stale revelation in this setting |
| C1. Arbitrary flow redistribution may mimic routing | quote-sensitive soft-rank allocation | identity-permuted allocation-weight multiset | execution, routing TV, weight-quote rank correlation | the dealer-quote-to-allocation mapping is necessary; similar arbitrary redistribution is insufficient |
| C2. The result may be a soft-rank implementation artifact | hard-best allocation family | hard-best exact random versus hard-best aligned | same primary endpoint and inference | alignment transfers across the tested soft-rank and hard-best contracts |
| E. Simulator findings may be impossible to audit independently | trace-first paired evaluation | deterministic exporter and independent verifier | per-path, per-seed, per-step records, hashes, fingerprints, replay checks | each reported contrast is recoverable from primitive records |
| Learner transfer | exact risk-only timing with official adaptive learners | matched random under MAPPO and IPPO | policy-seed t-CI is primary; sign test and two-way bootstrap are supporting | MAPPO is supported; IPPO remains inconclusive |

The paper must use `high-spread coordination`. The forced-undercut diagnostic
does not support punishment or retaliation, so it does not justify a stronger
learned-collusion label.

## Final Claim

> In a controlled N=10 ABIDES dealer market, aligning a fixed
> announced-contract multiset with lagged public market state improves
> low-risk customer execution relative to exact-multiset random, circularly
> shifted, and periodic schedules after dealer adaptation. Timely
> current-contract revelation outperforms one-step-lagged revelation, and
> quote-aligned allocation outperforms identity-permuted redistribution. The
> alignment effect also transfers to the tested hard-best allocation contract
> and has a positive policy-seed mean effect under the tested exact risk-only
> full-policy MAPPO implementation. IPPO remains inconclusive.

Follow immediately with:

> The method does not minimize all-state raw spread, the operating region
> includes a retained low-dose reversal, and the evidence does not establish
> universal learner, population-size, matching-engine, or real-market
> robustness.

## Replacement Abstract

Multi-dealer platforms can use order allocation to discourage high-spread
coordination, but blanket intervention can also compress quotes when wider
spreads reflect legitimate market risk. We study ACE-RL, which converts lagged
public market history into a bounded contract announced before dealers quote
and then reallocates a fixed customer volume toward competitive quotes after
quotes arrive. We evaluate the mechanism with a preregistered ABIDES Contract
Evaluation (ABCE) design that separates contract-state alignment, revelation
timing, and quote-sensitive allocation. Across 20 independently adapted policy
populations and common evaluation paths, aligned soft-rank routing improves
low-risk execution half-spread over exact-multiset random timing by 0.03996
ticks (95% policy-seed CI [0.03632, 0.04359], 20/20 positive). The effect also
holds against circular and periodic schedules. Current pre-quote revelation
outperforms one-step-lagged revelation, identity-permuted allocation removes
most of the gain, and state alignment transfers to a tested hard-best
contract. Under an exact risk-only official MAPPO holdout, the matched-random
minus risk-aware effect is 0.11386 [0.03351, 0.19420] with 9/10 positive policy
seeds; IPPO is directionally positive but inconclusive. Primitive per-path and
per-step records, deterministic exports, hashes, and independent verifiers
support the audit. The result is selective rather than an all-state spread or
universal robustness claim.

## Replacement Introduction Contributions

1. **Selective platform contract.** ACE-RL combines a simulator-supervised
   public-risk proxy, a bounded pre-quote announcement, and post-quote,
   demand-conserving allocation. Latent simulator state is used for offline
   supervision and post-hoc stratification, never as runtime input.
2. **Alignment identification.** ABCE fixes the full announced-strength
   multiset and compares state alignment with exact-random, circularly shifted,
   and periodic schedules after separate dealer adaptation. This isolates
   state-contingent timing from intervention dose.
3. **Information and allocation mechanisms.** One-step-lagged revelation tests
   whether the current contract must be actionable, while identity permutation
   tests whether gains require rewarding the dealers that actually quote more
   competitively. Hard-best provides a second tested allocation family.
4. **Audited adaptive evidence.** The primary study uses 20 policy populations,
   paired common paths, policy-seed inference, exact sign tests, two-way
   bootstrap intervals, multiplicity control for B/C, 180,000 primary step
   records, and deterministic independent verification. Exact risk-only MAPPO
   is positive; IPPO is retained as inconclusive.

## Main Results Narrative

### A. Alignment

All four timing schedules announce exactly 20.1% of the same contract-strength
multiset. Aligned soft-rank routing attains low-risk execution half-spread
6.3071, compared with 6.3471 for exact random, 6.3457 for circular shift, and
6.3672 for periodic timing. Comparator-minus-aligned effects are 0.03996,
0.03855, and 0.06001 ticks; both interval estimators exclude zero and every
contrast is positive in 20/20 populations. The headline effect is 0.03996,
not the larger earlier v5 estimate.

Alignment is selective. Its high-risk announced activation is 9.42%, compared
with 19.44% for exact random, and high-risk routing TV is approximately halved.
Aligned all-state spread is slightly worse than random, so the result is an
improvement in the preregistered low-risk endpoint, not universal spread
compression. Relative to random, aligned routing also has higher aggregate
dealer reward and lower inventory dispersion, but these components do not
constitute a total-welfare theorem.

In economic units, the 0.03996-tick effect is a 0.63% improvement over the
random arm and saves 47.92 low-risk customer tick-units per episode. High-risk
announced exposure falls by 51.6%, aggregate dealer reward is 39.05 units
higher, and inventory SD is 5.6% lower. All-state spread is nevertheless 0.0091
ticks worse, preserving the selective rather than Pareto-dominance reading.

### B. Timely Pre-Quote Revelation

One-step-lagged revelation worsens low-risk spread by 0.03389 ticks
([0.03073, 0.03705], 20/20 positive). The conclusion is limited to current
versus stale revelation under the tested contract. Aggregate dealer reward is
about 56 units higher under lagged revelation, so current revelation is not a
costless Pareto improvement.

### C. Quote-Sensitive Allocation And Family Transfer

Identity permutation preserves the allocation-weight multiset and similar
routing TV but changes mean weight-quote rank correlation from 0.893 to -0.028.
Its low-risk spread rises to 6.5314, close to the no-routing mechanism reference
at 6.5244. The paired C1 effect is 0.22426 ticks, 20/20 positive. The evidence
therefore identifies the quote-allocation mapping, not arbitrary redistribution,
as the operative channel.

Under hard-best allocation, exact random minus aligned is 0.04077 ticks
([0.03712, 0.04442], 20/20 positive). This supports transfer across the two
tested allocation families, not allocation-mechanism independence.

### Exact Learner Holdout

The exact risk-only official MAPPO effect is 0.11386, policy-seed 95% t-CI
[0.03351, 0.19420], 9/10 positive, sign-test p=0.02148, and two-way bootstrap
CI [0.04909, 0.18541]. MAPPO therefore passes the frozen rule. IPPO has mean
0.12352 and 8/10 positive, but its primary t-CI [-0.01658, 0.26363] crosses
zero. It remains inconclusive even though the supporting bootstrap interval is
positive.

### Proxy Validation And Implementation Cost

The simulator-supervised two-feature public-risk proxy uses disjoint 1,000-row
passive train and validation panels and attains validation AUC 0.8284;
volatility-only and return-only AUCs are 0.7573 and 0.7976. Runtime reads only
public volatility and absolute lagged return, never latent simulator risk.
Spread-only remains an outcome comparator rather than a risk classifier.

A frozen one-CPU `N=10` microbenchmark in the formal ABCE Docker image measures
median contract control-path latency of 230.45 microseconds (p95 244.77, p99
280.50). The announced no-routing path is 161.28 microseconds, giving a
69.18-microsecond median routing increment. The actor adapter has 560
parameters; response training updates 6,257 parameters including the critic.
Because the `linux/amd64` image runs under architecture emulation on the
recorded Apple Silicon host, these are reproducibility/accounting measurements,
not native production latency or a negligible-overhead claim.

## Limitations Paragraph

The evidence is intentionally narrow. Experiments use a controlled N=10
ABIDES dealer market and a simulator-supervised public-risk proxy; they do not
validate real-market collusion detection. The forced-undercut diagnostic does
not support punishment or retaliation, so we interpret the failure mode as
high-spread coordination. State alignment improves low-risk execution but does
not minimize all-state spread, and a retained low-dose cell reverses against
matched random. Exact risk-only MAPPO is positive while IPPO is inconclusive.
We test two allocation families, one population size, and one matching engine;
larger populations, heterogeneous dealers, alternate engines, and real-market
calibration remain open.

## Main-Text Asset Budget

Use exactly two figures and two tables in the eight-page main text:

1. `figure_framework.tex`: method and information order.
2. `table_main_results.tex`: nine-arm ABCE outcomes plus A/B/C contrasts.
3. `table_transfer_results.tex`: exact MAPPO/IPPO and the separately labeled
   earlier replication.
4. `figure_abce_trace.tex`: preregistered median-effect population/path trace.

Move `figure_paired_seeds.tex`, `figure_sensitivity.tex`,
`table_behavioral_boundary.tex`, and `table_efficiency_profile.tex` to the
supplement. Brero/Banerjee ABIDES adaptations remain descriptive supplement
material because their frozen admissibility gates did not pass.

## Source Map

- ABCE primary statistics:
  `revision/paper/artifacts/abides_abce_primary_followup_v1/statistics.json`
- ABCE primitive paths and trace:
  `revision/paper/artifacts/abides_abce_primary_followup_v1/`
- exact risk-only learner statistics:
  `revision/paper/artifacts/abides_abce_exact_risk_only_learner_holdout_v1/statistics.json`
- exact learner primitive paths and trace:
  `revision/paper/artifacts/abides_abce_exact_risk_only_learner_holdout_v1/`
- earlier v5 replication:
  `revision/paper/data/abides_mappo_adaptive_v5_supplement_v1_per_seed.csv`
- retained sensitivity and diagnostic boundaries:
  `revision/paper/artifacts/abides_dose_routing_sensitivity_v1/` and
  `revision/paper/artifacts/abides_collusion_diagnostic_v1/`
- frozen proxy/economic/efficiency export:
  `revision/paper/artifacts/abides_abce_mechanism_profile_v1/`

## Prohibited Upgrades

Do not write any of the following without new preregistered evidence:

- ACE-RL minimizes collusion or spread in every market state.
- Timely transparency is universally beneficial.
- The result is independent of the allocation mechanism.
- The method is learner-independent; IPPO is not confirmatory.
- ACE-RL outperforms the original Brero or Banerjee mechanisms.
- The result generalizes to larger N, another matching engine, or real markets.
- The mechanism has negligible overhead or production deployment latency.
