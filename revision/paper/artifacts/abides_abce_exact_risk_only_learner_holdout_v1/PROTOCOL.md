# ABCE Formal Experiment Protocol v1

**Status:** proposed protocol; must be committed before any formal outcome is opened.  
**Paper route:** controlled market-design and causal-identification study.  
**ABCE axes:**

- **A — Alignment:** budgeted state alignment of an otherwise fixed intervention-strength multiset.
- **B — Before-action commitment:** current contract strength is revealed before dealers quote.
- **C — Contract mechanism:** quote-sensitive allocation is separated from arbitrary redistribution, and the timing effect is tested under a second allocation family.
- **E — Evidence:** every claim is backed by paired, trace-first, replayable evidence.

The protocol follows a claim–metric–experiment–evidence structure. It deliberately includes remove, replace, and perturb controls, common random numbers, multiple policy seeds, uncertainty estimates, and explicit admissibility gates.

## 1. Research contract

### Setting

A controlled event-driven ABIDES dealer market with self-interested adaptive dealers, a fixed customer-flow budget, and a platform that may announce and apply an allocation contract but may not rewrite dealer rewards.

### Runtime inputs

The platform may use only lagged public market history. The latent risk regime is unavailable at runtime and is retained only for simulator-supervised calibration and post-hoc stratified evaluation.

### Platform outputs

At each decision \(t\):

1. an announced contract strength \(u_t\), generated before current quotes;
2. a post-quote allocation vector \(\mathbf w_t\), generated after observing current dealer spreads.

### Constraints

- fixed total customer quantity;
- no current-action or future-outcome leakage into \(u_t\);
- identical exogenous paths, action uniforms, training budgets, and final paths for paired arms;
- all formal protocols and gates frozen before final outcomes;
- failed cells and reversal seeds retained.

### Primary non-claims

The program does not claim real-market collusion detection, punishment-supported collusion, total-welfare improvement, superiority over the original Brero/Banerjee mechanisms, universal learner robustness, cross-engine validity, or cross-\(N\) generality.

## 2. Frozen primary environment

| Item | Frozen value |
|---|---:|
| Backend | ABIDES event-driven exchange |
| ABIDES source | `f9cbe51342b7dedd9587e4e069040d68a5c6477f` |
| Docker image | `ace-rl-abides-ace:adaptive-v3` |
| Dealers \(N\) | 10 |
| Decisions per episode \(T\) | 100 |
| Decision interval | 1 second |
| Half-spread action set | \(\{1,\ldots,7\}\) ticks |
| Customer events per decision \(Q\) | 20 unit events |
| Low/high fundamental sigma | 0.75 / 4.00 ticks |
| \(P(L\rightarrow H)\) / \(P(H\rightarrow L)\) | 0.04 / 0.10 |
| Initial high-risk probability | 0.30 |
| Informed-flow probability | 0.80 |
| Inventory penalty | 0.002 |
| Strategic background traders | none |
| Primary allocation | soft rank, \(\kappa=4,\phi=0.50\) |
| Proxy features | public volatility, absolute lagged return |
| Proxy threshold | 0.681 |
| Base episodes | 384 fixed horizon |
| Response episodes | 96 fixed horizon |
| Primary policy seeds | `530000–530019` |
| New final paths | `534000–534009` |

The main cell reuses the already frozen 20 base populations. This avoids retraining a new population merely because new controls were added and maximizes paired comparability. New arms receive separate response training from the same seed-specific base checkpoint.

## 3. Why \(N=10\) remains primary

The primary cell is a mechanism-identification setting, not a scale claim. With \(Q=20\), the uniform baseline gives \(Q/N=2\) customer events per dealer per decision, providing nontrivial ten-dealer competition without making dealer feedback extremely sparse. All title, abstract, tables, and conclusions must retain the \(N=10\) boundary unless a separate scale holdout is completed.

A scale extension is P2: \(N=20,Q=40\), so \(Q/N=2\) remains fixed. It uses fresh policies and is reported separately; a ten-seed screening cell cannot justify universal population robustness.

## 4. Claim–experiment map

### A. Budgeted state alignment

**Question:** Does state alignment help when the complete announced-strength multiset is fixed?

**Formal arms:**

- aligned announced soft rank;
- exact-multiset random soft rank;
- circular-shifted soft rank.

**Secondary cadence arm:**

- periodic exact-multiset soft rank, with the same nonzero values placed at evenly spaced decisions.

**Primary contrast:**

\[
H_A:
Y_L(\text{random})
-
Y_L(\text{aligned})
>0.
\]

Matched random must preserve each path's exact continuous signal multiset, not merely its activation rate or mean strength. Circular shift must preserve the same values and cyclic structure with a frozen nonzero offset.

### B. Before-action commitment

**Question:** Does access to the current contract before quoting matter, beyond knowing the standing platform rule?

**Formal arm:**

- `lagged_revelation_soft_rank`: the platform applies \(u_t\), while dealers observe \(u_{t-1}\); at \(t=1\), the observed signal is zero.

All network sizes, trainable parameters, response episodes, schedules, applied allocation, and random numbers remain unchanged. Only the signal available to the dealer at the current quote decision is stale.

**Contrast:**

\[
H_B:
Y_L(\text{lagged revelation})
-
Y_L(\text{current announcement})
>0.
\]

A positive result supports the value of timely pre-quote revelation. A null result implies that most benefit is mechanical allocation rather than strategic response. A reversal indicates possible contract gaming and must be retained.

### C1. Quote-sensitive mechanism replacement

**Question:** Is the gain caused by rewarding tighter quotes, or by arbitrary redistribution?

**Formal arm:**

- `identity_permuted_soft_rank`.

At every step, compute the same soft-rank target vector as the full method, then apply a frozen random permutation to dealer identities. This preserves, exactly and stepwise:

- the sorted allocation weights;
- realized routing activation;
- routing total variation;
- maximum share;
- targeted quantity;
- demand conservation.

It destroys only the association between quote competitiveness and allocation.

**Contrast:**

\[
H_{C1}:
Y_L(\text{identity-permuted})
-
Y_L(\text{quote-aligned})
>0.
\]

This is stronger than a no-routing ablation because it controls actual allocation intensity.

### C2. Allocation-family transfer

**Question:** Is the timing insight tied to the smooth soft-rank formula?

**Formal family:** hard-best allocation, splitting the routable portion equally among dealers tied at the best quote.

Run:

- hard-best with aligned schedule;
- hard-best with exact-multiset random schedule.

Both use the same maximum routable fraction \(\phi=0.50\). Their within-family timing contrast is valid even if hard-best has a different realized TV from soft rank. Cross-family effect magnitudes are descriptive only.

**Contrast:**

\[
H_{C2}:
Y_L(\text{hard-best random})
-
Y_L(\text{hard-best aligned})
>0.
\]

A positive result supports transfer across two allocation contracts; it does not establish mechanism-independent generality.

### E. Trace-first audit

Every efficacy claim must be recoverable from primitive traces. The audit layer is not a fourth market intervention module. It records the full chain:

\[
\text{public state}
\rightarrow p_t
\rightarrow u_t
\rightarrow \text{dealer quotes}
\rightarrow \mathbf w_t
\rightarrow \text{fills}
\rightarrow \text{customer/dealer outcomes}.
\]

## 5. Training and randomization requirements

### Compact follow-up

- Reuse frozen base checkpoints for policy seeds `530000–530019`.
- Train each new response arm for exactly 96 episodes.
- Evaluate response stability at episodes 48, 72, and 96 on a fixed two-path convergence panel.
- No early stopping, seed replacement, or post-outcome extension.
- Evaluate every arm once on fresh paths `534000–534009`.
- Use common exogenous paths and categorical action uniforms across paired arms.
- Use separate deterministic namespaces for signal permutations and allocation permutations.

### Exact-method learner holdout

The current learner holdout changes the signal specification. To support an exact-method transfer claim, rerun the **primary risk-only signal** with `include_lagged_spread_severity=false` under:

- official full-policy MAPPO;
- official IPPO.

Reuse the previously frozen hyperparameter selection: actor/critic learning rate \(5\times10^{-4}\), entropy 0.01. Use policy seeds `560000–560009`, 384 base episodes, 96 response episodes, and fresh paths `562000–562009`. No new hyperparameter search may use intervention outcomes.

## 6. Metric hierarchy

### Primary endpoint

\[
Y_L =
\frac{
\sum_t \mathbf 1[z_t=L]\sum_{e\in\mathcal E_t}q_e|p_e-m_t|
}{
\sum_t \mathbf 1[z_t=L]\sum_{e\in\mathcal E_t}q_e
}.
\]

Within each path, compute quantity-weighted low-risk execution half-spread. For audit statistics, give each final evaluation path equal weight within a policy seed.

### Intervention accounting

\[
A^{sig}=\frac1T\sum_t\mathbf1[u_t>0],
\quad
A^{route}=\frac1T\sum_t\mathbf1[TV_t>0],
\quad
\overline{TV}=\frac1T\sum_tTV_t.
\]

Also report high-risk versions of all three quantities and targeted customer quantity. The word `Activation` must not appear without specifying `announced signal` or `realized routing`.

### Customer and market outcomes

Mandatory:

- \(Y_L\), \(Y_H\), \(Y_{all}\);
- low-risk and all-state customer tick-units;
- fill ratio and fallback rate;
- exact volume conservation.

### Dealer and allocation guardrails

Mandatory:

- aggregate and per-dealer reward;
- reward dispersion;
- inventory standard deviation and maximum absolute inventory;
- normalized flow HHI and maximum dealer share;
- quote dispersion and best-quote tie rate.

### ABCE diagnostics

- A: pathwise strength-multiset hash, high-risk placement, run-length distribution;
- B: active-step quote response and policy-action divergence under current versus lagged revelation;
- C: weight–quote rank correlation, weighted routed quote, exact TV equality for the permuted control;
- E: trace completeness, replay success, raw-source correspondence, deterministic export.

## 7. Statistical plan

The policy seed is the primary inference unit.

For each formal contrast:

1. aggregate ten final paths within each policy seed;
2. report method mean \(\pm\) sample SD;
3. report paired seed-level mean difference;
4. report paired two-sided 95% Student-\(t\) interval;
5. report positive and reversal seed counts;
6. report exact sign test;
7. report a 10,000-replicate paired two-way bootstrap over policy seeds and paths;
8. retain failed and nonconverged runs;
9. apply Holm adjustment to the B/C key-secondary family.

The primary A contrast is supported only if its paired interval is strictly positive and at least 14/20 policy seeds are positive. B/C contrasts are classified as supported, directional-inconclusive, reversed, or descriptive-only according to the frozen gates.

## 8. Admissibility and integrity gates

A comparison is confirmatory only when all relevant gates pass:

- exact expected run grid;
- finite updates and metrics;
- fixed training budget;
- at least 18/20 common converged policy seeds for new compact arms;
- common path/action-uniform digests;
- exact pathwise signal-multiset matching for timing controls;
- exact per-step weight-multiset and TV matching for identity-permuted routing;
- valid weights and exact demand conservation;
- verified lagged-revelation semantics;
- no current-action/future leakage;
- complete trace schema and deterministic hashes.

A failed gate changes the result to descriptive-only. It does not trigger more search, seed replacement, or extended training.

## 9. Required trace schema

### Provenance

- protocol version/fingerprint;
- source commit and Docker image;
- method, policy seed, path seed, step;
- model digest, path digest, action-uniform and route-uniform IDs.

### Public-state estimator

- lagged public features;
- normalization artifact;
- proxy score and threshold;
- latent risk marked `audit_only`.

### Scheduler/revelation

- raw proposal;
- token balance before/after;
- platform-applied \(u_t\);
- dealer-observed signal;
- schedule source;
- permutation/shift/multiset IDs;
- announcement timestamp.

### Dealer response

- observation hash;
- action uniform;
- selected action and spread per dealer;
- base and response-policy digests.

### Allocation

- base weights;
- quote ranks and target scores;
- pre-permutation and final weights;
- identity permutation;
- routing TV;
- selected targets and route uniforms;
- targeted event count and quantity.

### Outcomes

- fills, prices, quantities, fallback;
- customer execution cost;
- reward components and total reward;
- inventories;
- flow shares, HHI, maximum share;
- next public state.

### Integrity

- no-lookahead;
- weights sum to one;
- fixed volume;
- gate-off recovery;
- missingness and nonfinite flags.

## 10. Artifact layout

```text
revision/paper/artifacts/abides_abce_primary_followup_v1/
  README.md
  manifest.json
  protocol.json
  checks.json
  per_path.csv
  per_seed.csv
  statistics.json
  step_records.jsonl.gz
  model_digests.json
  path_digests.json
  SHA256SUMS
  figures/
  tables/
```

Provide two verifiers:

```bash
python revision/experiments/verify_abides_abce_primary_followup_v1.py --strict-raw
python revision/experiments/verify_abides_abce_primary_followup_v1.py --bundle-only
```

The strict verifier checks raw-source correspondence locally. The bundle-only verifier allows CI to recompute statistics, hashes, conservation, and paper exports without private raw training outputs.

## 11. Representative trace figure

Pre-freeze the selection rule:

1. compute the A contrast for all policy seeds;
2. select the seed with the median seed-level effect, ties broken by lower seed;
3. within that seed, select the path with median path-level effect, ties broken by lower path seed.

Plot aligned, exact-random, lagged-revelation, identity-permuted, and no-routing traces on the same decision axis:

- latent regime and public score;
- platform-applied and dealer-observed contract;
- routing TV;
- mean/best quote;
- execution half-spread.

The figure is explanatory evidence, not a substitute for aggregate inference.

## 12. Two-week execution schedule

| Day | Work |
|---|---|
| 1 | Freeze protocol, configs, hypothesis order, gates, and seed namespaces |
| 2–3 | Implement lagged revelation, identity-permuted routing, hard-best family, logging fields |
| 4 | Unit tests, deterministic smoke, 2-seed convergence smoke; no formal outcome interpretation |
| 5–6 | Run compact ABCE formal arms and fresh common evaluation |
| 6–8 | Run exact risk-only MAPPO/IPPO holdout |
| 8–9 | Export primary v5 and ABCE per-step bundles; strict verifier |
| 10 | Statistical export, paired-seed plots, representative trace figure |
| 11 | Efficiency profiling and metric completeness audit |
| 12–13 | Rewrite Experimental Design and Results around A/B/C/E |
| 14 | Bundle-only CI, zero-diff paper exports, claim-evidence audit, freeze |

Based on the existing hours-scale supplement runtime, the compact arm execution should fit within a workday once implementation passes smoke; the schedule deliberately reserves most time for correctness, matching, audit, and writing.

## 13. Stop rules

Stop and narrow the paper rather than expanding search when:

- B is null or reversed: remove the claim that pre-quote revelation is beneficial;
- C1 is null: claim allocation is necessary but do not claim quote alignment is identified;
- C2 is null: keep soft-rank-specific timing, do not claim allocation-family transfer;
- exact risk-only MAPPO/IPPO are inconclusive: retain learner-specific scope;
- any method fails convergence or budget integrity: report descriptively;
- the primary A replication reverses: investigate implementation mismatch, but do not alter frozen settings.

## 14. Paper-facing evidence hierarchy

1. **Main table:** A, B, C1 and mechanism references, with mean \(\pm\) SD and paired effects.
2. **Transfer table:** exact risk-only MAPPO/IPPO and existing E1/E2 in separate panels.
3. **Sensitivity figure:** existing dose/routing results with explicit seed-\(t\) intervals and bootstrap caveat.
4. **Trace figure:** ABCE intermediate mechanism evidence.
5. **Audit paragraph:** row counts, schema, strict/raw and bundle-only verification.
6. **Limitations:** \(N=10\), one ABIDES engine, simulator-supervised proxy, local operating region, no punishment-supported collusion.
