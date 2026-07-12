# ABIDES/MAPPO Full-Main Readiness Protocol

Status: frozen before adapter efficacy runs  
Frozen upstream: `jpmorganchase/abides-jpmc-public@f9cbe51342b7dedd9587e4e069040d68a5c6477f`

## Claim Boundary

Passing this protocol permits a full N=10 ABIDES/MAPPO comparison of learned
risk-aware timing against equal-dose controls. Dependency smoke or a working
Gym step alone does not satisfy readiness. The local synthetic LOB result is
supporting evidence, not a substitute for any gate below.

## Architecture

- One coordinator is the only kernel-pausing Gym-style agent.
- Ten controlled market makers are distinct ABIDES `TradingAgent` instances.
- Each controlled agent owns its orders, fills, cash, and inventory.
- Actions travel from the coordinator to controlled agents as kernel messages.
- Exchange acknowledgements and fills travel directly to the controlled agent;
  the coordinator does not duplicate exchange messages.
- The adapter remains outside the frozen upstream source tree. Any matching
  extension must be a small, explicit, tested patch in the isolated image.

## Ordered Gates

### G0: Source and dependency integrity

- Runtime reports backend `abides` and the exact frozen commit.
- No local-LOB, synthetic-fill, or dependency fallback exists.
- `pip check` passes in the built image.

### G1: Ten-agent environment integrity

- Exactly ten distinct controlled IDs exist and action `i` changes only agent
  `i`'s two quotes.
- Quotes become real resting orders in the official `OrderBook`.
- A marketable order consumes resting quantity at its quoted price.
- Execution messages update the correct agents' cash and inventory.
- Inventory and cash conserve across counterparties.
- Same seed and action sequence reproduce the same economic fingerprint;
  different seeds produce a different exogenous path.
- Gate-off behavior matches the same adapter with intervention disabled.

### G2: Demand-conserving routing integrity

- Off, timed, shuffled, and oracle arms receive the same exogenous flow path and
  total quantity under common random numbers.
- Activation reallocates only a preregistered routable fraction; it never adds
  demand and never withholds cap-failing demand.
- Routed fills consume real resting orders at exchange execution prices.
- No self-trade, direct cash mutation, synthetic fill, or target-agent payment
  outside the order book is permitted.
- If no eligible quote exists, the order follows the preregistered base route.

### G3: Fixed-policy economics

- The risk regime changes expected adverse-selection cost, not merely an unused
  fundamental-volatility label.
- Wider quoting has a measurable legitimate benefit in high risk.
- Low-risk coordinated wide quoting creates positive rent and reduced demand or
  consumer surplus relative to competitive quoting.
- Conditional reallocation raises the marginal return to undercutting more in a
  high-spread state than in a competitive state.
- All four statements must hold across at least 8/10 fixed-policy seeds with
  directionally correct confidence intervals before MAPPO is introduced.

### G4: Oracle timing identification

- Compare oracle timing to an exact-dose within-risk shuffled schedule using the
  identical routing contract and common exogenous path.
- Activation must remain in `[0.05, 0.35]`.
- Oracle minus shuffled improves the preregistered low-risk spread/rent endpoint
  on at least 7/10 seeds and its paired 95% confidence interval must exclude zero.
- Reward and total-volume guardrails may not deteriorate by more than 10%.
- The no-routing ablation must return near no-intervention behavior.
- Failure stops learned-gate work; routing parameter search is not allowed as a
  substitute for this identification test.

### G5: Learned observable gate pilot

- Gate features exclude simulator risk labels, future fundamentals, rewards, and
  evaluation outcomes.
- Passive calibration, gate training, screen, and held-out seeds are disjoint.
- Matched random and shuffled comparators replay the learned arm's per-episode
  dose and burst structure while destroying state timing.
- A five-seed screen must have at least 4/5 positive paired effects; a fresh
  five-seed holdout must also have at least 4/5 before a 20-seed main launch.
- Activation remains in `[0.05, 0.35]`; reward and volume guardrails pass.

## Full-Main Launch

Only after G0-G5 pass may the full experiment start. The launch freezes one
configuration and runs at least 20 held-out seeds for no intervention,
equal-dose random, within-risk shuffled, learned gated, threshold/always-on
control, no-routing ablation, and oracle diagnostic. Primary inference is
paired and reports mean, standard error, t-based 95% confidence interval,
positive/tie/negative seed count, activation, reward, and volume.

No efficacy cell may be selected from a routing or environment grid. Structural
failures are repaired at the failed gate and all downstream gates are rerun.
