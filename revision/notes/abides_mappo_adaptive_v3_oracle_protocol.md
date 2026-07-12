# ABIDES/MAPPO Adaptive v3 Oracle Response Protocol

Status: frozen after the base gate passed and before oracle efficacy  
Date: 2026-07-12

Five converged no-intervention MAPPO base policies are loaded from the frozen
base-extension output. For each seed, the actor trunk and ten policy heads are
copied and frozen into two zero-initialized signal-only response adapters.

The oracle adapter observes a paced continuous contract signal before quoting.
The signal is positive only when the simulator's current risk state is low and
the lagged average spread exceeds the fixed four-tick reference. The comparator
receives the exact same signal-strength multiset shuffled within each true-risk
stratum and ten-step time block. Both arms share environment paths and action
uniforms. The oracle label is diagnostic only and cannot enter a learned-gate
claim.

Each adapter trains for 120 episodes in batches of eight using GAE MAPPO. Fixed
development evaluation begins at episode 72 and repeats every 24 episodes. The
last three paired panels must have per-method spread range at most `0.15`.

The mechanism gate passes only if:

- all five response-policy seeds converge;
- oracle beats block shuffle on at least four of five seeds with positive mean;
- activation is in `[0.05, 0.35]` and high-risk oracle activation is zero;
- customer demand is exactly conserved;
- oracle reward is no more than 10% below block shuffle;
- zero-signal adapters exactly reproduce their frozen base actors.

Failure stops learned-gate development. Passing permits, but does not imply, a
learned-gate result.
