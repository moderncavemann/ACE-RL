# ABIDES/MAPPO Adaptive v3 Base Extension Protocol

Status: frozen after the 192-episode base result and before extension  
Date: 2026-07-12

The adaptive-v3 development run passed market failure-mode, numerical,
entropy, KL, critic-scale, and reward-stability diagnostics on all five base
policy seeds. It failed only the preregistered fixed-panel spread-stability
rule: two seeds passed, one missed the `0.15` range by `0.0138` tick, and two
were still moving upward toward the same high-spread attractor.

No intervention timing outcome has been generated. The development protocol
therefore permits one base-only checkpoint extension:

- restore every exact episode-192 model, optimizer, RNG, and history;
- preserve all learner, environment, evaluation-panel, and stability fields;
- continue the deterministic training seed namespace through episode 288;
- retain the same three-panel `0.15` spread and 10% reward criteria;
- stop each seed as soon as the frozen convergence rule passes;
- keep the parent outputs immutable.

This extension cannot be repeated automatically. Failure at episode 288 stops
adaptive-v3 timing development and no learned/random efficacy comparison is
allowed.
