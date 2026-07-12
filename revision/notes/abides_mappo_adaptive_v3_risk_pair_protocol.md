# ABIDES/MAPPO Adaptive v3 Risk-Pair Development Protocol

Status: frozen after learned-full development and before risk-pair efficacy  
Date: 2026-07-12

Learned-full development passed its primary comparison on `5/5` policy seeds,
but the lagged spread-severity stage failed its preregistered ablation: it did
not reduce activation and increased low-risk spread by `0.01199` tick relative
to risk-only. The final candidate is therefore simplified before holdout.

This targeted development confirmation trains two new zero-initialized
response adapters from each frozen base:

- `risk_aware`: paced observable low-risk probability announced before quotes;
- `matched_random`: the exact risk-aware signal-strength multiset globally
  permuted within each episode.

The two arms share every environment path and action uniform. Training uses 96
episodes, GAE, eight episodes per update, and the unchanged soft-rank routing
contract. Fixed convergence panels occur at episodes 48, 72, and 96.

The simplified candidate passes only if it beats matched random on at least
four of five seeds with positive mean, high-risk activation is at most `0.15`,
reward and volume guardrails pass, all response policies converge, and
zero-signal actions exactly match the frozen base.

No further module or parameter change is allowed after this result. Passing
freezes the risk-only candidate for one run on the reserved `480xxx/481xxx`
holdout split.
