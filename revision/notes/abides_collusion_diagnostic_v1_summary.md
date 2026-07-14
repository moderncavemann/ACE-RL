# ABIDES Collusion Diagnostic v1 Summary

Status: formal diagnostic complete  
Protocol fingerprint: `a1623a3712c301bc08752311b49304308764a7c604f4c8b88483a5a5a8ed2a52`  
Execution commit: `09c6ee290961119c707d372698fe4cc6b4156bb7`

## Frozen Design

- 20 no-intervention source policies: `530000-530019`.
- 20 disjoint evaluation paths: `536000-536019`.
- First low-risk decision with sham joint proposal mean half-spread at least
  `6.0` and at least 11 decisions remaining.
- One deterministically hashed dealer per event; `H=10`.
- Same path and action uniforms for sham, one-level forced undercut, and
  one-level upward deviation where feasible.
- Frozen policies resume immediately after `t0`; routing and gate remain off.

All 20 policy seeds and all 400 policy/path cells completed. The run retained
400 eligible events and 13,200 `t0..t0+H` trajectory rows. Path, risk,
action-uniform, event-prefix, fixed-volume, routing-off, and gate-off checks
passed. A completed rerun started zero containers and reproduced aggregate
SHA-256 `fb083647e80baaca72313c2e1a05cf848b6c570749d43088a25341faa4649431`.

## Formal Result

| Frozen requirement | Policy-seed result | Pass |
|---|---:|---:|
| Rival spread compression after undercut | mean `0.000083`; 95% CI `[-0.000012, 0.000179]`; 3/20 positive | No |
| Deviator continuation flow loss | mean `0.000000`; 95% CI `[0.000000, 0.000000]`; 0/20 positive | No |
| Deviator continuation reward loss | mean `0.027500`; 95% CI `[-0.001385, 0.056385]`; 5/20 positive | No |
| High-spread recovery by H | mean `0.962500`; 95% CI `[0.944098, 0.980902]`; 20/20 positive | Yes |
| Flow-loss upward asymmetry | mean `0.000000`; 95% CI `[0.000000, 0.000000]`; 0/20 positive | No |
| Reward-loss upward asymmetry | mean `0.086667`; 95% CI `[-0.027008, 0.200341]`; 3/20 positive | No |

The implementation and sample-coverage checks pass, but four of the required
mechanism conditions do not. The formal scientific label is therefore:

```text
high_spread_coordination_only
```

The paper and repository must use **high-spread coordination** or
**supracompetitive/high-spread outcome in the controlled simulator**. This
diagnostic does not support `punishment-supported collusion`, learned
punishment, or mechanism-enforcement wording.

## Interpretation Boundary

The forced undercut is usually followed by a return to the high-spread state,
but that fact alone is not punishment evidence. Rivals do not measurably
compress quotes, the deviator loses no continuation flow, continuation reward
loss is not supported by its policy-seed CI, and upward-deviation asymmetry is
absent. This is consistent with persistent high-spread policies rather than a
demonstrated retaliation mechanism.

The committed audit source is
`revision/paper/artifacts/abides_collusion_diagnostic_v1/`. Verify it with:

```bash
python revision/experiments/verify_abides_collusion_diagnostic.py
python revision/experiments/export_abides_collusion_audit_bundle.py --check
```
