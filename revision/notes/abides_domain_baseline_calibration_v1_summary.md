# ABIDES Domain Baseline Calibration v1

Status: complete and frozen  
Date: 2026-07-14

The calibration used all 20 frozen v5 base policies and ten disjoint passive
paths `535000-535009`, producing 20,000 decision records. Records contain
public quotes, lagged spread severity, the frozen risk signal, path/model
digests, and a post-hoc risk label. The calibrator rejects customer spread,
reward, customer cost, or rank fields and records
`selection_uses_efficacy_outcomes=false`.

The risk-aware reference budget was active on `0.17865` of steps with mean
routing TV `0.05634192012609473`.

| Candidate | Selected parameters | Active rate | Mean TV | TV ratio | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Brero threshold family | `k=4`, `m=0` | 0.24660 | 0.0684725 | 1.21530 | unmatched |
| Spread-only | target 0.19, threshold 0.0, scale 0.75 | 0.18155 | 0.0559202 | 0.99251 | matched |

No Brero order-statistic candidate met both frozen tolerances. The nearest
candidate therefore remains named `brero_threshold_buybox_budget_matched` for
protocol continuity but carries `budget_match_status=unmatched`. It can appear
only as a trade-off operating point and cannot support same-budget superiority.
The candidate family was not expanded after this result.

The compressed passive-record hash is
`6ba0def94d27f88dde2b916ce9c8d431e3dd3687258da2f4e7acd8e332156728`.
The committed calibration artifact hash is
`6f6405f5f9f6dfc226756ebabe60ef071785d4ef3930c5d58a2d9f241506d96e`.
A completed rerun took 1.35 seconds and did not launch calibration workers.
