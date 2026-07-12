# ABIDES Learned Gate Calibration v1

Status: classification gate passed and frozen before intervention screen  
Date: 2026-07-12

## Data Separation

| Split | Seeds | Passive episodes per seed | Records |
| --- | --- | ---: | ---: |
| gate train | 300000-300004 | 5 | 1,000 |
| gate validation | 310000-310004 | 5 | 1,000 |

All records use the frozen passive MAPPO checkpoint with routing disabled.
No screen, holdout, or intervention outcome is used to fit the classifier or
select its probability threshold.

## Observable Feature Contract

The gate uses current submitted quote statistics, lagged public volatility and
return, lagged flow concentration, and maker inventory. It does not use the
simulator risk label, current fundamental shock, reward, future state, or the
current window's fills. Risk labels are used only to construct passive offline
targets and validation false-positive metrics.

## Frozen Validation Result

| Metric | Value |
| --- | ---: |
| probability threshold | 0.8420 |
| activation rate | 0.2750 |
| target prevalence | 0.2770 |
| precision | 0.8655 |
| recall | 0.8592 |
| F1 | 0.8623 |
| negative FPR | 0.0512 |
| high-risk FPR | 0.1486 |

The frozen constraints were activation in `[0.05, 0.35]`, high-risk FPR at
most 0.15, and F1 at least 0.50. All passed. High-risk FPR is close to its
boundary; the threshold is nevertheless frozen and will not be changed based
on intervention screen outcomes.

Artifact: `revision/configs/abides_learned_gate_v1.json`  
SHA-256: `f05d4c9f2ca16b8219f9786659ad076da81d4452e3d1cdd4a56ccae040c572cc`

## Boundary

This establishes that observable features recover the passive target with the
specified error/activation constraints. It does not establish causal timing
value. G5 remains blocked until the frozen gate beats exact-dose comparators on
the preregistered screen and fresh holdout seeds.
