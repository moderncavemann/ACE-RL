# ABIDES Oracle Timing v1

Status: G0-G4 passed  
Date: 2026-07-12  
Source: `jpmorganchase/abides-jpmc-public@f9cbe51342b7dedd9587e4e069040d68a5c6477f`

## Question

Does the demand-conserving routing contract have an identifiable timing effect
before MAPPO and a learned gate add training variance?

## Protocol

- Seeds: 200-209.
- Episode length: 100 decisions.
- State schedule: one wide-with-undercutter profile every four decisions; the
  other decisions use a competitive profile.
- Oracle: active only in low-risk wide windows.
- Comparator: the exact oracle activation vector shuffled within each risk
  regime. It preserves activation count and the exogenous path while destroying
  state timing.
- Primary endpoint: shuffled minus oracle mean customer execution half-spread
  in low-risk windows. Positive values favor oracle timing.
- Additional arms: oracle schedule with routing disabled and no intervention.

## Results

| Metric | Result |
| --- | ---: |
| shuffled minus oracle execution half-spread | 0.06642 ticks |
| standard error | 0.00117 |
| 95% t interval | [0.06378, 0.06905] |
| positive/tie/negative seeds | 10 / 0 / 0 |
| oracle activation range | 0.14-0.25 |
| oracle mean reward | 1.9724 |
| shuffled mean reward | 2.0892 |
| relative reward decrease | about 5.6% |
| realized volume | exactly conserved |
| no-routing vs no-intervention | exactly equal |

All preregistered G4 checks passed. The result identifies timing value in the
contract itself; it is not yet evidence that a learned observable gate or MAPPO
policies recover that value.

## Next Gate

G5 requires a host-side MAPPO connection and a gate trained only from observable
features. Passive calibration, screen, and held-out seeds must remain disjoint.
The learned schedule must beat exact-dose shuffled/random controls on both the
five-seed screen and a fresh five-seed holdout before the 20-seed full main may
start.
