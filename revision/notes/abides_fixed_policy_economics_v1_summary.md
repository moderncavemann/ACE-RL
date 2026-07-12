# ABIDES Fixed-Policy Economics v1

Status: G0-G3 passed  
Date: 2026-07-12  
Source: `jpmorganchase/abides-jpmc-public@f9cbe51342b7dedd9587e4e069040d68a5c6477f`

## Structural Changes

The assay uses one kernel-pausing coordinator and ten distinct controlled
`TradingAgent` market makers. All quotes rest in the official ABIDES exchange
book and all customer flow consumes those orders. A routed order may select a
predeclared dealer, but the fill price, quantity decrement, execution messages,
cash, and inventory remain exchange-managed.

Risk now enters the economics through informed flow. At each decision, a
seeded next-period fundamental shock is drawn with low- or high-risk volatility.
With probability 0.8, customer direction is aligned with that shock. Quotes
execute around the current fundamental and rewards mark inventory at the next
fundamental. This creates adverse-selection cost in high risk without making
the simulator risk label available to a learned gate.

Quote refresh is two-phase: all ten agents cancel first, then all ten submit new
quotes, and customer flow arrives afterward. This prevents a new quote around a
moved fundamental from crossing another agent's stale quote. Before this fix,
internal market-maker crosses inflated measured flow above the planned 20
units; after the fix every fixed-policy arm executes exactly 20 units per step.

## Frozen Parameters

| Parameter | Value |
| --- | ---: |
| N | 10 |
| half-spread actions | 1, 2, 3, 4, 5, 6, 7 ticks |
| flow events per step | 20 |
| quantity per event | 1 |
| informed-flow probability | 0.80 |
| low-risk shock sigma | 0.75 ticks |
| high-risk shock sigma | 4.00 ticks |
| routable fraction | 0.50 |
| excess-spread reference | 4.0 ticks |
| excess-spread scale | 2.0 ticks |
| undercut requirement | 1.0 tick below mean |
| validation seeds | 100-109 |
| steps per fixed-policy episode | 30 |

These values were set from the adverse-selection calculation before the
ten-seed acceptance run. The run is not a routing or environment grid.

## Acceptance Results

| Check | Result |
| --- | ---: |
| high-risk reward: half-spread 4 minus 1 | +6.0000 |
| positive seeds for high-risk widening | 10/10 |
| low-risk half-spread 5 mean reward | 8.7781 |
| low-risk wide-minus-competitive half-spread | +4.0000 ticks |
| wide-state undercutter routing gain | +8.5085 |
| positive seeds for wide-state routing gain | 10/10 |
| competitive-state routing gain | 0.0000 |
| planned/realized flow | 20/20 every step and arm |
| same-seed exogenous path | identical |
| different-seed exogenous path | distinct |

## Boundary

This passes environment integrity, demand conservation, and fixed-policy
economic identification. It does not yet show that timing matters. Full main
remains blocked until oracle timing beats an exact-dose shuffled schedule under
the same contract and a learned observable gate passes fresh screen and holdout
seeds.
