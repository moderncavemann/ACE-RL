# ABIDES Routing Contract Modal Eligibility v2

Status: current G0-G4 contract  
Date: 2026-07-12

## Trigger

The first passive MAPPO policy converged to a high-spread attractor with a mean
held-out half-spread of 6.9455. Most actions were 7, with occasional one-tick
deviations to 6. Under the v1 rule, eligibility required a quote to be at least
one tick below the cross-agent mean. A single 6 against nine 7s is only 0.9
ticks below that mean, so the rule failed to recognize the economically literal
one-tick undercut.

On 200 passive held-out decisions:

| Eligibility reference | Routable wide windows |
| --- | ---: |
| mean minus 1 tick | 7 / 200 (3.5%) |
| modal quote minus 1 tick | 66 / 200 (33.0%) |

After conditioning on low risk, the mean-based rule falls below the protocol's
minimum 5% activation rate. This is a representation failure, not an efficacy
screen result.

## Change

The undercut magnitude remains frozen at one tick. Only its reference changes:
an eligible agent must quote at least one tick below the modal controlled-maker
half-spread. If no such agent exists, the original base dealer route is used.
The excess-spread severity, routable fraction, real-order execution, and demand
path are unchanged.

## Mandatory Revalidation

Both downstream gates were rerun after the change.

- G3 fixed-policy economics reproduced exactly: high-risk widening +6.0000
  with 10/10 positive seeds; wide-state undercutter gain +8.5085 with 10/10
  positive seeds; competitive-state subsidy 0; flow 20/20.
- G4 oracle timing reproduced exactly: shuffled-minus-oracle execution
  half-spread 0.06642, 95% CI [0.06378, 0.06905], 10/10 positive seeds,
  activation 0.14-0.25, and all reward/volume/no-routing guardrails passed.

The modal-reference implementation is therefore the current contract for G5.
The mean-reference v1 result remains historical and must not be used as the
current learned-gate readiness evidence.
