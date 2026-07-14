# ABIDES Collusion Diagnostic v1 Protocol

Status: frozen before event selection  
Date: 2026-07-14

## Purpose

The existing endpoint establishes a controllable high-spread outcome, not by
itself punishment-supported collusion. This event study tests whether a forced
undercut induces asymmetric rival punishment and a continuation loss for the
deviator. Its definitions are independent of domain-benchmark outcomes.

## Frozen Sample And Paths

- Source: the 20 no-intervention base policies `530000-530019`.
- Evaluation paths: `536000-536019`, disjoint from all training,
  calibration, and domain-benchmark evaluation paths.
- Event eligibility: post-hoc simulator risk is low, current mean half-spread
  is at least `6.0`, and at least 11 decisions remain.
- Horizon: `H=10` decisions after the event.
- Event cap: the first eligible event per policy-seed/path pair only.
- Dealer choice: deterministic hash of policy seed, path seed, and event step.
- No future outcome is used to select an event.

The "current mean half-spread" is the mean of the sham frozen policy's joint
proposal at that decision state, not the lagged quote in the observation. The
continuation estimands use `t1..tH`, excluding the mechanically altered `t0`.
Recovery is the first `t1..tH` at which the realized joint mean half-spread is
at least the pre-event sham proposal mean. Its policy-seed 95% CI lower bound
must exceed `0.5`. Continuation loss requires both flow-share and reward-loss
95% CI lower bounds above zero; upward asymmetry requires both corresponding
undercut-minus-upward loss CI lower bounds above zero. Every policy seed must
contribute at least one feasible event to every required estimand.

If too few eligible events exist, the diagnostic reports its realized sample
and remains inconclusive; the threshold, paths, or cap are not changed.

## Paired Arms

Each selected pre-event state is cloned with the same exogenous continuation
and action uniforms:

- `sham`: execute the original joint action;
- `forced_undercut`: replace the selected dealer action with
  `max(original_action-1,0)`;
- `upward_deviation`: replace it with
  `min(original_action+1,6)`.

After the event action all dealers return to their frozen policies. There is
no routing intervention. Infeasible undercut or upward actions are retained
as explicit no-change events and excluded only from the corresponding
directional estimand by a predeclared feasibility flag.

## Measures

Store one event row plus step-level trajectories from `t0` through `t0+H`:
rival and deviator half-spreads, deviator flow share, immediate reward,
cumulative continuation reward, aggregate dealer reward, customer execution
half-spread, time to return to the pre-event mean spread, and rival spread
compression relative to sham.

Policy-seed aggregation precedes paired t CI and exact sign reporting. A
punishment-supported interpretation requires all of the following in the
frozen direction: rivals compress after an undercut relative to sham; the
deviator loses continuation flow/reward; the high-spread state recovers;
upward deviations are not punished symmetrically; and policy-seed CIs support
the pattern.

If any required component fails, the repository and paper must use
"high-spread coordination" or "supracompetitive/high-spread outcome in the
controlled simulator" rather than a strong collusion mechanism claim. No
normalized collusion index is constructed unless a later, separately audited
myopic reference passes unilateral-deviation consistency checks.
