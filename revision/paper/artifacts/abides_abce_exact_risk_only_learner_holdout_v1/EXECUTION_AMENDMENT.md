# ABCE v1 Execution Amendment

**Status:** frozen with the protocol before formal outcomes are opened.

This amendment changes no hypothesis, market parameter, endpoint, seed, budget,
or admissibility gate in the ABCE protocol. It records two execution controls
needed to preserve provenance.

## Derived runtime image

The protocol names `ace-rl-abides-ace:adaptive-v3`, but the new C1 and C2
allocation variants do not exist in that immutable image. The executable image
is therefore `ace-rl-abides-ace:abce-v1`, built `FROM` the exact adaptive-v3
image and replacing only the ACE adapter, worker, and mechanism module. The
runner records and verifies:

- the frozen upstream ABIDES commit;
- the adaptive-v3 parent image ID;
- the ABCE derived image ID;
- the ABCE adapter source-bundle SHA-256;
- the host Git commit and protocol/config fingerprints.

The adaptive-v3 tag is never overwritten. A deterministic parity smoke must
show identical no-routing and ordinary soft-rank behavior before formal launch.

## Paired response retraining

All nine compact ABCE arms are retrained for 96 response episodes from the
same frozen seed-specific base checkpoint. They use the same response
environment-seed and action-uniform namespaces in the same run. Existing v5
response checkpoints remain prior replication evidence but are not reused in
the A/B/C causal contrasts. This removes response-training history as a
possible difference between newly added controls and earlier arms; it changes
neither the base populations nor any formal budget or endpoint.

## Development and formal isolation

Outcome-guided tuning is permitted only in the development lane, using frozen
base populations `510000-510019` and development-only path namespaces. The
formal populations `530000-530019`, final paths `534000-534009`, official
learner populations `560000-560009`, and learner final paths `562000-562009`
must not be read by development selection code.

The development lane may repair implementation errors and select among an
explicit finite parameter list recorded in its config. It may not replace
failed seeds or silently extend a budget. Once the formal config is committed
and a formal result file exists, no intervention-outcome tuning is allowed.
Weak, null, or reversed formal effects trigger the protocol's claim-narrowing
rules, not another search.
