# ABIDES Collusion Diagnostic v1 Audit Bundle

This bundle is the repo-visible, audit-derived export of the preregistered
forced-undercut event study. It contains 20 policy seeds, 400 paired events,
and step records for sham, forced-undercut, and upward-deviation arms from
`t0` through `t0+10`.

External interpretation is fail-closed. The result supports only
`high-spread coordination`; it does not establish punishment-supported
collusion. Run `python revision/experiments/verify_abides_collusion_diagnostic.py`
to recompute path integrity, policy-seed intervals, exact sign tests, and the
claim boundary.
