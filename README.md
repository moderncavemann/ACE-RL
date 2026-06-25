# ACE-RL Revision Workspace

This repository root is now the clean workspace for the post-rejection
revision.

On this workstation, the original submission-era project has been archived in:

- `archive/v1_rejected_submission/`

That archive is intentionally ignored by git. Treat it as read-only local
evidence unless a file is deliberately migrated into `revision/`.

## Current Layout

- `archive/v1_rejected_submission/` - local-only v1 rejected-submission assets
  moved from the old repository root. This directory is ignored by git.
- `revision/` - clean working area for the revised paper/codebase.
- `REVISION_TRIAGE.md` - current evidence audit and rewrite checklist.
- `docs/` - lightweight notes for the revised project.

## Working Rule

Do not reuse v1 paper claims until they are re-audited against the relevant
experiment backend.

The main known issue is that the strong mock/calibrated result in v1 does not
carry over to the long Gymnasium LOB runs. The revision should either repair
that gap experimentally or change the paper claim.

## Next Steps

1. Add the rejection reviews to `REVISION_TRIAGE.md`.
2. Decide the revised core claim before porting code.
3. Migrate only the minimum needed implementation from the v1 archive.
4. Rebuild reproducibility from small tests first, then long runs.

## Toy LOB Diagnostics

The current toy-scale validation notes are under `revision/notes/`.

- `revision/notes/toy_lob_confirmation_sweep_v1_summary.md` records the
  toy-scale confirmation sweep.
- `revision/notes/toy_lob_robustness_sweep_v1_summary.md` records the
  controlled robustness sweep and the N=20 caveat.
- `revision/notes/toy_lob_n20_followup_v1_summary.md` records the longer N=20
  follow-up.
- `revision/notes/toy_lob_n_invariant_gate_diagnostic_v1_summary.md` records
  the N-invariant feature diagnostic and gate-v2 plumbing.

Gate-v2 feature diagnostic and calibration:

```bash
python revision/experiments/diagnose_toy_lob_n_invariant_gate.py --inputs N5=revision/experiments/outputs/observation_n5_n_invariant_v1,N10=revision/experiments/outputs/observation_n10_n_invariant_v1,N20=revision/experiments/outputs/observation_n20_n_invariant_v1 --calibration-label N10 --base-gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output-dir revision/experiments/outputs/n_invariant_gate_diagnostic_v1
```

N=20 gate-v2 engineering smoke:

```bash
python revision/experiments/run_toy_lob_robustness.py --episodes 5 --eval-episodes 1 --seeds 85 --scenarios n20 --methods no_intervention,gated_intervention,spread_only_gate,v2_entropy_gate,v2_fraction_near_min_gate,v2_two_stage_gate --gate-config revision/experiments/outputs/n_invariant_gate_diagnostic_v1/v2_gate_config.json --jobs 1 --output revision/experiments/outputs/n20_v2_gate_smoke_v1/summary.json --metrics-csv revision/experiments/outputs/n20_v2_gate_smoke_v1/metrics.csv
```
