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
- `revision/notes/toy_full_n10_experiment_v1_summary.md` records the
  paper-ready N=10 toy full confirmation package.
- `revision/notes/toy_lob_robustness_sweep_v1_summary.md` records the
  controlled robustness sweep and the N=20 caveat.
- `revision/notes/toy_lob_n20_followup_v1_summary.md` records the longer N=20
  follow-up.
- `revision/notes/toy_lob_n_invariant_gate_diagnostic_v1_summary.md` records
  the N-invariant feature diagnostic and gate-v2 plumbing.
- `revision/notes/toy_lob_n20_v2_stress_protocol.md` records the formal N=20
  gate-v2 stress-test protocol.
- `revision/notes/lob_marl_pilot_v1_summary.md` records the first local
  matching-engine LOB/MARL pilot bridge. This is not a full MAPPO result.
- `revision/notes/lob_mappo_pilot_v1_summary.md` records the first local
  LOB/MAPPO-style pilot. The pipeline runs, but full main experiments are
  blocked by MAPPO gate activation drift.

Gate-v2 feature diagnostic and calibration:

```bash
python revision/experiments/diagnose_toy_lob_n_invariant_gate.py --inputs N5=revision/experiments/outputs/observation_n5_n_invariant_v1,N10=revision/experiments/outputs/observation_n10_n_invariant_v1,N20=revision/experiments/outputs/observation_n20_n_invariant_v1 --calibration-label N10 --base-gate-config revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json --output-dir revision/experiments/outputs/n_invariant_gate_diagnostic_v1
```

N=20 gate-v2 engineering smoke:

```bash
python revision/experiments/run_toy_lob_robustness.py --episodes 5 --eval-episodes 1 --seeds 85 --scenarios n20 --methods no_intervention,gated_intervention,spread_only_gate,v2_entropy_gate,v2_fraction_near_min_gate,v2_two_stage_gate --gate-config revision/experiments/outputs/n_invariant_gate_diagnostic_v1/v2_gate_config.json --jobs 1 --output revision/experiments/outputs/n20_v2_gate_smoke_v1/summary.json --metrics-csv revision/experiments/outputs/n20_v2_gate_smoke_v1/metrics.csv
```

Formal N=20 gate-v2 stress-test runner:

```bash
python revision/experiments/run_toy_lob_n20_v2_sweep.py --episodes 800 --eval-episodes 50 --seeds 90-109 --jobs 6 --gate-config revision/experiments/outputs/n_invariant_gate_diagnostic_v1/v2_gate_config.json --output-dir revision/experiments/outputs/n20_v2_formal_sweep_v1
```

## Local LOB/MARL Pilot

The first migration pilot uses a local price-time priority matching engine and
independent Q learners to verify the LOB measurement/intervention path before a
MAPPO adapter is attached.

Calibrate the LOB gate from no-intervention records:

```bash
python revision/experiments/calibrate_lob_gate.py --episodes 30 --eval-episodes 5 --episode-length 200 --n-agents 10 --seeds 0-4 --target-activation 0.05,0.35 --output-dir revision/experiments/outputs/lob_gate_calibration_v1
```

Run the pilot with the frozen gate config and matched random baseline:

```bash
python revision/experiments/run_lob_marl_pilot.py --episodes 30 --eval-episodes 5 --episode-length 200 --n-agents 10 --seeds 0-4 --jobs 5 --gate-config revision/experiments/outputs/lob_gate_calibration_v1/lob_gate_config.json --output revision/experiments/outputs/lob_marl_pilot_v1/summary.json --metrics-csv revision/experiments/outputs/lob_marl_pilot_v1/metrics.csv
```
