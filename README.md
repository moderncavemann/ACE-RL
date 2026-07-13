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
- `revision/notes/lob_mappo_structural_redesign_v2_summary.md` records the
  risk-coupled, no-spill platform-routing redesign and the final timing
  decision. Economic acceptance and seller response pass; MAPPO timing versus
  a risk-stratified equal-dose shuffled schedule fails, so full main remains
  no-go.
- `revision/notes/lob_mappo_state_contingent_contract_v3_summary.md` records the
  superseding state-contingent soft-rank contract. Fixed-policy, scripted
  oracle, fresh MAPPO screen, and held-out oracle timing all pass. This clears
  the mechanism gate for a full-main learned-gate experiment; it is not itself
  a completed full-main result.
- `revision/notes/abides_mappo_full_main_v1_summary.md` records the completed
  20-seed N=10 ABIDES/MAPPO full-main timing experiment, including equal-dose
  controls, paired inference, claim boundaries, runtime, and recovery records.
- `revision/notes/abides_mappo_adaptive_pilot_protocol.md` freezes the next
  five-training-seed adaptive-policy acceptance gate. It separates the
  completed frozen-policy timing result from the still-missing paper main.
- `revision/notes/abides_mappo_adaptive_pilot_v1_summary.md` records why the
  original joint gate fails after separate adaptive MAPPO training.
- `revision/notes/abides_mappo_adaptive_two_stage_protocol.md` freezes the
  fresh-seed follow-up that separates exogenous risk inference from current
  routing opportunity.
- `revision/notes/abides_mappo_adaptive_two_stage_extension_v1_summary.md`
  records the final checkpoint extension decision. Risk control remains good,
  but learned timing falls to 3/5 positive policy seeds versus exact-dose
  matched random and training remains nonconverged; adaptive paper main is
  therefore no-go under the frozen protocol.
- `revision/notes/abides_mappo_adaptive_v3_holdout_v1_summary.md` records the
  superseding structural redesign and fresh adaptive holdout. Pre-action
  risk-aware soft-rank routing beats exact-dose random after separate MAPPO
  adaptation on 5/5 fresh policy seeds; the 95% CI is strictly positive.
- `revision/notes/abides_mappo_adaptive_v3_readiness_v1.md` records that the
  repaired method may enter a frozen full-scale package with governance
  baselines, a routing ablation, and a second market ecology. It is not itself
  a completed adaptive paper main.
- `revision/notes/abides_mappo_adaptive_v3_full_scale_v1_summary.md` records
  the resulting frozen package. The primary base gate fails at 18/20 converged
  policies, so primary efficacy is not run; the independently preregistered
  fast-risk ecology passes all six-arm checks on 10/10 policy seeds. The
  overall adaptive paper main remains no-go.

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

For MAPPO-style local LOB pilot scripts, `--device` defaults to `auto`.
On Apple Silicon this selects PyTorch MPS when available; pass `--device cpu`
to exactly reproduce CPU pilot runs recorded in the notes.
