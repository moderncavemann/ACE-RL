# ACE-RL Revision Triage

Date: 2026-06-19

## Status

The v1 submission assets are archived locally under
`archive/v1_rejected_submission/`. That directory is ignored by git; paths below
are local workstation references. The root directory is now reserved for the
revised project.

## Evidence Audit

| Area | v1 Location | Current Assessment | Revision Action |
| --- | --- | --- | --- |
| Mock/calibrated result | `archive/v1_rejected_submission/evidence_summaries/experiments/final_numerical_validation/summary_table.csv` | Supports the old headline result: ACE-RL has low `M` and high reward in the calibrated/mock path. | Keep as diagnostic or toy evidence only unless the paper explicitly scopes it as such. |
| Long Gymnasium LOB result | `archive/v1_rejected_submission/evidence_summaries/experiments/final_real_lob_50k/summary_table.csv` | Does not support the old headline claim: ACE-RL and baseline have near-equal `final_M`. | This is the main blocker for the old paper narrative. |
| ABIDES claim | `archive/v1_rejected_submission/EXPERIMENT_SETUP.md` | ABIDES failed at runtime and the main long runs used repository-local Gymnasium LOB. | Do not describe the main 50k runs as official ABIDES results. |
| ACE-RL implementation | `archive/v1_rejected_submission/ace_rl_project/src/agent.py` | Contains real gradient-surgery code, not only reward shaping. | Candidate code to migrate after tests and claim are clarified. |
| Root quick test | `archive/v1_rejected_submission/test_installation.py` | Failed at LOB assertion due to exact float comparison. | Fix only if this path remains part of the revision. |

## Claims To Freeze

Do not reuse these v1 claims in the revised paper without new evidence:

- ACE-RL is Pareto optimal in the realistic LOB setting.
- ACE-RL improves profitability by 3.9x in the main environment.
- Reward-shaping baselines universally converge to the same washout equilibrium.
- The main long-run evidence used official ABIDES.

## Candidate Revision Directions

1. Repair the algorithm and rerun realistic LOB experiments.
2. Reframe the contribution around when gradient surgery works and when it
   fails under more realistic microstructure.
3. Separate toy/mechanism evidence from realistic-market evidence with strict
   scope labels.
4. Turn the project into a benchmark/evaluation paper if the realistic result
   remains negative.

## Review Intake

Paste the rejection reviews below before rewriting.

### Reviewer 1

TBD

### Reviewer 2

TBD

### Reviewer 3

TBD
