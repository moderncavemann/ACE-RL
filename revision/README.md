# Revision Workspace

This directory is the clean work area for the revised ACE-RL project.

Use the local-only `archive/v1_rejected_submission/` as a source archive, not as
active code. Migrate files here only after deciding they support the revised
claim.

Suggested structure:

- `src/` - revised implementation.
- `experiments/` - runnable experiment scripts. Generated outputs go under
  `experiments/outputs/` and are ignored by git.
- `paper/` - revised manuscript assets.
- `notes/` - review responses, experiment plans, and evidence audits.

## Toy LOB Sequence

The toy stage follows a strict order:

1. Calibrate canonical parameters:
   `python3 revision/experiments/calibrate_toy_lob_params.py`
2. Run no-intervention observation only:
   `python3 revision/experiments/run_toy_lob_observation.py --episodes 1500 --eval-episodes 50 --n-sweep 2,5,10 --n-sweep-episodes 1000`
3. If observation passes, run simple baselines:
   `python3 revision/experiments/run_toy_lob_baselines.py --seeds 0,1,2,3,4,5,6,7,8,9 --random-rate auto`
4. If baselines justify a gate, run gate ablations:
   `python3 revision/experiments/run_toy_lob_ablation.py`

Do not use intervention results as paper evidence until the no-intervention
observation shows the toy environment actually exhibits the target problem.

The tabular learner state includes risk, average spread, own spread, market
minimum spread, and spread range. The last two terms are required so a one-agent
undercut can be visible in an N-agent market.
