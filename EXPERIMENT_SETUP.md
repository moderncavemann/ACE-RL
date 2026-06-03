# ACE-RL Experiment Setup

This note records the experiment configuration found in this repository, with
emphasis on the long-running simulation batches.

## Main Long-Running LOB Experiment

Primary results directory:

- `experiments/final_real_lob_50k/`
- Config: `experiments/final_real_lob_50k/config.json`
- Results: `experiments/final_real_lob_50k/experiment_results.csv`
- Main log: `experiments/final_real_lob_50k/training.log`

Configuration:

- Environment: `use_gymnasium_lob=true`, `use_real_lob=false`
- Backend used: repository-local Gymnasium LOB environment, not official ABIDES
- Environment implementation: `ace_rl_project/envs/lob_env.py`
- Matching rule: price-time priority limit order book
- Agents: 3
- Methods: 8
  - `baseline`
  - `mappo`
  - `fixed_penalty`
  - `ppo_lagrangian`
  - `exponential_reward`
  - `periodic_auditing`
  - `action_shielding`
  - `ace_rl`
- Seeds: 42, 43, 44
- Episodes per method-seed run: 50,000
- Steps per episode: 2,000
- Total runs: 8 methods x 3 seeds = 24
- Total simulated environment steps: 24 x 50,000 x 2,000 = 2.4 billion
- Parallel workers: 21
- Warmup steps: 100
- Checkpoint interval: 1,000 episodes
- Snapshot interval: 50 episodes
- Snapshot trace length: 100 steps
- ACE-RL alpha: 50.0

Completion evidence:

- `experiments/final_real_lob_50k/logs/*_episodes.csv` contains 24 episode
  logs.
- Each episode log has 50,000 rows and ends at episode index 49,999.
- `experiments/final_real_lob_50k/experiment_results.csv` contains 24 completed
  method-seed rows.
- The main training log reports total runtime of `2225.2 minutes`, about
  37.1 hours.
- The directory is about 77GB, mostly due to dense snapshot logging.
- Snapshot count found locally: 47,952 files under
  `experiments/final_real_lob_50k/snapshots/`.

ABIDES note:

- `experiments/final_real_lob_50k/training.log` shows that ABIDES environments
  were registered but failed to initialize because of:
  `cannot import name 'GeneralMixtureModel' from 'pomegranate'`.
- The same log then records:
  `Using GYMNASIUM LOB environment (Original lob_env.py)`.
- Therefore the main 50k LOB experiment did not run on official ABIDES. It ran
  on the repository's own Gymnasium LOB matching engine.

## Full 50k Per-Job Run

Primary directory:

- `experiments/final_real_lob_50k_full/`
- Script: `run_final_real_lob.py`

Configuration found in `run_final_real_lob.py`:

- Methods: same 8 methods as above
- Seeds: 42, 43, 44
- Episodes per run: 50,000
- Steps per episode: 2,000
- Total runs: 24
- Max concurrent jobs: 21
- Environment flag: `--use_gymnasium_lob`
- Warmup steps: 100
- Snapshot interval: 1,000 episodes
- Output base: `experiments/final_real_lob_50k_full`
- ACE-RL alpha: 1000.0

Completion evidence:

- There are 24 per-run `training.log` files under
  `experiments/final_real_lob_50k_full/*/training.log`.
- Each log reaches `Ep 50000/50000` and has a final `DONE` line.
- The directory is about 13GB.

## Alpha Sensitivity Runs

Primary directories:

- `experiments/alpha_sensitivity/`
- `experiments/alpha_sensitivity_fast/`
- Script: `run_alpha_sensitivity.py`

Configuration found in `run_alpha_sensitivity.py`:

- Method: ACE-RL only
- Alpha values: 10 and 100
- Seeds: 42, 43, 44
- Episodes per run: 50,000
- Steps per episode: 2,000
- Total runs per directory: 2 alpha values x 3 seeds = 6
- Max concurrent jobs: 6
- Environment flag: `--use_gymnasium_lob`
- Snapshot interval: 1,000 episodes

Completion evidence:

- Each alpha sensitivity episode log has 50,000 rows and ends at episode index
  49,999.
- Each run log reaches `Ep 50000/50000` and has a final `DONE` line.

## Shorter Validation Runs

Additional completed experiment directories include:

- `experiments/gymnasium_lob/`
  - 8 methods x 3 seeds
  - 500 episodes
  - 200 steps per episode
  - `use_gymnasium_lob=true`
- `experiments/gymnasium_lob_10k/`
  - 8 methods x 3 seeds
  - 10,000 episodes
  - 1,000 steps per episode
  - `use_gymnasium_lob=true`
  - Main log reports total runtime of `307.0 minutes`, about 5.1 hours.
- `experiments/real_lob_abides/` and `experiments/real_lob_production/`
  - 8 methods x 3 seeds
  - 500 episodes
  - 200 steps per episode
  - `use_real_lob=true`
  - Results record `regime=real_lob`, but the persisted result CSV does not
    include the selected backend. The wrapper tries ABIDES first, then pure
    Python LOB, then fallback.
- `experiments/final_numerical_validation/`
  - 8 methods x 3 seeds
  - 500 episodes
  - 200 steps per episode
  - mock/calibrated environment path, not Gymnasium LOB.

## Relevant Code Paths

- Main production trainer:
  - `ace_rl_project/src/main_trainer.py`
- Gymnasium LOB wrapper in trainer:
  - `ace_rl_project/src/main_trainer.py`
  - class `GymnasiumLOBWrapper`
- Repository-local Gymnasium LOB:
  - `ace_rl_project/envs/lob_env.py`
- ABIDES / pure-Python / fallback wrapper:
  - `ace_rl_project/src/real_lob_wrapper.py`
- Pure Python LOB engine:
  - `ace_rl_project/src/pure_python_lob.py`
- Top-level simpler LOB environment:
  - `src/env/lob_sim.py`
  - `src/env/market_env.py`

## Reproduction Commands

Main 50k LOB experiment:

```bash
python ace_rl_project/src/main_trainer.py \
  --method all \
  --n_episodes 50000 \
  --n_seeds 3 \
  --n_workers 21 \
  --use_gymnasium_lob \
  --episode_length 2000 \
  --warmup_steps 100 \
  --checkpoint_interval 1000 \
  --snapshot_interval 50 \
  --ace_alpha 50.0 \
  --output_dir experiments/final_real_lob_50k
```

Per-job full 50k run:

```bash
python run_final_real_lob.py
```

Alpha sensitivity:

```bash
python run_alpha_sensitivity.py
```

## Summary

The long-running experiments were large-scale multi-agent market-making
simulations on the repository-local Gymnasium limit order book matching engine.
The main completed setting was 8 methods x 3 seeds x 50,000 episodes x 2,000
steps, or 2.4 billion simulated environment steps. ABIDES was attempted in the
runtime environment but failed to initialize, so the main long LOB experiments
used the local Gymnasium matching engine rather than official ABIDES.
