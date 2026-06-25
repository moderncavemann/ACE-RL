#!/usr/bin/env python3
"""Step 1: no-intervention observation for the toy LOB environment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.independent_q_learning import IndependentQLearners, QLearningConfig
from src.envs.toy_lob_pricing_env import ToyLOBConfig, ToyLOBPricingEnv
from src.utils.metrics import enrich_step, paired_recovery_summary, summarize_by_risk, summarize_records


def train_no_intervention(config: ToyLOBConfig, episodes: int, seed: int) -> Tuple[IndependentQLearners, List[Dict[str, object]]]:
    env = ToyLOBPricingEnv(config)
    agents = IndependentQLearners(
        QLearningConfig(n_agents=config.n_agents, n_actions=config.n_actions, seed=seed)
    )
    records: List[Dict[str, object]] = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
            actions = {i: agents.act(i, states[i], explore=True) for i in range(config.n_agents)}
            _, rewards, done, _, info = env.step(actions)
            for i in range(config.n_agents):
                agents.update(i, states[i], actions[i], rewards[i], env.state_for_agent(i), done)
            agents.step_clock()
            if ep >= max(0, episodes - 50):
                row = enrich_step(config, info)
                row["episode"] = ep
                records.append(row)
    return agents, records


def evaluate_no_intervention(config: ToyLOBConfig, agents: IndependentQLearners, episodes: int, seed: int) -> List[Dict[str, object]]:
    env = ToyLOBPricingEnv(config)
    records: List[Dict[str, object]] = []
    for ep in range(episodes):
        env.reset(seed=seed + 10_000 + ep)
        done = False
        while not done:
            states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
            actions = {i: agents.act(i, states[i], explore=False) for i in range(config.n_agents)}
            _, _, done, _, info = env.step(actions)
            row = enrich_step(config, info)
            row["episode"] = ep
            records.append(row)
    return records


def _collect_probe_path(
    env: ToyLOBPricingEnv,
    agents: IndependentQLearners,
    config: ToyLOBConfig,
    deviator: int,
    horizon: int,
    force_first_deviation: bool,
) -> tuple[List[Dict[str, object]], float]:
    actual_deviation = 0.0
    records: List[Dict[str, object]] = []
    for k in range(horizon):
        states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
        actions = {i: agents.act(i, states[i], explore=False) for i in range(config.n_agents)}
        if k == 0 and force_first_deviation:
            pre_action = actions[deviator]
            post_action = max(0, pre_action - 1)
            actions[deviator] = post_action
            actual_deviation = (
                config.spread_levels[pre_action] - config.spread_levels[post_action]
            )
        _, _, _, _, info = env.step(actions)
        row = enrich_step(config, info)
        spreads = np.array(row["spreads"], dtype=float)
        rewards = np.array(row["rewards"], dtype=float)
        row["deviator_spread"] = float(spreads[deviator])
        row["other_avg_spread"] = float(np.mean(np.delete(spreads, deviator)))
        row["deviator_reward"] = float(rewards[deviator])
        records.append(row)
    return records, float(actual_deviation)


def deviation_probe(config: ToyLOBConfig, agents: IndependentQLearners, seed: int, horizon: int = 30) -> Dict[str, object]:
    warmup_seed = seed + 50_000
    warmup_steps = 40
    deviator = 0

    warm_env = ToyLOBPricingEnv(config.with_risk_mode("low"))
    warm_env.reset(seed=warmup_seed)
    warmup_records: List[Dict[str, object]] = []
    for _ in range(warmup_steps):
        states = {i: warm_env.state_for_agent(i) for i in range(config.n_agents)}
        actions = {i: agents.act(i, states[i], explore=False) for i in range(config.n_agents)}
        _, _, _, _, info = warm_env.step(actions)
        warmup_records.append(enrich_step(config, info))
    pre_avg_spread = float(np.mean([float(row["avg_spread"]) for row in warmup_records[-10:]]))

    # Recreate the same warmup state twice by replaying deterministic greedy
    # actions from the same seed. Path A is control, path B forces one real
    # one-level spread cut for the deviator at the first probe step.
    control_env = ToyLOBPricingEnv(config.with_risk_mode("low"))
    forced_env = ToyLOBPricingEnv(config.with_risk_mode("low"))
    control_env.reset(seed=warmup_seed)
    forced_env.reset(seed=warmup_seed)
    for _ in range(warmup_steps):
        for replay_env in (control_env, forced_env):
            states = {i: replay_env.state_for_agent(i) for i in range(config.n_agents)}
            actions = {i: agents.act(i, states[i], explore=False) for i in range(config.n_agents)}
            replay_env.step(actions)

    control_records, _ = _collect_probe_path(control_env, agents, config, deviator, horizon, False)
    forced_records, actual_deviation = _collect_probe_path(forced_env, agents, config, deviator, horizon, True)
    if actual_deviation <= 0.0:
        return {
            "actual_deviation": 0.0,
            "probe_skipped": 1.0,
            "reason": "deviator already at minimum spread",
        }
    out = paired_recovery_summary(forced_records, control_records, pre_avg_spread, actual_deviation)
    out["probe_skipped"] = 0.0
    return out


def write_records(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_rows = []
    for row in records:
        scalar_rows.append({
            k: v for k, v in row.items()
            if isinstance(v, (str, int, float, bool, np.integer, np.floating))
        })
    if not scalar_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(scalar_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scalar_rows)


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _records_name(risk_mode: str, seed: int, legacy_names: bool) -> str:
    if legacy_names:
        return f"{risk_mode}_eval_records.csv"
    return f"seed{seed}_{risk_mode}_eval_records.csv"


def _summary_name(seed: int, legacy_names: bool) -> str:
    if legacy_names:
        return "summary.json"
    return f"seed{seed}_summary.json"


def run_observation_seed(
    seed: int,
    episodes: int,
    eval_episodes: int,
    output_dir: Path,
    n_sweep: str,
    n_sweep_episodes: int,
    legacy_names: bool,
) -> Dict[str, object]:
    base = ToyLOBConfig(seed=seed)
    summary: Dict[str, object] = {"metadata": {"seed": seed}}

    for risk_mode in ["low", "high", "mixed"]:
        config = base.with_risk_mode(risk_mode)  # type: ignore[arg-type]
        agents, train_tail = train_no_intervention(config, episodes, seed)
        eval_records = evaluate_no_intervention(config, agents, eval_episodes, seed)
        write_records(output_dir / _records_name(risk_mode, seed, legacy_names), eval_records)
        summary[risk_mode] = {
            "train_tail": summarize_by_risk(train_tail),
            "eval_all": summarize_records(eval_records),
            "eval_by_risk": summarize_by_risk(eval_records),
        }
        if risk_mode == "low":
            summary[risk_mode]["deviation_probe"] = deviation_probe(config, agents, seed)

    if n_sweep:
        n_values = [int(x.strip()) for x in n_sweep.split(",") if x.strip()]
        sweep = {}
        for n_agents in n_values:
            config = replace(base, n_agents=n_agents, risk_mode="low")
            agents, _ = train_no_intervention(config, n_sweep_episodes, seed + n_agents)
            eval_records = evaluate_no_intervention(config, agents, eval_episodes, seed + n_agents)
            sweep[str(n_agents)] = {
                "eval_all": summarize_records(eval_records),
                "eval_by_risk": summarize_by_risk(eval_records),
            }
        summary["n_sweep_low_risk"] = sweep

    (output_dir / _summary_name(seed, legacy_names)).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _run_observation_job(job: Tuple[int, int, int, Path, str, int, bool]) -> Tuple[int, Dict[str, object]]:
    seed, episodes, eval_episodes, output_dir, n_sweep, n_sweep_episodes, legacy_names = job
    return seed, run_observation_seed(
        seed,
        episodes,
        eval_episodes,
        output_dir,
        n_sweep,
        n_sweep_episodes,
        legacy_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seeds. Overrides --seed.")
    parser.add_argument("--output-dir", default="revision/experiments/outputs/observation")
    parser.add_argument("--n-sweep", default="", help="Optional comma-separated N values for low-risk observation.")
    parser.add_argument("--n-sweep-episodes", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes across seeds.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds) if args.seeds else [args.seed]
    legacy_names = args.seeds is None and len(seeds) == 1
    observation_jobs = [
        (seed, args.episodes, args.eval_episodes, output_dir, args.n_sweep, args.n_sweep_episodes, legacy_names)
        for seed in seeds
    ]
    if args.jobs <= 1 or len(observation_jobs) <= 1:
        seed_summaries = [_run_observation_job(job) for job in observation_jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            seed_summaries = list(executor.map(_run_observation_job, observation_jobs))

    if legacy_names:
        summary = seed_summaries[0][1]
    else:
        summary = {
            "metadata": {
                "episodes": args.episodes,
                "eval_episodes": args.eval_episodes,
                "seeds": seeds,
                "jobs": args.jobs,
                "n_sweep": args.n_sweep,
                "n_sweep_episodes": args.n_sweep_episodes,
            },
            "seeds": {str(seed): seed_summary for seed, seed_summary in sorted(seed_summaries)},
        }

    out = output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved observation summary to {out}")


if __name__ == "__main__":
    main()
