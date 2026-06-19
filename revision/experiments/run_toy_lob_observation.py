#!/usr/bin/env python3
"""Step 1: no-intervention observation for the toy LOB environment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.independent_q_learning import IndependentQLearners, QLearningConfig
from src.envs.toy_lob_pricing_env import ToyLOBConfig, ToyLOBPricingEnv
from src.utils.metrics import enrich_step, recovery_summary, summarize_by_risk, summarize_records


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


def deviation_probe(config: ToyLOBConfig, agents: IndependentQLearners, seed: int, horizon: int = 30) -> Dict[str, float]:
    env = ToyLOBPricingEnv(config.with_risk_mode("low"))
    env.reset(seed=seed + 50_000)
    warmup_records: List[Dict[str, object]] = []
    for _ in range(40):
        states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
        actions = {i: agents.act(i, states[i], explore=False) for i in range(config.n_agents)}
        _, _, _, _, info = env.step(actions)
        warmup_records.append(enrich_step(config, info))
    pre_avg_spread = float(np.mean([float(row["avg_spread"]) for row in warmup_records[-10:]]))

    records: List[Dict[str, object]] = []
    deviator = 0
    for k in range(horizon):
        states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
        actions = {i: agents.act(i, states[i], explore=False) for i in range(config.n_agents)}
        if k == 0:
            actions[deviator] = max(0, actions[deviator] - 1)
        _, _, _, _, info = env.step(actions)
        row = enrich_step(config, info)
        spreads = np.array(row["spreads"], dtype=float)
        rewards = np.array(row["rewards"], dtype=float)
        row["deviator_spread"] = float(spreads[deviator])
        row["other_avg_spread"] = float(np.mean(np.delete(spreads, deviator)))
        row["deviator_reward"] = float(rewards[deviator])
        records.append(row)
    return recovery_summary(records, pre_avg_spread)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="revision/experiments/outputs/observation")
    parser.add_argument("--n-sweep", default="", help="Optional comma-separated N values for low-risk observation.")
    parser.add_argument("--n-sweep-episodes", type=int, default=300)
    args = parser.parse_args()

    base = ToyLOBConfig(seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {}

    for risk_mode in ["low", "high", "mixed"]:
        config = base.with_risk_mode(risk_mode)  # type: ignore[arg-type]
        agents, train_tail = train_no_intervention(config, args.episodes, args.seed)
        eval_records = evaluate_no_intervention(config, agents, args.eval_episodes, args.seed)
        write_records(output_dir / f"{risk_mode}_eval_records.csv", eval_records)
        summary[risk_mode] = {
            "train_tail": summarize_by_risk(train_tail),
            "eval_all": summarize_records(eval_records),
            "eval_by_risk": summarize_by_risk(eval_records),
        }
        if risk_mode == "low":
            summary[risk_mode]["deviation_probe"] = deviation_probe(config, agents, args.seed)

    if args.n_sweep:
        n_values = [int(x.strip()) for x in args.n_sweep.split(",") if x.strip()]
        sweep = {}
        for n_agents in n_values:
            config = replace(base, n_agents=n_agents, risk_mode="low")
            agents, _ = train_no_intervention(config, args.n_sweep_episodes, args.seed + n_agents)
            eval_records = evaluate_no_intervention(config, agents, args.eval_episodes, args.seed + n_agents)
            sweep[str(n_agents)] = {
                "eval_all": summarize_records(eval_records),
                "eval_by_risk": summarize_by_risk(eval_records),
            }
        summary["n_sweep_low_risk"] = sweep

    out = output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved observation summary to {out}")


if __name__ == "__main__":
    main()
