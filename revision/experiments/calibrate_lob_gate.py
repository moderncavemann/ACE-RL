#!/usr/bin/env python3
"""Calibrate the local LOB pilot gate from no-intervention records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.independent_q_learning import IndependentQLearners, QLearningConfig
from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv
from src.interventions.lob_gates import LOBGateConfig, LOBInterventionGate


Record = Dict[str, float | str]


def parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in [x.strip() for x in raw.split(",") if x.strip()]:
        if "-" in part:
            lo, hi = [int(x.strip()) for x in part.split("-", 1)]
            step = 1 if hi >= lo else -1
            seeds.extend(range(lo, hi + step, step))
        else:
            seeds.append(int(part))
    return seeds


def parse_target(raw: str) -> Tuple[float, float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError("--target-activation must be low,high")
    low, high = vals
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("--target-activation values must satisfy 0 <= low <= high <= 1")
    return low, high


def train_no_intervention(config: LocalLOBConfig, episodes: int, seed: int) -> IndependentQLearners:
    env = LocalLOBMarketEnv(config)
    agents = IndependentQLearners(
        QLearningConfig(
            n_agents=config.n_agents,
            n_actions=config.n_actions,
            seed=seed,
            epsilon_decay_steps=max(1, episodes * config.episode_length),
        )
    )
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
            actions = {i: agents.act(i, states[i], explore=True) for i in range(config.n_agents)}
            _, rewards, done, _, _ = env.step(actions)
            for i in range(config.n_agents):
                agents.update(i, states[i], actions[i], rewards[i], env.state_for_agent(i), done)
            agents.step_clock()
    return agents


def collect_records(
    config: LocalLOBConfig,
    agents: IndependentQLearners,
    episodes: int,
    seed: int,
    gate_config: LOBGateConfig,
) -> List[Record]:
    env = LocalLOBMarketEnv(config)
    gate = LOBInterventionGate(config, gate_config, "risk_aware")
    records: List[Record] = []
    for ep in range(episodes):
        env.reset(seed=seed + 20_000 + ep)
        gate.reset()
        done = False
        while not done:
            states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
            actions = {i: agents.act(i, states[i], explore=False) for i in range(config.n_agents)}
            spreads = [config.half_spread_levels[actions[i]] for i in range(config.n_agents)]
            decision = gate.decide(env.risk_state, spreads, env.last_flow_shares, env.last_total_volume)
            _, _, done, _, info = env.step(actions)
            records.append({
                "seed": float(seed),
                "episode": float(ep),
                "risk_state": str(info["risk_state"]),
                "avg_agent_spread": float(info["avg_agent_spread"]),
                "best_spread_ticks": float(info["best_spread_ticks"]),
                "total_volume": float(info["total_volume"]),
                "gate_score": float(decision["score"]),
                "gate_excess": float(decision["excess"]),
                "gate_spread_entropy": float(decision["spread_entropy"]),
                "gate_flow_concentration": float(decision["flow_concentration"]),
                "gate_volume_suppression": float(decision["volume_suppression"]),
            })
    return records


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return {"n": 0.0}
    out = {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    for q in (10, 25, 50, 75, 90, 95):
        out[f"p{q}"] = float(np.percentile(arr, q))
    return out


def evaluate_threshold(records: List[Record], threshold: float, high_spread_tau: float) -> Dict[str, float]:
    scores = np.array([float(row["gate_score"]) for row in records], dtype=float)
    active = scores > threshold
    risk = np.array([str(row["risk_state"]) for row in records])
    spread = np.array([float(row["avg_agent_spread"]) for row in records], dtype=float)
    low_high_spread = (risk == "low") & (spread > high_spread_tau)
    high_normal = (risk == "high") & (spread <= high_spread_tau)

    def rate(mask: np.ndarray) -> float:
        return float(np.mean(active[mask])) if bool(np.any(mask)) else 0.0

    return {
        "threshold": float(threshold),
        "activation_rate": float(np.mean(active)) if len(active) else 0.0,
        "tpr_low_high_spread": rate(low_high_spread),
        "tpr_low_high_spread_n": float(np.sum(low_high_spread)),
        "fpr_high_normal_spread": rate(high_normal),
        "fpr_high_normal_spread_n": float(np.sum(high_normal)),
    }


def choose_threshold(
    records: List[Record],
    high_spread_tau: float,
    target_activation: Tuple[float, float],
    max_fpr: float,
    grid_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    scores = np.array([float(row["gate_score"]) for row in records], dtype=float)
    if len(scores) == 0:
        raise ValueError("No records for calibration")
    candidates = set(float(np.percentile(scores, q)) for q in np.linspace(0, 100, grid_size))
    candidates.add(float(np.min(scores) - 1e-9))
    candidates.add(float(np.max(scores) + 1e-9))
    table = [evaluate_threshold(records, threshold, high_spread_tau) for threshold in sorted(candidates)]
    target_low, target_high = target_activation
    target_mid = (target_low + target_high) / 2.0

    def activation_penalty(row: Dict[str, float]) -> float:
        activation = row["activation_rate"]
        if target_low <= activation <= target_high:
            return 0.0
        return min(abs(activation - target_low), abs(activation - target_high))

    for row in table:
        row["in_target_activation"] = float(target_low <= row["activation_rate"] <= target_high)
        row["objective"] = (
            row["tpr_low_high_spread"]
            - 1.25 * row["fpr_high_normal_spread"]
            - 0.75 * activation_penalty(row)
            - 0.05 * abs(row["activation_rate"] - target_mid)
        )
    feasible = [
        row for row in table
        if row["in_target_activation"] and row["fpr_high_normal_spread"] <= max_fpr
    ]
    pool = feasible if feasible else table
    selected = max(
        pool,
        key=lambda row: (
            row["objective"],
            row["tpr_low_high_spread"],
            -row["fpr_high_normal_spread"],
            -abs(row["activation_rate"] - target_mid),
        ),
    )
    selected["selection_feasible"] = float(bool(feasible))
    return selected, table


def write_records(path: Path, records: List[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_table(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--target-activation", default="0.05,0.40")
    parser.add_argument("--max-fpr", type=float, default=0.25)
    parser.add_argument("--high-spread-tau", type=float, default=4.0)
    parser.add_argument("--grid-size", type=int, default=501)
    parser.add_argument("--output-dir", default="revision/experiments/outputs/lob_gate_calibration_v1")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    base_config = LocalLOBConfig(n_agents=args.n_agents, episode_length=args.episode_length, seed=seeds[0] if seeds else 0)
    base_gate_config = LOBGateConfig()
    all_records: List[Record] = []
    for seed in seeds:
        config = replace(base_config, seed=seed)
        agents = train_no_intervention(config, args.episodes, seed)
        all_records.extend(collect_records(config, agents, args.eval_episodes, seed, base_gate_config))

    target_activation = parse_target(args.target_activation)
    selected, table = choose_threshold(
        all_records,
        high_spread_tau=args.high_spread_tau,
        target_activation=target_activation,
        max_fpr=args.max_fpr,
        grid_size=args.grid_size,
    )
    calibrated_gate_config = LOBGateConfig(**{**vars(base_gate_config), "risk_threshold": float(selected["threshold"])})

    output_dir = Path(args.output_dir)
    write_records(output_dir / "calibration_records.csv", all_records)
    write_table(output_dir / "threshold_grid.csv", table)
    output = {
        "metadata": {
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "episode_length": args.episode_length,
            "n_agents": args.n_agents,
            "seeds": seeds,
            "target_activation": {"low": target_activation[0], "high": target_activation[1]},
            "max_fpr": args.max_fpr,
            "high_spread_tau": args.high_spread_tau,
            "grid_size": args.grid_size,
            "n_records": len(all_records),
        },
        "selected_threshold": selected,
        "score_distribution": summarize(float(row["gate_score"]) for row in all_records),
        "gate_config": asdict(calibrated_gate_config),
    }
    (output_dir / "lob_gate_config.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nSaved LOB gate calibration to {output_dir}")


if __name__ == "__main__":
    main()
