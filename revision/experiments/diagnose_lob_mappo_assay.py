#!/usr/bin/env python3
"""No-intervention assay diagnostics for the local LOB/MAPPO pilot.

This is not an intervention experiment. It checks whether the local LOB/MAPPO
environment reliably produces a high-spread failure mode before routing/gating
comparisons are treated as meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lob_mappo_pilot import (  # noqa: E402
    agent_observations,
    choose_device,
    critic_observations,
    rollout_training_episode,
)
from run_lob_marl_pilot import mean_se, parse_seeds  # noqa: E402
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv  # noqa: E402
from src.interventions.lob_gates import LOBGateConfig, LOBInterventionGate  # noqa: E402


@dataclass(frozen=True)
class ScenarioSpec:
    env: Dict[str, object]
    mappo: Dict[str, object]
    episodes: int | None = None


SCENARIOS: Dict[str, ScenarioSpec] = {
    "base": ScenarioSpec(env={}, mappo={}),
    "low_entropy": ScenarioSpec(env={}, mappo={"entropy_coef": 0.0}),
    "entropy_001": ScenarioSpec(env={}, mappo={"entropy_coef": 0.001}),
    "lr_1e3": ScenarioSpec(env={}, mappo={"learning_rate": 1e-3}),
    "lr_1e4": ScenarioSpec(env={}, mappo={"learning_rate": 1e-4}),
    "inventory_high": ScenarioSpec(env={"inventory_penalty": 0.006}, mappo={}),
    "inventory_low": ScenarioSpec(env={"inventory_penalty": 0.0005}, mappo={}),
    "flow_low": ScenarioSpec(env={"base_flow_intensity": 1.0}, mappo={}),
    "flow_high": ScenarioSpec(env={"base_flow_intensity": 4.0}, mappo={}),
    "risk_persistent": ScenarioSpec(env={"p_low_to_high": 0.02, "p_high_to_low": 0.05}, mappo={}),
    "longer100": ScenarioSpec(env={}, mappo={}, episodes=100),
}


Record = Dict[str, float | str]


def parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_csv(raw: str) -> List[int]:
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def flow_concentration(flow_shares: Sequence[float]) -> float:
    arr = np.asarray(flow_shares, dtype=float)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    arr = arr / total
    return float(np.sum(arr ** 2))


def normalized_entropy(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    probs = arr[arr > 0.0] / total
    if len(probs) <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / np.log(len(arr)))


def action_distribution(actions: np.ndarray, config: LocalLOBConfig) -> Dict[str, float]:
    counts = np.bincount(actions.astype(int), minlength=config.n_actions).astype(float)
    shares = counts / max(float(np.sum(counts)), 1.0)
    out = {
        "action_mean": float(np.mean(actions)),
        "action_std": float(np.std(actions)),
        "action_alignment": float(np.max(shares)),
        "action_entropy": normalized_entropy(counts),
        "spread_min": float(np.min([config.half_spread_levels[int(a)] for a in actions])),
        "spread_max": float(np.max([config.half_spread_levels[int(a)] for a in actions])),
    }
    for idx, level in enumerate(config.half_spread_levels):
        out[f"spread_frac_{level}"] = float(shares[idx])
    return out


def summarize_records(records: Sequence[Record]) -> Dict[str, float]:
    if not records:
        return {}
    fields = sorted({
        key for row in records for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    })
    out: Dict[str, float] = {"n_steps": float(len(records))}
    for field in fields:
        values = [float(row[field]) for row in records if field in row]
        if values:
            out[f"{field}_mean"] = float(np.mean(values))
            out[f"{field}_std"] = float(np.std(values))
    return out


def evaluate_policy(
    config: LocalLOBConfig,
    trainer: MAPPOTrainer,
    seed: int,
    eval_episodes: int,
) -> List[Record]:
    env = LocalLOBMarketEnv(config)
    records: List[Record] = []
    for ep in range(eval_episodes):
        env.reset(seed=seed + 20_000 + ep)
        done = False
        while not done:
            obs = agent_observations(env)
            critic_obs = critic_observations(env, obs)
            actions, _, _ = trainer.act(obs, critic_obs, deterministic=True)
            _, _, done, _, info = env.step(
                {i: int(actions[i]) for i in range(config.n_agents)},
                routing_flow_boost=0.0,
                gate_active=False,
                gate_score=0.0,
            )
            row: Record = {
                "seed": float(seed),
                "episode": float(ep),
                "t": float(info.get("t", 0.0)),
                "risk_state": str(info["risk_state"]),
                "risk_high": 1.0 if info["risk_state"] == "high" else 0.0,
                "avg_agent_spread": float(info["avg_agent_spread"]),
                "best_spread_ticks": float(info["best_spread_ticks"]),
                "total_volume": float(info["total_volume"]),
                "mean_reward": float(info["mean_reward"]),
                "mean_abs_inventory": float(info["mean_abs_inventory"]),
                "flow_concentration": flow_concentration(info["flow_shares"]),  # type: ignore[arg-type]
            }
            row.update(action_distribution(actions, config))
            records.append(row)
    return records


def make_trainer(config: LocalLOBConfig, seed: int, mappo_base: MAPPOConfig) -> MAPPOTrainer:
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    obs = agent_observations(env)
    critic_obs = critic_observations(env, obs)
    return MAPPOTrainer(MAPPOConfig(
        **{
            **asdict(mappo_base),
            "obs_dim": int(obs.shape[1]),
            "critic_obs_dim": int(critic_obs.shape[1]),
            "n_actions": config.n_actions,
            "seed": seed,
        }
    ))


def run_seed(
    config: LocalLOBConfig,
    mappo_base: MAPPOConfig,
    seed: int,
    episodes: int,
    eval_episodes: int,
    checkpoint_episodes: Sequence[int],
    checkpoint_eval_episodes: int,
) -> Dict[str, object]:
    env = LocalLOBMarketEnv(config)
    trainer = make_trainer(config, seed, mappo_base)
    gate = LOBInterventionGate(config, LOBGateConfig(), "none")
    training_rows: List[Dict[str, float]] = []
    learning_curve: List[Dict[str, float]] = []
    checkpoints = set(int(x) for x in checkpoint_episodes if int(x) > 0)

    for ep in range(1, episodes + 1):
        stats = rollout_training_episode(env, trainer, gate, routing_rho=0.0, seed=seed + ep - 1)
        training_rows.append(stats)
        if ep in checkpoints:
            eval_records = evaluate_policy(config, trainer, seed + 100_000 + ep, checkpoint_eval_episodes)
            eval_summary = summarize_records(eval_records)
            learning_curve.append({
                "seed": float(seed),
                "episode": float(ep),
                "avg_agent_spread": float(eval_summary.get("avg_agent_spread_mean", 0.0)),
                "best_spread_ticks": float(eval_summary.get("best_spread_ticks_mean", 0.0)),
                "mean_reward": float(eval_summary.get("mean_reward_mean", 0.0)),
                "action_alignment": float(eval_summary.get("action_alignment_mean", 0.0)),
                "action_entropy": float(eval_summary.get("action_entropy_mean", 0.0)),
            })

    final_records = evaluate_policy(config, trainer, seed, eval_episodes)
    final_summary = summarize_records(final_records)
    by_risk = {
        risk: summarize_records([row for row in final_records if row["risk_state"] == risk])
        for risk in ("low", "high")
    }
    return {
        "seed": seed,
        "training": {
            key: float(np.mean([row[key] for row in training_rows])) if training_rows else 0.0
            for key in ("loss", "policy_loss", "value_loss", "entropy", "return_mean")
        },
        "final_eval": final_summary,
        "final_eval_by_risk": by_risk,
        "learning_curve": learning_curve,
    }


def aggregate_seed_results(
    seed_results: Sequence[Dict[str, object]],
    failure_threshold: float,
    strong_failure_threshold: float,
) -> Dict[str, object]:
    final_rows = [dict(row["final_eval"]) for row in seed_results]
    training_rows = [dict(row["training"]) for row in seed_results]
    final_keys = sorted({key for row in final_rows for key in row})
    training_keys = sorted({key for row in training_rows for key in row})
    spreads = [float(row.get("avg_agent_spread_mean", 0.0)) for row in final_rows]
    return {
        "n_seeds": float(len(seed_results)),
        "failure_threshold": float(failure_threshold),
        "strong_failure_threshold": float(strong_failure_threshold),
        "failure_rate": float(np.mean([spread >= failure_threshold for spread in spreads])) if spreads else 0.0,
        "strong_failure_rate": float(np.mean([spread >= strong_failure_threshold for spread in spreads])) if spreads else 0.0,
        "failure_seed_count": float(np.sum([spread >= failure_threshold for spread in spreads])) if spreads else 0.0,
        "strong_failure_seed_count": float(np.sum([spread >= strong_failure_threshold for spread in spreads])) if spreads else 0.0,
        "final_eval": {
            key: mean_se(float(row.get(key, 0.0)) for row in final_rows)
            for key in final_keys
        },
        "training": {
            key: mean_se(float(row.get(key, 0.0)) for row in training_rows)
            for key in training_keys
        },
    }


def write_seed_csv(path: Path, summary: Dict[str, object]) -> None:
    rows: List[Dict[str, object]] = []
    for scenario, block in dict(summary["scenarios"]).items():
        for result in dict(block)["seed_results"]:  # type: ignore[assignment]
            final_eval = dict(dict(result)["final_eval"])
            training = dict(dict(result)["training"])
            row: Dict[str, object] = {
                "scenario": scenario,
                "seed": dict(result)["seed"],
                "avg_agent_spread": final_eval.get("avg_agent_spread_mean", 0.0),
                "best_spread_ticks": final_eval.get("best_spread_ticks_mean", 0.0),
                "mean_reward": final_eval.get("mean_reward_mean", 0.0),
                "total_volume": final_eval.get("total_volume_mean", 0.0),
                "flow_concentration": final_eval.get("flow_concentration_mean", 0.0),
                "action_alignment": final_eval.get("action_alignment_mean", 0.0),
                "action_entropy": final_eval.get("action_entropy_mean", 0.0),
                "spread_frac_1": final_eval.get("spread_frac_1_mean", 0.0),
                "spread_frac_7": final_eval.get("spread_frac_7_mean", 0.0),
                "train_entropy": training.get("entropy", 0.0),
                "train_return": training.get("return_mean", 0.0),
            }
            rows.append(row)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_learning_curve_csv(path: Path, summary: Dict[str, object]) -> None:
    rows: List[Dict[str, object]] = []
    for scenario, block in dict(summary["scenarios"]).items():
        for result in dict(block)["seed_results"]:  # type: ignore[assignment]
            for curve in dict(result).get("learning_curve", []):  # type: ignore[union-attr]
                rows.append({"scenario": scenario, **dict(curve)})
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--scenarios", default="base")
    parser.add_argument("--checkpoint-episodes", default="")
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=2)
    parser.add_argument("--failure-spread-threshold", type=float, default=3.0)
    parser.add_argument("--strong-failure-spread-threshold", type=float, default=4.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--output-dir", default="revision/experiments/outputs/lob_mappo_no_intervention_assay_v1")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    scenario_names = parse_csv(args.scenarios)
    unknown = [name for name in scenario_names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenarios: {', '.join(unknown)}")
    checkpoints = parse_int_csv(args.checkpoint_episodes)
    device = choose_device(args.device)

    base_config = LocalLOBConfig(
        n_agents=args.n_agents,
        episode_length=args.episode_length,
        seed=seeds[0] if seeds else 0,
    )
    base_mappo = MAPPOConfig(
        obs_dim=1,
        critic_obs_dim=1,
        n_actions=base_config.n_actions,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )

    scenarios_out: Dict[str, object] = {}
    for scenario_name in scenario_names:
        spec = SCENARIOS[scenario_name]
        config = replace(base_config, **spec.env)
        mappo_base = MAPPOConfig(**{**asdict(base_mappo), **spec.mappo})
        episodes = max(args.episodes, int(spec.episodes)) if spec.episodes is not None else args.episodes
        seed_results: List[Dict[str, object]] = []
        for seed in seeds:
            print(f"[{scenario_name}] seed={seed} episodes={episodes}", flush=True)
            seed_results.append(
                run_seed(
                    replace(config, seed=seed),
                    mappo_base,
                    seed,
                    episodes,
                    args.eval_episodes,
                    checkpoints,
                    args.checkpoint_eval_episodes,
                )
            )
        aggregate = aggregate_seed_results(
            seed_results,
            args.failure_spread_threshold,
            args.strong_failure_spread_threshold,
        )
        scenarios_out[scenario_name] = {
            "config": asdict(config),
            "mappo_config": asdict(mappo_base),
            "episodes": episodes,
            "seed_results": seed_results,
            "aggregate": aggregate,
            "passes_assay_failure_gate": bool(float(aggregate["failure_rate"]) >= 0.70),
        }

    ranked = sorted(
        [
            {
                "scenario": name,
                "episodes": dict(block)["episodes"],
                "failure_rate": dict(dict(block)["aggregate"])["failure_rate"],
                "strong_failure_rate": dict(dict(block)["aggregate"])["strong_failure_rate"],
                "avg_agent_spread": dict(dict(dict(block)["aggregate"])["final_eval"])["avg_agent_spread_mean"]["mean"],
                "action_alignment": dict(dict(dict(block)["aggregate"])["final_eval"]).get("action_alignment_mean", {}).get("mean", 0.0),
                "action_entropy": dict(dict(dict(block)["aggregate"])["final_eval"]).get("action_entropy_mean", {}).get("mean", 0.0),
            }
            for name, block in scenarios_out.items()
        ],
        key=lambda row: (-float(row["failure_rate"]), -float(row["avg_agent_spread"])),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {
        "metadata": {
            "stage": "local_lob_mappo_no_intervention_assay_not_full_main",
            "device": device,
            "seeds": seeds,
            "scenarios": scenario_names,
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "episode_length": args.episode_length,
            "failure_spread_threshold": args.failure_spread_threshold,
            "strong_failure_spread_threshold": args.strong_failure_spread_threshold,
            "checkpoint_episodes": checkpoints,
            "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
        },
        "ranked_scenarios": ranked,
        "scenarios": scenarios_out,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_seed_csv(output_dir / "seed_metrics.csv", summary)
    write_learning_curve_csv(output_dir / "learning_curve.csv", summary)
    print(json.dumps({
        "output_dir": str(output_dir),
        "device": device,
        "ranked_scenarios": ranked,
    }, indent=2))


if __name__ == "__main__":
    main()
