#!/usr/bin/env python3
"""Diagnose MAPPO gate activation drift in the local LOB pilot.

This script is not a full experiment. It trains short MAPPO policies under
selected intervention methods, then records the passive risk-aware gate score
distribution observed under each trained policy. The goal is to separate score
distribution drift from the mechanical effect of the intervention controller.
"""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lob_mappo_pilot import (  # noqa: E402
    METHODS,
    agent_observations,
    choose_device,
    critic_observations,
    rollout_training_episode,
)
from run_lob_marl_pilot import load_lob_gate_config, parse_seeds  # noqa: E402
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv  # noqa: E402
from src.interventions.lob_gates import LOBGateConfig, LOBInterventionGate  # noqa: E402


Record = Dict[str, float | str | bool]


def mean_se(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "se": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
    }


def quantiles(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return {"n": 0.0}
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def make_trainer(config: LocalLOBConfig, seed: int, mappo_base: MAPPOConfig) -> MAPPOTrainer:
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    obs = agent_observations(env)
    critic_obs = critic_observations(env, obs)
    mappo_config = MAPPOConfig(
        **{
            **asdict(mappo_base),
            "obs_dim": int(obs.shape[1]),
            "critic_obs_dim": int(critic_obs.shape[1]),
            "n_actions": config.n_actions,
            "seed": seed,
        }
    )
    return MAPPOTrainer(mappo_config)


def train_policy(
    config: LocalLOBConfig,
    method: str,
    episodes: int,
    seed: int,
    gate_config: LOBGateConfig,
    mappo_base: MAPPOConfig,
    routing_rho_override: float | None = None,
) -> Tuple[MAPPOTrainer, Dict[str, float]]:
    trainer = make_trainer(config, seed, mappo_base)
    env = LocalLOBMarketEnv(config)
    gate_mode, default_routing_rho = METHODS[method]
    routing_rho = default_routing_rho
    if routing_rho_override is not None and gate_mode != "none":
        routing_rho = float(routing_rho_override)
    gate = LOBInterventionGate(config, gate_config, gate_mode)
    stats: List[Dict[str, float]] = []
    for ep in range(episodes):
        stats.append(rollout_training_episode(env, trainer, gate, routing_rho, seed=seed + ep))
    return trainer, {
        key: float(np.mean([row[key] for row in stats])) if stats else 0.0
        for key in (
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "return_mean",
            "gate_activation_rate",
            "gate_raw_activation_rate",
            "gate_score_mean",
            "routing_boost_mean",
        )
    }


def collect_records(
    config: LocalLOBConfig,
    trainer: MAPPOTrainer,
    method: str,
    eval_episodes: int,
    seed: int,
    gate_config: LOBGateConfig,
    routing_rho_override: float | None = None,
) -> List[Record]:
    gate_mode, default_routing_rho = METHODS[method]
    routing_rho = default_routing_rho
    if routing_rho_override is not None and gate_mode != "none":
        routing_rho = float(routing_rho_override)
    env = LocalLOBMarketEnv(config)
    actual_gate = LOBInterventionGate(config, gate_config, gate_mode)
    passive_gate = LOBInterventionGate(config, gate_config, "risk_aware")
    records: List[Record] = []
    max_spread = float(max(config.half_spread_levels))

    for ep in range(eval_episodes):
        env.reset(seed=seed + 20_000 + ep)
        actual_gate.reset()
        passive_gate.reset()
        done = False
        while not done:
            obs = agent_observations(env)
            critic_obs = critic_observations(env, obs)
            actions, _, _ = trainer.act(obs, critic_obs, deterministic=True)
            spreads = np.array([config.half_spread_levels[int(actions[i])] for i in range(config.n_agents)], dtype=float)
            passive = passive_gate.decide(env.risk_state, spreads, env.last_flow_shares, env.last_total_volume)
            actual = actual_gate.decide(env.risk_state, spreads, env.last_flow_shares, env.last_total_volume)
            boost = routing_rho if bool(actual["active"]) else 0.0
            _, _, done, _, info = env.step(
                {i: int(actions[i]) for i in range(config.n_agents)},
                routing_flow_boost=boost,
                gate_active=bool(actual["active"]),
                gate_score=float(actual["score"]),
            )
            row: Record = {
                "method": method,
                "seed": float(seed),
                "episode": float(ep),
                "t": float(info.get("t", 0.0)),
                "risk_state": str(info["risk_state"]),
                "avg_agent_spread": float(info["avg_agent_spread"]),
                "best_spread_ticks": float(info["best_spread_ticks"]),
                "total_volume": float(info["total_volume"]),
                "mean_reward": float(info["mean_reward"]),
                "routing_effective_multiplier": float(info["routing_effective_multiplier"]),
                "routing_spread_cap_met": bool(info["routing_spread_cap_met"]),
                "routing_targeted_volume": float(info["routing_targeted_volume"]),
                "routing_targeted_orders": float(info["routing_targeted_orders"]),
                "routing_eligible_agents": float(info["routing_eligible_agents"]),
                "actual_gate_active": bool(actual["active"]),
                "actual_gate_raw_active": bool(actual["raw_active"]),
                "actual_gate_score": float(actual["score"]),
                "actual_gate_controlled_recent_rate": float(actual["controlled_recent_rate"]),
                "actual_gate_control_budget_open": bool(actual["control_budget_open"]),
                "actual_gate_control_cooldown_open": bool(actual["control_cooldown_open"]),
                "passive_gate_active": bool(passive["active"]),
                "passive_gate_raw_active": bool(passive["raw_active"]),
                "passive_gate_score": float(passive["score"]),
                "passive_gate_excess": float(passive["excess"]),
                "passive_gate_spread_entropy": float(passive["spread_entropy"]),
                "passive_gate_flow_concentration": float(passive["flow_concentration"]),
                "passive_gate_volume_suppression": float(passive["volume_suppression"]),
                "action_mean": float(np.mean(actions)),
                "action_std": float(np.std(actions)),
                "action_min": float(np.min(actions)),
                "action_max": float(np.max(actions)),
                "spread_mean": float(np.mean(spreads)),
                "spread_std": float(np.std(spreads)),
                "spread_min": float(np.min(spreads)),
                "spread_max": float(np.max(spreads)),
            }
            for level in config.half_spread_levels:
                row[f"spread_frac_{level}"] = float(np.mean(spreads == float(level)))
            row["spread_frac_max"] = float(np.mean(spreads == max_spread))
            records.append(row)
    return records


def summarize(records: List[Record]) -> Dict[str, object]:
    methods = sorted({str(row["method"]) for row in records})
    fields = [
        "avg_agent_spread",
        "best_spread_ticks",
        "total_volume",
        "mean_reward",
        "routing_effective_multiplier",
        "routing_targeted_volume",
        "routing_targeted_orders",
        "routing_eligible_agents",
        "actual_gate_active",
        "actual_gate_raw_active",
        "actual_gate_score",
        "actual_gate_controlled_recent_rate",
        "actual_gate_control_budget_open",
        "actual_gate_control_cooldown_open",
        "passive_gate_active",
        "passive_gate_raw_active",
        "passive_gate_score",
        "passive_gate_excess",
        "passive_gate_spread_entropy",
        "passive_gate_flow_concentration",
        "passive_gate_volume_suppression",
        "action_mean",
        "action_std",
        "spread_mean",
        "spread_std",
        "spread_frac_max",
    ]

    by_method: Dict[str, object] = {}
    for method in methods:
        scoped = [row for row in records if row["method"] == method]
        by_method[method] = {
            "n_steps": float(len(scoped)),
            "metrics": {
                field: mean_se(float(row[field]) for row in scoped if field in row)
                for field in fields
            },
            "passive_score_quantiles": quantiles(float(row["passive_gate_score"]) for row in scoped),
            "by_risk": {
                risk: {
                    "n_steps": float(len([row for row in scoped if row["risk_state"] == risk])),
                    "passive_score_quantiles": quantiles(
                        float(row["passive_gate_score"]) for row in scoped if row["risk_state"] == risk
                    ),
                    "passive_activation_rate": mean_se(
                        float(row["passive_gate_active"]) for row in scoped if row["risk_state"] == risk
                    ),
                    "actual_activation_rate": mean_se(
                        float(row["actual_gate_active"]) for row in scoped if row["risk_state"] == risk
                    ),
                    "avg_agent_spread": mean_se(
                        float(row["avg_agent_spread"]) for row in scoped if row["risk_state"] == risk
                    ),
                }
                for risk in ("low", "high")
            },
            "by_spread_bucket": summarize_spread_buckets(scoped),
        }
    return {"by_method": by_method}


def summarize_spread_buckets(records: List[Record]) -> Dict[str, object]:
    buckets = {
        "spread_le_2": lambda x: x <= 2.0,
        "spread_2_4": lambda x: 2.0 < x <= 4.0,
        "spread_gt_4": lambda x: x > 4.0,
    }
    out: Dict[str, object] = {}
    for name, predicate in buckets.items():
        scoped = [row for row in records if predicate(float(row["avg_agent_spread"]))]
        out[name] = {
            "n_steps": float(len(scoped)),
            "passive_activation_rate": mean_se(float(row["passive_gate_active"]) for row in scoped),
            "actual_activation_rate": mean_se(float(row["actual_gate_active"]) for row in scoped),
            "passive_score_quantiles": quantiles(float(row["passive_gate_score"]) for row in scoped),
        }
    return out


def write_records(path: Path, records: List[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_auto_random_rate(path: str | None, fallback: float) -> float:
    if not path:
        return fallback
    candidate = Path(path)
    if not candidate.exists():
        return fallback
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    return float(dict(raw.get("metadata", {})).get("matched_random_rate", fallback))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--methods", default="no_intervention,gated_intervention,random_gate")
    parser.add_argument("--random-rate", default="auto")
    parser.add_argument("--pilot-summary-for-random-rate", default="revision/experiments/outputs/lob_mappo_pilot_conservative_v1/summary.json")
    parser.add_argument("--gate-config", default="revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json")
    parser.add_argument(
        "--routing-mode",
        choices=["book_multiplier", "eligible_priority", "relative_priority", "relative_reallocate"],
        default=None,
    )
    parser.add_argument("--routing-rho", type=float, default=None)
    parser.add_argument("--routing-spread-cap-ticks", type=float, default=None)
    parser.add_argument("--routing-withhold-rate", type=float, default=None)
    parser.add_argument("--routing-relative-improvement-ticks", type=float, default=None)
    parser.add_argument("--routing-target-kappa", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--output-dir", default="revision/experiments/outputs/lob_mappo_gate_drift_diagnostic_v1")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    for method in methods:
        if method not in METHODS:
            raise SystemExit(f"Unknown method '{method}'")

    device = choose_device(args.device)
    config_kwargs = {
        "n_agents": args.n_agents,
        "episode_length": args.episode_length,
        "seed": seeds[0] if seeds else 0,
    }
    if args.routing_mode is not None:
        config_kwargs["routing_mode"] = args.routing_mode
    if args.routing_spread_cap_ticks is not None:
        config_kwargs["routing_spread_cap_ticks"] = float(args.routing_spread_cap_ticks)
    if args.routing_withhold_rate is not None:
        config_kwargs["routing_withhold_rate"] = float(args.routing_withhold_rate)
    if args.routing_relative_improvement_ticks is not None:
        config_kwargs["routing_relative_improvement_ticks"] = float(args.routing_relative_improvement_ticks)
    if args.routing_target_kappa is not None:
        config_kwargs["routing_target_kappa"] = float(args.routing_target_kappa)
    config = LocalLOBConfig(**config_kwargs)
    gate_config = load_lob_gate_config(args.gate_config)
    if args.random_rate == "auto":
        matched_rate = load_auto_random_rate(args.pilot_summary_for_random_rate, gate_config.random_rate)
    else:
        matched_rate = float(args.random_rate)
    gate_config_by_method = {
        method: (
            LOBGateConfig(**{**vars(gate_config), "random_rate": float(np.clip(matched_rate, 0.0, 1.0))})
            if method == "random_gate" else gate_config
        )
        for method in methods
    }
    mappo_base = MAPPOConfig(
        obs_dim=1,
        critic_obs_dim=1,
        n_actions=config.n_actions,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )

    records: List[Record] = []
    training: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in methods:
        training[method] = {}
        for seed in seeds:
            method_gate_config = gate_config_by_method[method]
            if method == "random_gate":
                method_gate_config = LOBGateConfig(**{
                    **vars(method_gate_config),
                    "seed": int(gate_config.seed + 10_000 + 9973 * seed),
                })
            trainer, stats = train_policy(
                replace(config, seed=seed),
                method,
                args.episodes,
                seed,
                method_gate_config,
                mappo_base,
                routing_rho_override=args.routing_rho,
            )
            training[method][str(seed)] = stats
            records.extend(
                collect_records(
                    replace(config, seed=seed),
                    trainer,
                    method,
                    args.eval_episodes,
                    seed,
                    method_gate_config,
                    routing_rho_override=args.routing_rho,
                )
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records(output_dir / "records.csv", records)
    summary = {
        "metadata": {
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "episode_length": args.episode_length,
            "n_agents": args.n_agents,
            "seeds": seeds,
            "methods": methods,
            "device": device,
            "matched_random_rate": float(np.clip(matched_rate, 0.0, 1.0)),
            "routing_rho_override": args.routing_rho,
            "config": asdict(config),
            "gate_config": asdict(gate_config),
            "gate_config_path": args.gate_config,
            "mappo_config": asdict(mappo_base),
            "stage": "local_lob_mappo_gate_drift_diagnostic_not_full_main",
        },
        "training": training,
        "diagnostics": summarize(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "device": device,
        "matched_random_rate": float(np.clip(matched_rate, 0.0, 1.0)),
        "methods": {
            method: dict(summary["diagnostics"]["by_method"])[method]["metrics"]  # type: ignore[index]
            for method in methods
        },
    }, indent=2))


if __name__ == "__main__":
    main()
