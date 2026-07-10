#!/usr/bin/env python3
"""MAPPO-style pilot on the local price-time priority LOB.

This runner is the next bridge after the tabular local LOB pilot. It is not a
full LOB/MAPPO main experiment; its job is to verify that the matching-engine
environment, MAPPO-style learner, gate, matched random baseline, and routing
diagnostics can run together.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lob_marl_pilot import (  # noqa: E402
    aggregate_seed_results,
    gate_metrics,
    load_lob_gate_config,
    paired_delta,
    parse_seeds,
    summarize_by_risk,
    summarize_records,
)
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv  # noqa: E402
from src.interventions.lob_gates import LOBGateConfig, LOBGateMode, LOBInterventionGate  # noqa: E402


METHODS: Dict[str, Tuple[LOBGateMode, float]] = {
    "no_intervention": ("none", 0.0),
    "always_on": ("always", 0.75),
    "threshold": ("threshold", 0.75),
    "gated_intervention": ("risk_aware", 0.75),
    "gated_intervention_controlled": ("risk_aware_controlled", 0.75),
    "oracle_low_high_spread": ("oracle_low_high_spread", 0.75),
    "random_gate": ("random", 0.75),
}


def choose_device(raw: str) -> str:
    if raw != "auto":
        return raw
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def flow_concentration(flow_shares: np.ndarray) -> float:
    total = float(np.sum(flow_shares))
    if total <= 0.0:
        return 0.0
    shares = flow_shares / total
    return float(np.sum(shares ** 2))


def agent_observations(env: LocalLOBMarketEnv, include_agent_id: bool = False) -> np.ndarray:
    cfg = env.config
    max_spread = float(max(cfg.half_spread_levels))
    risk = 1.0 if env.risk_state == "high" else 0.0
    denom_actions = max(cfg.n_actions - 1, 1)
    avg_action = float(np.mean(env.last_actions)) / denom_actions
    best_spread = env.book.spread_ticks() / max(2.0 * max_spread, 1.0)
    total_volume = env.last_total_volume / max(cfg.base_flow_intensity * (1.0 + cfg.routing_flow_multiplier), 1.0)
    t_frac = env.t / max(cfg.episode_length, 1)
    obs = []
    for agent_id in range(cfg.n_agents):
        row = [
            risk,
            float(env.last_actions[agent_id]) / denom_actions,
            avg_action,
            float(env.last_agent_spreads[agent_id]) / max_spread,
            float(env.inventories[agent_id]) / max(cfg.max_inventory, 1),
            float(env.last_flow_shares[agent_id]),
            best_spread,
            total_volume,
            t_frac,
        ]
        if include_agent_id:
            row.append(float(agent_id) / max(cfg.n_agents - 1, 1))
        obs.append(row)
    return np.array(obs, dtype=np.float32)


def critic_observations(env: LocalLOBMarketEnv, obs: np.ndarray) -> np.ndarray:
    cfg = env.config
    max_spread = float(max(cfg.half_spread_levels))
    global_features = np.array([
        1.0 if env.risk_state == "high" else 0.0,
        float(np.mean(env.last_agent_spreads)) / max_spread,
        float(np.std(env.last_agent_spreads)) / max_spread,
        env.book.spread_ticks() / max(2.0 * max_spread, 1.0),
        env.last_total_volume / max(cfg.base_flow_intensity * (1.0 + cfg.routing_flow_multiplier), 1.0),
        flow_concentration(env.last_flow_shares),
        float(np.mean(np.abs(env.inventories))) / max(cfg.max_inventory, 1),
        env.t / max(cfg.episode_length, 1),
    ], dtype=np.float32)
    tiled = np.repeat(global_features[None, :], cfg.n_agents, axis=0)
    return np.concatenate([obs, tiled], axis=1).astype(np.float32)


def rollout_training_episode(
    env: LocalLOBMarketEnv,
    trainer: MAPPOTrainer,
    gate: LOBInterventionGate,
    routing_rho: float,
    seed: int,
    include_agent_id: bool = False,
) -> Dict[str, float]:
    env.reset(seed=seed)
    gate.reset()
    done = False
    obs_rows: list[np.ndarray] = []
    critic_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    logp_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    gate_active_rows: list[float] = []
    gate_raw_active_rows: list[float] = []
    gate_score_rows: list[float] = []
    routing_boost_rows: list[float] = []

    while not done:
        obs = agent_observations(env, include_agent_id=include_agent_id)
        critic_obs = critic_observations(env, obs)
        actions, logp, values = trainer.act(obs, critic_obs, deterministic=False)
        spreads = [env.config.half_spread_levels[int(actions[i])] for i in range(env.config.n_agents)]
        decision = gate.decide(env.risk_state, spreads, env.last_flow_shares, env.last_total_volume)
        boost = routing_rho if bool(decision["active"]) else 0.0
        _, rewards, done, _, _ = env.step(
            {i: int(actions[i]) for i in range(env.config.n_agents)},
            routing_flow_boost=boost,
            gate_active=bool(decision["active"]),
            gate_score=float(decision["score"]),
        )
        obs_rows.append(obs)
        critic_rows.append(critic_obs)
        action_rows.append(actions.astype(np.int64))
        logp_rows.append(logp.astype(float))
        value_rows.append(values.astype(float))
        reward_rows.append(np.array([rewards[i] for i in range(env.config.n_agents)], dtype=float))
        gate_active_rows.append(float(bool(decision["active"])))
        gate_raw_active_rows.append(float(bool(decision["raw_active"])))
        gate_score_rows.append(float(decision["score"]))
        routing_boost_rows.append(float(boost))

    stats = trainer.update_episode(
        np.stack(obs_rows),
        np.stack(critic_rows),
        np.stack(action_rows),
        np.stack(logp_rows),
        np.stack(reward_rows),
        np.stack(value_rows),
    )
    stats.update({
        "gate_activation_rate": float(np.mean(gate_active_rows)) if gate_active_rows else 0.0,
        "gate_raw_activation_rate": float(np.mean(gate_raw_active_rows)) if gate_raw_active_rows else 0.0,
        "gate_score_mean": float(np.mean(gate_score_rows)) if gate_score_rows else 0.0,
        "routing_boost_mean": float(np.mean(routing_boost_rows)) if routing_boost_rows else 0.0,
    })
    return stats


def evaluate_policy(
    env: LocalLOBMarketEnv,
    trainer: MAPPOTrainer,
    gate: LOBInterventionGate,
    routing_rho: float,
    method: str,
    seed: int,
    eval_episodes: int,
    include_agent_id: bool = False,
    gate_off: bool = False,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for ep in range(eval_episodes):
        env.reset(seed=seed + 20_000 + ep)
        gate.reset()
        done = False
        while not done:
            obs = agent_observations(env, include_agent_id=include_agent_id)
            critic_obs = critic_observations(env, obs)
            actions, _, _ = trainer.act(obs, critic_obs, deterministic=True)
            spreads = [env.config.half_spread_levels[int(actions[i])] for i in range(env.config.n_agents)]
            decision = gate.decide(env.risk_state, spreads, env.last_flow_shares, env.last_total_volume)
            active = bool(decision["active"]) and not gate_off
            boost = routing_rho if active else 0.0
            _, _, done, _, info = env.step(
                {i: int(actions[i]) for i in range(env.config.n_agents)},
                routing_flow_boost=boost,
                gate_active=active,
                gate_score=float(decision["score"]),
            )
            row = {
                key: value for key, value in info.items()
                if isinstance(value, (str, int, float, bool, np.integer, np.floating))
            }
            row["method"] = method
            row["seed"] = seed
            row["gate_excess"] = float(decision["excess"])
            row["gate_spread_entropy"] = float(decision["spread_entropy"])
            row["gate_flow_concentration"] = float(decision["flow_concentration"])
            row["gate_volume_suppression"] = float(decision["volume_suppression"])
            row["gate_raw_active"] = bool(decision["raw_active"])
            row["gate_would_active"] = bool(decision["active"])
            row["gate_controlled_recent_rate"] = float(decision["controlled_recent_rate"])
            row["gate_control_budget_open"] = bool(decision["control_budget_open"])
            row["gate_control_cooldown_open"] = bool(decision["control_cooldown_open"])
            records.append(row)
    return records


def run_method_seed(
    config: LocalLOBConfig,
    method: str,
    gate_mode: LOBGateMode,
    routing_rho: float,
    episodes: int,
    eval_episodes: int,
    seed: int,
    gate_config: LOBGateConfig,
    mappo_base: MAPPOConfig,
    include_agent_id: bool = False,
    eval_gate_off: bool = False,
) -> Dict[str, object]:
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    obs = agent_observations(env, include_agent_id=include_agent_id)
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
    trainer = MAPPOTrainer(mappo_config)
    gate = LOBInterventionGate(config, gate_config, gate_mode)
    training_stats = []
    for ep in range(episodes):
        training_stats.append(
            rollout_training_episode(
                env,
                trainer,
                gate,
                routing_rho,
                seed=seed + ep,
                include_agent_id=include_agent_id,
            )
        )

    records = evaluate_policy(
        env,
        trainer,
        gate,
        routing_rho,
        method,
        seed,
        eval_episodes,
        include_agent_id=include_agent_id,
        gate_off=eval_gate_off,
    )
    return {
        "method": method,
        "gate_mode": gate_mode,
        "seed": seed,
        "routing_rho": routing_rho,
        "training": {
            key: float(np.mean([row[key] for row in training_stats])) if training_stats else 0.0
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
        },
        "eval_all": summarize_records(records),
        "eval_by_risk": summarize_by_risk(records),
        "gate_metrics": gate_metrics(records),
    }


def run_grid(
    config: LocalLOBConfig,
    methods: List[str],
    episodes: int,
    eval_episodes: int,
    seeds: List[int],
    gate_config: LOBGateConfig,
    mappo_config: MAPPOConfig,
    routing_rho_override: float | None = None,
    include_agent_id: bool = False,
    eval_gate_off: bool = False,
) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {method: [] for method in methods}
    for method in methods:
        gate_mode, default_routing_rho = METHODS[method]
        routing_rho = default_routing_rho
        if routing_rho_override is not None and gate_mode != "none":
            routing_rho = float(routing_rho_override)
        for seed in seeds:
            seed_gate_config = gate_config
            if gate_mode == "random":
                seed_gate_config = LOBGateConfig(**{
                    **vars(gate_config),
                    "seed": int(gate_config.seed + 10_000 + 9973 * seed),
                })
            grouped[method].append(
                run_method_seed(
                    replace(config, seed=seed),
                    method,
                    gate_mode,
                    routing_rho,
                    episodes,
                    eval_episodes,
                    seed,
                    seed_gate_config,
                    mappo_config,
                    include_agent_id=include_agent_id,
                    eval_gate_off=eval_gate_off,
                )
            )
    return grouped


def metric_mean(aggregate: Dict[str, object], block: str, metric: str) -> float:
    value = dict(aggregate.get(block, {})).get(metric, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def primary_gated_method(grouped_or_methods: Dict[str, object]) -> str:
    for method in ("gated_intervention_controlled", "gated_intervention", "oracle_low_high_spread"):
        if method in grouped_or_methods:
            return method
    return "gated_intervention"


def pass_diagnostics(summary: Dict[str, object]) -> Dict[str, object]:
    methods = dict(summary["methods"])
    gated_method = primary_gated_method(methods)

    def mean(method: str, block: str, metric: str) -> float:
        if method not in methods:
            return 0.0
        return metric_mean(dict(dict(methods[method])["aggregate"]), block, metric)

    no_spread = mean("no_intervention", "eval_all", "avg_agent_spread_mean")
    gated_active = mean(gated_method, "gate_metrics", "gate_activation_rate")
    random_active = mean("random_gate", "gate_metrics", "gate_activation_rate")
    always_active = mean("always_on", "gate_metrics", "gate_activation_rate")
    gated_volume = mean(gated_method, "eval_all", "total_volume_mean")
    no_volume = mean("no_intervention", "eval_all", "total_volume_mean")
    gated_reward = mean(gated_method, "eval_all", "mean_reward_mean")
    no_reward = mean("no_intervention", "eval_all", "mean_reward_mean")

    return {
        "no_intervention_spread_elevated": bool(no_spread > 2.0),
        "gate_in_target_activation_range": bool(0.05 <= gated_active <= 0.40),
        "gate_nonzero": bool(gated_active > 0.0),
        "gate_overactive": bool(gated_active > 0.40),
        "always_on_saturated": bool(always_active >= 0.95),
        "matched_random_runs": bool(random_active > 0.0),
        "routing_changes_volume": bool(abs(gated_volume - no_volume) > 1e-6),
        "reward_not_obviously_collapsed": bool(gated_reward > no_reward - 5.0),
        "observed": {
            "primary_gated_method": gated_method,
            "no_intervention_avg_agent_spread": no_spread,
            "gated_activation_rate": gated_active,
            "random_activation_rate": random_active,
            "always_activation_rate": always_active,
            "no_intervention_volume": no_volume,
            "gated_volume": gated_volume,
            "no_intervention_reward": no_reward,
            "gated_reward": gated_reward,
        },
    }


def write_metrics_csv(path: Path, summary: Dict[str, object]) -> None:
    rows: List[Dict[str, object]] = []
    for method, method_block in dict(summary["methods"]).items():
        aggregate = dict(dict(method_block)["aggregate"])
        for block, metric in [
            ("eval_all", "avg_agent_spread_mean"),
            ("eval_all", "best_spread_ticks_mean"),
            ("eval_all", "total_volume_mean"),
            ("eval_all", "mean_reward_mean"),
            ("eval_all", "mean_spread_revenue_mean"),
            ("eval_all", "mean_adverse_selection_loss_mean"),
            ("eval_all", "mean_inventory_penalty_mean"),
            ("eval_all", "mean_abs_inventory_mean"),
            ("eval_all", "routing_effective_multiplier_mean"),
            ("eval_all", "routing_targeted_volume_mean"),
            ("eval_all", "routing_targeted_orders_mean"),
            ("eval_all", "routing_eligible_agents_mean"),
            ("gate_metrics", "gate_activation_rate"),
            ("gate_metrics", "tpr_low_high_spread"),
            ("gate_metrics", "fpr_high_normal_spread"),
        ]:
            rows.append({
                "method": method,
                "block": block,
                "metric": metric,
                "mean": metric_mean(aggregate, block, metric),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "block", "metric", "mean"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default="no_intervention,always_on,threshold,gated_intervention")
    parser.add_argument("--random-rate", default="auto")
    parser.add_argument("--gate-config", default=None)
    parser.add_argument(
        "--routing-mode",
        choices=[
            "book_multiplier",
            "eligible_priority",
            "relative_priority",
            "relative_reallocate",
            "continuous_reallocate",
            "balanced_reallocate",
        ],
        default=None,
    )
    parser.add_argument("--routing-rho", type=float, default=None)
    parser.add_argument("--routing-spread-cap-ticks", type=float, default=None)
    parser.add_argument("--routing-withhold-rate", type=float, default=None)
    parser.add_argument("--routing-relative-improvement-ticks", type=float, default=None)
    parser.add_argument("--routing-target-kappa", type=float, default=None)
    parser.add_argument("--inventory-penalty", type=float, default=None)
    parser.add_argument("--base-flow-intensity", type=float, default=None)
    parser.add_argument("--p-low-to-high", type=float, default=None)
    parser.add_argument("--p-high-to-low", type=float, default=None)
    parser.add_argument("--low-sigma", type=float, default=None)
    parser.add_argument("--high-sigma", type=float, default=None)
    parser.add_argument("--seed-liquidity-levels", type=int, default=None)
    parser.add_argument("--seed-liquidity-qty", type=int, default=None)
    parser.add_argument("--risk-coupled-economics", action="store_true")
    parser.add_argument("--mark-to-fundamental", action="store_true")
    parser.add_argument("--informed-flow-prob", type=float, default=None)
    parser.add_argument("--spread-elasticity", type=float, default=None)
    parser.add_argument("--targeted-fill-pricing", choices=["agent_quote", "competitive_benchmark"], default=None)
    parser.add_argument("--targeted-competitive-half-spread-ticks", type=float, default=None)
    parser.add_argument("--enforce-max-inventory", action="store_true")
    parser.add_argument("--agent-id-feature", action="store_true")
    parser.add_argument("--eval-gate-off", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--output", default="revision/experiments/outputs/lob_mappo_pilot_v1/summary.json")
    parser.add_argument("--metrics-csv", default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    for method in methods:
        if method not in METHODS or method == "random_gate":
            raise SystemExit(f"Unknown or invalid non-random method '{method}'")

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
    if args.inventory_penalty is not None:
        config_kwargs["inventory_penalty"] = float(args.inventory_penalty)
    if args.base_flow_intensity is not None:
        config_kwargs["base_flow_intensity"] = float(args.base_flow_intensity)
    if args.p_low_to_high is not None:
        config_kwargs["p_low_to_high"] = float(args.p_low_to_high)
    if args.p_high_to_low is not None:
        config_kwargs["p_high_to_low"] = float(args.p_high_to_low)
    if args.low_sigma is not None:
        config_kwargs["low_sigma"] = float(args.low_sigma)
    if args.high_sigma is not None:
        config_kwargs["high_sigma"] = float(args.high_sigma)
    if args.seed_liquidity_levels is not None:
        config_kwargs["seed_liquidity_levels"] = int(args.seed_liquidity_levels)
    if args.seed_liquidity_qty is not None:
        config_kwargs["seed_liquidity_qty"] = int(args.seed_liquidity_qty)
    if args.risk_coupled_economics:
        config_kwargs["risk_coupled_economics"] = True
        config_kwargs["mark_to_fundamental"] = True
    if args.mark_to_fundamental:
        config_kwargs["mark_to_fundamental"] = True
    if args.informed_flow_prob is not None:
        config_kwargs["informed_flow_prob"] = float(args.informed_flow_prob)
    if args.spread_elasticity is not None:
        config_kwargs["spread_elasticity"] = float(args.spread_elasticity)
    if args.targeted_fill_pricing is not None:
        config_kwargs["targeted_fill_pricing"] = args.targeted_fill_pricing
    if args.targeted_competitive_half_spread_ticks is not None:
        config_kwargs["targeted_competitive_half_spread_ticks"] = float(args.targeted_competitive_half_spread_ticks)
    if args.enforce_max_inventory:
        config_kwargs["enforce_max_inventory"] = True
    config = LocalLOBConfig(**config_kwargs)
    gate_config = load_lob_gate_config(args.gate_config)
    mappo_config = MAPPOConfig(
        obs_dim=1,
        critic_obs_dim=1,
        n_actions=config.n_actions,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )
    grouped = run_grid(
        config,
        methods,
        args.episodes,
        args.eval_episodes,
        seeds,
        gate_config,
        mappo_config,
        routing_rho_override=args.routing_rho,
        include_agent_id=args.agent_id_feature,
        eval_gate_off=args.eval_gate_off,
    )
    gated_method = primary_gated_method(grouped)

    rate_block = "training" if args.eval_gate_off else "gate_metrics"
    gated_rates = [
        float(result[rate_block]["gate_activation_rate"])  # type: ignore[index]
        for result in grouped.get(gated_method, [])
    ]
    matched_rate = float(np.mean(gated_rates)) if args.random_rate == "auto" and gated_rates else float(args.random_rate)
    matched_rate = float(np.clip(matched_rate, 0.0, 1.0))
    random_gate_config = LOBGateConfig(**{**vars(gate_config), "random_rate": matched_rate})
    grouped["random_gate"] = run_grid(
        config,
        ["random_gate"],
        args.episodes,
        args.eval_episodes,
        seeds,
        random_gate_config,
        mappo_config,
        routing_rho_override=args.routing_rho,
        include_agent_id=args.agent_id_feature,
        eval_gate_off=args.eval_gate_off,
    )["random_gate"]

    ordered_methods = [
        "no_intervention",
        "always_on",
        "threshold",
        "random_gate",
        "gated_intervention",
        "gated_intervention_controlled",
        "oracle_low_high_spread",
    ]
    methods_out: Dict[str, object] = {}
    for method in ordered_methods:
        if method not in grouped:
            continue
        methods_out[method] = {
            "seed_results": grouped[method],
            "aggregate": aggregate_seed_results(grouped[method]),
        }

    summary: Dict[str, object] = {
        "metadata": {
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "episode_length": args.episode_length,
            "seeds": seeds,
            "matched_random_rate": matched_rate,
            "routing_rho_override": args.routing_rho,
            "config": asdict(config),
            "gate_config": asdict(gate_config),
            "gate_config_path": args.gate_config,
            "mappo_config": asdict(mappo_config),
            "primary_gated_method": gated_method,
            "agent_id_feature": args.agent_id_feature,
            "eval_gate_off": args.eval_gate_off,
            "stage": "local_lob_mappo_pilot_not_full_main",
        },
        "methods": methods_out,
        "paired_random_minus_gated_spread": paired_delta(grouped["random_gate"], grouped.get(gated_method, []), "avg_agent_spread_mean"),
    }
    summary["pilot_diagnostics"] = pass_diagnostics(summary)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else output.with_name("metrics.csv")
    write_metrics_csv(metrics_csv, summary)
    print(json.dumps({
        "output": str(output),
        "metrics_csv": str(metrics_csv),
        "matched_random_rate": matched_rate,
        "paired_random_minus_gated_spread": summary["paired_random_minus_gated_spread"],
        "pilot_diagnostics": summary["pilot_diagnostics"],
    }, indent=2))


if __name__ == "__main__":
    main()
