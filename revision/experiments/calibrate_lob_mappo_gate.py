#!/usr/bin/env python3
"""Calibrate the local LOB gate from MAPPO no-intervention records."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_lob_gate import (  # noqa: E402
    choose_threshold,
    parse_seeds,
    parse_target,
    summarize,
    write_records,
    write_table,
)
from run_lob_mappo_pilot import (  # noqa: E402
    agent_observations,
    choose_device,
    critic_observations,
    evaluation_episode_seed,
    rollout_training_episode,
)
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv  # noqa: E402
from src.interventions.lob_gates import LOBGateConfig, LOBInterventionGate  # noqa: E402


Record = Dict[str, float | str]


def train_no_intervention_mappo(
    config: LocalLOBConfig,
    episodes: int,
    seed: int,
    mappo_base: MAPPOConfig,
    include_agent_id: bool = False,
) -> MAPPOTrainer:
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
            "n_agents": config.n_agents,
            "seed": seed,
        }
    )
    trainer = MAPPOTrainer(mappo_config)
    no_gate = LOBInterventionGate(config, LOBGateConfig(), "none")
    for ep in range(episodes):
        rollout_training_episode(
            env,
            trainer,
            no_gate,
            routing_rho=0.0,
            seed=seed + ep,
            include_agent_id=include_agent_id,
        )
    return trainer


def collect_passive_records(
    config: LocalLOBConfig,
    trainer: MAPPOTrainer,
    episodes: int,
    seed: int,
    gate_config: LOBGateConfig,
    include_agent_id: bool = False,
) -> List[Record]:
    env = LocalLOBMarketEnv(config)
    gate = LOBInterventionGate(config, gate_config, "risk_aware")
    records: List[Record] = []
    for ep in range(episodes):
        env.reset(seed=evaluation_episode_seed(seed, ep))
        gate.reset()
        done = False
        while not done:
            obs = agent_observations(env, include_agent_id=include_agent_id)
            critic_obs = critic_observations(env, obs)
            actions, _, _ = trainer.act(obs, critic_obs, deterministic=True)
            spreads = [config.half_spread_levels[int(actions[i])] for i in range(config.n_agents)]
            decision = gate.decide(env.risk_state, spreads, env.last_flow_shares, env.last_total_volume)
            _, _, done, _, info = env.step(
                {i: int(actions[i]) for i in range(config.n_agents)},
                routing_flow_boost=0.0,
                gate_active=False,
                gate_score=float(decision["score"]),
            )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--target-activation", default="0.05,0.35")
    parser.add_argument("--max-fpr", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--high-spread-tau", type=float, default=4.0)
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
    parser.add_argument("--targeted-fill-pricing", choices=["agent_quote", "competitive_benchmark", "resting_quote"], default=None)
    parser.add_argument("--targeted-competitive-half-spread-ticks", type=float, default=None)
    parser.add_argument("--routing-mode", choices=["book_multiplier", "platform_reallocate"], default=None)
    parser.add_argument("--routing-routable-fraction", type=float, default=None)
    parser.add_argument("--routing-relative-improvement-ticks", type=float, default=None)
    parser.add_argument("--routing-target-kappa", type=float, default=None)
    parser.add_argument("--enforce-max-inventory", action="store_true")
    parser.add_argument("--quote-priority-mode", choices=["fixed", "rotating"], default=None)
    parser.add_argument("--agent-quote-quantity", type=int, default=None)
    parser.add_argument("--common-random-numbers", action="store_true")
    parser.add_argument("--agent-id-feature", action="store_true")
    parser.add_argument("--policy-mode", choices=["shared", "multihead", "independent"], default="shared")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--output-dir", default="revision/experiments/outputs/lob_mappo_gate_calibration_v1")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    device = choose_device(args.device)
    config_kwargs = {
        "n_agents": args.n_agents,
        "episode_length": args.episode_length,
        "seed": seeds[0] if seeds else 0,
    }
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
    if args.routing_mode is not None:
        config_kwargs["routing_mode"] = args.routing_mode
    if args.routing_routable_fraction is not None:
        config_kwargs["routing_routable_fraction"] = float(args.routing_routable_fraction)
    if args.routing_relative_improvement_ticks is not None:
        config_kwargs["routing_relative_improvement_ticks"] = float(args.routing_relative_improvement_ticks)
    if args.routing_target_kappa is not None:
        config_kwargs["routing_target_kappa"] = float(args.routing_target_kappa)
    if args.enforce_max_inventory:
        config_kwargs["enforce_max_inventory"] = True
    if args.quote_priority_mode is not None:
        config_kwargs["quote_priority_mode"] = args.quote_priority_mode
    if args.agent_quote_quantity is not None:
        config_kwargs["agent_quote_quantity"] = int(args.agent_quote_quantity)
    if args.common_random_numbers:
        config_kwargs["common_random_numbers"] = True
    config = LocalLOBConfig(**config_kwargs)
    base_gate_config = LOBGateConfig()
    mappo_base = MAPPOConfig(
        obs_dim=1,
        critic_obs_dim=1,
        n_actions=config.n_actions,
        n_agents=config.n_agents,
        policy_mode=args.policy_mode,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )

    records: List[Record] = []
    for seed in seeds:
        trainer = train_no_intervention_mappo(
            replace(config, seed=seed),
            args.episodes,
            seed,
            mappo_base,
            include_agent_id=args.agent_id_feature,
        )
        records.extend(
            collect_passive_records(
                replace(config, seed=seed),
                trainer,
                args.eval_episodes,
                seed,
                base_gate_config,
                include_agent_id=args.agent_id_feature,
            )
        )

    selected, table = choose_threshold(
        records,  # type: ignore[arg-type]
        high_spread_tau=args.high_spread_tau,
        target_activation=parse_target(args.target_activation),
        max_fpr=args.max_fpr,
        grid_size=args.grid_size,
    )
    calibrated_gate_config = LOBGateConfig(**{**vars(base_gate_config), "risk_threshold": float(selected["threshold"])})
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "episode_length": args.episode_length,
            "n_agents": args.n_agents,
            "seeds": seeds,
            "target_activation": args.target_activation,
            "max_fpr": args.max_fpr,
            "high_spread_tau": args.high_spread_tau,
            "config": asdict(config),
            "device": device,
            "mappo_config": asdict(mappo_base),
            "agent_id_feature": args.agent_id_feature,
            "score_summary": summarize(float(row["gate_score"]) for row in records),
            "stage": "local_lob_mappo_no_intervention_gate_calibration",
        },
        "gate_config": asdict(calibrated_gate_config),
        "selected_threshold": selected,
    }
    (out_dir / "lob_mappo_gate_config.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_records(out_dir / "calibration_records.csv", records)  # type: ignore[arg-type]
    write_table(out_dir / "threshold_grid.csv", table)
    print(json.dumps(output, indent=2))
    print(f"\nSaved MAPPO LOB gate calibration to {out_dir}")


if __name__ == "__main__":
    main()
