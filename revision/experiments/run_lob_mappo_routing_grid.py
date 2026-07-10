#!/usr/bin/env python3
"""Small routing-parameter grid for the local LOB/MAPPO pilot.

This is a targeted pilot diagnostic, not a full main experiment. It compares
activation-controlled gated routing against a matched random gate under a small
grid over routing intensity, spread cap, and withholding strength.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lob_mappo_pilot import (  # noqa: E402
    agent_observations,
    choose_device,
    critic_observations,
    evaluate_policy,
    rollout_training_episode,
)
from run_lob_marl_pilot import (  # noqa: E402
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


RANDOM_METHOD = "random_gate"
TIMING_METHODS: Dict[str, LOBGateMode] = {
    "gated_intervention_controlled": "risk_aware_controlled",
    "oracle_low_high_spread": "oracle_low_high_spread",
    "oracle_low_high_spread_controlled": "oracle_low_high_spread_controlled",
    "threshold": "threshold",
}


def parse_floats(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"Expected at least one float value, got '{raw}'")
    return values


def parse_strings(raw: str) -> List[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"Expected at least one value, got '{raw}'")
    return values


def mean_se(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "se": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
    }


def metric_mean(aggregate: Dict[str, object], block: str, metric: str) -> float:
    value = dict(aggregate.get(block, {})).get(metric, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def aggregate_seed_results(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"n_seeds": float(len(results))}
    for block_name in ("eval_all", "gate_metrics"):
        keys = sorted({key for result in results for key in dict(result[block_name]).keys()})  # type: ignore[arg-type]
        out[block_name] = {
            key: mean_se(float(dict(result[block_name]).get(key, 0.0)) for result in results)  # type: ignore[arg-type]
            for key in keys
        }
    return out


def make_trainer(
    config: LocalLOBConfig,
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
    return MAPPOTrainer(mappo_config)


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
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    env = LocalLOBMarketEnv(config)
    trainer = make_trainer(config, seed, mappo_base, include_agent_id=include_agent_id)
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
    result = {
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
    return result, records


def routing_split(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    active = [row for row in records if bool(row.get("gate_active", False))]
    active_cap = [row for row in active if bool(row.get("routing_spread_cap_met", False))]
    active_no_cap = [row for row in active if not bool(row.get("routing_spread_cap_met", False))]
    inactive = [row for row in records if not bool(row.get("gate_active", False))]

    def mean(rows: Sequence[Dict[str, object]], field: str) -> float:
        if not rows:
            return 0.0
        return float(np.mean([float(row.get(field, 0.0)) for row in rows]))

    n = max(1, len(records))
    active_n = max(1, len(active))
    return {
        "n_steps": float(len(records)),
        "active_n": float(len(active)),
        "active_share": float(len(active) / n),
        "active_cap_met_share_all": float(len(active_cap) / n),
        "active_cap_met_share_active": float(len(active_cap) / active_n),
        "active_cap_not_met_share_all": float(len(active_no_cap) / n),
        "active_cap_not_met_share_active": float(len(active_no_cap) / active_n),
        "active_avg_spread": mean(active, "avg_agent_spread"),
        "active_best_spread": mean(active, "best_spread_ticks"),
        "active_volume": mean(active, "total_volume"),
        "active_routing_multiplier": mean(active, "routing_effective_multiplier"),
        "active_targeted_volume": mean(active, "routing_targeted_volume"),
        "active_targeted_orders": mean(active, "routing_targeted_orders"),
        "active_eligible_agents": mean(active, "routing_eligible_agents"),
        "active_reward": mean(active, "mean_reward"),
        "active_cap_met_routing_multiplier": mean(active_cap, "routing_effective_multiplier"),
        "active_cap_not_met_routing_multiplier": mean(active_no_cap, "routing_effective_multiplier"),
        "inactive_avg_spread": mean(inactive, "avg_agent_spread"),
        "inactive_volume": mean(inactive, "total_volume"),
        "inactive_targeted_volume": mean(inactive, "routing_targeted_volume"),
        "inactive_targeted_orders": mean(inactive, "routing_targeted_orders"),
        "inactive_eligible_agents": mean(inactive, "routing_eligible_agents"),
        "inactive_reward": mean(inactive, "mean_reward"),
    }


def run_cell(
    base_config: LocalLOBConfig,
    base_gate_config: LOBGateConfig,
    mappo_config: MAPPOConfig,
    seeds: Sequence[int],
    episodes: int,
    eval_episodes: int,
    routing_mode: str,
    routing_rho: float,
    spread_cap: float,
    withhold_rate: float,
    relative_improvement: float,
    target_kappa: float,
    timing_method: str,
    timing_gate_mode: LOBGateMode,
    include_agent_id: bool,
    eval_gate_off: bool,
    match_random_rate_source: str,
) -> Dict[str, object]:
    config = replace(
        base_config,
        routing_mode=routing_mode,  # type: ignore[arg-type]
        routing_spread_cap_ticks=float(spread_cap),
        routing_withhold_rate=float(withhold_rate),
        routing_relative_improvement_ticks=float(relative_improvement),
        routing_target_kappa=float(target_kappa),
    )

    timed_results: List[Dict[str, object]] = []
    timed_records: List[Dict[str, object]] = []
    for seed in seeds:
        result, records = run_method_seed(
            replace(config, seed=seed),
            timing_method,
            timing_gate_mode,
            routing_rho,
            episodes,
            eval_episodes,
            seed,
            base_gate_config,
            mappo_config,
            include_agent_id=include_agent_id,
            eval_gate_off=eval_gate_off,
        )
        timed_results.append(result)
        timed_records.extend(records)

    rate_block = "training" if match_random_rate_source == "train" else "gate_metrics"
    timed_activation = float(np.mean([
        float(result[rate_block]["gate_activation_rate"])  # type: ignore[index]
        for result in timed_results
    ]))
    random_gate_config = LOBGateConfig(**{
        **vars(base_gate_config),
        "random_rate": float(np.clip(timed_activation, 0.0, 1.0)),
    })
    random_results: List[Dict[str, object]] = []
    random_records: List[Dict[str, object]] = []
    for seed in seeds:
        seeded_random_config = LOBGateConfig(**{
            **vars(random_gate_config),
            "seed": int(base_gate_config.seed + 10_000 + 9973 * seed),
        })
        result, records = run_method_seed(
            replace(config, seed=seed),
            RANDOM_METHOD,
            "random",
            routing_rho,
            episodes,
            eval_episodes,
            seed,
            seeded_random_config,
            mappo_config,
            include_agent_id=include_agent_id,
            eval_gate_off=eval_gate_off,
        )
        random_results.append(result)
        random_records.extend(records)

    timed_aggregate = aggregate_seed_results(timed_results)
    random_aggregate = aggregate_seed_results(random_results)
    paired = paired_delta(random_results, timed_results, "avg_agent_spread_mean")
    timed_spread = metric_mean(timed_aggregate, "eval_all", "avg_agent_spread_mean")
    random_spread = metric_mean(random_aggregate, "eval_all", "avg_agent_spread_mean")
    timed_reward = metric_mean(timed_aggregate, "eval_all", "mean_reward_mean")
    random_reward = metric_mean(random_aggregate, "eval_all", "mean_reward_mean")
    timed_volume = metric_mean(timed_aggregate, "eval_all", "total_volume_mean")
    random_volume = metric_mean(random_aggregate, "eval_all", "total_volume_mean")
    split = routing_split(timed_records)
    activation_ok = 0.05 <= timed_activation <= 0.40
    spread_delta = float(paired.get("mean", 0.0))
    reward_delta = timed_reward - random_reward

    return {
        "params": {
            "routing_rho": float(routing_rho),
            "routing_mode": routing_mode,
            "routing_spread_cap_ticks": float(spread_cap),
            "routing_withhold_rate": float(withhold_rate),
            "routing_relative_improvement_ticks": float(relative_improvement),
            "routing_target_kappa": float(target_kappa),
            "timing_method": timing_method,
            "matched_random_rate_source": match_random_rate_source,
            "matched_random_rate": timed_activation,
        },
        "methods": {
            timing_method: {
                "seed_results": timed_results,
                "aggregate": timed_aggregate,
                "routing_split": split,
            },
            RANDOM_METHOD: {
                "seed_results": random_results,
                "aggregate": random_aggregate,
            },
        },
        "paired_random_minus_gated_spread": paired,
        "screen": {
            "activation_ok": bool(activation_ok),
            "timed_activation_rate": timed_activation,
            "timed_spread": timed_spread,
            "random_spread": random_spread,
            "spread_delta_random_minus_gated": spread_delta,
            "timed_reward": timed_reward,
            "random_reward": random_reward,
            "reward_delta_gated_minus_random": reward_delta,
            "timed_volume": timed_volume,
            "random_volume": random_volume,
            "active_cap_met_share_active": split["active_cap_met_share_active"],
            "active_cap_not_met_share_active": split["active_cap_not_met_share_active"],
            "active_routing_multiplier": split["active_routing_multiplier"],
            "active_targeted_volume": split["active_targeted_volume"],
            "active_targeted_orders": split["active_targeted_orders"],
            "active_eligible_agents": split["active_eligible_agents"],
            "candidate": bool(activation_ok and spread_delta >= 0.0 and reward_delta >= -0.005),
        },
    }


def timed_method_from_cell(cell: Dict[str, object]) -> str:
    methods = dict(cell["methods"])
    for method in TIMING_METHODS:
        if method in methods:
            return method
    raise KeyError("No timing method found in cell")


def flatten_cell(index: int, cell: Dict[str, object]) -> Dict[str, object]:
    params = dict(cell["params"])
    screen = dict(cell["screen"])
    paired = dict(cell["paired_random_minus_gated_spread"])
    timed_method = timed_method_from_cell(cell)
    split = dict(dict(dict(cell["methods"])[timed_method])["routing_split"])
    return {
        "cell": index,
        **params,
        **screen,
        "paired_positive": paired.get("positive", 0.0),
        "paired_n": paired.get("n", 0.0),
        "paired_se": paired.get("se", 0.0),
        "active_share": split.get("active_share", 0.0),
        "active_cap_met_share_all": split.get("active_cap_met_share_all", 0.0),
        "active_cap_not_met_share_all": split.get("active_cap_not_met_share_all", 0.0),
        "active_targeted_volume": split.get("active_targeted_volume", 0.0),
        "active_targeted_orders": split.get("active_targeted_orders", 0.0),
        "active_eligible_agents": split.get("active_eligible_agents", 0.0),
    }


def write_grid_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--seeds", default="0-2")
    parser.add_argument("--routing-modes", default="book_multiplier")
    parser.add_argument("--routing-rhos", default="0.25,0.75")
    parser.add_argument("--routing-spread-caps", default="4,8,10")
    parser.add_argument("--routing-withhold-rates", default="0.0,0.5")
    parser.add_argument("--routing-relative-improvements", default="1.0")
    parser.add_argument("--routing-target-kappas", default="0.0")
    parser.add_argument("--timing-method", choices=sorted(TIMING_METHODS), default="gated_intervention_controlled")
    parser.add_argument("--gate-config", default="revision/experiments/outputs/lob_mappo_gate_calibration_conservative_v1/lob_mappo_gate_config.json")
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
    parser.add_argument("--routing-routable-fraction", type=float, default=None)
    parser.add_argument("--enforce-max-inventory", action="store_true")
    parser.add_argument("--quote-priority-mode", choices=["fixed", "rotating"], default=None)
    parser.add_argument("--agent-quote-quantity", type=int, default=None)
    parser.add_argument("--common-random-numbers", action="store_true")
    parser.add_argument("--agent-id-feature", action="store_true")
    parser.add_argument("--policy-mode", choices=["shared", "multihead", "independent"], default="shared")
    parser.add_argument("--eval-gate-off", action="store_true")
    parser.add_argument("--match-random-rate-source", choices=["eval", "train"], default="eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--output-dir", default="revision/experiments/outputs/lob_mappo_routing_grid_v1")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    routing_modes = parse_strings(args.routing_modes)
    for mode in routing_modes:
        if mode not in {
            "book_multiplier",
            "eligible_priority",
            "relative_priority",
            "relative_reallocate",
            "conditional_reallocate",
            "platform_reallocate",
            "continuous_reallocate",
            "balanced_reallocate",
        }:
            raise SystemExit(f"Unknown routing mode '{mode}'")
    routing_rhos = parse_floats(args.routing_rhos)
    spread_caps = parse_floats(args.routing_spread_caps)
    withhold_rates = parse_floats(args.routing_withhold_rates)
    relative_improvements = parse_floats(args.routing_relative_improvements)
    target_kappas = parse_floats(args.routing_target_kappas)
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
    if args.routing_routable_fraction is not None:
        config_kwargs["routing_routable_fraction"] = float(args.routing_routable_fraction)
    if args.enforce_max_inventory:
        config_kwargs["enforce_max_inventory"] = True
    if args.quote_priority_mode is not None:
        config_kwargs["quote_priority_mode"] = args.quote_priority_mode
    if args.agent_quote_quantity is not None:
        config_kwargs["agent_quote_quantity"] = int(args.agent_quote_quantity)
    if args.common_random_numbers:
        config_kwargs["common_random_numbers"] = True
    base_config = LocalLOBConfig(**config_kwargs)
    gate_config = load_lob_gate_config(args.gate_config)
    timing_gate_mode = TIMING_METHODS[args.timing_method]
    mappo_config = MAPPOConfig(
        obs_dim=1,
        critic_obs_dim=1,
        n_actions=base_config.n_actions,
        n_agents=base_config.n_agents,
        policy_mode=args.policy_mode,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )

    cells: List[Dict[str, object]] = []
    total = (
        len(routing_modes)
        * len(routing_rhos)
        * len(spread_caps)
        * len(withhold_rates)
        * len(relative_improvements)
        * len(target_kappas)
    )
    for routing_mode in routing_modes:
        for routing_rho in routing_rhos:
            for spread_cap in spread_caps:
                for withhold_rate in withhold_rates:
                    for relative_improvement in relative_improvements:
                        for target_kappa in target_kappas:
                            print(
                                (
                                    f"[{len(cells) + 1}/{total}] mode={routing_mode} "
                                    f"rho={routing_rho} cap={spread_cap} withhold={withhold_rate} "
                                    f"rel={relative_improvement} kappa={target_kappa}"
                                ),
                                flush=True,
                            )
                            cells.append(
                                run_cell(
                                    base_config,
                                    gate_config,
                                    mappo_config,
                                    seeds,
                                    args.episodes,
                                    args.eval_episodes,
                                    routing_mode,
                                    routing_rho,
                                    spread_cap,
                                    withhold_rate,
                                    relative_improvement,
                                    target_kappa,
                                    args.timing_method,
                                    timing_gate_mode,
                                    args.agent_id_feature,
                                    args.eval_gate_off,
                                    args.match_random_rate_source,
                                )
                            )

    rows = [flatten_cell(index, cell) for index, cell in enumerate(cells)]
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            not bool(row["candidate"]),
            -float(row["spread_delta_random_minus_gated"]),
            -float(row["reward_delta_gated_minus_random"]),
            -float(row["active_cap_met_share_active"]),
        ),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metadata": {
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "episode_length": args.episode_length,
            "n_agents": args.n_agents,
            "seeds": seeds,
            "routing_modes": routing_modes,
            "routing_rhos": routing_rhos,
            "routing_spread_caps": spread_caps,
            "routing_withhold_rates": withhold_rates,
            "routing_relative_improvements": relative_improvements,
            "routing_target_kappas": target_kappas,
            "timing_method": args.timing_method,
            "agent_id_feature": args.agent_id_feature,
            "eval_gate_off": args.eval_gate_off,
            "match_random_rate_source": args.match_random_rate_source,
            "device": device,
            "config": asdict(base_config),
            "gate_config": asdict(gate_config),
            "gate_config_path": args.gate_config,
            "mappo_config": asdict(mappo_config),
            "stage": "local_lob_mappo_routing_grid_pilot_not_full_main",
        },
        "cells": cells,
        "ranked_grid": rows_sorted,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_grid_csv(output_dir / "grid.csv", rows_sorted)
    print(json.dumps({
        "output_dir": str(output_dir),
        "device": device,
        "n_cells": len(cells),
        "top_rows": rows_sorted[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
