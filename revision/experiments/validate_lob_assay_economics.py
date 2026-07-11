#!/usr/bin/env python3
"""Validate local-LOB economics with fixed policies before MARL training.

The script first checks passive market economics, then applies a controlled
routing deviation to test whether the intervention changes seller incentives
before a learned policy or gate is allowed to enter the experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv  # noqa: E402


def parse_seeds(raw: str) -> list[int]:
    if "-" in raw and "," not in raw:
        start, end = raw.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def action_for_half_spread(config: LocalLOBConfig, half_spread: int) -> int:
    try:
        return config.half_spread_levels.index(int(half_spread))
    except ValueError as exc:
        raise SystemExit(
            f"half-spread {half_spread} is not in {config.half_spread_levels}"
        ) from exc


def run_fixed_profile(
    config: LocalLOBConfig,
    risk_state: str,
    actions: Sequence[int],
    seeds: Iterable[int],
    routing_flow_boost: float = 0.0,
    routing_state_severity: float | None = None,
) -> Dict[str, object]:
    per_agent_rewards: list[np.ndarray] = []
    metrics: Dict[str, list[float]] = {
        "mean_reward": [],
        "mean_spread_revenue": [],
        "mean_adverse_selection_loss": [],
        "mean_inventory_penalty": [],
        "total_volume": [],
        "mean_abs_inventory": [],
        "routing_contract_alpha": [],
        "routing_effective_alpha": [],
    }
    for seed in seeds:
        env = LocalLOBMarketEnv(replace(config, seed=int(seed)))
        env.reset(seed=int(seed))
        env.risk_state = risk_state  # Transitions are disabled in this assay.
        reward_rows: list[np.ndarray] = []
        done = False
        while not done:
            _, rewards, done, _, info = env.step(
                actions,
                routing_flow_boost=routing_flow_boost,
                gate_active=routing_flow_boost > 0.0,
                routing_state_severity=routing_state_severity,
            )
            reward_rows.append(np.array([rewards[i] for i in range(config.n_agents)], dtype=float))
            for key in metrics:
                metrics[key].append(float(info[key]))
        per_agent_rewards.append(np.mean(np.stack(reward_rows), axis=0))

    agent_rewards = np.mean(np.stack(per_agent_rewards), axis=0)
    return {
        "risk_state": risk_state,
        "actions": [int(action) for action in actions],
        "half_spreads": [int(config.half_spread_levels[action]) for action in actions],
        "mean_reward": float(np.mean(metrics["mean_reward"])),
        "mean_spread_revenue": float(np.mean(metrics["mean_spread_revenue"])),
        "mean_adverse_selection_loss": float(np.mean(metrics["mean_adverse_selection_loss"])),
        "mean_inventory_penalty": float(np.mean(metrics["mean_inventory_penalty"])),
        "total_volume": float(np.mean(metrics["total_volume"])),
        "mean_abs_inventory": float(np.mean(metrics["mean_abs_inventory"])),
        "routing_contract_alpha": float(np.mean(metrics["routing_contract_alpha"])),
        "routing_effective_alpha": float(np.mean(metrics["routing_effective_alpha"])),
        "agent_rewards": [float(value) for value in agent_rewards],
        "priority_reward_gap": float(np.max(agent_rewards) - np.min(agent_rewards)),
    }


def common_spread_curve(
    config: LocalLOBConfig,
    risk_state: str,
    seeds: Sequence[int],
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for action, half_spread in enumerate(config.half_spread_levels):
        result = run_fixed_profile(
            config,
            risk_state,
            [action] * config.n_agents,
            seeds,
        )
        rows.append({"action": action, "half_spread": half_spread, **result})
    return rows


def best_curve_row(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return max(rows, key=lambda row: float(row["mean_reward"]))


def unilateral_response_curve(
    config: LocalLOBConfig,
    risk_state: str,
    other_action: int,
    seeds: Sequence[int],
    routing_flow_boost: float,
    routing_state_severity: float | None = None,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for action, half_spread in enumerate(config.half_spread_levels):
        actions = [other_action] * config.n_agents
        actions[0] = action
        result = run_fixed_profile(
            config,
            risk_state,
            actions,
            seeds,
            routing_flow_boost=routing_flow_boost,
            routing_state_severity=routing_state_severity,
        )
        rows.append({
            "agent_zero_action": action,
            "agent_zero_half_spread": half_spread,
            "agent_zero_reward": float(result["agent_rewards"][0]),  # type: ignore[index]
            "profile_mean_reward": float(result["mean_reward"]),
            "total_volume": float(result["total_volume"]),
        })
    return rows


def best_response_row(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return max(rows, key=lambda row: float(row["agent_zero_reward"]))


def contract_severity(config: LocalLOBConfig, half_spread: float) -> float:
    if config.routing_contract != "excess_spread_pool":
        return 1.0
    scale = max(float(config.routing_excess_scale_ticks), 1e-8)
    return float(np.clip(
        (float(half_spread) - float(config.routing_excess_reference_half_spread))
        / scale,
        0.0,
        1.0,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--seeds", default="0-29")
    parser.add_argument("--inventory-penalty", type=float, default=0.0005)
    parser.add_argument("--base-flow-intensity", type=float, default=2.0)
    parser.add_argument("--low-sigma", type=float, default=0.01)
    parser.add_argument("--high-sigma", type=float, default=0.04)
    parser.add_argument("--informed-flow-prob", type=float, default=0.5)
    parser.add_argument("--spread-elasticity", type=float, default=0.05)
    parser.add_argument("--seed-liquidity-levels", type=int, default=5)
    parser.add_argument("--seed-liquidity-qty", type=int, default=0)
    parser.add_argument("--max-inventory", type=int, default=20)
    parser.add_argument("--agent-quote-quantity", type=int, default=3)
    parser.add_argument("--routing-mode", choices=["book_multiplier", "platform_reallocate"], default="platform_reallocate")
    parser.add_argument("--routing-routable-fraction", type=float, default=0.5)
    parser.add_argument(
        "--routing-platform-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--routing-contract",
        choices=["fixed_share", "excess_spread_pool"],
        default="fixed_share",
    )
    parser.add_argument("--routing-excess-reference-half-spread", type=float, default=4.0)
    parser.add_argument("--routing-excess-scale-ticks", type=float, default=1.0)
    parser.add_argument(
        "--routing-eligibility-mode",
        choices=["relative", "absolute_cap", "soft_rank"],
        default="relative",
    )
    parser.add_argument("--common-random-numbers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quote-priority-mode", choices=["fixed", "rotating"], default="rotating")
    parser.add_argument("--collusive-half-spread", type=int, default=5)
    parser.add_argument("--undercut-half-spread", type=int, default=3)
    parser.add_argument("--min-risk-widening-ticks", type=float, default=1.0)
    parser.add_argument("--max-priority-reward-gap", type=float, default=0.002)
    parser.add_argument("--routing-rho", type=float, default=0.75)
    parser.add_argument("--routing-relative-improvement-ticks", type=float, default=0.5)
    parser.add_argument("--routing-target-kappa", type=float, default=4.0)
    parser.add_argument("--max-baseline-undercut-rent-ratio", type=float, default=1.0)
    parser.add_argument("--min-routing-undercut-gain-multiplier", type=float, default=2.0)
    parser.add_argument("--min-routing-undercut-gain-increase", type=float, default=0.002)
    parser.add_argument("--min-routed-best-response-gain", type=float, default=0.002)
    parser.add_argument("--min-best-response-shift-ticks", type=float, default=1.0)
    parser.add_argument("--max-normal-state-routing-gain-change", type=float, default=0.0005)
    parser.add_argument("--min-state-selective-gain-increase", type=float, default=0.004)
    parser.add_argument("--routing-fallback-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--output-dir",
        default="revision/experiments/outputs/lob_assay_economic_validation_v1",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    config = LocalLOBConfig(
        n_agents=args.n_agents,
        episode_length=args.episode_length,
        max_inventory=args.max_inventory,
        inventory_penalty=args.inventory_penalty,
        base_flow_intensity=args.base_flow_intensity,
        routing_mode=args.routing_mode,
        routing_routable_fraction=args.routing_routable_fraction,
        routing_platform_balanced=args.routing_platform_balanced,
        routing_contract=args.routing_contract,
        routing_excess_reference_half_spread=args.routing_excess_reference_half_spread,
        routing_excess_scale_ticks=args.routing_excess_scale_ticks,
        routing_eligibility_mode=args.routing_eligibility_mode,
        routing_relative_improvement_ticks=args.routing_relative_improvement_ticks,
        routing_target_kappa=args.routing_target_kappa,
        common_random_numbers=args.common_random_numbers,
        low_sigma=args.low_sigma,
        high_sigma=args.high_sigma,
        p_low_to_high=0.0,
        p_high_to_low=0.0,
        risk_coupled_economics=True,
        mark_to_fundamental=True,
        informed_flow_prob=args.informed_flow_prob,
        spread_elasticity=args.spread_elasticity,
        enforce_max_inventory=True,
        quote_priority_mode=args.quote_priority_mode,
        agent_quote_quantity=args.agent_quote_quantity,
        seed_liquidity_levels=args.seed_liquidity_levels,
        seed_liquidity_qty=args.seed_liquidity_qty,
    )

    low_curve = common_spread_curve(config, "low", seeds)
    high_curve = common_spread_curve(config, "high", seeds)
    low_best = best_curve_row(low_curve)
    high_best = best_curve_row(high_curve)

    collusive_action = action_for_half_spread(config, args.collusive_half_spread)
    undercut_action = action_for_half_spread(config, args.undercut_half_spread)
    collusive = run_fixed_profile(
        config,
        "low",
        [collusive_action] * config.n_agents,
        seeds,
    )
    deviation_actions = [collusive_action] * config.n_agents
    deviation_actions[0] = undercut_action
    deviation = run_fixed_profile(config, "low", deviation_actions, seeds)
    symmetric_agent_zero_reward = float(collusive["agent_rewards"][0])  # type: ignore[index]
    deviator_reward = float(deviation["agent_rewards"][0])  # type: ignore[index]
    undercut_gain = deviator_reward - symmetric_agent_zero_reward

    routed_config = replace(config, routing_mode="platform_reallocate")
    high_state_severity = contract_severity(config, args.collusive_half_spread)
    routed_collusive = run_fixed_profile(
        routed_config,
        "low",
        [collusive_action] * config.n_agents,
        seeds,
        routing_flow_boost=args.routing_rho,
        routing_state_severity=high_state_severity,
    )
    routed_deviation = run_fixed_profile(
        routed_config,
        "low",
        deviation_actions,
        seeds,
        routing_flow_boost=args.routing_rho,
        routing_state_severity=high_state_severity,
    )
    routed_symmetric_agent_zero_reward = float(routed_collusive["agent_rewards"][0])  # type: ignore[index]
    routed_deviator_reward = float(routed_deviation["agent_rewards"][0])  # type: ignore[index]
    routed_undercut_gain = routed_deviator_reward - routed_symmetric_agent_zero_reward
    routing_gain_increase = routed_undercut_gain - undercut_gain
    low_risk_collusive_rent = float(collusive["mean_reward"])
    baseline_undercut_rent_ratio = undercut_gain / max(low_risk_collusive_rent, 1e-12)
    routing_undercut_gain_multiplier = (
        routed_undercut_gain / undercut_gain if undercut_gain > 0.0 else None
    )
    baseline_response_curve = unilateral_response_curve(
        config,
        "low",
        collusive_action,
        seeds,
        routing_flow_boost=0.0,
    )
    routed_response_curve = unilateral_response_curve(
        routed_config,
        "low",
        collusive_action,
        seeds,
        routing_flow_boost=args.routing_rho,
        routing_state_severity=high_state_severity,
    )
    baseline_best_response = best_response_row(baseline_response_curve)
    routed_best_response = best_response_row(routed_response_curve)
    best_response_shift = (
        float(baseline_best_response["agent_zero_half_spread"])
        - float(routed_best_response["agent_zero_half_spread"])
    )
    routed_collusive_reward = next(
        float(row["agent_zero_reward"])
        for row in routed_response_curve
        if int(row["agent_zero_action"]) == collusive_action
    )
    routed_best_response_gain = (
        float(routed_best_response["agent_zero_reward"]) - routed_collusive_reward
    )
    normal_action = int(low_best["action"])
    normal_half_spread = float(low_best["half_spread"])
    normal_actions = [normal_action] * config.n_agents
    normal_deviation_actions = list(normal_actions)
    normal_deviation_actions[0] = undercut_action
    normal_profile = run_fixed_profile(config, "low", normal_actions, seeds)
    normal_deviation = run_fixed_profile(
        config,
        "low",
        normal_deviation_actions,
        seeds,
    )
    normal_state_severity = contract_severity(config, normal_half_spread)
    routed_normal_profile = run_fixed_profile(
        routed_config,
        "low",
        normal_actions,
        seeds,
        routing_flow_boost=args.routing_rho,
        routing_state_severity=normal_state_severity,
    )
    routed_normal_deviation = run_fixed_profile(
        routed_config,
        "low",
        normal_deviation_actions,
        seeds,
        routing_flow_boost=args.routing_rho,
        routing_state_severity=normal_state_severity,
    )
    normal_undercut_gain = (
        float(normal_deviation["agent_rewards"][0])  # type: ignore[index]
        - float(normal_profile["agent_rewards"][0])  # type: ignore[index]
    )
    routed_normal_undercut_gain = (
        float(routed_normal_deviation["agent_rewards"][0])  # type: ignore[index]
        - float(routed_normal_profile["agent_rewards"][0])  # type: ignore[index]
    )
    normal_routing_gain_change = routed_normal_undercut_gain - normal_undercut_gain
    state_selective_gain_increase = routing_gain_increase - normal_routing_gain_change
    normal_response_curve = unilateral_response_curve(
        config,
        "low",
        normal_action,
        seeds,
        routing_flow_boost=0.0,
    )
    routed_normal_response_curve = unilateral_response_curve(
        routed_config,
        "low",
        normal_action,
        seeds,
        routing_flow_boost=args.routing_rho,
        routing_state_severity=normal_state_severity,
    )
    normal_best_response = best_response_row(normal_response_curve)
    routed_normal_best_response = best_response_row(routed_normal_response_curve)

    checks = {
        "risk_conditioned_optimum": bool(
            float(high_best["half_spread"]) - float(low_best["half_spread"])
            >= args.min_risk_widening_ticks
        ),
        "positive_low_risk_collusive_rent": bool(float(collusive["mean_reward"]) > 0.0),
        "baseline_collusion_not_trivially_unstable": bool(
            baseline_undercut_rent_ratio <= args.max_baseline_undercut_rent_ratio
        ),
        "routing_makes_undercut_profitable": bool(routed_undercut_gain > 0.0),
        "routing_strengthens_undercut_incentive": bool(
            routing_gain_increase >= args.min_routing_undercut_gain_increase
        ),
        "routing_amplifies_undercut_incentive": bool(
            routed_undercut_gain > 0.0
            and (
                undercut_gain <= 0.0
                or (
                    routing_undercut_gain_multiplier is not None
                    and routing_undercut_gain_multiplier
                    >= args.min_routing_undercut_gain_multiplier
                )
            )
        ),
        "routing_changes_best_response_downward": bool(
            best_response_shift >= args.min_best_response_shift_ticks
        ),
        "routed_best_response_satisfies_contract_target": bool(
            (
                args.routing_eligibility_mode == "soft_rank"
                and float(routed_best_response["agent_zero_half_spread"])
                < args.collusive_half_spread
            )
            or (
                args.routing_eligibility_mode != "soft_rank"
                and float(routed_best_response["agent_zero_half_spread"])
                <= args.undercut_half_spread
            )
        ),
        "routed_best_response_gain_is_material": bool(
            routed_best_response_gain >= args.min_routed_best_response_gain
        ),
        "normal_state_contract_is_economically_inactive": bool(
            abs(normal_routing_gain_change)
            <= args.max_normal_state_routing_gain_change
        ),
        "normal_state_best_response_is_unchanged": bool(
            int(normal_best_response["agent_zero_action"])
            == int(routed_normal_best_response["agent_zero_action"])
        ),
        "high_state_gain_increase_exceeds_normal_state": bool(
            state_selective_gain_increase
            >= args.min_state_selective_gain_increase
        ),
        "symmetric_routing_does_not_subsidize_collusive_reward": bool(
            abs(float(routed_collusive["mean_reward"]) - float(collusive["mean_reward"]))
            <= args.routing_fallback_tolerance
        ),
        "symmetric_priority_fairness": bool(
            float(collusive["priority_reward_gap"]) <= args.max_priority_reward_gap
        ),
        "high_risk_has_more_adverse_selection": bool(
            float(high_curve[0]["mean_adverse_selection_loss"])
            > float(low_curve[0]["mean_adverse_selection_loss"])
        ),
        "wide_spread_suppresses_flow": bool(
            float(low_curve[-1]["total_volume"]) < float(low_curve[0]["total_volume"])
        ),
    }
    summary = {
        "metadata": {
            "stage": "local_lob_fixed_policy_economic_acceptance",
            "seeds": seeds,
            "config": asdict(config),
            "collusive_half_spread": args.collusive_half_spread,
            "undercut_half_spread": args.undercut_half_spread,
        },
        "low_risk_curve": low_curve,
        "high_risk_curve": high_curve,
        "low_risk_best": low_best,
        "high_risk_best": high_best,
        "collusive_profile": collusive,
        "undercut_profile": deviation,
        "undercut_gain": undercut_gain,
        "baseline_undercut_rent_ratio": baseline_undercut_rent_ratio,
        "routed_collusive_profile": routed_collusive,
        "routed_undercut_profile": routed_deviation,
        "routed_undercut_gain": routed_undercut_gain,
        "routing_undercut_gain_increase": routing_gain_increase,
        "routing_undercut_gain_multiplier": routing_undercut_gain_multiplier,
        "baseline_response_curve": baseline_response_curve,
        "routed_response_curve": routed_response_curve,
        "baseline_best_response": baseline_best_response,
        "routed_best_response": routed_best_response,
        "best_response_shift_ticks": best_response_shift,
        "routed_best_response_gain": routed_best_response_gain,
        "high_state_contract_severity": high_state_severity,
        "normal_state_contract_severity": normal_state_severity,
        "normal_profile": normal_profile,
        "normal_deviation_profile": normal_deviation,
        "normal_undercut_gain": normal_undercut_gain,
        "routed_normal_profile": routed_normal_profile,
        "routed_normal_deviation_profile": routed_normal_deviation,
        "routed_normal_undercut_gain": routed_normal_undercut_gain,
        "normal_routing_gain_change": normal_routing_gain_change,
        "state_selective_gain_increase": state_selective_gain_increase,
        "normal_response_curve": normal_response_curve,
        "routed_normal_response_curve": routed_normal_response_curve,
        "normal_best_response": normal_best_response,
        "routed_normal_best_response": routed_normal_best_response,
        "checks": checks,
        "passes_all": bool(all(checks.values())),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "spread_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "risk_state",
            "action",
            "half_spread",
            "mean_reward",
            "mean_spread_revenue",
            "mean_adverse_selection_loss",
            "mean_inventory_penalty",
            "total_volume",
            "mean_abs_inventory",
            "priority_reward_gap",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in [*low_curve, *high_curve]:
            writer.writerow({key: row[key] for key in fields})

    print(json.dumps({
        "output_dir": str(output_dir),
        "low_risk_best_half_spread": low_best["half_spread"],
        "high_risk_best_half_spread": high_best["half_spread"],
        "undercut_gain": undercut_gain,
        "baseline_undercut_rent_ratio": baseline_undercut_rent_ratio,
        "routed_undercut_gain": routed_undercut_gain,
        "routing_undercut_gain_increase": routing_gain_increase,
        "routing_undercut_gain_multiplier": routing_undercut_gain_multiplier,
        "baseline_best_response_half_spread": baseline_best_response["agent_zero_half_spread"],
        "routed_best_response_half_spread": routed_best_response["agent_zero_half_spread"],
        "best_response_shift_ticks": best_response_shift,
        "routed_best_response_gain": routed_best_response_gain,
        "high_state_contract_severity": high_state_severity,
        "normal_state_contract_severity": normal_state_severity,
        "normal_routing_gain_change": normal_routing_gain_change,
        "state_selective_gain_increase": state_selective_gain_increase,
        "normal_best_response_half_spread": normal_best_response["agent_zero_half_spread"],
        "routed_normal_best_response_half_spread": routed_normal_best_response["agent_zero_half_spread"],
        "checks": checks,
        "passes_all": summary["passes_all"],
    }, indent=2))


if __name__ == "__main__":
    main()
