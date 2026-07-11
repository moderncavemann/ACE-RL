#!/usr/bin/env python3
"""Validate state-contingent routing timing before any MAPPO training.

The assay alternates low-risk normal-spread and high-spread market regimes.
Timed and shuffled arms receive the same nominal activation count on identical
CRN paths. A fixed rational responder undercuts only when the announced
contract has a positive excess-spread-funded routing pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_lob_assay_economics import (  # noqa: E402
    action_for_half_spread,
    contract_severity,
    parse_seeds,
)
from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv  # noqa: E402


def mean_se(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "se": (
            float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            if len(arr) > 1
            else 0.0
        ),
    }


def high_regime_mask(length: int, block_length: int) -> np.ndarray:
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    return np.asarray([
        (step // block_length) % 2 == 1
        for step in range(length)
    ], dtype=bool)


def timing_schedules(
    high_mask: np.ndarray,
    activation_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(high_mask)
    active_count = int(round(float(np.clip(activation_rate, 0.0, 1.0)) * length))
    high_indices = np.flatnonzero(high_mask)
    if active_count > len(high_indices):
        raise ValueError("timed activation count exceeds available high-spread windows")
    rng = np.random.default_rng(seed)
    rng.shuffle(high_indices)
    timed = np.zeros(length, dtype=bool)
    timed[high_indices[:active_count]] = True
    shuffled = timed[rng.permutation(length)]
    return timed, shuffled


def run_schedule(
    config: LocalLOBConfig,
    seed: int,
    high_mask: np.ndarray,
    schedule: np.ndarray,
    normal_action: int,
    high_action: int,
    undercut_action: int,
) -> Dict[str, float]:
    env = LocalLOBMarketEnv(replace(config, seed=seed))
    env.reset(seed=seed)
    env.risk_state = "low"
    rows: list[Dict[str, float]] = []
    agent_zero_rewards: list[float] = []

    for step, active in enumerate(schedule):
        high_regime = bool(high_mask[step])
        baseline_action = high_action if high_regime else normal_action
        baseline_half_spread = float(config.half_spread_levels[baseline_action])
        severity = contract_severity(config, baseline_half_spread)
        actions = np.full(config.n_agents, baseline_action, dtype=np.int64)
        if bool(active) and severity > 0.0:
            actions[0] = undercut_action
        _, rewards, _, _, info = env.step(
            actions,
            routing_flow_boost=1.0 if bool(active) else 0.0,
            gate_active=bool(active),
            gate_score=severity,
            routing_state_severity=severity,
        )
        rows.append({
            "high_regime": float(high_regime),
            "active": float(active),
            "avg_agent_spread": float(info["avg_agent_spread"]),
            "mean_reward": float(info["mean_reward"]),
            "total_volume": float(info["total_volume"]),
            "targeted_volume": float(info["routing_targeted_volume"]),
            "contract_alpha": float(info["routing_contract_alpha"]),
            "effective_alpha": float(info["routing_effective_alpha"]),
        })
        agent_zero_rewards.append(float(rewards[0]))

    def mean_for(key: str, mask: np.ndarray | None = None) -> float:
        values = np.asarray([row[key] for row in rows], dtype=float)
        if mask is not None:
            values = values[mask]
        return float(np.mean(values)) if len(values) else 0.0

    return {
        "avg_agent_spread": mean_for("avg_agent_spread"),
        "high_regime_avg_agent_spread": mean_for(
            "avg_agent_spread", high_mask
        ),
        "normal_regime_avg_agent_spread": mean_for(
            "avg_agent_spread", ~high_mask
        ),
        "mean_reward": mean_for("mean_reward"),
        "agent_zero_reward": float(np.mean(agent_zero_rewards)),
        "total_volume": mean_for("total_volume"),
        "targeted_volume": mean_for("targeted_volume"),
        "nominal_activation_rate": mean_for("active"),
        "high_regime_activation_rate": mean_for("active", high_mask),
        "normal_regime_activation_rate": mean_for("active", ~high_mask),
        "contract_alpha": mean_for("contract_alpha"),
        "effective_alpha": mean_for("effective_alpha"),
        "normal_regime_effective_alpha": mean_for("effective_alpha", ~high_mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-99")
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--activation-rate", type=float, default=0.25)
    parser.add_argument("--normal-half-spread", type=int, default=4)
    parser.add_argument("--high-half-spread", type=int, default=5)
    parser.add_argument("--undercut-half-spread", type=int, default=3)
    parser.add_argument(
        "--routing-eligibility-mode",
        choices=["relative", "absolute_cap", "soft_rank"],
        default="absolute_cap",
    )
    parser.add_argument("--routing-target-kappa", type=float, default=4.0)
    parser.add_argument("--min-high-spread-reduction", type=float, default=None)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.90)
    parser.add_argument("--normal-spread-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--output-dir",
        default="revision/experiments/outputs/lob_state_contingent_routing_timing_v1",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    config = LocalLOBConfig(
        n_agents=10,
        episode_length=args.episode_length,
        inventory_penalty=0.0001,
        base_flow_intensity=2.0,
        routing_mode="platform_reallocate",
        routing_routable_fraction=0.90,
        routing_platform_balanced=False,
        routing_contract="excess_spread_pool",
        routing_excess_reference_half_spread=4.0,
        routing_excess_scale_ticks=1.0,
        routing_eligibility_mode=args.routing_eligibility_mode,
        routing_target_half_spread_cap=3.0,
        routing_target_kappa=args.routing_target_kappa,
        risk_coupled_economics=True,
        mark_to_fundamental=True,
        informed_flow_prob=0.7,
        spread_elasticity=0.14,
        low_sigma=0.005,
        high_sigma=0.08,
        p_low_to_high=0.0,
        p_high_to_low=0.0,
        seed_liquidity_qty=0,
        agent_quote_quantity=1,
        quote_priority_mode="rotating",
        enforce_max_inventory=True,
        common_random_numbers=True,
        seed=seeds[0] if seeds else 0,
    )
    normal_action = action_for_half_spread(config, args.normal_half_spread)
    high_action = action_for_half_spread(config, args.high_half_spread)
    undercut_action = action_for_half_spread(config, args.undercut_half_spread)
    high_mask = high_regime_mask(args.episode_length, args.block_length)
    active_count = min(
        int(round(args.activation_rate * args.episode_length)),
        int(np.sum(high_mask)),
    )
    high_active_share = active_count / max(int(np.sum(high_mask)), 1)
    shuffled_high_active_share = active_count / max(args.episode_length, 1)
    theoretical_timing_delta = (
        (args.high_half_spread - args.undercut_half_spread)
        / config.n_agents
        * (high_active_share - shuffled_high_active_share)
    )
    min_high_spread_reduction = (
        float(args.min_high_spread_reduction)
        if args.min_high_spread_reduction is not None
        else 0.75 * theoretical_timing_delta
    )

    seed_results: list[Dict[str, object]] = []
    for seed in seeds:
        timed_schedule, shuffled_schedule = timing_schedules(
            high_mask,
            args.activation_rate,
            seed=700_001 + seed,
        )
        timed = run_schedule(
            config,
            seed,
            high_mask,
            timed_schedule,
            normal_action,
            high_action,
            undercut_action,
        )
        shuffled = run_schedule(
            config,
            seed,
            high_mask,
            shuffled_schedule,
            normal_action,
            high_action,
            undercut_action,
        )
        seed_results.append({
            "seed": seed,
            "nominal_dose_exact": bool(
                int(np.sum(timed_schedule)) == int(np.sum(shuffled_schedule))
            ),
            "timed": timed,
            "shuffled": shuffled,
            "high_spread_reduction": (
                float(shuffled["high_regime_avg_agent_spread"])
                - float(timed["high_regime_avg_agent_spread"])
            ),
            "normal_spread_delta": (
                float(shuffled["normal_regime_avg_agent_spread"])
                - float(timed["normal_regime_avg_agent_spread"])
            ),
            "agent_zero_reward_delta": (
                float(timed["agent_zero_reward"])
                - float(shuffled["agent_zero_reward"])
            ),
        })

    high_reductions = [
        float(result["high_spread_reduction"]) for result in seed_results
    ]
    normal_deltas = [
        float(result["normal_spread_delta"]) for result in seed_results
    ]
    reward_deltas = [
        float(result["agent_zero_reward_delta"]) for result in seed_results
    ]
    timed_effective = [
        float(dict(result["timed"])["effective_alpha"]) for result in seed_results
    ]
    shuffled_effective = [
        float(dict(result["shuffled"])["effective_alpha"]) for result in seed_results
    ]
    positive_rate = float(np.mean(np.asarray(high_reductions) > 0.0))
    checks = {
        "nominal_dose_exact_every_seed": bool(
            all(bool(result["nominal_dose_exact"]) for result in seed_results)
        ),
        "timed_high_spread_reduction_is_material": bool(
            float(np.mean(high_reductions)) >= min_high_spread_reduction
        ),
        "high_spread_reduction_positive_in_required_seed_rate": bool(
            positive_rate >= args.min_positive_seed_rate
        ),
        "normal_regime_spread_is_unchanged": bool(
            abs(float(np.mean(normal_deltas))) <= args.normal_spread_tolerance
        ),
        "timed_contract_allocates_more_effective_pool": bool(
            float(np.mean(timed_effective)) > float(np.mean(shuffled_effective))
        ),
        "timed_responder_reward_is_not_lower": bool(
            float(np.mean(reward_deltas)) >= 0.0
        ),
    }
    summary = {
        "metadata": {
            "stage": "fixed_policy_state_contingent_routing_oracle_timing",
            "seeds": seeds,
            "config": asdict(config),
            "episode_length": args.episode_length,
            "block_length": args.block_length,
            "activation_rate": args.activation_rate,
            "normal_half_spread": args.normal_half_spread,
            "high_half_spread": args.high_half_spread,
            "undercut_half_spread": args.undercut_half_spread,
            "theoretical_timing_delta": theoretical_timing_delta,
            "min_high_spread_reduction": min_high_spread_reduction,
        },
        "seed_results": seed_results,
        "aggregate": {
            "shuffled_minus_timed_high_regime_spread": {
                **mean_se(high_reductions),
                "positive_seed_count": int(np.sum(np.asarray(high_reductions) > 0.0)),
                "n_seeds": len(high_reductions),
            },
            "shuffled_minus_timed_normal_regime_spread": mean_se(normal_deltas),
            "timed_minus_shuffled_agent_zero_reward": mean_se(reward_deltas),
            "timed_effective_alpha": mean_se(timed_effective),
            "shuffled_effective_alpha": mean_se(shuffled_effective),
            "checks": checks,
            "passes_all": bool(all(checks.values())),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "high_spread_reduction": summary["aggregate"][
            "shuffled_minus_timed_high_regime_spread"
        ],
        "agent_zero_reward_delta": summary["aggregate"][
            "timed_minus_shuffled_agent_zero_reward"
        ],
        "timed_effective_alpha": summary["aggregate"]["timed_effective_alpha"],
        "shuffled_effective_alpha": summary["aggregate"]["shuffled_effective_alpha"],
        "checks": checks,
        "passes_all": summary["aggregate"]["passes_all"],
    }, indent=2))


if __name__ == "__main__":
    main()
