#!/usr/bin/env python3
"""Preregistered fixed-policy checks for the ABIDES dealer-market economics."""

import json
from dataclasses import replace
from statistics import mean
from typing import Dict, Iterable, Sequence

import numpy as np

from ace_abides.economic_env import (
    ABIDESDealerMarketConfig,
    ABIDESDealerMarketEnv,
    generate_exogenous_path,
    path_digest,
)


def run_profile(
    config: ABIDESDealerMarketConfig,
    seed: int,
    actions: Sequence[int],
    gate_active: bool,
) -> Dict[str, float]:
    env = ABIDESDealerMarketEnv(config)
    state = env.reset(seed=seed)
    rewards = []
    agent_zero_rewards = []
    realized_flow = []
    targeted = []
    consumer_half_spreads = []
    while not state["done"]:
        state = env.step(actions, gate_active=gate_active)
        rewards.extend(state["rewards"])
        agent_zero_rewards.append(state["rewards"][0])
        info = state["info"]
        realized_flow.append(info["realized_flow_quantity"])
        targeted.append(info.get("targeted_event_count", 0))
        consumer_half_spreads.append(info["avg_half_spread"])
    env.close()
    return {
        "mean_reward": float(mean(rewards)),
        "agent_zero_reward": float(mean(agent_zero_rewards)),
        "mean_realized_flow": float(mean(realized_flow)),
        "targeted_events": float(sum(targeted)),
        "mean_half_spread": float(mean(consumer_half_spreads)),
    }


def paired(values_a: Iterable[float], values_b: Iterable[float]) -> Dict[str, float]:
    delta = np.asarray(list(values_a), dtype=float) - np.asarray(
        list(values_b), dtype=float
    )
    return {
        "mean": float(np.mean(delta)),
        "positive": int(np.sum(delta > 1e-9)),
        "negative": int(np.sum(delta < -1e-9)),
        "ties": int(np.sum(np.abs(delta) <= 1e-9)),
    }


def main() -> None:
    seeds = list(range(100, 110))
    base = ABIDESDealerMarketConfig(episode_steps=30)
    low = replace(base, fixed_risk_state="low")
    high = replace(base, fixed_risk_state="high")

    action_h1 = [0] * base.n_agents
    action_h4 = [3] * base.n_agents
    action_h5 = [4] * base.n_agents
    wide_deviation = [2] + [4] * (base.n_agents - 1)
    competitive_deviation = [0] + [1] * (base.n_agents - 1)

    results = {
        "low_h1": [],
        "low_h5": [],
        "high_h1": [],
        "high_h4": [],
        "wide_off": [],
        "wide_on": [],
        "competitive_off": [],
        "competitive_on": [],
    }
    for seed in seeds:
        results["low_h1"].append(run_profile(low, seed, action_h1, False))
        results["low_h5"].append(run_profile(low, seed, action_h5, False))
        results["high_h1"].append(run_profile(high, seed, action_h1, False))
        results["high_h4"].append(run_profile(high, seed, action_h4, False))
        results["wide_off"].append(
            run_profile(low, seed, wide_deviation, False)
        )
        results["wide_on"].append(run_profile(low, seed, wide_deviation, True))
        results["competitive_off"].append(
            run_profile(low, seed, competitive_deviation, False)
        )
        results["competitive_on"].append(
            run_profile(low, seed, competitive_deviation, True)
        )

    high_widening = paired(
        (row["mean_reward"] for row in results["high_h4"]),
        (row["mean_reward"] for row in results["high_h1"]),
    )
    low_wide_rent = {
        "mean_reward": float(
            mean(row["mean_reward"] for row in results["low_h5"])
        ),
        "half_spread_delta": float(
            mean(row["mean_half_spread"] for row in results["low_h5"])
            - mean(row["mean_half_spread"] for row in results["low_h1"])
        ),
    }
    wide_marginal = paired(
        (row["agent_zero_reward"] for row in results["wide_on"]),
        (row["agent_zero_reward"] for row in results["wide_off"]),
    )
    competitive_marginal = paired(
        (row["agent_zero_reward"] for row in results["competitive_on"]),
        (row["agent_zero_reward"] for row in results["competitive_off"]),
    )

    all_rows = [row for group in results.values() for row in group]
    expected_flow = float(base.flow_events_per_step * base.flow_quantity)
    flow_conserved = all(
        abs(row["mean_realized_flow"] - expected_flow) < 1e-9
        for row in all_rows
    )
    same_path = path_digest(generate_exogenous_path(base, 123)) == path_digest(
        generate_exogenous_path(base, 123)
    )
    different_path = path_digest(generate_exogenous_path(base, 123)) != path_digest(
        generate_exogenous_path(base, 124)
    )

    checks = {
        "high_risk_widening": high_widening["mean"] > 0
        and high_widening["positive"] >= 8,
        "low_risk_wide_positive_rent": low_wide_rent["mean_reward"] > 0
        and low_wide_rent["half_spread_delta"] >= 3.0,
        "wide_state_marginal_routing_gain": wide_marginal["mean"] > 0
        and wide_marginal["positive"] >= 8,
        "competitive_state_no_routing_subsidy": abs(
            competitive_marginal["mean"]
        ) < 1e-9,
        "flow_conserved": flow_conserved,
        "same_seed_path_deterministic": same_path,
        "different_seed_path_distinct": different_path,
    }
    if not all(checks.values()):
        raise AssertionError(
            json.dumps(
                {
                    "checks": checks,
                    "high_widening": high_widening,
                    "low_wide_rent": low_wide_rent,
                    "wide_marginal": wide_marginal,
                    "competitive_marginal": competitive_marginal,
                },
                sort_keys=True,
            )
        )

    print(
        json.dumps(
            {
                "status": "pass",
                "backend": "abides",
                "seeds": seeds,
                "checks": checks,
                "high_widening": high_widening,
                "low_wide_rent": low_wide_rent,
                "wide_marginal": wide_marginal,
                "competitive_marginal": competitive_marginal,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
