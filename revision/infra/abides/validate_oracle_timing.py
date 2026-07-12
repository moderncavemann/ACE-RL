#!/usr/bin/env python3
"""G4 oracle-versus-within-risk-shuffled timing identification."""

import json
from math import sqrt
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import t

from ace_abides.economic_env import (
    ABIDESDealerMarketConfig,
    ABIDESDealerMarketEnv,
    ExogenousStep,
    generate_exogenous_path,
)


def action_schedule(config: ABIDESDealerMarketConfig) -> List[List[int]]:
    wide = [2] + [4] * (config.n_agents - 1)
    competitive = [0] + [1] * (config.n_agents - 1)
    return [
        list(wide if step % 4 == 0 else competitive)
        for step in range(config.episode_steps)
    ]


def timing_schedules(
    path: Sequence[ExogenousStep], seed: int
) -> Tuple[List[bool], List[bool]]:
    oracle = [
        step.risk_state == "low" and index % 4 == 0
        for index, step in enumerate(path)
    ]
    shuffled = [False] * len(path)
    for risk_state in ("low", "high"):
        indices = [
            index
            for index, step in enumerate(path)
            if step.risk_state == risk_state
        ]
        values = np.asarray([oracle[index] for index in indices], dtype=bool)
        rng = np.random.RandomState(seed ^ (0x510 if risk_state == "low" else 0xA17))
        rng.shuffle(values)
        for index, value in zip(indices, values):
            shuffled[index] = bool(value)
    assert sum(oracle) == sum(shuffled)
    return oracle, shuffled


def run_arm(
    config: ABIDESDealerMarketConfig,
    seed: int,
    schedule: Sequence[bool],
    routing_enabled: bool,
) -> Dict[str, float]:
    env = ABIDESDealerMarketEnv(config)
    state = env.reset(seed=seed)
    actions = action_schedule(config)
    low_execution_spreads = []
    rewards = []
    volumes = []
    effective_activations = []
    for step in range(config.episode_steps):
        state = env.step(
            actions[step],
            gate_active=schedule[step],
            routing_enabled=routing_enabled,
        )
        info = state["info"]
        rewards.append(info["mean_reward"])
        volumes.append(info["realized_flow_quantity"])
        effective_activations.append(
            1.0 if schedule[step] else 0.0
        )
        if info["last_step_risk"] == "low":
            low_execution_spreads.append(info["mean_execution_half_spread"])
    assert state["done"] is True
    env.close()
    return {
        "low_execution_half_spread": float(mean(low_execution_spreads)),
        "mean_reward": float(mean(rewards)),
        "mean_volume": float(mean(volumes)),
        "activation_rate": float(mean(effective_activations)),
    }


def paired_summary(delta: np.ndarray) -> Dict[str, float]:
    n = len(delta)
    avg = float(np.mean(delta))
    se = float(np.std(delta, ddof=1) / sqrt(n))
    critical = float(t.ppf(0.975, df=n - 1))
    return {
        "mean": avg,
        "se": se,
        "ci_low": avg - critical * se,
        "ci_high": avg + critical * se,
        "positive": int(np.sum(delta > 1e-9)),
        "ties": int(np.sum(np.abs(delta) <= 1e-9)),
        "negative": int(np.sum(delta < -1e-9)),
    }


def main() -> None:
    config = ABIDESDealerMarketConfig(episode_steps=100)
    seeds = list(range(200, 210))
    rows = []
    for seed in seeds:
        path = generate_exogenous_path(config, seed)
        oracle_schedule, shuffled_schedule = timing_schedules(path, seed)
        oracle = run_arm(config, seed, oracle_schedule, True)
        shuffled = run_arm(config, seed, shuffled_schedule, True)
        no_routing = run_arm(config, seed, oracle_schedule, False)
        no_intervention = run_arm(
            config, seed, [False] * config.episode_steps, True
        )
        rows.append(
            {
                "seed": seed,
                "oracle": oracle,
                "shuffled": shuffled,
                "no_routing": no_routing,
                "no_intervention": no_intervention,
            }
        )

    primary_delta = np.asarray(
        [
            row["shuffled"]["low_execution_half_spread"]
            - row["oracle"]["low_execution_half_spread"]
            for row in rows
        ],
        dtype=float,
    )
    primary = paired_summary(primary_delta)
    activation_rates = [row["oracle"]["activation_rate"] for row in rows]
    oracle_reward = float(mean(row["oracle"]["mean_reward"] for row in rows))
    shuffled_reward = float(
        mean(row["shuffled"]["mean_reward"] for row in rows)
    )
    reward_guardrail = (
        oracle_reward >= shuffled_reward - 0.10 * max(1.0, abs(shuffled_reward))
    )
    volume_equal = all(
        abs(row["oracle"]["mean_volume"] - config.flow_events_per_step) < 1e-9
        and abs(
            row["oracle"]["mean_volume"]
            - row["shuffled"]["mean_volume"]
        )
        < 1e-9
        for row in rows
    )
    no_routing_exact = all(
        abs(
            row["no_routing"]["low_execution_half_spread"]
            - row["no_intervention"]["low_execution_half_spread"]
        )
        < 1e-9
        and abs(
            row["no_routing"]["mean_reward"]
            - row["no_intervention"]["mean_reward"]
        )
        < 1e-9
        for row in rows
    )
    checks = {
        "activation_in_range": all(0.05 <= rate <= 0.35 for rate in activation_rates),
        "oracle_primary_positive": primary["positive"] >= 7,
        "oracle_ci_excludes_zero": primary["ci_low"] > 0.0,
        "reward_guardrail": reward_guardrail,
        "volume_conserved": volume_equal,
        "no_routing_exact_no_intervention": no_routing_exact,
    }
    output = {
        "status": "pass" if all(checks.values()) else "fail",
        "backend": "abides",
        "seeds": seeds,
        "checks": checks,
        "primary_shuffled_minus_oracle": primary,
        "activation_rate_min": min(activation_rates),
        "activation_rate_max": max(activation_rates),
        "oracle_mean_reward": oracle_reward,
        "shuffled_mean_reward": shuffled_reward,
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(output, sort_keys=True))
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
