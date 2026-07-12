#!/usr/bin/env python3
"""Fixed-policy economics check for the announced ABIDES soft-rank contract."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)


def run_fixed(
    env: ABIDESDockerDealerEnv,
    seed: int,
    actions: Sequence[int],
    strength: float,
    episode_steps: int,
) -> dict[str, float]:
    rewards = []
    agent_zero_rewards = []
    execution_spreads = []
    flows = []
    targeted = []
    env.reset(seed=seed)
    for _ in range(episode_steps):
        state = env.step(actions, routing_strength=strength)
        rewards.append(float(np.mean(state["rewards"])))
        agent_zero_rewards.append(float(state["rewards"][0]))
        execution_spreads.append(
            float(state["info"]["mean_execution_half_spread"])
        )
        flows.append(float(state["info"]["realized_flow_quantity"]))
        targeted.append(float(state["info"]["targeted_event_count"]))
    env.close_episode()
    return {
        "mean_reward": mean(rewards),
        "agent_zero_reward": mean(agent_zero_rewards),
        "execution_half_spread": mean(execution_spreads),
        "mean_flow": mean(flows),
        "targeted_events": sum(targeted),
    }


def run_seed(seed: int, episode_steps: int, image: str) -> dict[str, Any]:
    market = {
        "episode_steps": episode_steps,
        "fixed_risk_state": "low",
        "routing_allocation_mode": "soft_rank",
        "routing_target_kappa": 4.0,
    }
    row: dict[str, Any] = {"seed": seed}
    with ABIDESDockerDealerEnv(
        market,
        ABIDESDockerConfig(image=image),
    ) as env:
        for name, actions, strength in (
            ("wide_base", [5] * 10, 0.0),
            ("wide_routed", [5] * 10, 1.0),
            ("undercut_base", [3] + [5] * 9, 0.0),
            ("undercut_routed", [3] + [5] * 9, 1.0),
        ):
            row[name] = run_fixed(
                env,
                seed,
                actions,
                strength,
                episode_steps,
            )
    return row


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    symmetric_reward_deltas = [
        row["wide_routed"]["mean_reward"] - row["wide_base"]["mean_reward"]
        for row in rows
    ]
    undercutter_reward_deltas = [
        row["undercut_routed"]["agent_zero_reward"]
        - row["undercut_base"]["agent_zero_reward"]
        for row in rows
    ]
    execution_improvements = [
        row["undercut_base"]["execution_half_spread"]
        - row["undercut_routed"]["execution_half_spread"]
        for row in rows
    ]
    flows = [
        method["mean_flow"]
        for row in rows
        for method in row.values()
        if isinstance(method, dict) and "mean_flow" in method
    ]
    checks = {
        "volume_conserved": all(abs(value - 20.0) < 1e-9 for value in flows),
        "symmetric_market_not_subsidized": abs(mean(symmetric_reward_deltas)) <= 0.10,
        "undercutter_reward_gain_positive": mean(undercutter_reward_deltas) > 0.0,
        "undercutter_gain_positive_all_seeds": all(
            value > 0.0 for value in undercutter_reward_deltas
        ),
        "customer_execution_improves": mean(execution_improvements) > 0.0,
        "customer_execution_improves_all_seeds": all(
            value > 0.0 for value in execution_improvements
        ),
    }
    return {
        "symmetric_reward_delta": {
            "mean": mean(symmetric_reward_deltas),
            "values": symmetric_reward_deltas,
        },
        "undercutter_reward_delta": {
            "mean": mean(undercutter_reward_deltas),
            "values": undercutter_reward_deltas,
        },
        "execution_spread_improvement": {
            "mean": mean(execution_improvements),
            "values": execution_improvements,
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="470100-470109")
    parser.add_argument("--episode-steps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--image", default="ace-rl-abides-ace:adaptive-v3")
    parser.add_argument(
        "--output",
        default=(
            "revision/experiments/outputs/"
            "abides_announced_contract_economics_dev_v1/summary.json"
        ),
    )
    args = parser.parse_args()
    start, end = (int(value) for value in args.seeds.split("-", 1))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_seed, seed, args.episode_steps, args.image): seed
            for seed in range(start, end + 1)
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                json.dumps({"seed": row["seed"], "status": "complete"}),
                flush=True,
            )
    rows.sort(key=lambda row: int(row["seed"]))
    result = {"rows": rows, **aggregate(rows)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checks")}, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
