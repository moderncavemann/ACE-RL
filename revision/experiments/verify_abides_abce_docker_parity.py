#!/usr/bin/env python3
"""Verify that ABCE preserves parent economics for unchanged mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_mappo_adaptive_v3_response import market_config  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)


CORE_INFO_FIELDS = (
    "path_digest",
    "completed_steps",
    "fundamental",
    "last_step_risk",
    "avg_half_spread",
    "mean_execution_half_spread",
    "mean_reward",
    "flow_shares",
    "mean_spread_revenue",
    "mean_adverse_selection_component",
    "mean_inventory_penalty",
)


def digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parity_rollout(
    config: Mapping[str, Any], image: str, routing_enabled: bool
) -> list[dict[str, Any]]:
    rows = []
    with ABIDESDockerDealerEnv(
        market_config(config), ABIDESDockerConfig(image=image)
    ) as env:
        env.reset(seed=695123)
        for step in range(5):
            state = env.step(
                [
                    (step + dealer) % len(config["market"]["half_spread_levels"])
                    for dealer in range(int(config["market"]["n_agents"]))
                ],
                routing_enabled=routing_enabled,
                routing_strength=0.63 if routing_enabled else 0.0,
                routing_mechanism={"kind": "soft_rank"},
            )
            rows.append(
                {
                    "observations": state["observations"],
                    "rewards": state["rewards"],
                    "agents": state["agents"],
                    "info": {
                        field: state["info"][field] for field in CORE_INFO_FIELDS
                    },
                }
            )
    return rows


def verify_parity(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for label, enabled in (("no_routing", False), ("soft_rank", True)):
        parent = parity_rollout(config, str(config["parent_docker_image"]), enabled)
        derived = parity_rollout(config, str(config["docker_image"]), enabled)
        if parent != derived:
            differing_steps = [
                step
                for step, (left, right) in enumerate(zip(parent, derived))
                if left != right
            ]
            raise RuntimeError(
                f"ABCE Docker economic parity failed for {label}: {differing_steps}"
            )
        result[label] = {
            "status": "pass",
            "steps": len(parent),
            "economic_digest": digest(parent),
        }
    return {
        "status": "pass",
        "development_path_seed": 695123,
        "checks": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="revision/configs/abides_abce_primary_followup_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    print(json.dumps(verify_parity(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
