#!/usr/bin/env python3
"""Host-to-container smoke for the persistent ABIDES training bridge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs.abides_docker_env import ABIDESDockerDealerEnv  # noqa: E402


def main() -> None:
    config = {"episode_steps": 3}
    with ABIDESDockerDealerEnv(config) as env:
        first = env.reset(seed=700)
        first_digest = first["info"]["path_digest"]
        state = env.step([0] * 10)
        assert state["info"]["realized_flow_quantity"] == 20.0
        assert len(state["rewards"]) == 10
        while not state["done"]:
            state = env.step([0] * 10)
        second = env.reset(seed=701)
        second_digest = second["info"]["path_digest"]
        assert first_digest != second_digest
        env.close_episode()
    print(
        json.dumps(
            {
                "status": "pass",
                "backend": "abides",
                "persistent_worker": True,
                "ten_agent_observations": True,
                "flow_per_step": 20.0,
                "different_seed_path_distinct": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
