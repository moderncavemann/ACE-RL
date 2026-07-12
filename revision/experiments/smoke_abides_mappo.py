#!/usr/bin/env python3
"""Three-episode host-MPS MAPPO smoke against real ABIDES steps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.abides_mappo import make_abides_trainer, rollout_episode  # noqa: E402
from src.envs.abides_docker_env import ABIDESDockerDealerEnv  # noqa: E402


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the frozen local MAPPO path")
    episode_steps = 10
    trainer = make_abides_trainer(
        seed=801,
        episode_steps=episode_steps,
        device="mps",
    )
    rows = []
    with ABIDESDockerDealerEnv({"episode_steps": episode_steps}) as env:
        for episode in range(3):
            row = rollout_episode(
                env,
                trainer,
                seed=810 + episode,
                episode_steps=episode_steps,
                train=True,
            )
            rows.append(row)
    print(
        json.dumps(
            {
                "status": "pass",
                "backend": "abides",
                "trainer_device": str(trainer.device),
                "episodes": len(rows),
                "finite_updates": True,
                "mean_half_spread": sum(row["mean_half_spread"] for row in rows)
                / len(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
