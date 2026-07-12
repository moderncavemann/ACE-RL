#!/usr/bin/env python3
"""Verify that the archived ABIDES-Gym package can initialize and step."""

from __future__ import annotations

import json

import abides_gym  # noqa: F401
import gym


def main() -> None:
    environment_id = "markets-daily_investor-v0"
    env = gym.make(
        environment_id,
        background_config="rmsc04",
        mkt_close="09:35:00",
        timestep_duration="60s",
        first_interval="00:01:00",
        background_config_extra_kvargs={
            "num_noise_agents": 20,
            "num_value_agents": 5,
            "num_momentum_agents": 2,
        },
    )
    try:
        env.seed(11)
        observation = env.reset()
        next_observation, reward, done, info = env.step(0)
        payload = {
            "status": "pass",
            "backend": "abides_gym",
            "environment_id": environment_id,
            "observation_shape": list(observation.shape),
            "next_observation_shape": list(next_observation.shape),
            "reward": float(reward),
            "done": bool(done),
            "info_keys": sorted(str(key) for key in info),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
