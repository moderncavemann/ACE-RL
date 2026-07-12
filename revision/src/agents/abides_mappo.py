"""MAPPO rollout helpers for the host-to-ABIDES Docker bridge."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.agents.mappo import MAPPOConfig, MAPPOTrainer
from src.envs.abides_docker_env import ABIDESDockerDealerEnv

GateDecision = Callable[[Mapping[str, Any], np.ndarray, int], bool]


def actor_observations(
    state: Mapping[str, Any], gate_signal: float = 0.0
) -> np.ndarray:
    base = np.asarray(state["observations"], dtype=np.float32)
    if base.shape != (10, 6):
        raise ValueError(f"expected ABIDES observations with shape (10, 6), got {base.shape}")
    signal = np.full((base.shape[0], 1), float(gate_signal), dtype=np.float32)
    return np.concatenate([base, signal], axis=1)


def centralized_critic_observations(
    state: Mapping[str, Any], observations: np.ndarray, episode_steps: int
) -> np.ndarray:
    base = observations[:, :-1]
    info = state["info"]
    global_features = np.concatenate(
        [
            np.mean(base, axis=0),
            np.std(base, axis=0),
            np.asarray(
                [
                    float(info.get("avg_half_spread", 1.0)) / 7.0,
                    float(info.get("public_volatility", 0.0)) / 4.0,
                    float(info.get("completed_steps", 0)) / max(episode_steps, 1),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    tiled = np.repeat(global_features[None, :], observations.shape[0], axis=0)
    return np.concatenate([observations, tiled], axis=1).astype(np.float32)


def make_abides_trainer(
    seed: int,
    episode_steps: int,
    device: str,
    entropy_coef: float = 0.02,
) -> MAPPOTrainer:
    dummy_state = {
        "observations": [[0.0] * 6 for _ in range(10)],
        "info": {
            "avg_half_spread": 1.0,
            "public_volatility": 0.0,
            "completed_steps": 0,
        },
    }
    obs = actor_observations(dummy_state)
    critic = centralized_critic_observations(dummy_state, obs, episode_steps)
    return MAPPOTrainer(
        MAPPOConfig(
            obs_dim=obs.shape[1],
            critic_obs_dim=critic.shape[1],
            n_actions=7,
            n_agents=10,
            policy_mode="multihead",
            policy_advantage_mode="relative_immediate",
            hidden_dim=64,
            learning_rate=1e-3,
            gamma=0.96,
            gae_lambda=0.95,
            clip_ratio=0.20,
            entropy_coef=entropy_coef,
            value_coef=0.25,
            max_grad_norm=0.50,
            update_epochs=4,
            minibatch_size=512,
            seed=seed,
            device=device,
        )
    )


def rollout_episode(
    env: ABIDESDockerDealerEnv,
    trainer: MAPPOTrainer,
    seed: int,
    episode_steps: int,
    train: bool,
    gate_schedule: Sequence[bool] | None = None,
    gate_decision: GateDecision | None = None,
    routing_enabled: bool = True,
    reward_scale: float = 10.0,
    action_namespace: int = 73_117,
    deterministic: bool = False,
) -> dict[str, Any]:
    if gate_schedule is not None and gate_decision is not None:
        raise ValueError("provide either gate_schedule or gate_decision, not both")
    if gate_schedule is not None and len(gate_schedule) != episode_steps:
        raise ValueError("gate schedule length must match episode_steps")

    state = env.reset(seed=seed)
    obs_rows = []
    critic_rows = []
    action_rows = []
    logp_rows = []
    value_rows = []
    reward_rows = []
    records = []
    for step in range(episode_steps):
        obs = actor_observations(state, gate_signal=0.0)
        critic = centralized_critic_observations(state, obs, episode_steps)
        action_uniforms = None
        if not deterministic:
            action_uniforms = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(step), int(action_namespace)])
            ).random(10)
        actions, logp, values = trainer.act(
            obs,
            critic,
            deterministic=deterministic,
            action_uniforms=action_uniforms,
        )
        if gate_schedule is not None:
            active = bool(gate_schedule[step])
        elif gate_decision is not None:
            active = bool(gate_decision(state, actions, step))
        else:
            active = False
        next_state = env.step(
            actions,
            gate_active=active,
            routing_enabled=routing_enabled,
        )
        rewards = np.asarray(next_state["rewards"], dtype=float) / reward_scale
        obs_rows.append(obs)
        critic_rows.append(critic)
        action_rows.append(actions.astype(np.int64))
        logp_rows.append(logp.astype(float))
        value_rows.append(values.astype(float))
        reward_rows.append(rewards)
        records.append(
            {
                "step": step,
                "actions": actions.tolist(),
                "gate_active": active,
                "pre_info": dict(state["info"]),
                "pre_observations": state["observations"],
                "info": dict(next_state["info"]),
                "mean_reward_unscaled": float(np.mean(next_state["rewards"])),
            }
        )
        state = next_state
    if not state["done"]:
        raise RuntimeError("ABIDES episode did not finish at configured horizon")
    env.close_episode()

    update = {}
    if train:
        update = trainer.update_episode(
            np.stack(obs_rows),
            np.stack(critic_rows),
            np.stack(action_rows),
            np.stack(logp_rows),
            np.stack(reward_rows),
            np.stack(value_rows),
        )
        if not all(np.isfinite(float(value)) for value in update.values()):
            raise FloatingPointError(f"non-finite MAPPO update: {update}")
    return {
        "update": update,
        "records": records,
        "mean_action": float(np.mean(action_rows)),
        "mean_half_spread": float(np.mean(np.asarray(action_rows) + 1)),
        "mean_reward": float(
            np.mean([record["mean_reward_unscaled"] for record in records])
        ),
        "activation_rate": float(
            np.mean([record["gate_active"] for record in records])
        ),
    }
