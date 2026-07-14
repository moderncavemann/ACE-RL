from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from src.agents.official_on_policy_mappo import (  # noqa: E402
    OFFICIAL_COMMIT,
    OfficialMAPPO,
    OfficialMAPPOConfig,
    verify_official_source,
)


SOURCE = Path.home() / ".cache" / "ace-rl" / "on-policy" / OFFICIAL_COMMIT


def config(centralized: bool = True) -> OfficialMAPPOConfig:
    return OfficialMAPPOConfig(
        source_root=str(SOURCE),
        seed=123,
        episode_steps=4,
        centralized_value=centralized,
        hidden_size=16,
        data_chunk_length=2,
        ppo_epoch=1,
    )


def test_official_source_and_payload_roundtrip_are_pinned() -> None:
    source = verify_official_source(SOURCE)
    assert source["commit"] == OFFICIAL_COMMIT
    learner = OfficialMAPPO(config())
    clone = learner.clone()
    assert learner.source_commit == OFFICIAL_COMMIT
    assert learner.model_digest() == clone.model_digest()


def test_official_recurrent_buffer_update_uses_explicit_action_uniforms() -> None:
    learner = OfficialMAPPO(config())
    obs = np.zeros((10, 7), dtype=np.float32)
    critic = np.zeros((10, 22), dtype=np.float32)
    buffer = learner.new_buffer(obs, critic)
    rnn_actor, rnn_critic, masks = learner.initial_recurrent_states()
    uniforms = np.linspace(0.01, 0.99, 10, dtype=np.float32)
    first_actions = None
    for step in range(4):
        actions, logp, values, rnn_actor, rnn_critic = learner.act(
            obs,
            critic,
            rnn_actor,
            rnn_critic,
            masks,
            uniforms,
        )
        if first_actions is None:
            first_actions = actions.copy()
        terminal_masks = (
            np.zeros_like(masks) if step == 3 else np.ones_like(masks)
        )
        learner.insert(
            buffer,
            obs,
            critic,
            rnn_actor,
            rnn_critic,
            actions,
            logp,
            values,
            np.linspace(-0.1, 0.1, 10, dtype=np.float32),
            terminal_masks,
        )
        masks = terminal_masks
    assert first_actions is not None
    assert first_actions.shape == (10, 1)
    metrics = learner.update(buffer, update_seed=991)
    assert metrics
    assert all(np.isfinite(value) for value in metrics.values())


def test_official_ippo_uses_local_critic_observations() -> None:
    learner = OfficialMAPPO(config(centralized=False))
    assert learner.config.critic_obs_dim == 7
    obs = np.zeros((10, 7), dtype=np.float32)
    buffer = learner.new_buffer(obs, obs)
    assert buffer.share_obs.shape[-1] == 7
