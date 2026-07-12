"""Regression tests for the state-contingent MAPPO response adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402


class InterventionAdapterTest(unittest.TestCase):
    def make_trainer(self, policy_mode: str) -> MAPPOTrainer:
        return MAPPOTrainer(MAPPOConfig(
            obs_dim=3,
            critic_obs_dim=4,
            n_actions=4,
            n_agents=3,
            policy_mode=policy_mode,
            hidden_dim=8,
            learning_rate=1e-2,
            entropy_coef=0.0,
            update_epochs=1,
            minibatch_size=32,
            seed=7,
        ))

    def test_adapter_preserves_frozen_base_outside_active_low_risk(self) -> None:
        base = self.make_trainer(policy_mode="multihead")
        adapter = self.make_trainer(policy_mode="intervention_adapter")
        adapter.initialize_intervention_adapter(base)
        obs = np.array([
            [0.0, 0.2, 0.0],
            [1.0, 0.4, 1.0],
            [0.0, 0.6, 1.0],
        ], dtype=np.float32)
        agent_ids = np.arange(3, dtype=np.int64)
        obs_t = torch.as_tensor(obs)
        agent_ids_t = torch.as_tensor(agent_ids)

        with torch.no_grad():
            base_logits = base.model.distribution(obs_t, agent_ids_t).logits.clone()
            initial_logits = adapter.model.distribution(obs_t, agent_ids_t).logits.clone()
        self.assertTrue(torch.equal(base_logits, initial_logits))
        frozen_before = [
            parameter.detach().clone()
            for parameter in list(adapter.model.actor_trunk.parameters())
            + list(adapter.model.actor_heads.parameters())
        ]

        active_low_obs = np.repeat(obs[2:3], 24, axis=0)
        active_low_agent_ids = np.tile(agent_ids, 8)
        target_actions = (active_low_agent_ids + 2) % 4
        adapter.pretrain_actor(
            active_low_obs,
            target_actions,
            active_low_agent_ids,
            epochs=4,
        )

        with torch.no_grad():
            trained_logits = adapter.model.distribution(obs_t, agent_ids_t).logits
        self.assertTrue(torch.equal(base_logits[:2], trained_logits[:2]))
        self.assertFalse(torch.equal(base_logits[2], trained_logits[2]))
        frozen_after = (
            list(adapter.model.actor_trunk.parameters())
            + list(adapter.model.actor_heads.parameters())
        )
        self.assertTrue(all(
            torch.equal(before, after.detach())
            for before, after in zip(frozen_before, frozen_after)
        ))

    def test_action_uniforms_make_stochastic_sampling_reproducible(self) -> None:
        trainer = self.make_trainer(policy_mode="multihead")
        obs = np.array([
            [0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0],
            [1.0, 0.6, 0.0],
        ], dtype=np.float32)
        critic_obs = np.zeros((3, 4), dtype=np.float32)
        uniforms = np.array([0.05, 0.50, 0.95], dtype=np.float32)
        first, _, _ = trainer.act(
            obs,
            critic_obs,
            action_uniforms=uniforms,
        )
        second, _, _ = trainer.act(
            obs,
            critic_obs,
            action_uniforms=uniforms,
        )
        np.testing.assert_array_equal(first, second)

    def test_relative_immediate_advantage_removes_common_market_shock(self) -> None:
        trainer = MAPPOTrainer(MAPPOConfig(
            obs_dim=3,
            critic_obs_dim=4,
            n_actions=4,
            n_agents=3,
            policy_mode="intervention_adapter",
            policy_advantage_mode="relative_immediate",
            hidden_dim=8,
            seed=9,
        ))
        rewards = np.array([
            [10.0, 11.0, 12.0],
            [-5.0, -5.0, -2.0],
        ])
        gae = np.full_like(rewards, 99.0)
        advantages = trainer._actor_advantages(rewards, gae)
        np.testing.assert_allclose(
            advantages,
            np.array([[-1.0, 0.0, 1.0], [-1.0, -1.0, 2.0]]),
        )
        np.testing.assert_allclose(np.mean(advantages, axis=1), 0.0)

    def test_signal_only_adapter_is_exactly_off_at_zero_signal(self) -> None:
        base = self.make_trainer(policy_mode="multihead")
        adapter = MAPPOTrainer(MAPPOConfig(
            obs_dim=3,
            critic_obs_dim=4,
            n_actions=4,
            n_agents=3,
            policy_mode="intervention_adapter",
            adapter_risk_feature_index=None,
            hidden_dim=8,
            seed=9,
        ))
        adapter.initialize_intervention_adapter(base)
        obs = torch.tensor([
            [1.0, 0.2, 0.0],
            [1.0, 0.4, 0.0],
            [1.0, 0.6, 0.0],
        ])
        agent_ids = torch.arange(3)
        base_logits = base.model.distribution(obs, agent_ids).logits
        adapter_logits = adapter.model.distribution(obs, agent_ids).logits
        torch.testing.assert_close(adapter_logits, base_logits)

        with torch.no_grad():
            adapter.model.intervention_adapter_heads[0].bias[0] = 2.0
        signaled = obs.clone()
        signaled[:, -1] = 1.0
        active_logits = adapter.model.distribution(signaled, agent_ids).logits
        self.assertFalse(torch.allclose(active_logits, base_logits))

    def test_adapter_freeze_contract_can_be_restored_after_construction(self) -> None:
        adapter = MAPPOTrainer(MAPPOConfig(
            obs_dim=3,
            critic_obs_dim=4,
            n_actions=4,
            n_agents=3,
            policy_mode="intervention_adapter",
            adapter_risk_feature_index=None,
            hidden_dim=8,
            seed=9,
        ))
        adapter.freeze_intervention_base()
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in list(adapter.model.actor_trunk.parameters())
            + list(adapter.model.actor_heads.parameters())
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in list(adapter.model.intervention_adapter_heads.parameters())
            + list(adapter.model.critic.parameters())
        ))


if __name__ == "__main__":
    unittest.main()
