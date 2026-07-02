"""Minimal MAPPO-style trainer for local LOB pilot runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class MAPPOConfig:
    obs_dim: int
    critic_obs_dim: int
    n_actions: int
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.96
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    update_epochs: int = 4
    minibatch_size: int = 512
    seed: int = 42
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, config: MAPPOConfig):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(config.critic_obs_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)


class MAPPOTrainer:
    """Shared actor with centralized critic observations.

    This is intentionally compact: it is for pilot plumbing and diagnostics, not
    a tuned research-grade MAPPO implementation.
    """

    def __init__(self, config: MAPPOConfig):
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.device = torch.device(config.device)
        self.model = ActorCritic(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        critic_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        dist = self.model.distribution(obs_t)
        actions_t = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        logp_t = dist.log_prob(actions_t)
        values_t = self.model.value(critic_t)
        return (
            actions_t.cpu().numpy().astype(int),
            logp_t.cpu().numpy().astype(float),
            values_t.cpu().numpy().astype(float),
        )

    def update_episode(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        actions: np.ndarray,
        old_logp: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
    ) -> Dict[str, float]:
        advantages, returns = self._gae(rewards, values)
        flat_obs = obs.reshape((-1, self.config.obs_dim))
        flat_critic = critic_obs.reshape((-1, self.config.critic_obs_dim))
        flat_actions = actions.reshape(-1)
        flat_old_logp = old_logp.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        adv_std = float(np.std(flat_adv))
        if adv_std > 1e-8:
            flat_adv = (flat_adv - float(np.mean(flat_adv))) / (adv_std + 1e-8)

        n = len(flat_actions)
        batch_size = min(self.config.minibatch_size, n)
        idx = np.arange(n)
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(self.config.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                loss, p_loss, v_loss, entropy = self._update_minibatch(
                    flat_obs[batch],
                    flat_critic[batch],
                    flat_actions[batch],
                    flat_old_logp[batch],
                    flat_adv[batch],
                    flat_returns[batch],
                )
                losses.append(loss)
                policy_losses.append(p_loss)
                value_losses.append(v_loss)
                entropies.append(entropy)

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "advantage_mean": float(np.mean(advantages)) if advantages.size else 0.0,
            "return_mean": float(np.mean(returns)) if returns.size else 0.0,
        }

    def _update_minibatch(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        actions: np.ndarray,
        old_logp: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> tuple[float, float, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        critic_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        dist = self.model.distribution(obs_t)
        logp = dist.log_prob(actions_t)
        ratio = torch.exp(logp - old_logp_t)
        clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
        policy_loss = -torch.mean(torch.minimum(ratio * adv_t, clipped * adv_t))
        values = self.model.value(critic_t)
        value_loss = torch.mean((returns_t - values) ** 2)
        entropy = torch.mean(dist.entropy())
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return (
            float(loss.detach().cpu()),
            float(policy_loss.detach().cpu()),
            float(value_loss.detach().cpu()),
            float(entropy.detach().cpu()),
        )

    def _gae(self, rewards: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=float)
        last_gae = np.zeros(rewards.shape[1], dtype=float)
        for t in range(rewards.shape[0] - 1, -1, -1):
            next_value = np.zeros(rewards.shape[1], dtype=float) if t == rewards.shape[0] - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns
