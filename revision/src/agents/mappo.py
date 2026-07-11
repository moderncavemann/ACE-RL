"""Minimal MAPPO-style trainer for local LOB pilot runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class MAPPOConfig:
    obs_dim: int
    critic_obs_dim: int
    n_actions: int
    n_agents: int = 1
    policy_mode: Literal[
        "shared",
        "multihead",
        "intervention_adapter",
        "independent",
    ] = "shared"
    policy_advantage_mode: Literal["gae", "relative_immediate"] = "gae"
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
        self.policy_mode = config.policy_mode
        self.n_agents = config.n_agents
        self.n_actions = config.n_actions
        if config.policy_mode == "independent":
            self.actors = nn.ModuleList([
                self._actor_network(config) for _ in range(config.n_agents)
            ])
        elif config.policy_mode in {"multihead", "intervention_adapter"}:
            self.actor_trunk = nn.Sequential(
                nn.Linear(config.obs_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Tanh(),
            )
            self.actor_heads = nn.ModuleList([
                nn.Linear(config.hidden_dim, config.n_actions)
                for _ in range(config.n_agents)
            ])
            if config.policy_mode == "intervention_adapter":
                self.intervention_adapter_heads = nn.ModuleList([
                    nn.Linear(config.obs_dim, config.n_actions)
                    for _ in range(config.n_agents)
                ])
                for head in self.intervention_adapter_heads:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
        else:
            self.actor = self._actor_network(config)
        self.critic = nn.Sequential(
            nn.Linear(config.critic_obs_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )

    @staticmethod
    def _actor_network(config: MAPPOConfig) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.n_actions),
        )

    def distribution(self, obs: torch.Tensor, agent_ids: torch.Tensor | None = None) -> Categorical:
        if self.policy_mode == "shared":
            return Categorical(logits=self.actor(obs))
        if agent_ids is None:
            raise ValueError("agent_ids are required for agent-specific actor policies")
        if self.policy_mode in {"multihead", "intervention_adapter"}:
            hidden = self.actor_trunk(obs)
            all_logits = torch.stack([head(hidden) for head in self.actor_heads], dim=1)
            if self.policy_mode == "intervention_adapter":
                adapter_logits = torch.stack([
                    head(obs) for head in self.intervention_adapter_heads
                ], dim=1)
                low_risk = (obs[:, 0] < 0.5).to(dtype=all_logits.dtype)
                announced_strength = torch.clamp(obs[:, -1], 0.0, 1.0)
                adapter_strength = low_risk * announced_strength
                all_logits = (
                    all_logits
                    + adapter_strength[:, None, None] * adapter_logits
                )
        else:
            all_logits = torch.stack([actor(obs) for actor in self.actors], dim=1)
        row_ids = torch.arange(obs.shape[0], device=obs.device)
        logits = all_logits[row_ids, agent_ids]
        return Categorical(logits=logits)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)


class MAPPOTrainer:
    """MAPPO-style actor variants with centralized critic observations.

    This is intentionally compact: it is for pilot plumbing and diagnostics, not
    a tuned research-grade MAPPO implementation.
    """

    def __init__(self, config: MAPPOConfig):
        if config.policy_mode != "shared" and config.n_agents < 2:
            raise ValueError(f"{config.policy_mode} policy_mode requires n_agents >= 2")
        self.config = config
        torch.manual_seed(config.seed)
        self.np_rng = np.random.default_rng(config.seed)
        self.action_call_count = 0
        self.device = torch.device(config.device)
        self.model = ActorCritic(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def initialize_intervention_adapter(self, base: "MAPPOTrainer") -> None:
        """Copy and freeze a trained multihead actor under a zero-logit adapter."""
        if self.config.policy_mode != "intervention_adapter":
            raise ValueError("target trainer must use intervention_adapter policy_mode")
        if base.config.policy_mode != "multihead":
            raise ValueError("base trainer must use multihead policy_mode")
        dimensions = (
            "obs_dim",
            "critic_obs_dim",
            "n_actions",
            "n_agents",
            "hidden_dim",
        )
        if any(getattr(self.config, key) != getattr(base.config, key) for key in dimensions):
            raise ValueError("base and adapter trainer dimensions must match")

        self.model.actor_trunk.load_state_dict(base.model.actor_trunk.state_dict())
        self.model.actor_heads.load_state_dict(base.model.actor_heads.state_dict())
        self.model.critic.load_state_dict(base.model.critic.state_dict())
        for parameter in self.model.actor_trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.actor_heads.parameters():
            parameter.requires_grad_(False)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self.config.learning_rate)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        deterministic: bool = False,
        action_uniforms: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        critic_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        agent_ids = None
        if self.config.policy_mode != "shared":
            if obs_t.shape[0] != self.config.n_agents:
                raise ValueError(
                    f"Expected {self.config.n_agents} agent observations, got {obs_t.shape[0]}"
                )
            agent_ids = torch.arange(self.config.n_agents, dtype=torch.long, device=self.device)
        dist = self.model.distribution(obs_t, agent_ids)
        if deterministic:
            if action_uniforms is not None:
                raise ValueError("action_uniforms cannot be used with deterministic actions")
            actions_t = torch.argmax(dist.logits, dim=-1)
        elif action_uniforms is not None:
            uniforms_t = torch.as_tensor(
                action_uniforms,
                dtype=dist.probs.dtype,
                device=self.device,
            )
            if uniforms_t.shape != dist.probs.shape[:-1]:
                raise ValueError(
                    f"Expected action_uniforms shape {dist.probs.shape[:-1]}, "
                    f"got {uniforms_t.shape}"
                )
            cumulative = torch.cumsum(dist.probs, dim=-1)
            actions_t = torch.sum(
                uniforms_t[..., None] > cumulative,
                dim=-1,
            ).clamp(max=self.config.n_actions - 1)
        else:
            torch.manual_seed(self.config.seed + 104_729 * self.action_call_count)
            self.action_call_count += 1
            actions_t = dist.sample()
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
        gae_advantages, returns = self._gae(rewards, values)
        advantages = self._actor_advantages(rewards, gae_advantages)
        flat_obs = obs.reshape((-1, self.config.obs_dim))
        flat_critic = critic_obs.reshape((-1, self.config.critic_obs_dim))
        flat_actions = actions.reshape(-1)
        flat_old_logp = old_logp.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_agent_ids = np.tile(np.arange(obs.shape[1], dtype=np.int64), obs.shape[0])

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
            self.np_rng.shuffle(idx)
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                loss, p_loss, v_loss, entropy = self._update_minibatch(
                    flat_obs[batch],
                    flat_critic[batch],
                    flat_actions[batch],
                    flat_old_logp[batch],
                    flat_adv[batch],
                    flat_returns[batch],
                    flat_agent_ids[batch],
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
            "gae_advantage_mean": (
                float(np.mean(gae_advantages)) if gae_advantages.size else 0.0
            ),
            "return_mean": float(np.mean(returns)) if returns.size else 0.0,
        }

    def pretrain_actor(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        agent_ids: np.ndarray,
        epochs: int,
        minibatch_size: int | None = None,
    ) -> Dict[str, float]:
        """Behavior-clone a risk-conditioned quote prior before RL updates."""
        if epochs <= 0 or len(actions) == 0:
            return {"loss": 0.0, "accuracy": 0.0, "epochs": 0.0}
        n = len(actions)
        batch_size = min(minibatch_size or self.config.minibatch_size, n)
        idx = np.arange(n)
        losses: list[float] = []
        accuracies: list[float] = []
        for _ in range(epochs):
            self.np_rng.shuffle(idx)
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                obs_t = torch.as_tensor(obs[batch], dtype=torch.float32, device=self.device)
                actions_t = torch.as_tensor(actions[batch], dtype=torch.long, device=self.device)
                agent_ids_t = torch.as_tensor(agent_ids[batch], dtype=torch.long, device=self.device)
                dist = self.model.distribution(obs_t, agent_ids_t)
                loss = -torch.mean(dist.log_prob(actions_t))
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
                accuracies.append(float(torch.mean((torch.argmax(dist.logits, dim=-1) == actions_t).float()).cpu()))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            agent_ids_t = torch.as_tensor(agent_ids, dtype=torch.long, device=self.device)
            final_dist = self.model.distribution(obs_t, agent_ids_t)
            final_loss = -torch.mean(final_dist.log_prob(actions_t))
            final_accuracy = torch.mean((torch.argmax(final_dist.logits, dim=-1) == actions_t).float())
            final_entropy = torch.mean(final_dist.entropy())
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "epochs": float(epochs),
            "final_loss": float(final_loss.cpu()),
            "final_accuracy": float(final_accuracy.cpu()),
            "final_entropy": float(final_entropy.cpu()),
        }

    def _update_minibatch(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        actions: np.ndarray,
        old_logp: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        agent_ids: np.ndarray,
    ) -> tuple[float, float, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        critic_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        agent_ids_t = torch.as_tensor(agent_ids, dtype=torch.long, device=self.device)

        dist = self.model.distribution(obs_t, agent_ids_t)
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

    def _actor_advantages(
        self,
        rewards: np.ndarray,
        gae_advantages: np.ndarray,
    ) -> np.ndarray:
        if self.config.policy_advantage_mode == "relative_immediate":
            return rewards - np.mean(rewards, axis=1, keepdims=True)
        return gae_advantages
