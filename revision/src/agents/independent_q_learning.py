"""Independent tabular Q-learning agents for toy observation work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

State = Tuple[int, ...]


@dataclass
class QLearningConfig:
    n_agents: int = 10
    n_actions: int = 7
    learning_rate: float = 0.08
    gamma: float = 0.96
    epsilon_start: float = 0.35
    epsilon_end: float = 0.03
    epsilon_decay_steps: int = 80_000
    seed: int = 42


class IndependentQLearners:
    """One tabular Q table per agent."""

    def __init__(self, config: QLearningConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.q_tables: list[Dict[State, np.ndarray]] = [
            {} for _ in range(config.n_agents)
        ]
        self.total_steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, agent_id: int, state: State, explore: bool = True) -> int:
        q = self._values(agent_id, state)
        if explore and self.rng.random() < self.epsilon():
            return int(self.rng.integers(0, self.config.n_actions))
        max_q = np.max(q)
        candidates = np.flatnonzero(np.isclose(q, max_q))
        return int(self.rng.choice(candidates))

    def update(self, agent_id: int, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        q = self._values(agent_id, state)
        next_q = self._values(agent_id, next_state)
        target = reward if done else reward + self.config.gamma * float(np.max(next_q))
        q[action] += self.config.learning_rate * (target - q[action])

    def step_clock(self) -> None:
        self.total_steps += 1

    def _values(self, agent_id: int, state: State) -> np.ndarray:
        table = self.q_tables[agent_id]
        if state not in table:
            table[state] = np.zeros(self.config.n_actions, dtype=float)
        return table[state]
