"""LOB-inspired spread-choice pricing environment.

This is intentionally simpler than a matching-engine LOB. Agents choose one of
seven spread levels. Lower spreads attract more routed order flow; higher
spreads earn more margin but suppress demand. A persistent risk state changes
the competitive spread benchmark, so high spread is not automatically collusion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Literal, Tuple

import numpy as np

RiskMode = Literal["low", "high", "mixed"]
RiskState = Literal["low", "high"]


@dataclass(frozen=True)
class ToyLOBConfig:
    n_agents: int = 10
    spread_levels: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    q0: float = 100.0
    demand_alpha: float = 0.20
    allocation_eta: float = 0.80
    c_low: float = 0.50
    c_high: float = 2.50
    demand_noise_frac: float = 0.03
    risk_mode: RiskMode = "mixed"
    p_low_to_high: float = 0.05
    p_high_to_low: float = 0.10
    episode_length: int = 200
    seed: int = 42

    def with_risk_mode(self, risk_mode: RiskMode) -> "ToyLOBConfig":
        return replace(self, risk_mode=risk_mode)

    @property
    def n_actions(self) -> int:
        return len(self.spread_levels)

    def cost_for_risk(self, risk_state: RiskState) -> float:
        return self.c_low if risk_state == "low" else self.c_high


class ToyLOBPricingEnv:
    """Repeated spread-choice market-making game."""

    def __init__(self, config: ToyLOBConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.t = 0
        self.risk_state: RiskState = "low"
        self.last_spreads = np.full(config.n_agents, config.spread_levels[0], dtype=float)
        self.last_flows = np.zeros(config.n_agents, dtype=float)
        self.last_rewards = np.zeros(config.n_agents, dtype=float)
        self.recent_avg_spreads = []

    def reset(self, seed: int | None = None) -> Tuple[Dict[int, np.ndarray], Dict[str, object]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.risk_state = self._initial_risk_state()
        initial_spread = self.config.spread_levels[0]
        self.last_spreads = np.full(self.config.n_agents, initial_spread, dtype=float)
        self.last_flows = np.zeros(self.config.n_agents, dtype=float)
        self.last_rewards = np.zeros(self.config.n_agents, dtype=float)
        self.recent_avg_spreads = [float(initial_spread)]
        return self._observations(), self._info_stub()

    def step(
        self,
        actions: Dict[int, int] | Iterable[int],
        routing_eta_boost: float = 0.0,
        gate_active: bool = False,
        gate_score: float = 0.0,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, Dict[str, object]]:
        action_array = self._coerce_actions(actions)
        spreads = np.array([self.config.spread_levels[a] for a in action_array], dtype=float)
        avg_spread = float(np.mean(spreads))
        cost = self.config.cost_for_risk(self.risk_state)

        base_demand = self.config.q0 * np.exp(-self.config.demand_alpha * avg_spread)
        noise = self.rng.normal(0.0, self.config.demand_noise_frac * self.config.q0)
        total_demand = max(0.0, float(base_demand + noise))

        effective_eta = self.config.allocation_eta + max(0.0, routing_eta_boost)
        raw_weights = np.exp(-effective_eta * spreads)
        weights = raw_weights / np.sum(raw_weights)
        flows = total_demand * weights
        rewards_array = (spreads - cost) * flows
        rewards = {i: float(rewards_array[i]) for i in range(self.config.n_agents)}

        info = {
            "t": self.t,
            "risk_state": self.risk_state,
            "cost": cost,
            "spreads": spreads.copy(),
            "actions": action_array.copy(),
            "avg_spread": avg_spread,
            "min_spread": float(np.min(spreads)),
            "max_spread": float(np.max(spreads)),
            "total_demand": total_demand,
            "base_demand": float(base_demand),
            "flows": flows.copy(),
            "weights": weights.copy(),
            "rewards": rewards_array.copy(),
            "mean_reward": float(np.mean(rewards_array)),
            "effective_eta": float(effective_eta),
            "routing_eta_boost": float(routing_eta_boost),
            "gate_active": bool(gate_active),
            "gate_score": float(gate_score),
        }

        self.last_spreads = spreads
        self.last_flows = flows
        self.last_rewards = rewards_array
        self.recent_avg_spreads.append(avg_spread)
        self.recent_avg_spreads = self.recent_avg_spreads[-20:]
        self.t += 1
        done = self.t >= self.config.episode_length
        self._advance_risk_state()
        return self._observations(), rewards, done, False, info

    def state_for_agent(self, agent_id: int) -> Tuple[int, int, int, int, int]:
        """Tabular state with enough market detail to reveal undercutting.

        The first version only used risk, average spread, and own spread. With
        N=10, one deviator can lower the best spread while barely moving the
        average, making punishment/recovery dynamics invisible. The min-spread
        and range bins expose that deviation signal to other agents.
        """
        levels = np.array(self.config.spread_levels)
        avg = float(np.mean(self.last_spreads))
        min_spread = float(np.min(self.last_spreads))
        spread_range = float(np.max(self.last_spreads) - min_spread)
        avg_bin = int(np.argmin(np.abs(levels - avg)))
        own_bin = int(np.argmin(np.abs(levels - self.last_spreads[agent_id])))
        min_bin = int(np.argmin(np.abs(levels - min_spread)))
        range_bin = int(np.clip(round(spread_range / self._grid_step()), 0, self.config.n_actions - 1))
        risk_bin = 0 if self.risk_state == "low" else 1
        return risk_bin, avg_bin, own_bin, min_bin, range_bin

    def _grid_step(self) -> float:
        levels = np.array(self.config.spread_levels, dtype=float)
        if len(levels) < 2:
            return 1.0
        diffs = np.diff(np.sort(levels))
        return float(np.min(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0

    def _observations(self) -> Dict[int, np.ndarray]:
        risk_bin = 0.0 if self.risk_state == "low" else 1.0
        avg_spread = float(np.mean(self.last_spreads))
        recent_avg = float(np.mean(self.recent_avg_spreads)) if self.recent_avg_spreads else avg_spread
        obs = {}
        for i in range(self.config.n_agents):
            obs[i] = np.array(
                [
                    self.last_spreads[i],
                    avg_spread,
                    risk_bin,
                    self.last_flows[i],
                    self.last_rewards[i],
                    recent_avg,
                ],
                dtype=np.float32,
            )
        return obs

    def _initial_risk_state(self) -> RiskState:
        if self.config.risk_mode == "low":
            return "low"
        if self.config.risk_mode == "high":
            return "high"
        return "high" if self.rng.random() < 0.33 else "low"

    def _advance_risk_state(self) -> None:
        if self.config.risk_mode == "low":
            self.risk_state = "low"
            return
        if self.config.risk_mode == "high":
            self.risk_state = "high"
            return
        if self.risk_state == "low" and self.rng.random() < self.config.p_low_to_high:
            self.risk_state = "high"
        elif self.risk_state == "high" and self.rng.random() < self.config.p_high_to_low:
            self.risk_state = "low"

    def _coerce_actions(self, actions: Dict[int, int] | Iterable[int]) -> np.ndarray:
        if isinstance(actions, dict):
            arr = np.array([actions[i] for i in range(self.config.n_agents)], dtype=int)
        else:
            arr = np.array(list(actions), dtype=int)
        if arr.shape != (self.config.n_agents,):
            raise ValueError(f"Expected {self.config.n_agents} actions, got shape {arr.shape}")
        return np.clip(arr, 0, self.config.n_actions - 1)

    def _info_stub(self) -> Dict[str, object]:
        return {
            "risk_state": self.risk_state,
            "avg_spread": float(np.mean(self.last_spreads)),
            "spreads": self.last_spreads.copy(),
        }
