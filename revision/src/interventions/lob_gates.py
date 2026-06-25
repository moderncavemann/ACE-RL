"""Gate rules for the local matching-engine LOB pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence

import numpy as np

from src.envs.local_lob_market_env import LocalLOBConfig, RiskState

LOBGateMode = Literal["none", "always", "threshold", "random", "risk_aware"]


@dataclass
class LOBGateConfig:
    threshold_spread: float = 4.0
    random_rate: float = 0.20
    risk_threshold: float = 0.35
    low_comp_spread: float = 1.5
    low_coll_spread: float = 5.0
    high_comp_spread: float = 2.5
    high_coll_spread: float = 6.5
    expected_volume: float = 2.5
    w_excess: float = 0.65
    w_entropy: float = 0.15
    w_flow: float = 0.10
    w_volume_suppression: float = 0.10
    w_recent_intervention: float = 0.10
    seed: int = 123


class LOBInterventionGate:
    def __init__(self, config: LocalLOBConfig, gate_config: LOBGateConfig | None = None, mode: LOBGateMode = "risk_aware"):
        self.config = config
        self.gate_config = gate_config or LOBGateConfig()
        self.mode = mode
        self.rng = np.random.default_rng(self.gate_config.seed)
        self.recent_intervention = 0.0

    def reset(self) -> None:
        self.recent_intervention = 0.0

    def decide(
        self,
        risk_state: RiskState,
        agent_spreads: Sequence[float],
        flow_shares: Sequence[float] | None = None,
        recent_volume: float = 0.0,
    ) -> Dict[str, float | bool]:
        avg_spread = float(np.mean(agent_spreads))
        excess = self._risk_adjusted_excess(risk_state, avg_spread)
        entropy = self._spread_entropy(agent_spreads)
        flow_concentration = self._flow_concentration(flow_shares)
        volume_suppression = float(np.clip(1.0 - recent_volume / max(self.gate_config.expected_volume, 1e-8), 0.0, 1.0))

        if self.mode == "none":
            score, active = 0.0, False
        elif self.mode == "always":
            score, active = 1.0, True
        elif self.mode == "threshold":
            score = avg_spread - self.gate_config.threshold_spread
            active = avg_spread > self.gate_config.threshold_spread
        elif self.mode == "random":
            score = self.gate_config.random_rate
            active = bool(self.rng.random() < self.gate_config.random_rate)
        else:
            score = (
                self.gate_config.w_excess * excess
                + self.gate_config.w_entropy * entropy
                + self.gate_config.w_flow * flow_concentration
                + self.gate_config.w_volume_suppression * volume_suppression
                - self.gate_config.w_recent_intervention * self.recent_intervention
            )
            active = score > self.gate_config.risk_threshold

        self.recent_intervention = 0.9 * self.recent_intervention + 0.1 * float(active)
        return {
            "active": bool(active),
            "score": float(score),
            "excess": float(excess),
            "spread_entropy": float(entropy),
            "flow_concentration": float(flow_concentration),
            "volume_suppression": float(volume_suppression),
            "avg_spread": float(avg_spread),
        }

    def _risk_adjusted_excess(self, risk_state: RiskState, avg_spread: float) -> float:
        if risk_state == "high":
            comp = self.gate_config.high_comp_spread
            coll = self.gate_config.high_coll_spread
        else:
            comp = self.gate_config.low_comp_spread
            coll = self.gate_config.low_coll_spread
        return float((avg_spread - comp) / max(coll - comp, 1e-8))

    def _spread_entropy(self, agent_spreads: Sequence[float]) -> float:
        levels = np.array(self.config.half_spread_levels, dtype=float)
        arr = np.array(agent_spreads, dtype=float)
        nearest = np.argmin(np.abs(arr[:, None] - levels[None, :]), axis=1)
        counts = np.bincount(nearest, minlength=len(levels)).astype(float)
        probs = counts[counts > 0.0] / max(float(np.sum(counts)), 1e-8)
        if len(probs) <= 1 or len(levels) <= 1:
            return 0.0
        entropy = -float(np.sum(probs * np.log(probs)))
        return float(np.clip(entropy / np.log(len(levels)), 0.0, 1.0))

    def _flow_concentration(self, flow_shares: Sequence[float] | None) -> float:
        if flow_shares is None:
            return 0.0
        shares = np.array(flow_shares, dtype=float)
        total = float(np.sum(shares))
        if total <= 0.0 or len(shares) <= 1:
            return 0.0
        shares = shares / total
        hhi = float(np.sum(shares ** 2))
        uniform_hhi = 1.0 / len(shares)
        return float(np.clip((hhi - uniform_hhi) / max(1.0 - uniform_hhi, 1e-8), 0.0, 1.0))
