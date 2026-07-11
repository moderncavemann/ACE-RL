"""Serializable learned gate for the state-contingent local LOB contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, cast

import numpy as np

from src.envs.local_lob_market_env import LocalLOBMarketEnv


FEATURE_NAMES = (
    "low_risk",
    "contract_severity",
    "spread_entropy",
    "flow_concentration",
    "volume_suppression",
    "low_risk_x_severity",
    "low_risk_x_volume_suppression",
)


def learned_gate_features(
    env: LocalLOBMarketEnv,
    expected_volume: float,
) -> np.ndarray:
    """Build pre-action features available to the platform."""
    low_risk = float(env.risk_state == "low")
    severity = env.current_routing_contract_severity()
    levels = np.asarray(env.config.half_spread_levels, dtype=float)
    spreads = np.asarray(env.last_agent_spreads, dtype=float)
    nearest = np.argmin(np.abs(spreads[:, None] - levels[None, :]), axis=1)
    counts = np.bincount(nearest, minlength=len(levels)).astype(float)
    probabilities = counts[counts > 0.0] / max(float(np.sum(counts)), 1e-8)
    if len(probabilities) <= 1 or len(levels) <= 1:
        entropy = 0.0
    else:
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        entropy /= float(np.log(len(levels)))

    flow_shares = np.asarray(env.last_flow_shares, dtype=float)
    flow_total = float(np.sum(flow_shares))
    if flow_total <= 0.0 or len(flow_shares) <= 1:
        concentration = 0.0
    else:
        normalized = flow_shares / flow_total
        hhi = float(np.sum(normalized ** 2))
        uniform_hhi = 1.0 / len(normalized)
        concentration = float(np.clip(
            (hhi - uniform_hhi) / max(1.0 - uniform_hhi, 1e-8),
            0.0,
            1.0,
        ))

    volume_suppression = float(np.clip(
        1.0 - env.last_total_volume / max(expected_volume, 1e-8),
        0.0,
        1.0,
    ))
    return np.asarray([
        low_risk,
        severity,
        entropy,
        concentration,
        volume_suppression,
        low_risk * severity,
        low_risk * volume_suppression,
    ], dtype=float)


def learned_gate_target(env: LocalLOBMarketEnv) -> bool:
    """Oracle target used only for passive gate training and diagnostics."""
    return bool(
        env.risk_state == "low"
        and env.current_routing_contract_severity() > 0.0
    )


@dataclass(frozen=True)
class LearnedLOBGateModel:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    probability_threshold: float
    expected_volume: float

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("learned gate feature schema does not match runtime schema")
        if not all(len(values) == size for values in (
            self.feature_mean,
            self.feature_scale,
            self.coefficients,
        )):
            raise ValueError("learned gate parameter dimensions do not match features")
        if any(scale <= 0.0 for scale in self.feature_scale):
            raise ValueError("learned gate feature scales must be positive")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("learned gate probability threshold must be in [0, 1]")

    def probability(self, features: Sequence[float]) -> float:
        values = np.asarray(features, dtype=float)
        if values.shape != (len(self.feature_names),):
            raise ValueError(
                f"expected {len(self.feature_names)} gate features, got {values.shape}"
            )
        mean = np.asarray(self.feature_mean, dtype=float)
        scale = np.asarray(self.feature_scale, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        logit = float(np.dot((values - mean) / scale, coefficients) + self.intercept)
        if logit >= 0.0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exp_logit = float(np.exp(logit))
        return float(exp_logit / (1.0 + exp_logit))

    def decide(self, features: Sequence[float]) -> Dict[str, float | bool]:
        probability = self.probability(features)
        return {
            "probability": probability,
            "active": bool(probability >= self.probability_threshold),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "probability_threshold": self.probability_threshold,
            "expected_volume": self.expected_volume,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LearnedLOBGateModel":
        raw_block = data.get("model", data)
        if not isinstance(raw_block, Mapping):
            raise ValueError("learned gate artifact must contain a model object")
        block = cast(Mapping[str, Any], raw_block)
        return cls(
            feature_names=tuple(str(value) for value in block["feature_names"]),
            feature_mean=tuple(float(value) for value in block["feature_mean"]),
            feature_scale=tuple(float(value) for value in block["feature_scale"]),
            coefficients=tuple(float(value) for value in block["coefficients"]),
            intercept=float(block["intercept"]),
            probability_threshold=float(block["probability_threshold"]),
            expected_volume=float(block["expected_volume"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedLOBGateModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
