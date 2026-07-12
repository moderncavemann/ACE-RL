"""Observable learned gate for the ABIDES dealer-market routing contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FEATURE_NAMES = (
    "average_half_spread",
    "contract_severity",
    "spread_dispersion",
    "spread_entropy",
    "modal_undercut_gap",
    "eligible_fraction",
    "public_volatility",
    "absolute_lagged_return",
    "flow_concentration",
    "mean_absolute_inventory",
    "severity_x_calm",
)


def _normalized_entropy(actions: np.ndarray, n_actions: int = 7) -> float:
    counts = np.bincount(actions.astype(int), minlength=n_actions).astype(float)
    probabilities = counts[counts > 0] / max(float(np.sum(counts)), 1.0)
    if len(probabilities) <= 1:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(n_actions))


def _flow_concentration(shares: np.ndarray) -> float:
    total = float(np.sum(shares))
    if total <= 0.0:
        return 0.0
    normalized = shares / total
    hhi = float(np.sum(np.square(normalized)))
    uniform = 1.0 / len(normalized)
    return float(np.clip((hhi - uniform) / (1.0 - uniform), 0.0, 1.0))


def abides_gate_features(
    state: Mapping[str, Any], actions: Sequence[int]
) -> np.ndarray:
    action_array = np.asarray(actions, dtype=int)
    if action_array.shape != (10,):
        raise ValueError("gate requires ten current quote actions")
    half_spreads = action_array.astype(float) + 1.0
    average = float(np.mean(half_spreads))
    severity = float(np.clip((average - 4.0) / 2.0, 0.0, 1.0))
    values, counts = np.unique(half_spreads, return_counts=True)
    modal_candidates = values[counts == np.max(counts)]
    modal = float(np.max(modal_candidates))
    minimum = float(np.min(half_spreads))
    eligible = half_spreads <= modal - 1.0

    info = state["info"]
    observations = np.asarray(state["observations"], dtype=float)
    shares = np.asarray(info.get("flow_shares", [0.1] * 10), dtype=float)
    public_volatility = float(info.get("public_volatility", 0.0)) / 4.0
    lagged_return = float(np.mean(observations[:, 5]))
    inventories = observations[:, 0]
    calm = float(np.clip(1.0 - public_volatility, 0.0, 1.0))
    return np.asarray(
        [
            average / 7.0,
            severity,
            float(np.std(half_spreads)) / 7.0,
            _normalized_entropy(action_array),
            max(0.0, modal - minimum) / 6.0,
            float(np.mean(eligible)),
            public_volatility,
            abs(lagged_return),
            _flow_concentration(shares),
            float(np.mean(np.abs(inventories))),
            severity * calm,
        ],
        dtype=float,
    )


def abides_gate_target(post_info: Mapping[str, Any]) -> bool:
    return bool(
        post_info.get("last_step_risk") == "low"
        and float(post_info.get("routing_severity", 0.0)) > 0.0
        and int(post_info.get("eligible_agent_count", 0)) > 0
    )


@dataclass(frozen=True)
class ABIDESLearnedGateModel:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    probability_threshold: float

    def __post_init__(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("ABIDES gate feature schema mismatch")
        size = len(FEATURE_NAMES)
        if not all(
            len(values) == size
            for values in (
                self.feature_mean,
                self.feature_scale,
                self.coefficients,
            )
        ):
            raise ValueError("ABIDES gate parameter dimension mismatch")
        if any(value <= 0.0 for value in self.feature_scale):
            raise ValueError("ABIDES gate feature scales must be positive")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("ABIDES gate threshold outside [0, 1]")

    def probability(self, features: Sequence[float]) -> float:
        values = np.asarray(features, dtype=float)
        if values.shape != (len(FEATURE_NAMES),):
            raise ValueError("ABIDES gate feature vector has wrong shape")
        standardized = (
            values - np.asarray(self.feature_mean)
        ) / np.asarray(self.feature_scale)
        logit = float(
            np.dot(standardized, np.asarray(self.coefficients)) + self.intercept
        )
        if logit >= 0.0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exp_logit = float(np.exp(logit))
        return float(exp_logit / (1.0 + exp_logit))

    def decide(self, state: Mapping[str, Any], actions: Sequence[int]) -> dict[str, Any]:
        features = abides_gate_features(state, actions)
        probability = self.probability(features)
        return {
            "active": probability >= self.probability_threshold,
            "probability": probability,
            "features": features.tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ABIDESLearnedGateModel":
        return cls(
            feature_names=tuple(values["feature_names"]),
            feature_mean=tuple(float(x) for x in values["feature_mean"]),
            feature_scale=tuple(float(x) for x in values["feature_scale"]),
            coefficients=tuple(float(x) for x in values["coefficients"]),
            intercept=float(values["intercept"]),
            probability_threshold=float(values["probability_threshold"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ABIDESLearnedGateModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
