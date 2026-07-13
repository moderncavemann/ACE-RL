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

RISK_FEATURE_NAMES = (
    "public_volatility",
    "absolute_lagged_return",
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


def abides_lagged_actions(state: Mapping[str, Any]) -> np.ndarray:
    """Recover the previous quote actions available before the next decision."""
    observations = np.asarray(state["observations"], dtype=float)
    if observations.shape != (10, 6):
        raise ValueError("lagged ABIDES observations must have shape (10, 6)")
    actions = np.rint(observations[:, 1] * 6.0).astype(int)
    return np.clip(actions, 0, 6)


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
        standardized = (values - np.asarray(self.feature_mean)) / np.asarray(
            self.feature_scale
        )
        logit = float(
            np.dot(standardized, np.asarray(self.coefficients)) + self.intercept
        )
        if logit >= 0.0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exp_logit = float(np.exp(logit))
        return float(exp_logit / (1.0 + exp_logit))

    def decide(
        self, state: Mapping[str, Any], actions: Sequence[int]
    ) -> dict[str, Any]:
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


@dataclass(frozen=True)
class ABIDESTwoStageGateModel:
    """Separate exogenous risk inference from current routing opportunity."""

    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    probability_threshold: float

    def __post_init__(self) -> None:
        if self.feature_names != RISK_FEATURE_NAMES:
            raise ValueError("two-stage gate risk feature schema mismatch")
        size = len(RISK_FEATURE_NAMES)
        if not all(
            len(values) == size
            for values in (
                self.feature_mean,
                self.feature_scale,
                self.coefficients,
            )
        ):
            raise ValueError("two-stage gate parameter dimension mismatch")
        if any(value <= 0.0 for value in self.feature_scale):
            raise ValueError("two-stage gate feature scales must be positive")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("two-stage gate threshold outside [0, 1]")

    def probability(self, full_features: Sequence[float]) -> float:
        values_by_name = dict(zip(FEATURE_NAMES, full_features))
        values = np.asarray(
            [values_by_name[name] for name in self.feature_names], dtype=float
        )
        standardized = (values - np.asarray(self.feature_mean)) / np.asarray(
            self.feature_scale
        )
        logit = float(
            np.dot(standardized, np.asarray(self.coefficients)) + self.intercept
        )
        if logit >= 0.0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exp_logit = float(np.exp(logit))
        return float(exp_logit / (1.0 + exp_logit))

    def decide(
        self, state: Mapping[str, Any], actions: Sequence[int]
    ) -> dict[str, Any]:
        features = abides_gate_features(state, actions)
        values_by_name = dict(zip(FEATURE_NAMES, features))
        risk_probability = self.probability(features)
        opportunity = bool(
            float(values_by_name["contract_severity"]) > 0.0
            and float(values_by_name["eligible_fraction"]) > 0.0
        )
        return {
            "active": bool(
                risk_probability >= self.probability_threshold and opportunity
            ),
            "risk_probability": risk_probability,
            "opportunity": opportunity,
            "features": features.tolist(),
        }

    def announced_strength(
        self,
        state: Mapping[str, Any],
        include_spread_severity: bool = True,
    ) -> dict[str, Any]:
        """Compute a pre-action contract signal from lagged observable state."""
        features = abides_gate_features(state, abides_lagged_actions(state))
        values_by_name = dict(zip(FEATURE_NAMES, features))
        risk_probability = self.probability(features)
        severity = (
            float(values_by_name["contract_severity"])
            if include_spread_severity
            else 1.0
        )
        proposed = bool(
            risk_probability >= self.probability_threshold and severity > 0.0
        )
        strength = risk_probability * severity if proposed else 0.0
        return {
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "proposed": proposed,
            "risk_probability": risk_probability,
            "severity": severity,
            "features": features.tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ABIDESTwoStageGateModel":
        return cls(
            feature_names=tuple(values["feature_names"]),
            feature_mean=tuple(float(x) for x in values["feature_mean"]),
            feature_scale=tuple(float(x) for x in values["feature_scale"]),
            coefficients=tuple(float(x) for x in values["coefficients"]),
            intercept=float(values["intercept"]),
            probability_threshold=float(values["probability_threshold"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ABIDESTwoStageGateModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ABIDESPublicRiskProxyModel:
    """Single-feature, pre-action risk proxy used as a simple comparator."""

    feature_name: str
    feature_mean: float
    feature_scale: float
    coefficient: float
    intercept: float
    probability_threshold: float
    pacing_target_rate: float
    training_label: str = "simulator_low_risk"

    def __post_init__(self) -> None:
        if self.feature_name not in RISK_FEATURE_NAMES:
            raise ValueError("public risk proxy must use a public risk feature")
        if self.feature_scale <= 0.0:
            raise ValueError("public risk proxy feature scale must be positive")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("public risk proxy threshold outside [0, 1]")
        if not 0.0 < self.pacing_target_rate <= 1.0:
            raise ValueError("public risk proxy pacing rate outside (0, 1]")
        if self.training_label != "simulator_low_risk":
            raise ValueError("public risk proxy training label is not recognized")

    def probability(self, full_features: Sequence[float]) -> float:
        values_by_name = dict(zip(FEATURE_NAMES, full_features))
        standardized = (
            float(values_by_name[self.feature_name]) - self.feature_mean
        ) / self.feature_scale
        logit = self.coefficient * standardized + self.intercept
        if logit >= 0.0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exp_logit = float(np.exp(logit))
        return float(exp_logit / (1.0 + exp_logit))

    def announced_strength(self, state: Mapping[str, Any]) -> dict[str, Any]:
        features = abides_gate_features(state, abides_lagged_actions(state))
        probability = self.probability(features)
        proposed = probability >= self.probability_threshold
        return {
            "strength": probability if proposed else 0.0,
            "proposed": proposed,
            "probability": probability,
            "features": features.tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ABIDESPublicRiskProxyModel":
        return cls(
            feature_name=str(values["feature_name"]),
            feature_mean=float(values["feature_mean"]),
            feature_scale=float(values["feature_scale"]),
            coefficient=float(values["coefficient"]),
            intercept=float(values["intercept"]),
            probability_threshold=float(values["probability_threshold"]),
            pacing_target_rate=float(values["pacing_target_rate"]),
            training_label=str(values.get("training_label", "simulator_low_risk")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ABIDESPublicRiskProxyModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class ABIDESGateActivationController:
    """Enforce the preregistered per-episode intervention-rate ceiling."""

    episode_steps: int
    max_activation_rate: float = 0.35
    accepted: int = 0
    proposed: int = 0

    def __post_init__(self) -> None:
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        if not 0.0 < self.max_activation_rate <= 1.0:
            raise ValueError("max_activation_rate must be in (0, 1]")

    @property
    def budget(self) -> int:
        return int(np.floor(self.episode_steps * self.max_activation_rate + 1e-12))

    def decide(self, proposed: bool) -> bool:
        self.proposed += int(bool(proposed))
        if not proposed or self.accepted >= self.budget:
            return False
        self.accepted += 1
        return True

    @property
    def proposal_rate(self) -> float:
        return self.proposed / self.episode_steps

    @property
    def activation_rate(self) -> float:
        return self.accepted / self.episode_steps


@dataclass
class ABIDESPacedSignalController:
    """Pace continuous interventions without exhausting an episode budget early."""

    episode_steps: int
    target_rate: float = 0.35
    burst_capacity: float = 2.0
    tokens: float = 0.0
    steps: int = 0
    proposed: int = 0
    accepted: int = 0
    accepted_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        if not 0.0 < self.target_rate <= 1.0:
            raise ValueError("target_rate must be in (0, 1]")
        if self.burst_capacity < 1.0:
            raise ValueError("burst_capacity must be at least one")

    def decide(self, proposed_strength: float) -> float:
        if self.steps >= self.episode_steps:
            raise RuntimeError("signal controller exceeded the episode horizon")
        strength = float(np.clip(proposed_strength, 0.0, 1.0))
        self.steps += 1
        self.tokens = min(self.burst_capacity, self.tokens + self.target_rate)
        self.proposed += int(strength > 0.0)
        if strength <= 0.0 or self.tokens + 1e-12 < 1.0:
            return 0.0
        self.tokens -= 1.0
        self.accepted += 1
        self.accepted_strength += strength
        return strength

    @property
    def proposal_rate(self) -> float:
        return self.proposed / self.episode_steps

    @property
    def activation_rate(self) -> float:
        return self.accepted / self.episode_steps

    @property
    def mean_strength(self) -> float:
        if self.accepted == 0:
            return 0.0
        return self.accepted_strength / self.accepted
