"""Auditable public-quote intervention rules for the ABIDES benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RoutingDecision:
    mechanism: str
    threshold: float | None
    eligible_indices: tuple[int, ...]
    routing_weights: tuple[float, ...]
    routing_total_variation: float
    flow_hhi: float
    active_routing: bool
    parameters: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetMetrics:
    active_routing_rate: float
    mean_routing_total_variation: float
    mean_flow_hhi: float
    record_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporaryCommitmentDecision:
    mechanism: str
    active: bool
    takeup: bool
    reference_half_spread: float | None
    dealer_index: int | None
    frozen_half_spread: float | None
    protected_fraction: float
    remaining_periods_including_current: int
    routing_weights: tuple[float, ...]
    routing_total_variation: float
    parameters: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporaryCommitmentController:
    """Public-quote state machine for the Banerjee-style ABIDES adaptation."""

    def __init__(
        self,
        n_agents: int,
        reference_window: int = 10,
        qualification_steps: int = 3,
        protected_fraction: float = 0.5,
        duration: int = 5,
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if reference_window <= 0:
            raise ValueError("reference_window must be positive")
        if qualification_steps <= 0:
            raise ValueError("qualification_steps must be positive")
        if not 0.0 <= protected_fraction <= 1.0:
            raise ValueError("protected_fraction must be in [0, 1]")
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.n_agents = int(n_agents)
        self.reference_window = int(reference_window)
        self.qualification_steps = int(qualification_steps)
        self.protected_fraction = float(protected_fraction)
        self.duration = int(duration)
        self._history: list[float] = []
        self._dealer_index: int | None = None
        self._frozen_half_spread: float | None = None
        self._remaining = 0

    def reset(self) -> None:
        self._history.clear()
        self._dealer_index = None
        self._frozen_half_spread = None
        self._remaining = 0

    def decide(
        self,
        half_spreads: Sequence[float],
        tie_uniform: float,
        enabled: bool = True,
    ) -> TemporaryCommitmentDecision:
        spreads = _spread_array(half_spreads)
        if spreads.size != self.n_agents:
            raise ValueError("half_spreads size differs from n_agents")
        tie_value = float(tie_uniform)
        if not 0.0 <= tie_value < 1.0:
            raise ValueError("tie_uniform must be in [0, 1)")

        reference = (
            float(mean(self._history[-self.reference_window :]))
            if len(self._history) >= self.reference_window
            else None
        )
        takeup = False
        if enabled and self._remaining <= 0 and reference is not None:
            qualifying = np.flatnonzero(
                spreads <= reference - float(self.qualification_steps)
            )
            if qualifying.size:
                minimum = float(np.min(spreads[qualifying]))
                lowest = qualifying[spreads[qualifying] == minimum]
                slot = min(int(tie_value * len(lowest)), len(lowest) - 1)
                self._dealer_index = int(lowest[slot])
                self._frozen_half_spread = float(spreads[self._dealer_index])
                self._remaining = self.duration
                takeup = True

        active = bool(enabled and self._remaining > 0)
        remaining = int(self._remaining if active else 0)
        if active:
            assert self._dealer_index is not None
            assert self._frozen_half_spread is not None
            weights = eligible_routing_weights(
                self.n_agents,
                [self._dealer_index],
                self.protected_fraction,
            )
            dealer_index = self._dealer_index
            frozen_half_spread = self._frozen_half_spread
        else:
            weights = uniform_routing_weights(self.n_agents)
            dealer_index = None
            frozen_half_spread = None

        decision = TemporaryCommitmentDecision(
            mechanism="banerjee_temporary_commitment",
            active=active,
            takeup=takeup,
            reference_half_spread=reference,
            dealer_index=dealer_index,
            frozen_half_spread=frozen_half_spread,
            protected_fraction=self.protected_fraction,
            remaining_periods_including_current=remaining,
            routing_weights=tuple(float(value) for value in weights),
            routing_total_variation=routing_total_variation(weights),
            parameters={
                "reference_window": self.reference_window,
                "qualification_steps": self.qualification_steps,
                "protected_fraction": self.protected_fraction,
                "duration": self.duration,
            },
        )
        self._history.append(float(np.mean(spreads)))
        if active:
            self._remaining -= 1
            if self._remaining == 0:
                self._dealer_index = None
                self._frozen_half_spread = None
        return decision


def _spread_array(half_spreads: Sequence[float]) -> np.ndarray:
    values = np.asarray(half_spreads, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("half_spreads must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("half_spreads must contain positive finite values")
    return values


def _probability_array(weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("routing weights must be a nonempty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("routing weights must be finite and nonnegative")
    if not np.isclose(float(np.sum(values)), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("routing weights must sum to one")
    return values


def routing_total_variation(weights: Sequence[float]) -> float:
    values = _probability_array(weights)
    uniform = 1.0 / values.size
    return float(0.5 * np.sum(np.abs(values - uniform)))


def flow_hhi(weights: Sequence[float]) -> float:
    values = _probability_array(weights)
    return float(np.sum(np.square(values)))


def uniform_routing_weights(n_agents: int) -> np.ndarray:
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    return np.full(n_agents, 1.0 / n_agents, dtype=float)


def brero_threshold_eligibility(
    half_spreads: Sequence[float],
    order_statistic_k: int,
    margin_ticks: int,
    maximum_half_spread: float = 7.0,
) -> tuple[float, tuple[int, ...]]:
    """Return the threshold and all quote ties at or below it."""

    spreads = _spread_array(half_spreads)
    if not 1 <= int(order_statistic_k) <= spreads.size:
        raise ValueError("order_statistic_k must be in 1..N")
    if int(margin_ticks) not in (0, 1):
        raise ValueError("margin_ticks must be 0 or 1")
    threshold = min(
        float(maximum_half_spread),
        float(
            np.partition(spreads, int(order_statistic_k) - 1)[
                int(order_statistic_k) - 1
            ]
        )
        + int(margin_ticks),
    )
    eligible = tuple(int(index) for index in np.flatnonzero(spreads <= threshold))
    if not eligible:
        minimum = float(np.min(spreads))
        eligible = tuple(int(index) for index in np.flatnonzero(spreads == minimum))
        threshold = minimum
    return threshold, eligible


def eligible_routing_weights(
    n_agents: int,
    eligible_indices: Iterable[int],
    routable_fraction: float,
    enabled: bool = True,
) -> np.ndarray:
    """Mix uniform base allocation with uniform eligible-only allocation."""

    base = uniform_routing_weights(n_agents)
    if not enabled:
        return base
    fraction = float(routable_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("routable_fraction must be in [0, 1]")
    eligible = tuple(sorted(set(int(index) for index in eligible_indices)))
    if not eligible or any(index < 0 or index >= n_agents for index in eligible):
        raise ValueError("eligible_indices must be a nonempty subset of agents")
    target = np.zeros(n_agents, dtype=float)
    target[list(eligible)] = 1.0 / len(eligible)
    weights = (1.0 - fraction) * base + fraction * target
    _probability_array(weights)
    return weights


def brero_threshold_decision(
    half_spreads: Sequence[float],
    order_statistic_k: int,
    margin_ticks: int,
    routable_fraction: float,
    enabled: bool = True,
    active_epsilon: float = 1e-12,
) -> RoutingDecision:
    spreads = _spread_array(half_spreads)
    threshold, eligible = brero_threshold_eligibility(
        spreads, order_statistic_k, margin_ticks
    )
    weights = eligible_routing_weights(
        spreads.size, eligible, routable_fraction, enabled=enabled
    )
    tv = routing_total_variation(weights)
    return RoutingDecision(
        mechanism="brero_threshold_buybox",
        threshold=threshold,
        eligible_indices=eligible,
        routing_weights=tuple(float(value) for value in weights),
        routing_total_variation=tv,
        flow_hhi=flow_hhi(weights),
        active_routing=bool(tv > active_epsilon),
        parameters={
            "order_statistic_k": int(order_statistic_k),
            "margin_ticks": int(margin_ticks),
            "routable_fraction": float(routable_fraction),
        },
    )


def soft_rank_routing_weights(
    half_spreads: Sequence[float],
    signal_strength: float,
    kappa: float,
    routable_fraction: float,
    enabled: bool = True,
) -> np.ndarray:
    spreads = _spread_array(half_spreads)
    if kappa < 0.0:
        raise ValueError("kappa must be nonnegative")
    if not 0.0 <= routable_fraction <= 1.0:
        raise ValueError("routable_fraction must be in [0, 1]")
    strength = float(np.clip(signal_strength, 0.0, 1.0)) if enabled else 0.0
    centered = spreads - float(np.min(spreads))
    target = np.exp(-float(kappa) * centered)
    target /= float(np.sum(target))
    alpha = float(routable_fraction) * strength
    weights = (1.0 - alpha) * uniform_routing_weights(spreads.size) + alpha * target
    _probability_array(weights)
    return weights


def spread_only_proposal_strength(
    lagged_severity: float,
    severity_threshold: float,
    signal_scale: float,
) -> float:
    severity = float(np.clip(lagged_severity, 0.0, 1.0))
    threshold = float(severity_threshold)
    scale = float(signal_scale)
    if not 0.0 <= threshold < 1.0:
        raise ValueError("severity_threshold must be in [0, 1)")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("signal_scale must be in [0, 1]")
    excess = max(0.0, severity - threshold) / (1.0 - threshold)
    return float(scale * np.clip(excess, 0.0, 1.0))


def pace_signal_strengths(
    proposed_strengths: Sequence[float],
    target_rate: float,
    burst_capacity: float = 2.0,
) -> list[float]:
    if not 0.0 < target_rate <= 1.0:
        raise ValueError("target_rate must be in (0, 1]")
    if burst_capacity < 1.0:
        raise ValueError("burst_capacity must be at least one")
    tokens = 0.0
    accepted: list[float] = []
    for proposal in proposed_strengths:
        strength = float(np.clip(proposal, 0.0, 1.0))
        tokens = min(float(burst_capacity), tokens + float(target_rate))
        if strength <= 0.0 or tokens + 1e-12 < 1.0:
            accepted.append(0.0)
            continue
        tokens -= 1.0
        accepted.append(strength)
    return accepted


def summarize_budget(
    routing_weights: Sequence[Sequence[float]], active_epsilon: float = 1e-12
) -> BudgetMetrics:
    weights = [_probability_array(row) for row in routing_weights]
    if not weights:
        raise ValueError("at least one routing record is required")
    tv = np.asarray([routing_total_variation(row) for row in weights], dtype=float)
    return BudgetMetrics(
        active_routing_rate=float(np.mean(tv > active_epsilon)),
        mean_routing_total_variation=float(np.mean(tv)),
        mean_flow_hhi=float(np.mean([flow_hhi(row) for row in weights])),
        record_count=len(weights),
    )


def budget_match_status(
    candidate: BudgetMetrics,
    reference: BudgetMetrics,
    maximum_active_rate_difference: float,
    minimum_mean_tv_ratio: float,
    maximum_mean_tv_ratio: float,
) -> tuple[str, float | None]:
    if reference.mean_routing_total_variation <= 0.0:
        ratio = None
        matched = candidate.mean_routing_total_variation <= 1e-12
    else:
        ratio = (
            candidate.mean_routing_total_variation
            / reference.mean_routing_total_variation
        )
        matched = minimum_mean_tv_ratio <= ratio <= maximum_mean_tv_ratio
    matched = matched and (
        abs(candidate.active_routing_rate - reference.active_routing_rate)
        <= maximum_active_rate_difference + 1e-12
    )
    return ("matched" if matched else "unmatched"), ratio


def budget_distance(
    candidate: BudgetMetrics,
    reference: BudgetMetrics,
    maximum_active_rate_difference: float,
) -> float:
    active_scale = max(float(maximum_active_rate_difference), 1e-12)
    active_distance = (
        abs(candidate.active_routing_rate - reference.active_routing_rate)
        / active_scale
    )
    tv_scale = max(0.1 * reference.mean_routing_total_variation, 1e-12)
    tv_distance = (
        abs(
            candidate.mean_routing_total_variation
            - reference.mean_routing_total_variation
        )
        / tv_scale
    )
    return float(active_distance + tv_distance)


def reject_efficacy_fields(
    records: Sequence[Mapping[str, Any]], forbidden_fields: Iterable[str]
) -> None:
    forbidden = set(str(field) for field in forbidden_fields)
    for index, record in enumerate(records):
        overlap = forbidden.intersection(record)
        if overlap:
            raise ValueError(
                f"calibration record {index} contains forbidden efficacy fields: "
                f"{sorted(overlap)}"
            )


def _episode_groups(
    records: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for record in records:
        key = str(record["episode_id"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    for values in groups.values():
        values.sort(key=lambda row: int(row["step"]))
    return [groups[key] for key in order]


def calibrate_domain_baselines(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Select candidates using routing budgets only, never efficacy outcomes."""

    if not records:
        raise ValueError("passive calibration records are empty")
    reject_efficacy_fields(records, config["forbidden_calibration_fields"])
    kappa = float(config["routing_target_kappa"])
    fraction = float(config["routing_routable_fraction"])
    epsilon = float(config["active_routing_tv_epsilon"])
    tolerances = config["budget_tolerances"]

    reference_weights = [
        soft_rank_routing_weights(
            record["half_spreads"],
            float(record["risk_signal_strength"]),
            kappa,
            fraction,
        )
        for record in records
    ]
    reference = summarize_budget(reference_weights, epsilon)

    brero_results = []
    brero_config = config["brero_threshold_buybox"]
    for order_statistic_k in brero_config["order_statistic_k_candidates"]:
        for margin_ticks in brero_config["margin_tick_candidates"]:
            decisions = [
                brero_threshold_decision(
                    record["half_spreads"],
                    int(order_statistic_k),
                    int(margin_ticks),
                    fraction,
                    active_epsilon=epsilon,
                )
                for record in records
            ]
            metrics = summarize_budget(
                [decision.routing_weights for decision in decisions], epsilon
            )
            status, ratio = budget_match_status(
                metrics,
                reference,
                float(tolerances["maximum_active_rate_difference"]),
                float(tolerances["minimum_mean_tv_ratio"]),
                float(tolerances["maximum_mean_tv_ratio"]),
            )
            brero_results.append(
                {
                    "parameters": {
                        "order_statistic_k": int(order_statistic_k),
                        "margin_ticks": int(margin_ticks),
                    },
                    "metrics": metrics.to_dict(),
                    "mean_tv_ratio": ratio,
                    "budget_match_status": status,
                    "budget_distance": budget_distance(
                        metrics,
                        reference,
                        float(tolerances["maximum_active_rate_difference"]),
                    ),
                }
            )
    brero_results.sort(
        key=lambda row: (
            float(row["budget_distance"]),
            int(row["parameters"]["order_statistic_k"]),
            int(row["parameters"]["margin_ticks"]),
        )
    )

    spread_results = []
    spread_config = config["spread_only_budget_matched"]
    episodes = _episode_groups(records)
    for target_rate in spread_config["activation_target_candidates"]:
        for threshold in spread_config["severity_threshold_candidates"]:
            for scale in spread_config["signal_scale_candidates"]:
                strengths_by_key: dict[tuple[str, int], float] = {}
                for episode in episodes:
                    proposals = [
                        spread_only_proposal_strength(
                            float(record["lagged_contract_severity"]),
                            float(threshold),
                            float(scale),
                        )
                        for record in episode
                    ]
                    strengths = pace_signal_strengths(proposals, float(target_rate))
                    for record, strength in zip(episode, strengths):
                        strengths_by_key[
                            (str(record["episode_id"]), int(record["step"]))
                        ] = strength
                weights = [
                    soft_rank_routing_weights(
                        record["half_spreads"],
                        strengths_by_key[
                            (str(record["episode_id"]), int(record["step"]))
                        ],
                        kappa,
                        fraction,
                    )
                    for record in records
                ]
                metrics = summarize_budget(weights, epsilon)
                status, ratio = budget_match_status(
                    metrics,
                    reference,
                    float(tolerances["maximum_active_rate_difference"]),
                    float(tolerances["minimum_mean_tv_ratio"]),
                    float(tolerances["maximum_mean_tv_ratio"]),
                )
                spread_results.append(
                    {
                        "parameters": {
                            "activation_target": float(target_rate),
                            "severity_threshold": float(threshold),
                            "signal_scale": float(scale),
                        },
                        "metrics": metrics.to_dict(),
                        "mean_tv_ratio": ratio,
                        "budget_match_status": status,
                        "budget_distance": budget_distance(
                            metrics,
                            reference,
                            float(tolerances["maximum_active_rate_difference"]),
                        ),
                    }
                )
    spread_results.sort(
        key=lambda row: (
            float(row["budget_distance"]),
            float(row["parameters"]["activation_target"]),
            float(row["parameters"]["severity_threshold"]),
            float(row["parameters"]["signal_scale"]),
        )
    )

    return {
        "reference_budget": reference.to_dict(),
        "brero_threshold_buybox_budget_matched": {
            "selected": brero_results[0],
            "candidates": brero_results,
        },
        "spread_only_budget_matched": {
            "selected": spread_results[0],
            "candidates": spread_results,
        },
        "record_count": len(records),
        "episode_count": len(episodes),
        "selection_uses_efficacy_outcomes": False,
    }


def calibrate_temporary_commitment(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Budget-match commitment intensity using passive public quotes only."""

    if not records:
        raise ValueError("passive calibration records are empty")
    reject_efficacy_fields(records, config["forbidden_calibration_fields"])
    epsilon = float(config["active_routing_tv_epsilon"])
    reference = summarize_budget(
        [
            soft_rank_routing_weights(
                record["half_spreads"],
                float(record["risk_signal_strength"]),
                float(config["routing_target_kappa"]),
                float(config["routing_routable_fraction"]),
            )
            for record in records
        ],
        epsilon,
    )
    tolerances = config["budget_tolerances"]
    mechanism = config["banerjee_temporary_commitment"]
    episodes = _episode_groups(records)
    candidates = []
    for protected_fraction in mechanism["protected_fraction_candidates"]:
        for duration in mechanism["duration_candidates"]:
            weights = []
            takeups = 0
            for episode in episodes:
                controller = TemporaryCommitmentController(
                    n_agents=len(episode[0]["half_spreads"]),
                    reference_window=int(mechanism["reference_window"]),
                    qualification_steps=int(mechanism["qualification_steps"]),
                    protected_fraction=float(protected_fraction),
                    duration=int(duration),
                )
                for record in episode:
                    decision = controller.decide(
                        record["half_spreads"],
                        tie_uniform=float(record.get("tie_uniform", 0.0)),
                    )
                    weights.append(decision.routing_weights)
                    takeups += int(decision.takeup)
            metrics = summarize_budget(weights, epsilon)
            status, ratio = budget_match_status(
                metrics,
                reference,
                float(tolerances["maximum_active_rate_difference"]),
                float(tolerances["minimum_mean_tv_ratio"]),
                float(tolerances["maximum_mean_tv_ratio"]),
            )
            candidates.append(
                {
                    "parameters": {
                        "reference_window": int(mechanism["reference_window"]),
                        "qualification_steps": int(
                            mechanism["qualification_steps"]
                        ),
                        "protected_fraction": float(protected_fraction),
                        "duration": int(duration),
                    },
                    "metrics": metrics.to_dict(),
                    "takeup_count": takeups,
                    "mean_tv_ratio": ratio,
                    "budget_match_status": status,
                    "budget_distance": budget_distance(
                        metrics,
                        reference,
                        float(tolerances["maximum_active_rate_difference"]),
                    ),
                }
            )
    candidates.sort(
        key=lambda row: (
            float(row["budget_distance"]),
            float(row["parameters"]["protected_fraction"]),
            int(row["parameters"]["duration"]),
        )
    )
    return {
        "reference_budget": reference.to_dict(),
        "literature_form": dict(mechanism["literature_form"]),
        "budget_matched": {
            "selected": candidates[0],
            "candidates": candidates,
        },
        "record_count": len(records),
        "episode_count": len(episodes),
        "selection_uses_efficacy_outcomes": False,
    }
