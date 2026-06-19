"""Routing interventions for the toy LOB pricing game."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingInterventionConfig:
    rho: float = 0.75


def eta_boost_from_gate(gate_active: bool, config: RoutingInterventionConfig | None = None) -> float:
    cfg = config or RoutingInterventionConfig()
    return cfg.rho if gate_active else 0.0
