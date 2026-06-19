"""Analytic benchmarks for the toy LOB pricing game."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Literal

import numpy as np

from src.envs.toy_lob_pricing_env import ToyLOBConfig

RiskState = Literal["low", "high"]


def competitive_spread(config: ToyLOBConfig, cost: float) -> float:
    """Approximate one-shot competitive spread benchmark."""
    denom = (config.demand_alpha / config.n_agents) + (
        config.allocation_eta * (1.0 - 1.0 / config.n_agents)
    )
    return cost + 1.0 / denom


def collusive_spread(config: ToyLOBConfig, cost: float) -> float:
    """Approximate symmetric joint-profit spread benchmark."""
    return cost + 1.0 / config.demand_alpha


def demand_at_spread(config: ToyLOBConfig, spread: float) -> float:
    return float(config.q0 * np.exp(-config.demand_alpha * spread))


def nearest_grid_level(config: ToyLOBConfig, spread: float) -> float:
    levels = np.array(config.spread_levels)
    return float(levels[np.argmin(np.abs(levels - spread))])


def benchmark_for_risk(config: ToyLOBConfig, risk_state: RiskState) -> Dict[str, float]:
    cost = config.cost_for_risk(risk_state)
    s_comp = competitive_spread(config, cost)
    s_coll = collusive_spread(config, cost)
    return {
        "risk_state": risk_state,
        "cost": cost,
        "s_comp": s_comp,
        "s_comp_grid": nearest_grid_level(config, s_comp),
        "s_coll": s_coll,
        "s_coll_grid": nearest_grid_level(config, s_coll),
        "q_comp": demand_at_spread(config, s_comp),
        "q_coll": demand_at_spread(config, s_coll),
    }


def benchmark_table(config: ToyLOBConfig) -> Dict[str, object]:
    return {
        "config": asdict(config),
        "low": benchmark_for_risk(config, "low"),
        "high": benchmark_for_risk(config, "high"),
        "symmetric_grid_payoffs": symmetric_grid_payoffs(config),
    }


def symmetric_grid_payoffs(config: ToyLOBConfig) -> Dict[str, list[dict[str, float]]]:
    """Per-agent payoff if all agents choose the same grid spread."""
    out: Dict[str, list[dict[str, float]]] = {}
    for risk_state in ("low", "high"):
        cost = config.cost_for_risk(risk_state)
        rows = []
        for spread in config.spread_levels:
            demand = demand_at_spread(config, spread)
            reward = (spread - cost) * demand / config.n_agents
            rows.append({
                "spread": float(spread),
                "total_demand": float(demand),
                "per_agent_reward": float(reward),
            })
        out[risk_state] = rows
    return out
