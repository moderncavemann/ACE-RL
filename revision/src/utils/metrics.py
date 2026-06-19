"""Metrics for toy LOB observation and intervention checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.envs.toy_lob_pricing_env import ToyLOBConfig
from src.utils.benchmarks import benchmark_for_risk, demand_at_spread


def spread_alignment(spreads: Sequence[float], spread_levels: Sequence[float]) -> float:
    """1 means all agents choose the same spread, 0 means full grid range."""
    arr = np.array(spreads, dtype=float)
    max_range = max(spread_levels) - min(spread_levels)
    if max_range <= 0:
        return 1.0
    return float(np.clip(1.0 - (np.max(arr) - np.min(arr)) / max_range, 0.0, 1.0))


def risk_adjusted_collusion_gap(config: ToyLOBConfig, risk_state: str, avg_spread: float) -> float:
    bench = benchmark_for_risk(config, risk_state)  # type: ignore[arg-type]
    denom = max(bench["s_coll"] - bench["s_comp"], 1e-8)
    return float((avg_spread - bench["s_comp"]) / denom)


def demand_suppression(config: ToyLOBConfig, risk_state: str, total_demand: float) -> float:
    bench = benchmark_for_risk(config, risk_state)  # type: ignore[arg-type]
    q_comp = demand_at_spread(config, bench["s_comp"])
    return float((q_comp - total_demand) / max(q_comp, 1e-8))


def enrich_step(config: ToyLOBConfig, info: Dict[str, object]) -> Dict[str, object]:
    risk_state = str(info["risk_state"])
    avg_spread = float(info["avg_spread"])
    total_demand = float(info["total_demand"])
    spreads = np.array(info["spreads"], dtype=float)
    enriched = dict(info)
    enriched["alignment"] = spread_alignment(spreads, config.spread_levels)
    enriched["collusion_gap"] = risk_adjusted_collusion_gap(config, risk_state, avg_spread)
    enriched["demand_suppression"] = demand_suppression(config, risk_state, total_demand)
    return enriched


def summarize_records(records: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = list(records)
    if not rows:
        return {}
    fields = [
        "avg_spread",
        "collusion_gap",
        "alignment",
        "demand_suppression",
        "total_demand",
        "mean_reward",
        "gate_active",
        "routing_eta_boost",
    ]
    summary: Dict[str, float] = {"n_steps": float(len(rows))}
    for field in fields:
        vals = [float(row[field]) for row in rows if field in row]
        if vals:
            summary[f"{field}_mean"] = float(np.mean(vals))
            summary[f"{field}_std"] = float(np.std(vals))
    return summary


def summarize_by_risk(records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in records:
        buckets[str(row["risk_state"])].append(row)
    return {risk: summarize_records(rows) for risk, rows in sorted(buckets.items())}


def recovery_summary(probe_records: List[Dict[str, object]], pre_avg_spread: float) -> Dict[str, float]:
    if not probe_records:
        return {}
    avg_spreads = np.array([float(row["avg_spread"]) for row in probe_records])
    deviator_rewards = np.array([float(row["deviator_reward"]) for row in probe_records])
    other_spreads = np.array([float(row["other_avg_spread"]) for row in probe_records])
    recovery_level = pre_avg_spread - 0.25
    recovered = np.where(avg_spreads >= recovery_level)[0]
    return {
        "probe_steps": float(len(probe_records)),
        "pre_probe_avg_spread": float(pre_avg_spread),
        "min_market_avg_spread": float(np.min(avg_spreads)),
        "final_market_avg_spread": float(avg_spreads[-1]),
        "mean_other_avg_spread": float(np.mean(other_spreads)),
        "mean_deviator_reward": float(np.mean(deviator_rewards)),
        "recovery_step": float(recovered[0]) if len(recovered) else -1.0,
    }
