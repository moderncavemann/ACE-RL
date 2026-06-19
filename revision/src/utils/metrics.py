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
    # Passive score for threshold calibration during no-intervention
    # observation. This intentionally omits recent-intervention history because
    # no intervention is being applied.
    enriched["passive_risk_score"] = (
        0.65 * float(enriched["collusion_gap"])
        + 0.20 * float(enriched["alignment"])
        + 0.15 * float(enriched["demand_suppression"])
    )
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
        "gate_score",
        "passive_risk_score",
        "routing_eta_boost",
    ]
    summary: Dict[str, float] = {"n_steps": float(len(rows))}
    for field in fields:
        vals = [float(row[field]) for row in rows if field in row]
        if vals:
            summary[f"{field}_mean"] = float(np.mean(vals))
            summary[f"{field}_std"] = float(np.std(vals))
    return summary


def gate_classification_metrics(records: Iterable[Dict[str, object]], tau: float = 0.3) -> Dict[str, float]:
    """Risk-adjusted gate diagnostics.

    False positive is a gate firing in high-risk, normal-spread windows.
    True positive is a gate firing in low-risk, collusive-spread windows.
    High risk is not automatically a negative class; high-risk windows with
    large risk-adjusted collusion gaps are excluded from the FPR denominator.
    """
    rows = list(records)
    high_normal = [
        row for row in rows
        if row.get("risk_state") == "high" and float(row.get("collusion_gap", 0.0)) < tau
    ]
    low_collusive = [
        row for row in rows
        if row.get("risk_state") == "low" and float(row.get("collusion_gap", 0.0)) > tau
    ]
    all_gate = [float(row.get("gate_active", 0.0)) for row in rows]
    out = {
        "gate_activation_rate": float(np.mean(all_gate)) if all_gate else 0.0,
        "fpr_high_normal_tau": float(tau),
        "fpr_high_normal_n": float(len(high_normal)),
        "tpr_low_collusive_tau": float(tau),
        "tpr_low_collusive_n": float(len(low_collusive)),
    }
    out["fpr_high_normal"] = (
        float(np.mean([float(row.get("gate_active", 0.0)) for row in high_normal]))
        if high_normal else 0.0
    )
    out["tpr_low_collusive"] = (
        float(np.mean([float(row.get("gate_active", 0.0)) for row in low_collusive]))
        if low_collusive else 0.0
    )
    return out


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


def paired_recovery_summary(
    forced_records: List[Dict[str, object]],
    control_records: List[Dict[str, object]],
    pre_avg_spread: float,
    actual_deviation: float,
) -> Dict[str, float]:
    forced = recovery_summary(forced_records, pre_avg_spread)
    control = recovery_summary(control_records, pre_avg_spread)
    if not forced_records or not control_records:
        return {"actual_deviation": float(actual_deviation)}
    n = min(len(forced_records), len(control_records))
    forced_avg = np.array([float(row["avg_spread"]) for row in forced_records[:n]])
    control_avg = np.array([float(row["avg_spread"]) for row in control_records[:n]])
    forced_other = np.array([float(row["other_avg_spread"]) for row in forced_records[:n]])
    control_other = np.array([float(row["other_avg_spread"]) for row in control_records[:n]])
    forced_reward = np.array([float(row["deviator_reward"]) for row in forced_records[:n]])
    control_reward = np.array([float(row["deviator_reward"]) for row in control_records[:n]])
    return {
        "actual_deviation": float(actual_deviation),
        "forced_recovery_step": forced.get("recovery_step", -1.0),
        "control_recovery_step": control.get("recovery_step", -1.0),
        "mean_market_spread_delta_forced_minus_control": float(np.mean(forced_avg - control_avg)),
        "mean_other_spread_delta_forced_minus_control": float(np.mean(forced_other - control_other)),
        "mean_deviator_reward_delta_forced_minus_control": float(np.mean(forced_reward - control_reward)),
        "forced_min_market_avg_spread": forced.get("min_market_avg_spread", 0.0),
        "control_min_market_avg_spread": control.get("min_market_avg_spread", 0.0),
        "forced_final_market_avg_spread": forced.get("final_market_avg_spread", 0.0),
        "control_final_market_avg_spread": control.get("final_market_avg_spread", 0.0),
    }
