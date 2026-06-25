"""Gate rules for toy routing interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence

import numpy as np

from src.envs.toy_lob_pricing_env import ToyLOBConfig
from src.utils.benchmarks import benchmark_for_risk
from src.utils.metrics import demand_suppression, risk_adjusted_collusion_gap, spread_alignment

GateMode = Literal[
    "none",
    "always",
    "threshold",
    "random",
    "oracle",
    "risk_aware",
    "spread_only",
    "no_risk_adjustment",
    "no_alignment",
    "no_demand",
]


@dataclass
class GateConfig:
    threshold_spread: float = 3.5
    random_rate: float = 0.30
    risk_threshold: float = 0.85
    oracle_collusion_gap_tau: float = 0.30
    oracle_alignment_tau: float = 0.0
    w_excess: float = 0.65
    w_alignment: float = 0.20
    w_demand: float = 0.15
    w_recent_intervention: float = 0.10
    seed: int = 123


class InterventionGate:
    def __init__(self, config: ToyLOBConfig, gate_config: GateConfig | None = None, mode: GateMode = "risk_aware"):
        self.config = config
        self.gate_config = gate_config or GateConfig()
        self.mode = mode
        self.rng = np.random.default_rng(self.gate_config.seed)
        self.recent_intervention = 0.0

    def reset(self) -> None:
        self.recent_intervention = 0.0

    def decide(self, risk_state: str, spreads: Sequence[float], total_demand: float | None = None) -> Dict[str, float | bool]:
        avg_spread = float(np.mean(spreads))
        alignment = spread_alignment(spreads, self.config.spread_levels)
        if total_demand is None:
            total_demand = float(self.config.q0 * np.exp(-self.config.demand_alpha * avg_spread))
        ds = demand_suppression(self.config, risk_state, total_demand)
        excess = risk_adjusted_collusion_gap(self.config, risk_state, avg_spread)

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
        elif self.mode == "oracle":
            # Diagnostic upper bound: uses the true risk-adjusted excess spread.
            # This is not deployable evidence; it isolates whether routing can
            # work when intervention timing is correct.
            score = excess
            active = (
                excess > self.gate_config.oracle_collusion_gap_tau
                and alignment >= self.gate_config.oracle_alignment_tau
            )
        elif self.mode == "spread_only":
            score = excess
            active = score > self.gate_config.risk_threshold
        elif self.mode == "no_risk_adjustment":
            low_bench = benchmark_for_risk(self.config, "low")
            denom = max(float(low_bench["s_coll"] - low_bench["s_comp"]), 1e-8)
            unadjusted_excess = (avg_spread - float(low_bench["s_comp"])) / denom
            score = self._risk_aware_score(unadjusted_excess, alignment, ds)
            active = score > self.gate_config.risk_threshold
        else:
            score = self._risk_aware_score(excess, alignment, ds)
            active = score > self.gate_config.risk_threshold

        self.recent_intervention = 0.9 * self.recent_intervention + 0.1 * float(active)
        bench = benchmark_for_risk(self.config, risk_state)  # type: ignore[arg-type]
        return {
            "active": bool(active),
            "score": float(score),
            "excess": float(excess),
            "alignment": float(alignment),
            "demand_suppression": float(ds),
            "s_comp": float(bench["s_comp"]),
            "s_coll": float(bench["s_coll"]),
        }

    def _risk_aware_score(self, excess: float, alignment: float, ds: float) -> float:
        cfg = self.gate_config
        if self.mode == "no_alignment":
            return cfg.w_excess * excess + cfg.w_demand * ds - cfg.w_recent_intervention * self.recent_intervention
        if self.mode == "no_demand":
            return cfg.w_excess * excess + cfg.w_alignment * alignment - cfg.w_recent_intervention * self.recent_intervention
        return (
            cfg.w_excess * excess
            + cfg.w_alignment * alignment
            + cfg.w_demand * ds
            - cfg.w_recent_intervention * self.recent_intervention
        )
