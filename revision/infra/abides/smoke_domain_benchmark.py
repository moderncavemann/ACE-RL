#!/usr/bin/env python3
"""Deterministic non-formal smoke for domain benchmark routing primitives."""

from __future__ import annotations

import numpy as np

from ace_abides.economic_env import ABIDESDealerMarketConfig, ABIDESDealerMarketEnv


ACTIONS = [0, 1, 2, 3, 4, 5, 6, 3, 4, 5]
MECHANISM = {
    "kind": "brero_threshold_buybox",
    "order_statistic_k": 1,
    "margin_ticks": 0,
    "routable_fraction": 0.5,
}


def run_once(risk_state: str, enabled: bool = True) -> dict:
    config = ABIDESDealerMarketConfig(
        episode_steps=1,
        fixed_risk_state=risk_state,
        seed=880001,
    )
    env = ABIDESDealerMarketEnv(config)
    try:
        env.reset(seed=880001)
        return env.step(
            ACTIONS,
            gate_active=True,
            routing_enabled=enabled,
            routing_strength=1.0,
            routing_mechanism=MECHANISM,
        )
    finally:
        env.close()


def routing_audit(state: dict) -> dict:
    info = state["info"]
    return {
        "routing_threshold": info["routing_threshold"],
        "eligible_agent_indices": info["eligible_agent_indices"],
        "routing_weights": info["routing_weights"],
        "routing_total_variation": info["routing_total_variation"],
        "planned_flow_hhi": info["planned_flow_hhi"],
        "active_routing": info["active_routing"],
        "mean_volume": info["realized_flow_quantity"],
    }


def main() -> None:
    low = routing_audit(run_once("low"))
    high = routing_audit(run_once("high"))
    repeated = routing_audit(run_once("low"))
    off = routing_audit(run_once("low", enabled=False))

    if low != high or low != repeated:
        raise RuntimeError(
            "public-quote routing changed with latent risk or repetition"
        )
    if low["routing_threshold"] != 1.0:
        raise RuntimeError("unexpected threshold")
    if low["eligible_agent_indices"] != [0]:
        raise RuntimeError("unexpected eligible set")
    if not np.isclose(sum(low["routing_weights"]), 1.0):
        raise RuntimeError("routing weights do not sum to one")
    if any(value < 0.0 for value in low["routing_weights"]):
        raise RuntimeError("negative routing weight")
    if not np.isclose(low["mean_volume"], 20.0):
        raise RuntimeError("customer flow was not conserved")
    if not low["active_routing"] or low["routing_total_variation"] <= 0.0:
        raise RuntimeError("eligible-only routing did not activate")
    if off["routing_weights"] != [0.1] * 10:
        raise RuntimeError("mechanism-off routing is not exactly uniform")
    if off["active_routing"] or off["routing_total_variation"] != 0.0:
        raise RuntimeError("mechanism-off routing remained active")
    if not np.isclose(off["mean_volume"], 20.0):
        raise RuntimeError("mechanism-off customer flow was not conserved")
    print("ABIDES domain benchmark routing smoke passed")


if __name__ == "__main__":
    main()
