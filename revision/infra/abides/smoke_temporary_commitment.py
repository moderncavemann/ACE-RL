#!/usr/bin/env python3
"""Deterministic integrity smoke for temporary protected buyer commitments."""

from __future__ import annotations

import numpy as np

from ace_abides.economic_env import ABIDESDealerMarketConfig, ABIDESDealerMarketEnv


MECHANISM = {
    "kind": "banerjee_temporary_commitment",
    "reference_window": 10,
    "qualification_steps": 3,
    "protected_fraction": 0.5,
    "duration": 5,
}


def run_once(risk_state: str) -> list[dict]:
    config = ABIDESDealerMarketConfig(
        episode_steps=16,
        fixed_risk_state=risk_state,
        seed=881001,
    )
    env = ABIDESDealerMarketEnv(config)
    states = []
    try:
        env.reset(seed=881001)
        for step in range(config.episode_steps):
            if step < 10:
                actions = [4] * 10
            elif step == 10:
                actions = [1, 1, 4, 4, 4, 4, 4, 4, 4, 4]
            elif step == 11:
                actions = [6, 0, 6, 6, 6, 6, 6, 6, 6, 6]
            else:
                actions = [6] * 10
            states.append(
                env.step(
                    actions,
                    gate_active=True,
                    routing_enabled=True,
                    routing_strength=1.0,
                    routing_mechanism=MECHANISM,
                )
            )
    finally:
        env.close()
    return states


def public_contract_trace(states: list[dict]) -> list[tuple]:
    return [
        (
            state["info"]["active_routing"],
            state["info"]["routing_mechanism_parameters"].get("takeup"),
            state["info"]["routing_mechanism_parameters"].get("dealer_index"),
            state["info"]["routing_mechanism_parameters"].get(
                "frozen_half_spread"
            ),
            state["info"]["routing_mechanism_parameters"].get(
                "remaining_periods_including_current"
            ),
        )
        for state in states
    ]


def main() -> None:
    low = run_once("low")
    repeated = run_once("low")
    high = run_once("high")
    if public_contract_trace(low) != public_contract_trace(repeated):
        raise RuntimeError("temporary commitment is not deterministic")
    if public_contract_trace(low) != public_contract_trace(high):
        raise RuntimeError("latent risk changed the public-quote commitment rule")

    for state in low:
        if not np.isclose(state["info"]["realized_flow_quantity"], 20.0):
            raise RuntimeError("temporary commitment failed demand conservation")
    active = low[10:15]
    if [
        state["info"]["routing_mechanism_parameters"][
            "remaining_periods_including_current"
        ]
        for state in active
    ] != [5, 4, 3, 2, 1]:
        raise RuntimeError("temporary commitment duration is not exact")
    if not active[0]["info"]["routing_mechanism_parameters"]["takeup"]:
        raise RuntimeError("qualifying undercut did not take up a commitment")
    takeup_dealer = active[0]["info"]["routing_mechanism_parameters"][
        "dealer_index"
    ]
    if takeup_dealer not in (0, 1) or any(
        state["info"]["routing_mechanism_parameters"]["dealer_index"]
        != takeup_dealer
        for state in active
    ):
        raise RuntimeError("tie handling or no-overlap protection changed dealer")
    frozen_reference = active[0]["info"]["routing_mechanism_parameters"][
        "frozen_reference_price"
    ]
    for state in active:
        info = state["info"]
        if info["targeted_event_count"] != 10:
            raise RuntimeError("protected block is not exactly half of demand")
        if not np.isclose(info["protected_execution_quantity"], 10.0):
            raise RuntimeError("protected routed orders did not execute in full")
        if info["protected_routing_fallback_count"] != 0:
            raise RuntimeError("protected flow fell back to the lit book")
        if not all(
            tag.startswith("ace_protected_c10")
            for tag in info["protected_matched_order_tags"]
        ):
            raise RuntimeError("protected flow did not consume protected resting orders")
        if not all(
            abs(int(price) - int(frozen_reference)) == 2
            for price in info["protected_execution_prices"]
        ):
            raise RuntimeError("protected execution did not honor the take-up quote")
    if low[15]["info"]["active_routing"]:
        raise RuntimeError("temporary commitment did not expire")
    if low[15]["info"]["protected_execution_quantity"] != 0.0:
        raise RuntimeError("expired commitment retained protected flow")
    print("ABIDES temporary commitment smoke passed")


if __name__ == "__main__":
    main()
