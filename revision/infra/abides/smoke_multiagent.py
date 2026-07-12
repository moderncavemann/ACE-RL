#!/usr/bin/env python3
"""Fail-closed integrity smoke for the ten-agent ABIDES adapter."""

import json
import subprocess
from typing import Dict, List

from abides_markets.orders import Side

from ace_abides.adapter import build_integrity_config, make_kernel


EXPECTED_COMMIT = "f9cbe51342b7dedd9587e4e069040d68a5c6477f"


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", "/opt/abides", "rev-parse", "HEAD"], text=True
    ).strip()


def book_orders(exchange, symbol: str) -> List:
    book = exchange.order_books[symbol]
    orders = []
    for level in book.bids + book.asks:
        orders.extend(order for order, _ in level.visible_orders)
        orders.extend(order for order, _ in level.hidden_orders)
    return orders


def run_integrity_episode(seed: int) -> Dict:
    config, handles = build_integrity_config(seed=seed)
    kernel = make_kernel(config)
    kernel.initialize()

    first_pause = kernel.runner()
    assert first_pause["done"] is False
    assert first_pause["result"]["backend"] == "abides"

    quotes = [
        {"half_spread": half_spread, "quantity": 10}
        for half_spread in range(1, 11)
    ]
    second_pause = kernel.runner(
        (
            handles["coordinator"],
            {
                "reference_price": 100_000,
                "quotes": quotes,
                "flow": {"side": "BUY", "quantity": 3},
            },
        )
    )
    assert second_pause["done"] is False

    controlled = handles["controlled"]
    controlled_ids = [agent.id for agent in controlled]
    assert len(controlled_ids) == 10
    assert len(set(controlled_ids)) == 10
    assert controlled_ids == list(range(1, 11))

    for expected_half_spread, agent in enumerate(controlled, start=1):
        instruction = agent.last_instruction
        assert instruction is not None
        assert instruction.half_spread == expected_half_spread
        assert instruction.reference_price == 100_000

    resting = book_orders(handles["exchange"], handles["symbol"])
    controlled_resting = [order for order in resting if order.agent_id in controlled_ids]
    assert len(controlled_resting) == 20
    by_agent = {agent_id: [] for agent_id in controlled_ids}
    for order in controlled_resting:
        by_agent[order.agent_id].append(order)

    for expected_half_spread, agent_id in enumerate(controlled_ids, start=1):
        orders = by_agent[agent_id]
        assert len(orders) == 2
        prices = {(order.side, order.limit_price) for order in orders}
        assert prices == {
            (Side.BID, 100_000 - expected_half_spread),
            (Side.ASK, 100_000 + expected_half_spread),
        }

    best_mm = controlled[0]
    taker = handles["taker"]
    assert best_mm.holdings.get(handles["symbol"], 0) == -3
    assert taker.holdings.get(handles["symbol"], 0) == 3
    assert best_mm.holdings["CASH"] == handles["starting_cash"] + 3 * 100_001
    assert taker.holdings["CASH"] == handles["starting_cash"] - 3 * 100_001
    assert best_mm.executions[-1].fill_price == 100_001
    assert taker.executions[-1].fill_price == 100_001

    economic_agents = controlled + [taker]
    assert sum(agent.holdings.get(handles["symbol"], 0) for agent in economic_agents) == 0
    assert sum(agent.holdings["CASH"] for agent in economic_agents) == len(
        economic_agents
    ) * handles["starting_cash"]

    fingerprint = {
        "controlled_ids": controlled_ids,
        "best_mm_inventory": best_mm.holdings.get(handles["symbol"], 0),
        "taker_inventory": taker.holdings.get(handles["symbol"], 0),
        "fill_price": taker.executions[-1].fill_price,
        "resting": sorted(
            (order.agent_id, order.side.value, order.limit_price, order.quantity)
            for order in controlled_resting
        ),
    }

    completion = kernel.runner()
    while completion["done"] is False:
        completion = kernel.runner()
    kernel.terminate()
    return fingerprint


def main() -> None:
    commit = source_commit()
    assert commit == EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)
    first = run_integrity_episode(seed=17)
    second = run_integrity_episode(seed=17)
    assert first == second
    print(
        json.dumps(
            {
                "status": "pass",
                "backend": "abides",
                "source_commit": commit,
                "controlled_agents": len(first["controlled_ids"]),
                "fill_price": first["fill_price"],
                "inventory_conserved": True,
                "cash_conserved": True,
                "same_seed_deterministic": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
