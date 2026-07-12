#!/usr/bin/env python3
"""Fail-closed checks for dealer-directed, demand-conserving ABIDES routing."""

import json
from typing import Dict

from ace_abides.adapter import build_integrity_config, make_kernel
from ace_abides.routing import RoutedOrderBook


def run_route(destination_agent_id: int, route_kind: str) -> Dict:
    config, handles = build_integrity_config(seed=29, routed_book=True)
    kernel = make_kernel(config)
    kernel.initialize()
    pause = kernel.runner()
    assert pause["done"] is False

    quotes = [
        {"half_spread": half_spread, "quantity": 10}
        for half_spread in range(1, 11)
    ]
    pause = kernel.runner(
        (
            handles["coordinator"],
            {
                "reference_price": 100_000,
                "quotes": quotes,
                "flow": {
                    "side": "BUY",
                    "quantity": 1,
                    "eligible_agent_ids": [destination_agent_id],
                    "route_kind": route_kind,
                },
            },
        )
    )
    assert pause["done"] is False

    book = handles["exchange"].order_books[handles["symbol"]]
    assert isinstance(book, RoutedOrderBook)
    assert len(book.routed_executions) == 1
    record = book.routed_executions[0]
    assert record["matched_agent_id"] == destination_agent_id
    assert record["quantity"] == 1
    assert record["used_fallback"] is False

    matched = handles["controlled"][destination_agent_id - 1]
    taker = handles["taker"]
    assert matched.holdings.get(handles["symbol"], 0) == -1
    assert taker.holdings.get(handles["symbol"], 0) == 1
    assert matched.holdings["CASH"] + taker.holdings["CASH"] == 2 * handles[
        "starting_cash"
    ]

    completion = kernel.runner()
    while completion["done"] is False:
        completion = kernel.runner()
    kernel.terminate()
    return record


def main() -> None:
    base = run_route(destination_agent_id=5, route_kind="base")
    targeted = run_route(destination_agent_id=1, route_kind="targeted")
    assert base["quantity"] == targeted["quantity"] == 1
    assert base["price"] == 100_005
    assert targeted["price"] == 100_001
    assert targeted["price"] < base["price"]
    print(
        json.dumps(
            {
                "status": "pass",
                "backend": "abides",
                "base_quantity": base["quantity"],
                "targeted_quantity": targeted["quantity"],
                "base_price": base["price"],
                "targeted_price": targeted["price"],
                "real_resting_order_consumed": True,
                "cash_conserved": True,
                "demand_conserved": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
