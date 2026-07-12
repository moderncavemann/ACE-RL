#!/usr/bin/env python3
"""Fail-closed ABIDES core/markets smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from typing import Any, Dict

from abides_core import abides
from abides_markets.agents import ExchangeAgent


ABIDES_COMMIT = "f9cbe51342b7dedd9587e4e069040d68a5c6477f"


def source_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", "/opt/abides", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def build_config(
    name: str,
    seed: int,
    end_time: str,
    num_noise_agents: int,
    num_value_agents: int,
    num_momentum_agents: int,
) -> Dict[str, Any]:
    common = {
        "seed": seed,
        "end_time": end_time,
        "stdout_log_level": "WARNING",
        "log_orders": False,
        "book_logging": False,
        "num_noise_agents": num_noise_agents,
        "num_value_agents": num_value_agents,
        "num_momentum_agents": num_momentum_agents,
    }
    if name == "rmsc03":
        from abides_markets.configs import rmsc03

        return rmsc03.build_config(
            **common,
            exchange_log_orders=False,
            execution_agents=False,
        )
    if name == "rmsc04":
        from abides_markets.configs import rmsc04

        return rmsc04.build_config(
            **common,
            exchange_log_orders=False,
        )
    raise ValueError(f"unsupported config: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=("rmsc03", "rmsc04"), default="rmsc03")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--end-time", default="09:31:00")
    parser.add_argument("--num-noise-agents", type=int, default=20)
    parser.add_argument("--num-value-agents", type=int, default=5)
    parser.add_argument("--num-momentum-agents", type=int, default=2)
    args = parser.parse_args()

    revision = source_revision()
    if revision != ABIDES_COMMIT:
        raise RuntimeError(f"ABIDES revision mismatch: {revision}")

    config = build_config(
        args.config,
        args.seed,
        args.end_time,
        args.num_noise_agents,
        args.num_value_agents,
        args.num_momentum_agents,
    )
    exchanges = [agent for agent in config["agents"] if isinstance(agent, ExchangeAgent)]
    if len(exchanges) != 1:
        raise RuntimeError(f"expected one ExchangeAgent, found {len(exchanges)}")

    started = time.perf_counter()
    end_state = abides.run(config, kernel_seed=args.seed)
    elapsed = time.perf_counter() - started

    exchange = next(
        agent for agent in end_state["agents"] if isinstance(agent, ExchangeAgent)
    )
    if "ABM" not in exchange.order_books:
        raise RuntimeError("ABM order book missing from ExchangeAgent")
    book = exchange.order_books["ABM"]
    if not book.history:
        raise RuntimeError("ABIDES exchange produced no order history")
    if not book.bids and not book.asks:
        raise RuntimeError("ABIDES exchange ended with an empty two-sided book")

    payload = {
        "status": "pass",
        "backend": "abides",
        "abides_commit": revision,
        "config": args.config,
        "seed": args.seed,
        "end_time": args.end_time,
        "agent_count": len(end_state["agents"]),
        "num_noise_agents": args.num_noise_agents,
        "num_value_agents": args.num_value_agents,
        "num_momentum_agents": args.num_momentum_agents,
        "order_history_events": len(book.history),
        "bid_levels": len(book.bids),
        "ask_levels": len(book.asks),
        "last_trade": book.last_trade,
        "elapsed_seconds": elapsed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
