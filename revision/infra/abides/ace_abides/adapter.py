"""Multi-market-maker control plane for the frozen ABIDES event kernel.

The kernel supports one pausing Gym agent. This module uses that single agent as
an environment coordinator while keeping every controlled market maker as a
distinct TradingAgent with its own orders, fills, cash, and inventory.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from abides_core import Agent, Kernel, Message, NanosecondTime
from abides_core.utils import str_to_ns, subdict
from abides_markets.agents import AdaptiveMarketMakerAgent, ExchangeAgent
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.configs import rmsc04
from abides_markets.messages.marketdata import L2DataMsg, L2SubReqMsg
from abides_markets.messages.order import MarketOrderMsg
from abides_markets.orders import Order, Side
from abides_markets.utils import generate_latency_model

from .routing import RoutedMarketOrder, RoutedOrderBook


@dataclass
class QuoteInstructionMsg(Message):
    decision_id: int
    reference_price: int
    half_spread: int
    quantity: int
    cancel_existing: bool = True
    order_tag_prefix: Optional[str] = None


@dataclass
class CancelQuotesMsg(Message):
    decision_id: int


@dataclass
class FlowInstructionMsg(Message):
    decision_id: int
    side: str
    quantity: int
    eligible_agent_ids: Tuple[int, ...] = ()
    route_kind: str = "lit"
    required_order_tag_prefix: Optional[str] = None
    fallback_to_lit: bool = True


class CoreGymAgent(Agent, ABC):
    """Minimal local equivalent of the upstream ABIDES-Gym marker class."""

    @abstractmethod
    def apply_actions(self, actions: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_raw_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_raw_state(self) -> Dict[str, Any]:
        raise NotImplementedError


class ControlledMarketMaker(TradingAgent):
    """A real ABIDES trader whose quotes are supplied by the coordinator."""

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        random_state: np.random.RandomState,
    ) -> None:
        super().__init__(
            id=id,
            name=f"ACE_MARKET_MAKER_{id}",
            type="ACEControlledMarketMaker",
            random_state=random_state,
            starting_cash=starting_cash,
            log_orders=False,
        )
        self.symbol = symbol
        self.log_events = False
        self.log_to_file = False
        self.last_instruction: Optional[QuoteInstructionMsg] = None
        self.executions: List[Order] = []

    def get_wake_frequency(self) -> NanosecondTime:
        assert self.mkt_open is not None and self.mkt_close is not None
        return self.mkt_close - self.mkt_open + str_to_ns("1s")

    def wakeup(self, current_time: NanosecondTime) -> bool:
        return super().wakeup(current_time)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)
        if isinstance(message, CancelQuotesMsg):
            self.cancel_all_orders()
            return
        if not isinstance(message, QuoteInstructionMsg):
            return

        if message.half_spread < 1:
            raise ValueError("half_spread must be at least one tick")
        if message.quantity < 1:
            raise ValueError("quote quantity must be positive")

        self.last_instruction = message
        if message.cancel_existing:
            self.cancel_all_orders()
        bid_tag = (
            f"ace_bid_d{message.decision_id}"
            if message.order_tag_prefix is None
            else f"{message.order_tag_prefix}_bid"
        )
        ask_tag = (
            f"ace_ask_d{message.decision_id}"
            if message.order_tag_prefix is None
            else f"{message.order_tag_prefix}_ask"
        )
        self.place_limit_order(
            self.symbol,
            message.quantity,
            Side.BID,
            message.reference_price - message.half_spread,
            tag=bid_tag,
        )
        self.place_limit_order(
            self.symbol,
            message.quantity,
            Side.ASK,
            message.reference_price + message.half_spread,
            tag=ask_tag,
        )

    def order_executed(self, order: Order) -> None:
        super().order_executed(order)
        self.executions.append(deepcopy(order))


class MarketOrderTaker(TradingAgent):
    """Exogenous flow endpoint used by integrity and economics checks."""

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        random_state: np.random.RandomState,
    ) -> None:
        super().__init__(
            id=id,
            name="ACE_MARKET_ORDER_TAKER",
            type="ACEMarketOrderTaker",
            random_state=random_state,
            starting_cash=starting_cash,
            log_orders=False,
        )
        self.symbol = symbol
        self.log_events = False
        self.log_to_file = False
        self.executions: List[Order] = []

    def get_wake_frequency(self) -> NanosecondTime:
        assert self.mkt_open is not None and self.mkt_close is not None
        return self.mkt_close - self.mkt_open + str_to_ns("1s")

    def wakeup(self, current_time: NanosecondTime) -> bool:
        return super().wakeup(current_time)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)
        if not isinstance(message, FlowInstructionMsg):
            return

        side = Side.BID if message.side == "BUY" else Side.ASK
        if message.eligible_agent_ids:
            order = RoutedMarketOrder(
                agent_id=self.id,
                time_placed=self.current_time,
                symbol=self.symbol,
                quantity=message.quantity,
                side=side,
                eligible_agent_ids=message.eligible_agent_ids,
                fallback_to_lit=message.fallback_to_lit,
                route_kind=message.route_kind,
                required_order_tag_prefix=message.required_order_tag_prefix,
                tag=f"ace_flow_d{message.decision_id}",
            )
            self.orders[order.order_id] = deepcopy(order)
            self.send_message(self.exchange_id, MarketOrderMsg(order))
        else:
            self.place_market_order(
                self.symbol,
                message.quantity,
                side,
                tag=f"ace_flow_d{message.decision_id}",
            )

    def order_executed(self, order: Order) -> None:
        super().order_executed(order)
        self.executions.append(deepcopy(order))


class MultiMarketMakerCoordinator(CoreGymAgent):
    """The sole kernel-pausing agent controlling multiple real traders."""

    def __init__(
        self,
        id: int,
        symbol: str,
        controlled_agent_ids: Sequence[int],
        taker_agent_id: int,
        decision_start: NanosecondTime,
        decision_end: NanosecondTime,
        decision_interval: NanosecondTime,
        default_reference_price: int,
    ) -> None:
        super().__init__(
            id=id,
            name="ACE_MULTI_MM_COORDINATOR",
            type="ACEMultiMarketMakerCoordinator",
            random_state=np.random.RandomState(id),
            log_events=False,
            log_to_file=False,
        )
        self.symbol = symbol
        self.controlled_agent_ids = list(controlled_agent_ids)
        self.taker_agent_id = taker_agent_id
        self.decision_start = decision_start
        self.decision_end = decision_end
        self.decision_interval = decision_interval
        self.default_reference_price = default_reference_price
        self.exchange_id: Optional[int] = None
        self.latest_l2: Optional[L2DataMsg] = None
        self.decision_id = 0
        self.raw_state: Dict[str, Any] = {}

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        assert self.kernel is not None
        self.exchange_id = self.kernel.find_agents_by_type(ExchangeAgent)[0]
        self.send_message(
            self.exchange_id,
            L2SubReqMsg(symbol=self.symbol, freq=0, depth=10),
        )
        super().kernel_starting(start_time)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)
        if isinstance(message, L2DataMsg):
            self.latest_l2 = message

    def wakeup(self, current_time: NanosecondTime) -> Optional[Dict[str, Any]]:
        super().wakeup(current_time)
        if current_time < self.decision_start:
            self.set_wakeup(self.decision_start)
            return None

        self.update_raw_state()
        next_wakeup = current_time + self.decision_interval
        if next_wakeup <= self.decision_end:
            self.set_wakeup(next_wakeup)
        return deepcopy(self.raw_state)

    def _reference_price(self) -> int:
        if self.latest_l2 is None:
            return self.default_reference_price
        if self.latest_l2.bids and self.latest_l2.asks:
            return int(round((self.latest_l2.bids[0][0] + self.latest_l2.asks[0][0]) / 2))
        return int(self.latest_l2.last_transaction or self.default_reference_price)

    def apply_actions(self, actions: Dict[str, Any]) -> None:
        quotes = actions.get("quotes")
        if not isinstance(quotes, list) or len(quotes) != len(
            self.controlled_agent_ids
        ):
            raise ValueError("one quote action is required per controlled market maker")

        reference_price = int(actions.get("reference_price", self._reference_price()))
        for agent_id, quote in zip(self.controlled_agent_ids, quotes):
            self.send_message(
                agent_id,
                QuoteInstructionMsg(
                    decision_id=self.decision_id,
                    reference_price=reference_price,
                    half_spread=int(quote["half_spread"]),
                    quantity=int(quote["quantity"]),
                ),
            )

        flow = actions.get("flow")
        if flow is not None:
            side = str(flow["side"]).upper()
            if side not in {"BUY", "SELL"}:
                raise ValueError("flow side must be BUY or SELL")
            self.send_message(
                self.taker_agent_id,
                FlowInstructionMsg(
                    decision_id=self.decision_id,
                    side=side,
                    quantity=int(flow["quantity"]),
                    eligible_agent_ids=tuple(
                        int(agent_id) for agent_id in flow.get("eligible_agent_ids", ())
                    ),
                    route_kind=str(flow.get("route_kind", "lit")),
                    required_order_tag_prefix=flow.get(
                        "required_order_tag_prefix"
                    ),
                    fallback_to_lit=bool(flow.get("fallback_to_lit", True)),
                ),
                delay=int(flow.get("delay_ns", str_to_ns("1ms"))),
            )
        self.decision_id += 1

    def update_raw_state(self) -> None:
        assert self.kernel is not None
        agents = []
        for agent_id in self.controlled_agent_ids:
            agent = self.kernel.agents[agent_id]
            if not isinstance(agent, ControlledMarketMaker):
                raise TypeError(f"agent {agent_id} is not a controlled market maker")
            agents.append(
                {
                    "agent_id": agent.id,
                    "cash": agent.holdings["CASH"],
                    "inventory": agent.holdings.get(self.symbol, 0),
                    "active_order_count": len(agent.orders),
                    "execution_count": len(agent.executions),
                }
            )

        self.raw_state = {
            "backend": "abides",
            "decision_id": self.decision_id,
            "time": int(self.current_time),
            "reference_price": self._reference_price(),
            "bids": [] if self.latest_l2 is None else list(self.latest_l2.bids),
            "asks": [] if self.latest_l2 is None else list(self.latest_l2.asks),
            "agents": agents,
        }

    def get_raw_state(self) -> Dict[str, Any]:
        return self.raw_state


def _reindex_agents(agents: Sequence[Agent]) -> List[Agent]:
    reindexed = list(agents)
    for new_id, agent in enumerate(reindexed):
        agent.id = new_id
    return reindexed


def build_integrity_config(
    seed: int = 17,
    num_market_makers: int = 10,
    starting_cash: int = 10_000_000,
    symbol: str = "ABM",
    routed_book: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build a minimal real-exchange config for adapter integrity checks."""

    config = rmsc04.build_config(
        seed=seed,
        end_time="09:32:00",
        ticker=symbol,
        starting_cash=starting_cash,
        log_orders=False,
        book_logging=False,
        exchange_log_orders=False,
        num_noise_agents=0,
        num_value_agents=0,
        num_momentum_agents=0,
    )
    background = [
        agent
        for agent in config["agents"]
        if not isinstance(agent, AdaptiveMarketMakerAgent)
    ]
    agents = _reindex_agents(background)
    exchange = agents[0]
    if not isinstance(exchange, ExchangeAgent):
        raise RuntimeError("integrity config must retain the real ABIDES exchange")
    if routed_book:
        previous_book = exchange.order_books[symbol]
        routed = RoutedOrderBook(exchange, symbol)
        routed.last_trade = previous_book.last_trade
        exchange.order_books[symbol] = routed

    rng = np.random.RandomState(seed ^ 0xACE)
    controlled: List[ControlledMarketMaker] = []
    for _ in range(num_market_makers):
        agent_id = len(agents)
        mm = ControlledMarketMaker(
            id=agent_id,
            symbol=symbol,
            starting_cash=starting_cash,
            random_state=np.random.RandomState(rng.randint(0, 2**32, dtype="uint32")),
        )
        controlled.append(mm)
        agents.append(mm)

    taker = MarketOrderTaker(
        id=len(agents),
        symbol=symbol,
        starting_cash=starting_cash,
        random_state=np.random.RandomState(rng.randint(0, 2**32, dtype="uint32")),
    )
    agents.append(taker)

    decision_start = exchange.mkt_open + str_to_ns("1s")
    coordinator = MultiMarketMakerCoordinator(
        id=len(agents),
        symbol=symbol,
        controlled_agent_ids=[agent.id for agent in controlled],
        taker_agent_id=taker.id,
        decision_start=decision_start,
        decision_end=exchange.mkt_open + str_to_ns("61s"),
        decision_interval=str_to_ns("30s"),
        default_reference_price=100_000,
    )
    agents.append(coordinator)

    config["agents"] = agents
    config["agent_latency_model"] = generate_latency_model(
        len(agents), latency_type="no_latency"
    )
    config["default_computation_delay"] = 1
    config["stdout_log_level"] = "WARNING"

    handles = {
        "exchange": exchange,
        "controlled": controlled,
        "taker": taker,
        "coordinator": coordinator,
        "starting_cash": starting_cash,
        "symbol": symbol,
    }
    return config, handles


def make_kernel(config: Dict[str, Any]) -> Kernel:
    return Kernel(
        random_state=config["random_state_kernel"],
        **subdict(
            config,
            [
                "start_time",
                "stop_time",
                "agents",
                "agent_latency_model",
                "default_computation_delay",
                "custom_properties",
            ],
        ),
    )
