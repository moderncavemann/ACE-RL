"""Demand-conserving routed order support on the official ABIDES order book."""

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

from abides_core import NanosecondTime
from abides_markets.messages.orderbook import OrderExecutedMsg
from abides_markets.order_book import OrderBook
from abides_markets.orders import MarketOrder, Order, Side


class RoutedMarketOrder(MarketOrder):
    """A market order restricted to real resting orders from selected dealers."""

    def __init__(
        self,
        agent_id: int,
        time_placed: NanosecondTime,
        symbol: str,
        quantity: int,
        side: Side,
        eligible_agent_ids: Iterable[int],
        fallback_to_lit: bool,
        route_kind: str,
        order_id: Optional[int] = None,
        tag: Any = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            time_placed=time_placed,
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_id=order_id,
            tag=tag,
        )
        self.eligible_agent_ids: Tuple[int, ...] = tuple(eligible_agent_ids)
        self.fallback_to_lit = bool(fallback_to_lit)
        self.route_kind = str(route_kind)

    def __deepcopy__(self, memodict: Dict = {}) -> "RoutedMarketOrder":
        order = RoutedMarketOrder(
            agent_id=self.agent_id,
            time_placed=self.time_placed,
            symbol=self.symbol,
            quantity=self.quantity,
            side=self.side,
            eligible_agent_ids=self.eligible_agent_ids,
            fallback_to_lit=self.fallback_to_lit,
            route_kind=self.route_kind,
            order_id=self.order_id,
            tag=deepcopy(self.tag),
        )
        order.fill_price = self.fill_price
        return order


class RoutedOrderBook(OrderBook):
    """OrderBook extension for auditable dealer-directed customer flow.

    Ordinary limit and market orders delegate unchanged to upstream ABIDES.
    A RoutedMarketOrder selects the best-priced, FIFO resting order among the
    declared eligible dealers. If none is available, the preregistered fallback
    delegates to normal lit-book price-time priority.
    """

    def __init__(self, owner, symbol: str) -> None:
        super().__init__(owner, symbol)
        self.routed_executions = []
        self.routed_fallbacks = 0

    def execute_order(self, order: Order) -> Optional[Order]:
        if not isinstance(order, RoutedMarketOrder):
            return super().execute_order(order)

        selected = self._find_selected_order(order)
        if selected is None:
            if not order.fallback_to_lit:
                return None
            matched = super().execute_order(order)
            if matched is not None:
                self.routed_fallbacks += 1
                self._record_route(order, matched, used_fallback=True)
            return matched

        book, level_index, queue_name, queue_index = selected
        level = book[level_index]
        queue = getattr(level, queue_name)
        resting_order, metadata = queue[queue_index]
        if resting_order.is_price_to_comply:
            raise RuntimeError("routed flow does not support price-to-comply orders")

        if order.quantity >= resting_order.quantity:
            matched_order, _ = queue.pop(queue_index)
            if level.is_empty:
                del book[level_index]
        else:
            matched_order = deepcopy(resting_order)
            matched_order.quantity = order.quantity
            resting_order.quantity -= matched_order.quantity

        matched_order.fill_price = matched_order.limit_price
        if order.side.is_bid():
            self.buy_transactions.append(
                (self.owner.current_time, matched_order.quantity)
            )
        else:
            self.sell_transactions.append(
                (self.owner.current_time, matched_order.quantity)
            )

        self.history.append(
            {
                "time": self.owner.current_time,
                "type": "EXEC",
                "order_id": matched_order.order_id,
                "agent_id": matched_order.agent_id,
                "oppos_order_id": order.order_id,
                "oppos_agent_id": order.agent_id,
                "side": "SELL" if order.side.is_bid() else "BUY",
                "quantity": matched_order.quantity,
                "price": None,
                "route_kind": order.route_kind,
            }
        )

        filled_order = deepcopy(order)
        filled_order.quantity = matched_order.quantity
        filled_order.fill_price = matched_order.fill_price
        order.quantity -= filled_order.quantity

        self.owner.send_message(
            matched_order.agent_id, OrderExecutedMsg(matched_order)
        )
        self.owner.send_message(order.agent_id, OrderExecutedMsg(filled_order))
        if self.owner.book_logging:
            self.append_book_log2()
        self._record_route(order, matched_order, used_fallback=False)
        return matched_order

    def _find_selected_order(self, order: RoutedMarketOrder):
        eligible = set(order.eligible_agent_ids)
        eligible.discard(order.agent_id)
        if not eligible:
            return None

        book = self.asks if order.side.is_bid() else self.bids
        for level_index, level in enumerate(book):
            for queue_name in ("visible_orders", "hidden_orders"):
                queue = getattr(level, queue_name)
                for queue_index, (resting_order, _) in enumerate(queue):
                    if resting_order.agent_id in eligible:
                        return book, level_index, queue_name, queue_index
        return None

    def _record_route(
        self,
        order: RoutedMarketOrder,
        matched_order: Order,
        used_fallback: bool,
    ) -> None:
        self.last_trade = matched_order.fill_price
        self.routed_executions.append(
            {
                "time": int(self.owner.current_time),
                "route_kind": order.route_kind,
                "eligible_agent_ids": tuple(order.eligible_agent_ids),
                "matched_agent_id": matched_order.agent_id,
                "quantity": matched_order.quantity,
                "price": matched_order.fill_price,
                "used_fallback": used_fallback,
            }
        )
