"""Demand-conserving routed order support on the official ABIDES order book."""

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

from abides_core import NanosecondTime
from abides_markets.messages.orderbook import (
    OrderAcceptedMsg,
    OrderCancelledMsg,
    OrderExecutedMsg,
)
from abides_markets.order_book import OrderBook
from abides_markets.orders import LimitOrder, MarketOrder, Order, Side


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
        required_order_tag_prefix: Optional[str] = None,
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
        self.required_order_tag_prefix = (
            None
            if required_order_tag_prefix is None
            else str(required_order_tag_prefix)
        )

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
            required_order_tag_prefix=self.required_order_tag_prefix,
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
        self.protected_orders: Dict[int, LimitOrder] = {}

    def handle_limit_order(self, order: LimitOrder, quiet: bool = False) -> None:
        if not str(order.tag or "").startswith("ace_protected_"):
            super().handle_limit_order(order, quiet=quiet)
            return
        if order.symbol != self.symbol:
            raise ValueError("protected order symbol differs from routed book")
        if order.quantity <= 0 or int(order.quantity) != order.quantity:
            raise ValueError("protected order quantity must be a positive integer")
        if order.limit_price < 0 or int(order.limit_price) != order.limit_price:
            raise ValueError("protected order limit price must be a positive integer")
        if order.order_id in self.protected_orders:
            raise RuntimeError("duplicate protected order id")
        self.protected_orders[order.order_id] = deepcopy(order)
        if not quiet:
            self.owner.send_message(order.agent_id, OrderAcceptedMsg(order))

    def cancel_order(
        self,
        order: LimitOrder,
        tag: Optional[str] = None,
        cancellation_metadata: Optional[Dict] = None,
        quiet: bool = False,
    ) -> bool:
        protected = self.protected_orders.pop(order.order_id, None)
        if protected is None:
            return super().cancel_order(
                order,
                tag=tag,
                cancellation_metadata=cancellation_metadata,
                quiet=quiet,
            )
        if not quiet:
            self.history.append(
                {
                    "time": self.owner.current_time,
                    "type": "CANCEL",
                    "order_id": protected.order_id,
                    "tag": tag,
                    "protected_contract": True,
                }
            )
            self.owner.send_message(order.agent_id, OrderCancelledMsg(protected))
        self.last_update_ts = self.owner.current_time
        return True

    def execute_order(self, order: Order) -> Optional[Order]:
        if not isinstance(order, RoutedMarketOrder):
            return super().execute_order(order)

        if order.required_order_tag_prefix is not None:
            matched = self._execute_protected_order(order)
            if matched is None and order.fallback_to_lit:
                matched = super().execute_order(order)
                if matched is not None:
                    self.routed_fallbacks += 1
                    self._record_route(order, matched, used_fallback=True)
            return matched

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

    def _execute_protected_order(
        self, order: RoutedMarketOrder
    ) -> Optional[LimitOrder]:
        eligible = set(order.eligible_agent_ids)
        eligible.discard(order.agent_id)
        prefix = order.required_order_tag_prefix
        if not eligible or prefix is None:
            return None
        required_side = Side.ASK if order.side.is_bid() else Side.BID
        candidates = [
            resting
            for resting in self.protected_orders.values()
            if resting.agent_id in eligible
            and resting.side == required_side
            and str(resting.tag or "").startswith(prefix)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda resting: (resting.time_placed, resting.order_id))
        resting = candidates[0]
        quantity = min(int(order.quantity), int(resting.quantity))
        matched_order = deepcopy(resting)
        matched_order.quantity = quantity
        matched_order.fill_price = resting.limit_price
        resting.quantity -= quantity
        if resting.quantity == 0:
            del self.protected_orders[resting.order_id]

        filled_order = deepcopy(order)
        filled_order.quantity = quantity
        filled_order.fill_price = matched_order.fill_price
        order.quantity -= quantity
        self.owner.send_message(
            matched_order.agent_id, OrderExecutedMsg(matched_order)
        )
        self.owner.send_message(order.agent_id, OrderExecutedMsg(filled_order))
        self.history.append(
            {
                "time": self.owner.current_time,
                "type": "EXEC",
                "order_id": matched_order.order_id,
                "agent_id": matched_order.agent_id,
                "oppos_order_id": order.order_id,
                "oppos_agent_id": order.agent_id,
                "side": "SELL" if order.side.is_bid() else "BUY",
                "quantity": quantity,
                "price": matched_order.fill_price,
                "route_kind": order.route_kind,
                "protected_contract": True,
            }
        )
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
                    tag = str(resting_order.tag or "")
                    protected = tag.startswith("ace_protected_")
                    required = order.required_order_tag_prefix
                    tag_matches = (
                        tag.startswith(required)
                        if required is not None
                        else not protected
                    )
                    if resting_order.agent_id in eligible and tag_matches:
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
                "required_order_tag_prefix": order.required_order_tag_prefix,
                "matched_order_tag": matched_order.tag,
            }
        )
