"""Local matching-engine LOB market-making pilot environment.

This is a migration bridge between the toy spread-choice market and a full
LOB/MAPPO experiment. Agents submit two-sided limit quotes into a price-time
priority book. When the platform routing gate is active, the platform routes
more exogenous market flow only if the best spread is competitive; otherwise it
withholds part of the flow. This keeps the intervention at the order-flow layer
without directly changing agent rewards.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Sequence, Tuple

import numpy as np

RiskState = Literal["low", "high"]
RoutingMode = Literal[
    "book_multiplier",
    "eligible_priority",
    "relative_priority",
    "relative_reallocate",
    "continuous_reallocate",
    "balanced_reallocate",
]
TargetedFillPricing = Literal["agent_quote", "competitive_benchmark"]


@dataclass(frozen=True)
class LocalLOBConfig:
    n_agents: int = 10
    half_spread_levels: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    tick_size: float = 0.01
    initial_mid_price: float = 100.0
    episode_length: int = 200
    max_inventory: int = 20
    inventory_penalty: float = 0.002
    base_flow_intensity: float = 2.0
    routing_mode: RoutingMode = "book_multiplier"
    routing_flow_multiplier: float = 1.25
    routing_spread_cap_ticks: float = 4.0
    routing_withhold_rate: float = 0.50
    routing_relative_improvement_ticks: float = 1.0
    routing_target_kappa: float = 0.0
    risk_coupled_economics: bool = False
    mark_to_fundamental: bool = False
    informed_flow_prob: float = 0.0
    spread_elasticity: float = 0.0
    targeted_fill_pricing: TargetedFillPricing = "agent_quote"
    targeted_competitive_half_spread_ticks: float = 2.0
    enforce_max_inventory: bool = False
    seed_liquidity_levels: int = 5
    seed_liquidity_qty: int = 8
    p_low_to_high: float = 0.04
    p_high_to_low: float = 0.10
    low_sigma: float = 0.01
    high_sigma: float = 0.04
    seed: int = 42

    @property
    def n_actions(self) -> int:
        return len(self.half_spread_levels)


class Order:
    def __init__(self, order_id: int, agent_id: int, side: str, price: float, quantity: int, timestamp: int):
        self.order_id = order_id
        self.agent_id = agent_id
        self.side = side
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp
        self.filled = 0

    def remaining(self) -> int:
        return self.quantity - self.filled


class LocalOrderBook:
    def __init__(self, tick_size: float):
        self.tick_size = tick_size
        self.bids: Dict[float, deque[Order]] = {}
        self.asks: Dict[float, deque[Order]] = {}
        self.orders: Dict[int, Order] = {}
        self.next_order_id = 0
        self.current_time = 0
        self.trades: list[dict[str, object]] = []

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()
        self.trades.clear()
        self.next_order_id = 0
        self.current_time = 0

    def submit_order(self, agent_id: int, side: str, price: float, quantity: int) -> tuple[int, list[dict[str, object]]]:
        price = round(price / self.tick_size) * self.tick_size
        order = Order(self.next_order_id, agent_id, side, price, quantity, self.current_time)
        self.next_order_id += 1
        self.current_time += 1
        fills = self._match(order)
        if order.remaining() > 0:
            self.orders[order.order_id] = order
            book = self.bids if side == "bid" else self.asks
            book.setdefault(price, deque()).append(order)
        return order.order_id, fills

    def cancel_order(self, order_id: int) -> bool:
        order = self.orders.get(order_id)
        if order is None:
            return False
        book = self.bids if order.side == "bid" else self.asks
        level = book.get(order.price)
        if level is not None:
            try:
                level.remove(order)
            except ValueError:
                pass
            if not level:
                del book[order.price]
        del self.orders[order_id]
        return True

    def cancel_agent_orders(self, agent_id: int) -> None:
        for order_id, order in list(self.orders.items()):
            if order.agent_id == agent_id:
                self.cancel_order(order_id)

    def best_bid_ask(self) -> tuple[float | None, float | None]:
        best_bid = max(self.bids) if self.bids else None
        best_ask = min(self.asks) if self.asks else None
        return best_bid, best_ask

    def mid_price(self, fallback: float) -> float:
        best_bid, best_ask = self.best_bid_ask()
        if best_bid is not None and best_ask is not None:
            return float((best_bid + best_ask) / 2.0)
        return fallback

    def spread_ticks(self) -> float:
        best_bid, best_ask = self.best_bid_ask()
        if best_bid is None or best_ask is None:
            return 0.0
        return float((best_ask - best_bid) / self.tick_size)

    def _match(self, order: Order) -> list[dict[str, object]]:
        fills: list[dict[str, object]] = []
        contra = self.asks if order.side == "bid" else self.bids
        while order.remaining() > 0 and contra:
            if order.side == "bid":
                best_price = min(contra)
                if best_price > order.price:
                    break
            else:
                best_price = max(contra)
                if best_price < order.price:
                    break
            level = contra[best_price]
            while level and order.remaining() > 0:
                resting = level[0]
                fill_qty = min(order.remaining(), resting.remaining())
                fill = {
                    "price": best_price,
                    "quantity": fill_qty,
                    "aggressor_id": order.agent_id,
                    "aggressor_side": order.side,
                    "passive_id": resting.agent_id,
                    "passive_side": resting.side,
                }
                fills.append(fill)
                self.trades.append(fill)
                order.filled += fill_qty
                resting.filled += fill_qty
                if resting.remaining() == 0:
                    level.popleft()
                    self.orders.pop(resting.order_id, None)
            if not level:
                del contra[best_price]
        return fills


class LocalLOBMarketEnv:
    def __init__(self, config: LocalLOBConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.book = LocalOrderBook(config.tick_size)
        self.t = 0
        self.risk_state: RiskState = "low"
        self.fundamental = config.initial_mid_price
        self.agent_orders: Dict[int, list[int]] = {}
        self.inventories = np.zeros(config.n_agents, dtype=float)
        self.cash = np.zeros(config.n_agents, dtype=float)
        self.last_actions = np.zeros(config.n_agents, dtype=int)
        self.last_agent_spreads = np.ones(config.n_agents, dtype=float)
        self.last_flow_shares = np.full(config.n_agents, 1.0 / config.n_agents, dtype=float)
        self.last_total_volume = 0.0
        self.last_routing_multiplier = 1.0
        self.last_routing_cap_met = False
        self.last_routing_targeted_volume = 0.0
        self.last_routing_targeted_orders = 0
        self.last_routing_eligible_agents = 0
        self.last_wealth = np.zeros(config.n_agents, dtype=float)
        self.last_fundamental_shock = 0.0
        self.last_spread_revenue = np.zeros(config.n_agents, dtype=float)
        self.last_adverse_selection_loss = np.zeros(config.n_agents, dtype=float)
        self.last_inventory_penalty = np.zeros(config.n_agents, dtype=float)
        self.last_mark_price = config.initial_mid_price
        self._execution_mid = config.initial_mid_price

    def reset(self, seed: int | None = None) -> tuple[Dict[int, Tuple[int, ...]], dict[str, object]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.book.reset()
        self.t = 0
        self.risk_state = "high" if self.rng.random() < 0.30 else "low"
        self.fundamental = self.config.initial_mid_price
        self.agent_orders = {i: [] for i in range(self.config.n_agents)}
        self.inventories = np.zeros(self.config.n_agents, dtype=float)
        self.cash = np.zeros(self.config.n_agents, dtype=float)
        self.last_actions = np.zeros(self.config.n_agents, dtype=int)
        self.last_agent_spreads = np.full(self.config.n_agents, self.config.half_spread_levels[0], dtype=float)
        self.last_flow_shares = np.full(self.config.n_agents, 1.0 / self.config.n_agents, dtype=float)
        self.last_total_volume = 0.0
        self.last_routing_multiplier = 1.0
        self.last_routing_cap_met = False
        self.last_routing_targeted_volume = 0.0
        self.last_routing_targeted_orders = 0
        self.last_routing_eligible_agents = 0
        self.last_fundamental_shock = 0.0
        self.last_spread_revenue = np.zeros(self.config.n_agents, dtype=float)
        self.last_adverse_selection_loss = np.zeros(self.config.n_agents, dtype=float)
        self.last_inventory_penalty = np.zeros(self.config.n_agents, dtype=float)
        self.last_mark_price = self.fundamental
        self._execution_mid = self.fundamental
        self._seed_liquidity()
        self.last_wealth = self._wealth()
        return self._states(), self._info_stub()

    def step(
        self,
        actions: Dict[int, int] | Iterable[int],
        routing_flow_boost: float = 0.0,
        gate_active: bool = False,
        gate_score: float = 0.0,
    ) -> tuple[Dict[int, Tuple[int, ...]], Dict[int, float], bool, bool, dict[str, object]]:
        action_arr = self._coerce_actions(actions)
        self.last_actions = action_arr.copy()
        half_spreads = np.array([self.config.half_spread_levels[a] for a in action_arr], dtype=float)
        self.last_agent_spreads = half_spreads

        sigma = self.config.high_sigma if self.risk_state == "high" else self.config.low_sigma
        fundamental_shock = float(self.rng.normal(0.0, sigma))
        self.last_fundamental_shock = fundamental_shock
        before_wealth = self._wealth()
        if self.config.risk_coupled_economics:
            mid = self._reference_mid()
            self._execution_mid = mid
        else:
            self.fundamental += fundamental_shock
            mid = self.book.mid_price(self.fundamental)
            self._execution_mid = mid

        for agent_id in range(self.config.n_agents):
            self.book.cancel_agent_orders(agent_id)
            self.agent_orders[agent_id] = []
            half = half_spreads[agent_id] * self.config.tick_size
            bid_id, _ = self.book.submit_order(agent_id, "bid", mid - half, 1)
            ask_id, _ = self.book.submit_order(agent_id, "ask", mid + half, 1)
            self.agent_orders[agent_id] = [bid_id, ask_id]

        fills = self._route_external_flow(max(0.0, routing_flow_boost))
        fills = self._process_fills(fills)
        if self.config.risk_coupled_economics:
            self.fundamental += fundamental_shock

        after_wealth = self._wealth()
        self.last_inventory_penalty = self.config.inventory_penalty * (self.inventories ** 2)
        rewards_arr = after_wealth - before_wealth - self.last_inventory_penalty
        rewards = {i: float(rewards_arr[i]) for i in range(self.config.n_agents)}

        agent_flow = np.zeros(self.config.n_agents, dtype=float)
        total_volume = 0.0
        for fill in fills:
            qty = float(fill["quantity"])
            total_volume += qty
            passive_id = int(fill["passive_id"])
            if 0 <= passive_id < self.config.n_agents:
                agent_flow[passive_id] += qty
        if total_volume > 0:
            self.last_flow_shares = agent_flow / total_volume
        self.last_total_volume = total_volume

        info = self._info_stub()
        info.update({
            "t": self.t,
            "risk_state": self.risk_state,
            "fundamental": float(self.fundamental),
            "fundamental_shock": float(self.last_fundamental_shock),
            "mark_price": float(self.last_mark_price),
            "best_spread_ticks": self.book.spread_ticks(),
            "avg_agent_spread": float(np.mean(half_spreads)),
            "agent_spreads": half_spreads.copy(),
            "total_volume": float(total_volume),
            "flow_shares": self.last_flow_shares.copy(),
            "mean_reward": float(np.mean(rewards_arr)),
            "mean_spread_revenue": float(np.mean(self.last_spread_revenue)),
            "mean_adverse_selection_loss": float(np.mean(self.last_adverse_selection_loss)),
            "mean_inventory_penalty": float(np.mean(self.last_inventory_penalty)),
            "routing_flow_boost": float(routing_flow_boost),
            "routing_effective_multiplier": float(self.last_routing_multiplier),
            "routing_spread_cap_met": bool(self.last_routing_cap_met),
            "routing_targeted_volume": float(self.last_routing_targeted_volume),
            "routing_targeted_orders": float(self.last_routing_targeted_orders),
            "routing_eligible_agents": float(self.last_routing_eligible_agents),
            "gate_active": bool(gate_active),
            "gate_score": float(gate_score),
            "mean_abs_inventory": float(np.mean(np.abs(self.inventories))),
        })

        self.t += 1
        done = self.t >= self.config.episode_length
        self._advance_risk_state()
        return self._states(), rewards, done, False, info

    def state_for_agent(self, agent_id: int) -> Tuple[int, ...]:
        risk_bin = 1 if self.risk_state == "high" else 0
        own_action = int(self.last_actions[agent_id])
        avg_bin = int(np.clip(round(float(np.mean(self.last_actions))), 0, self.config.n_actions - 1))
        best_spread_bin = int(np.clip(round(self.book.spread_ticks() / 2.0), 0, self.config.n_actions - 1))
        inv = self.inventories[agent_id]
        inv_bin = int(np.clip(inv + 2, 0, 4))
        flow_bin = int(np.clip(np.floor(self.last_flow_shares[agent_id] * 5), 0, 4))
        return risk_bin, own_action, avg_bin, best_spread_bin, inv_bin, flow_bin

    def _states(self) -> Dict[int, Tuple[int, ...]]:
        return {i: self.state_for_agent(i) for i in range(self.config.n_agents)}

    def _info_stub(self) -> dict[str, object]:
        return {
            "risk_state": self.risk_state,
            "avg_agent_spread": float(np.mean(self.last_agent_spreads)),
            "agent_spreads": self.last_agent_spreads.copy(),
            "best_spread_ticks": self.book.spread_ticks(),
            "total_volume": float(self.last_total_volume),
            "flow_shares": self.last_flow_shares.copy(),
        }

    def _seed_liquidity(self) -> None:
        mid = self.fundamental
        for level in range(1, self.config.seed_liquidity_levels + 1):
            qty = self.config.seed_liquidity_qty
            self.book.submit_order(-1, "bid", mid - level * self.config.tick_size, qty)
            self.book.submit_order(-1, "ask", mid + level * self.config.tick_size, qty)

    def _route_external_flow(self, routing_flow_boost: float) -> list[dict[str, object]]:
        best_spread = self.book.spread_ticks()
        cap_met = 0.0 < best_spread <= self.config.routing_spread_cap_ticks
        self.last_routing_targeted_volume = 0.0
        self.last_routing_targeted_orders = 0
        eligible_agents = self._routing_eligible_agents()
        if self.config.routing_mode in {"continuous_reallocate", "balanced_reallocate"}:
            eligible_agents = list(range(self.config.n_agents))
        self.last_routing_eligible_agents = len(eligible_agents) if routing_flow_boost > 0.0 else 0
        if self.config.routing_mode in {"eligible_priority", "relative_priority"}:
            return self._route_external_flow_eligible_priority(routing_flow_boost, cap_met, eligible_agents)
        if self.config.routing_mode == "relative_reallocate":
            return self._route_external_flow_relative_reallocate(routing_flow_boost, cap_met, eligible_agents)
        if self.config.routing_mode == "continuous_reallocate":
            return self._route_external_flow_continuous_reallocate(
                routing_flow_boost,
                cap_met,
                eligible_agents,
                balanced=False,
            )
        if self.config.routing_mode == "balanced_reallocate":
            return self._route_external_flow_continuous_reallocate(
                routing_flow_boost,
                cap_met,
                eligible_agents,
                balanced=True,
            )

        return self._route_external_flow_book_multiplier(routing_flow_boost, cap_met)

    def _route_external_flow_book_multiplier(self, routing_flow_boost: float, cap_met: bool) -> list[dict[str, object]]:
        if routing_flow_boost > 0.0 and cap_met:
            multiplier = 1.0 + self.config.routing_flow_multiplier * routing_flow_boost
        elif routing_flow_boost > 0.0:
            multiplier = 1.0 - self.config.routing_withhold_rate * routing_flow_boost
        else:
            multiplier = 1.0
        multiplier = max(0.0, multiplier)
        self.last_routing_multiplier = multiplier
        self.last_routing_cap_met = bool(cap_met)
        intensity = self._flow_intensity() * multiplier
        n_orders = int(self.rng.poisson(intensity))
        return self._sample_book_external_orders(n_orders)

    def _route_external_flow_eligible_priority(
        self,
        routing_flow_boost: float,
        cap_met: bool,
        eligible_agents: list[int],
    ) -> list[dict[str, object]]:
        if routing_flow_boost > 0.0 and eligible_agents:
            normal_multiplier = 1.0
            targeted_multiplier = self.config.routing_flow_multiplier * routing_flow_boost
            effective_multiplier = 1.0 + targeted_multiplier
        elif routing_flow_boost > 0.0:
            normal_multiplier = max(0.0, 1.0 - self.config.routing_withhold_rate * routing_flow_boost)
            targeted_multiplier = 0.0
            effective_multiplier = normal_multiplier
        else:
            normal_multiplier = 1.0
            targeted_multiplier = 0.0
            effective_multiplier = 1.0

        self.last_routing_multiplier = effective_multiplier
        self.last_routing_cap_met = bool(cap_met)
        base_intensity = self._flow_intensity()
        normal_orders = int(self.rng.poisson(base_intensity * normal_multiplier))
        targeted_orders = int(self.rng.poisson(base_intensity * targeted_multiplier))
        fills = self._sample_book_external_orders(normal_orders)
        targeted_fills = self._sample_targeted_external_fills(targeted_orders, eligible_agents)
        self.last_routing_targeted_volume = float(sum(float(fill["quantity"]) for fill in targeted_fills))
        self.last_routing_targeted_orders = len(targeted_fills)
        fills.extend(targeted_fills)
        return fills

    def _route_external_flow_relative_reallocate(
        self,
        routing_flow_boost: float,
        cap_met: bool,
        eligible_agents: list[int],
    ) -> list[dict[str, object]]:
        if routing_flow_boost > 0.0 and eligible_agents:
            targeted_share = float(np.clip(routing_flow_boost, 0.0, 1.0))
            normal_multiplier = 1.0 - targeted_share
            targeted_multiplier = targeted_share
            effective_multiplier = 1.0
        elif routing_flow_boost > 0.0:
            normal_multiplier = max(0.0, 1.0 - self.config.routing_withhold_rate * routing_flow_boost)
            targeted_multiplier = 0.0
            effective_multiplier = normal_multiplier
        else:
            normal_multiplier = 1.0
            targeted_multiplier = 0.0
            effective_multiplier = 1.0

        self.last_routing_multiplier = effective_multiplier
        self.last_routing_cap_met = bool(cap_met)
        base_intensity = self._flow_intensity()
        normal_orders = int(self.rng.poisson(base_intensity * normal_multiplier))
        targeted_orders = int(self.rng.poisson(base_intensity * targeted_multiplier))
        fills = self._sample_book_external_orders(normal_orders)
        targeted_fills = self._sample_targeted_external_fills(targeted_orders, eligible_agents)
        self.last_routing_targeted_volume = float(sum(float(fill["quantity"]) for fill in targeted_fills))
        self.last_routing_targeted_orders = len(targeted_fills)
        fills.extend(targeted_fills)
        return fills

    def _route_external_flow_continuous_reallocate(
        self,
        routing_flow_boost: float,
        cap_met: bool,
        eligible_agents: list[int],
        *,
        balanced: bool,
    ) -> list[dict[str, object]]:
        if routing_flow_boost > 0.0 and eligible_agents:
            targeted_share = float(np.clip(routing_flow_boost, 0.0, 1.0))
            normal_multiplier = 1.0 - targeted_share
            targeted_multiplier = targeted_share
            effective_multiplier = 1.0
        elif routing_flow_boost > 0.0:
            normal_multiplier = max(0.0, 1.0 - self.config.routing_withhold_rate * routing_flow_boost)
            targeted_multiplier = 0.0
            effective_multiplier = normal_multiplier
        else:
            normal_multiplier = 1.0
            targeted_multiplier = 0.0
            effective_multiplier = 1.0

        self.last_routing_multiplier = effective_multiplier
        self.last_routing_cap_met = bool(cap_met)
        base_intensity = self._flow_intensity()
        normal_orders = int(self.rng.poisson(base_intensity * normal_multiplier))
        if balanced:
            targeted_orders = int(self.rng.poisson(0.5 * base_intensity * targeted_multiplier))
            targeted_fills = self._sample_targeted_balanced_external_fills(targeted_orders, eligible_agents)
        else:
            targeted_orders = int(self.rng.poisson(base_intensity * targeted_multiplier))
            targeted_fills = self._sample_targeted_external_fills(targeted_orders, eligible_agents)
        fills = self._sample_book_external_orders(normal_orders)
        self.last_routing_targeted_volume = float(sum(float(fill["quantity"]) for fill in targeted_fills))
        self.last_routing_targeted_orders = len(targeted_fills)
        fills.extend(targeted_fills)
        return fills

    def _sample_book_external_orders(self, n_orders: int) -> list[dict[str, object]]:
        fills: list[dict[str, object]] = []
        mid = self._reference_mid()
        for _ in range(n_orders):
            side = self._sample_external_side()
            qty = int(self.rng.integers(1, 4))
            if side == "bid":
                price = mid + (max(self.config.half_spread_levels) + 3) * self.config.tick_size
            else:
                price = mid - (max(self.config.half_spread_levels) + 3) * self.config.tick_size
            _, order_fills = self.book.submit_order(-2, side, price, qty)
            fills.extend(order_fills)
        return fills

    def _sample_targeted_external_fills(self, n_orders: int, eligible_agents: Sequence[int]) -> list[dict[str, object]]:
        eligible_agents = self._targetable_agents(eligible_agents)
        self.last_routing_eligible_agents = len(eligible_agents)
        if n_orders <= 0 or not eligible_agents:
            return []
        fills: list[dict[str, object]] = []
        mid = self._reference_mid()
        for _ in range(n_orders):
            agent_id = int(self.rng.choice(eligible_agents, p=self._target_agent_probabilities(eligible_agents)))
            half_ticks = self._target_fill_half_spread(agent_id)
            if half_ticks is None:
                continue
            half = half_ticks * self.config.tick_size
            qty = int(self.rng.integers(1, 4))
            if self.rng.random() < 0.5:
                fills.append({
                    "price": round((mid + half) / self.config.tick_size) * self.config.tick_size,
                    "quantity": qty,
                    "aggressor_id": -2,
                    "aggressor_side": "bid",
                    "passive_id": agent_id,
                    "passive_side": "ask",
                })
            else:
                fills.append({
                    "price": round((mid - half) / self.config.tick_size) * self.config.tick_size,
                    "quantity": qty,
                    "aggressor_id": -2,
                    "aggressor_side": "ask",
                    "passive_id": agent_id,
                    "passive_side": "bid",
                })
        return fills

    def _sample_targeted_balanced_external_fills(
        self,
        n_pairs: int,
        eligible_agents: Sequence[int],
    ) -> list[dict[str, object]]:
        eligible_agents = self._targetable_agents(eligible_agents)
        self.last_routing_eligible_agents = len(eligible_agents)
        if n_pairs <= 0 or not eligible_agents:
            return []
        fills: list[dict[str, object]] = []
        mid = self._reference_mid()
        for _ in range(n_pairs):
            agent_id = int(self.rng.choice(eligible_agents, p=self._target_agent_probabilities(eligible_agents)))
            half_ticks = self._target_fill_half_spread(agent_id)
            if half_ticks is None:
                continue
            half = half_ticks * self.config.tick_size
            qty = int(self.rng.integers(1, 4))
            fills.append({
                "price": round((mid + half) / self.config.tick_size) * self.config.tick_size,
                "quantity": qty,
                "aggressor_id": -2,
                "aggressor_side": "bid",
                "passive_id": agent_id,
                "passive_side": "ask",
            })
            fills.append({
                "price": round((mid - half) / self.config.tick_size) * self.config.tick_size,
                "quantity": qty,
                "aggressor_id": -2,
                "aggressor_side": "ask",
                "passive_id": agent_id,
                "passive_side": "bid",
            })
        return fills

    def _flow_intensity(self) -> float:
        intensity = float(self.config.base_flow_intensity)
        elasticity = float(self.config.spread_elasticity)
        if elasticity <= 0.0:
            return intensity
        full_avg_spread = 2.0 * float(np.mean(self.last_agent_spreads))
        competitive_full_spread = 2.0 * float(min(self.config.half_spread_levels))
        spread_markup = max(0.0, full_avg_spread - competitive_full_spread)
        return float(intensity * np.exp(-elasticity * spread_markup))

    def _reference_mid(self) -> float:
        if self.config.risk_coupled_economics:
            return float(self.fundamental)
        return self.book.mid_price(self.fundamental)

    def _sample_external_side(self) -> str:
        informed_prob = float(np.clip(self.config.informed_flow_prob, 0.0, 1.0))
        if informed_prob > 0.0 and self.rng.random() < informed_prob and self.last_fundamental_shock != 0.0:
            return "bid" if self.last_fundamental_shock > 0.0 else "ask"
        return "bid" if self.rng.random() < 0.5 else "ask"

    def _targetable_agents(self, eligible_agents: Sequence[int]) -> list[int]:
        if self.config.targeted_fill_pricing != "competitive_benchmark":
            return [int(agent_id) for agent_id in eligible_agents]
        cap = float(self.config.targeted_competitive_half_spread_ticks)
        return [
            int(agent_id)
            for agent_id in eligible_agents
            if float(self.last_agent_spreads[int(agent_id)]) <= cap
        ]

    def _target_fill_half_spread(self, agent_id: int) -> float | None:
        own_half = float(self.last_agent_spreads[int(agent_id)])
        if self.config.targeted_fill_pricing == "competitive_benchmark":
            benchmark = float(self.config.targeted_competitive_half_spread_ticks)
            if own_half > benchmark:
                return None
            return benchmark
        return own_half

    def _target_agent_probabilities(self, eligible_agents: Sequence[int]) -> np.ndarray | None:
        kappa = float(self.config.routing_target_kappa)
        if kappa <= 0.0 or len(eligible_agents) <= 1:
            return None
        spreads = np.array([self.last_agent_spreads[int(agent_id)] for agent_id in eligible_agents], dtype=float)
        logits = -kappa * spreads
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        total = float(np.sum(weights))
        if total <= 0.0:
            return None
        return weights / total

    def _routing_eligible_agents(self) -> list[int]:
        half_spreads = np.asarray(self.last_agent_spreads, dtype=float)
        if self.config.routing_mode in {"relative_priority", "relative_reallocate"}:
            threshold = float(np.mean(half_spreads) - self.config.routing_relative_improvement_ticks)
            return [int(i) for i, spread in enumerate(half_spreads) if spread < threshold]

        cap = float(self.config.routing_spread_cap_ticks)
        if cap <= 0.0:
            return []
        full_spreads = 2.0 * half_spreads
        return [int(i) for i, spread in enumerate(full_spreads) if spread <= cap]

    def _process_fills(self, fills: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        self.last_spread_revenue = np.zeros(self.config.n_agents, dtype=float)
        self.last_adverse_selection_loss = np.zeros(self.config.n_agents, dtype=float)
        processed: list[dict[str, object]] = []
        for fill in fills:
            qty = float(fill["quantity"])
            price = float(fill["price"])
            buyer = int(fill["aggressor_id"]) if fill["aggressor_side"] == "bid" else int(fill["passive_id"])
            seller = int(fill["passive_id"]) if fill["aggressor_side"] == "bid" else int(fill["aggressor_id"])
            if self.config.enforce_max_inventory:
                max_qty = qty
                if 0 <= buyer < self.config.n_agents:
                    max_qty = min(max_qty, float(self.config.max_inventory) - float(self.inventories[buyer]))
                if 0 <= seller < self.config.n_agents:
                    max_qty = min(max_qty, float(self.config.max_inventory) + float(self.inventories[seller]))
                if max_qty <= 0.0:
                    continue
                if max_qty < qty:
                    fill = {**fill, "quantity": max_qty}
                    qty = max_qty
            if 0 <= buyer < self.config.n_agents:
                self.inventories[buyer] += qty
                self.cash[buyer] -= price * qty
            if 0 <= seller < self.config.n_agents:
                self.inventories[seller] -= qty
                self.cash[seller] += price * qty
            passive_id = int(fill["passive_id"])
            if 0 <= passive_id < self.config.n_agents:
                passive_side = str(fill["passive_side"])
                if passive_side == "ask":
                    spread_revenue = price - self._execution_mid
                    inventory_delta = -qty
                else:
                    spread_revenue = self._execution_mid - price
                    inventory_delta = qty
                self.last_spread_revenue[passive_id] += spread_revenue * qty
                adverse_pnl = inventory_delta * self.last_fundamental_shock
                self.last_adverse_selection_loss[passive_id] += max(0.0, -adverse_pnl)
            processed.append(fill)
        return processed

    def _wealth(self) -> np.ndarray:
        if self.config.risk_coupled_economics or self.config.mark_to_fundamental:
            mark = float(self.fundamental)
        else:
            mark = self.book.mid_price(self.fundamental)
        self.last_mark_price = mark
        return self.cash + self.inventories * mark

    def _advance_risk_state(self) -> None:
        if self.risk_state == "low" and self.rng.random() < self.config.p_low_to_high:
            self.risk_state = "high"
        elif self.risk_state == "high" and self.rng.random() < self.config.p_high_to_low:
            self.risk_state = "low"

    def _coerce_actions(self, actions: Dict[int, int] | Iterable[int]) -> np.ndarray:
        if isinstance(actions, dict):
            arr = np.array([actions[i] for i in range(self.config.n_agents)], dtype=int)
        else:
            arr = np.array(list(actions), dtype=int)
        if arr.shape != (self.config.n_agents,):
            raise ValueError(f"Expected {self.config.n_agents} actions, got {arr.shape}")
        return np.clip(arr, 0, self.config.n_actions - 1)
