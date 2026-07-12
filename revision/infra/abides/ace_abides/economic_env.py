"""Risk-coupled N=10 dealer market running on the real ABIDES exchange."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from abides_core import Agent, Kernel, Message, NanosecondTime
from abides_core.utils import str_to_ns, subdict
from abides_markets.agents import AdaptiveMarketMakerAgent, ExchangeAgent
from abides_markets.configs import rmsc04
from abides_markets.messages.marketdata import L2DataMsg, L2SubReqMsg
from abides_markets.utils import generate_latency_model

from .adapter import (
    CancelQuotesMsg,
    ControlledMarketMaker,
    CoreGymAgent,
    FlowInstructionMsg,
    MarketOrderTaker,
    QuoteInstructionMsg,
)
from .routing import RoutedOrderBook

RiskState = Literal["low", "high"]


@dataclass(frozen=True)
class ABIDESDealerMarketConfig:
    n_agents: int = 10
    half_spread_levels: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    episode_steps: int = 40
    decision_interval_ns: int = str_to_ns("1s")
    initial_fundamental: int = 100_000
    starting_cash: int = 100_000_000
    quote_quantity: int = 1_000
    flow_events_per_step: int = 20
    flow_quantity: int = 1
    informed_flow_probability: float = 0.80
    low_sigma_ticks: float = 0.75
    high_sigma_ticks: float = 4.00
    p_low_to_high: float = 0.04
    p_high_to_low: float = 0.10
    initial_high_probability: float = 0.30
    fixed_risk_state: Optional[RiskState] = None
    inventory_penalty: float = 0.002
    routing_routable_fraction: float = 0.50
    routing_excess_reference_half_spread: float = 4.0
    routing_excess_scale_ticks: float = 2.0
    routing_undercut_delta_ticks: float = 1.0
    seed: int = 42

    @property
    def n_actions(self) -> int:
        return len(self.half_spread_levels)


@dataclass(frozen=True)
class FlowEvent:
    side: str
    quantity: int
    base_agent_index: int
    route_uniform: float
    target_uniform: float


@dataclass(frozen=True)
class ExogenousStep:
    risk_state: RiskState
    shock_ticks: int
    flow_events: Tuple[FlowEvent, ...]


@dataclass
class SettlementMsg(Message):
    step_id: int
    risk_state: RiskState
    previous_fundamental: int
    next_fundamental: int


def generate_exogenous_path(
    config: ABIDESDealerMarketConfig, seed: int
) -> Tuple[ExogenousStep, ...]:
    rng = np.random.RandomState(seed)
    if config.fixed_risk_state is not None:
        risk: RiskState = config.fixed_risk_state
    else:
        risk = "high" if rng.rand() < config.initial_high_probability else "low"

    path: List[ExogenousStep] = []
    for _ in range(config.episode_steps):
        sigma = (
            config.high_sigma_ticks if risk == "high" else config.low_sigma_ticks
        )
        shock = int(np.rint(rng.normal(0.0, sigma)))
        events = []
        for _ in range(config.flow_events_per_step):
            informed = rng.rand() < config.informed_flow_probability
            if informed and shock != 0:
                side = "BUY" if shock > 0 else "SELL"
            else:
                side = "BUY" if rng.rand() < 0.5 else "SELL"
            events.append(
                FlowEvent(
                    side=side,
                    quantity=config.flow_quantity,
                    base_agent_index=int(rng.randint(0, config.n_agents)),
                    route_uniform=float(rng.rand()),
                    target_uniform=float(rng.rand()),
                )
            )
        path.append(
            ExogenousStep(
                risk_state=risk,
                shock_ticks=shock,
                flow_events=tuple(events),
            )
        )
        if config.fixed_risk_state is None:
            if risk == "low" and rng.rand() < config.p_low_to_high:
                risk = "high"
            elif risk == "high" and rng.rand() < config.p_high_to_low:
                risk = "low"
    return tuple(path)


def path_digest(path: Sequence[ExogenousStep]) -> str:
    payload = repr(
        [
            (
                step.risk_state,
                step.shock_ticks,
                [asdict(event) for event in step.flow_events],
            )
            for step in path
        ]
    ).encode("ascii")
    return sha256(payload).hexdigest()


class EconomicCoordinator(CoreGymAgent):
    """Pauses the kernel, routes fixed customer flow, and computes rewards."""

    def __init__(
        self,
        id: int,
        symbol: str,
        controlled_agent_ids: Sequence[int],
        taker_agent_id: int,
        decision_start: NanosecondTime,
        config: ABIDESDealerMarketConfig,
        path: Sequence[ExogenousStep],
    ) -> None:
        super().__init__(
            id=id,
            name="ACE_ECONOMIC_COORDINATOR",
            type="ACEEconomicCoordinator",
            random_state=np.random.RandomState(config.seed ^ 0xC001),
            log_events=False,
            log_to_file=False,
        )
        self.symbol = symbol
        self.controlled_agent_ids = list(controlled_agent_ids)
        self.taker_agent_id = taker_agent_id
        self.decision_start = decision_start
        self.config = config
        self.path = tuple(path)
        self.exchange_id: Optional[int] = None
        self.latest_l2: Optional[L2DataMsg] = None
        self.step_id = 0
        self.completed_steps = 0
        self.fundamental = config.initial_fundamental
        self.previous_fundamental = config.initial_fundamental
        self.previous_wealth = np.full(
            config.n_agents, float(config.starting_cash), dtype=float
        )
        self.last_actions = np.zeros(config.n_agents, dtype=int)
        self.last_half_spreads = np.full(
            config.n_agents, config.half_spread_levels[0], dtype=float
        )
        self.last_rewards = np.zeros(config.n_agents, dtype=float)
        self.execution_cursors = np.zeros(config.n_agents, dtype=int)
        self.last_flow_shares = np.full(config.n_agents, 1.0 / config.n_agents)
        self.last_step_info: Dict[str, Any] = {}
        self.raw_state: Dict[str, Any] = {}
        self.last_reward_completed_step = 0
        self.public_shock_history: List[int] = []

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
        elif isinstance(message, SettlementMsg):
            if message.step_id != self.completed_steps:
                raise RuntimeError("settlement step arrived out of order")
            self.previous_fundamental = message.previous_fundamental
            self.fundamental = message.next_fundamental
            self.public_shock_history.append(
                message.next_fundamental - message.previous_fundamental
            )
            self.last_step_info.update(
                {
                    "last_step_id": message.step_id,
                    "last_step_risk": message.risk_state,
                    "previous_fundamental": message.previous_fundamental,
                    "fundamental": message.next_fundamental,
                    "fundamental_shock": (
                        message.next_fundamental - message.previous_fundamental
                    ),
                }
            )
            self.completed_steps += 1

    def wakeup(self, current_time: NanosecondTime) -> Optional[Dict[str, Any]]:
        super().wakeup(current_time)
        if current_time < self.decision_start:
            self.set_wakeup(self.decision_start)
            return None
        self.update_raw_state()
        return self.get_raw_state()

    def apply_actions(self, actions: Dict[str, Any]) -> None:
        if self.step_id >= self.config.episode_steps:
            raise RuntimeError("episode is complete")
        action_indices = np.asarray(actions["actions"], dtype=int)
        if action_indices.shape != (self.config.n_agents,):
            raise ValueError("actions must contain one discrete action per agent")
        if np.any(action_indices < 0) or np.any(
            action_indices >= self.config.n_actions
        ):
            raise ValueError("action index outside half-spread grid")

        path_step = self.path[self.step_id]
        half_spreads = np.asarray(
            [self.config.half_spread_levels[index] for index in action_indices],
            dtype=float,
        )
        self.last_actions = action_indices.copy()
        self.last_half_spreads = half_spreads.copy()

        for agent_id in self.controlled_agent_ids:
            self.send_message(
                agent_id,
                CancelQuotesMsg(decision_id=self.step_id),
            )
        for agent_id, half_spread in zip(
            self.controlled_agent_ids, half_spreads
        ):
            self.send_message(
                agent_id,
                QuoteInstructionMsg(
                    decision_id=self.step_id,
                    reference_price=self.fundamental,
                    half_spread=int(half_spread),
                    quantity=self.config.quote_quantity,
                    cancel_existing=False,
                ),
                delay=str_to_ns("2ms"),
            )

        severity = float(
            np.clip(
                (
                    float(np.mean(half_spreads))
                    - self.config.routing_excess_reference_half_spread
                )
                / self.config.routing_excess_scale_ticks,
                0.0,
                1.0,
            )
        )
        gate_active = bool(actions.get("gate_active", False))
        routing_enabled = bool(actions.get("routing_enabled", True))
        effective_alpha = (
            self.config.routing_routable_fraction * severity
            if gate_active and routing_enabled
            else 0.0
        )
        eligible_indices = np.flatnonzero(
            half_spreads
            <= float(np.mean(half_spreads))
            - self.config.routing_undercut_delta_ticks
        )
        if eligible_indices.size:
            minimum = float(np.min(half_spreads[eligible_indices]))
            target_indices = eligible_indices[half_spreads[eligible_indices] == minimum]
        else:
            target_indices = np.asarray([], dtype=int)

        target_count = 0
        total_quantity = 0
        flow_start_delay = str_to_ns("10ms")
        flow_spacing = max(
            1,
            int(
                (self.config.decision_interval_ns - str_to_ns("20ms"))
                / max(1, len(path_step.flow_events))
            ),
        )
        for event_index, event in enumerate(path_step.flow_events):
            destination_index = event.base_agent_index
            route_kind = f"base_s{self.step_id}"
            if (
                target_indices.size
                and event.route_uniform < effective_alpha
            ):
                target_slot = min(
                    int(event.target_uniform * len(target_indices)),
                    len(target_indices) - 1,
                )
                destination_index = int(target_indices[target_slot])
                route_kind = f"targeted_s{self.step_id}"
                target_count += 1
            total_quantity += event.quantity
            self.send_message(
                self.taker_agent_id,
                FlowInstructionMsg(
                    decision_id=self.step_id,
                    side=event.side,
                    quantity=event.quantity,
                    eligible_agent_ids=(
                        self.controlled_agent_ids[destination_index],
                    ),
                    route_kind=route_kind,
                ),
                delay=flow_start_delay + event_index * flow_spacing,
            )

        next_fundamental = self.fundamental + path_step.shock_ticks
        self.send_message(
            self.id,
            SettlementMsg(
                step_id=self.step_id,
                risk_state=path_step.risk_state,
                previous_fundamental=self.fundamental,
                next_fundamental=next_fundamental,
            ),
            delay=self.config.decision_interval_ns - str_to_ns("2ms"),
        )
        self.last_step_info = {
            "gate_active": gate_active,
            "routing_enabled": routing_enabled,
            "routing_severity": severity,
            "routing_effective_alpha": effective_alpha,
            "eligible_agent_count": int(target_indices.size),
            "targeted_event_count": target_count,
            "planned_flow_quantity": total_quantity,
            "avg_half_spread": float(np.mean(half_spreads)),
        }
        self.step_id += 1
        self.set_wakeup(
            self.decision_start + self.step_id * self.config.decision_interval_ns
        )

    def update_raw_state(self) -> None:
        assert self.kernel is not None
        controlled = []
        wealth = np.zeros(self.config.n_agents, dtype=float)
        new_execution_quantity = np.zeros(self.config.n_agents, dtype=float)
        adverse_selection = np.zeros(self.config.n_agents, dtype=float)
        spread_revenue = np.zeros(self.config.n_agents, dtype=float)

        for index, agent_id in enumerate(self.controlled_agent_ids):
            agent = self.kernel.agents[agent_id]
            if not isinstance(agent, ControlledMarketMaker):
                raise TypeError("controlled agent type changed")
            inventory = agent.holdings.get(self.symbol, 0)
            cash = agent.holdings["CASH"]
            wealth[index] = cash + inventory * self.fundamental
            new_executions = agent.executions[self.execution_cursors[index] :]
            self.execution_cursors[index] = len(agent.executions)
            for execution in new_executions:
                quantity = float(execution.quantity)
                new_execution_quantity[index] += quantity
                if execution.side.is_bid():
                    spread_revenue[index] += quantity * (
                        self.previous_fundamental - execution.fill_price
                    )
                    adverse_selection[index] += quantity * (
                        self.fundamental - self.previous_fundamental
                    )
                else:
                    spread_revenue[index] += quantity * (
                        execution.fill_price - self.previous_fundamental
                    )
                    adverse_selection[index] += quantity * (
                        self.previous_fundamental - self.fundamental
                    )
            controlled.append(
                {
                    "agent_id": agent_id,
                    "cash": cash,
                    "inventory": inventory,
                    "active_order_count": len(agent.orders),
                    "execution_count": len(agent.executions),
                }
            )

        if self.completed_steps > self.last_reward_completed_step:
            inventory = np.asarray(
                [entry["inventory"] for entry in controlled], dtype=float
            )
            penalty = self.config.inventory_penalty * np.square(inventory)
            self.last_rewards = wealth - self.previous_wealth - penalty
            self.previous_wealth = wealth.copy()
            self.last_reward_completed_step = self.completed_steps
        else:
            penalty = np.zeros(self.config.n_agents, dtype=float)
            if self.completed_steps == 0:
                self.last_rewards = np.zeros(self.config.n_agents, dtype=float)

        total_execution_quantity = float(np.sum(new_execution_quantity))
        total_spread_revenue = float(np.sum(spread_revenue))
        if total_execution_quantity > 0:
            self.last_flow_shares = (
                new_execution_quantity / total_execution_quantity
            )

        public_volatility = float(
            np.std(self.public_shock_history[-10:])
            if len(self.public_shock_history) >= 2
            else 0.0
        )
        observations = []
        for index, entry in enumerate(controlled):
            observations.append(
                [
                    float(entry["inventory"]) / 100.0,
                    float(self.last_actions[index])
                    / max(1, self.config.n_actions - 1),
                    float(np.mean(self.last_half_spreads))
                    / max(self.config.half_spread_levels),
                    public_volatility / max(1.0, self.config.high_sigma_ticks),
                    float(self.last_flow_shares[index]),
                    float(self.public_shock_history[-1])
                    / max(1.0, self.config.high_sigma_ticks)
                    if self.public_shock_history
                    else 0.0,
                ]
            )

        info = dict(self.last_step_info)
        info.update(
            {
                "backend": "abides",
                "path_digest": path_digest(self.path),
                "step_id": self.step_id,
                "completed_steps": self.completed_steps,
                "fundamental": self.fundamental,
                "public_volatility": public_volatility,
                "realized_flow_quantity": total_execution_quantity,
                "mean_reward": float(np.mean(self.last_rewards)),
                "mean_spread_revenue": float(np.mean(spread_revenue)),
                "mean_execution_half_spread": (
                    total_spread_revenue / total_execution_quantity
                    if total_execution_quantity > 0
                    else 0.0
                ),
                "mean_adverse_selection_component": float(
                    np.mean(adverse_selection)
                ),
                "mean_inventory_penalty": float(np.mean(penalty)),
                "flow_shares": self.last_flow_shares.tolist(),
                "avg_half_spread": float(np.mean(self.last_half_spreads)),
            }
        )
        self.raw_state = {
            "backend": "abides",
            "observations": observations,
            "rewards": self.last_rewards.tolist(),
            "done": self.completed_steps >= self.config.episode_steps,
            "agents": controlled,
            "info": info,
        }

    def get_raw_state(self) -> Dict[str, Any]:
        return self.raw_state


def _reindex_agents(agents: Sequence[Agent]) -> List[Agent]:
    result = list(agents)
    for new_id, agent in enumerate(result):
        agent.id = new_id
    return result


def build_economic_config(
    config: ABIDESDealerMarketConfig,
    seed: int,
    symbol: str = "ABM",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if config.n_agents != 10:
        raise ValueError("the preregistered primary assay requires N=10")
    if config.episode_steps > 240:
        raise ValueError("episode_steps must fit inside the frozen five-minute session")

    base = rmsc04.build_config(
        seed=seed,
        end_time="09:35:00",
        ticker=symbol,
        starting_cash=config.starting_cash,
        log_orders=False,
        book_logging=False,
        exchange_log_orders=False,
        num_noise_agents=0,
        num_value_agents=0,
        num_momentum_agents=0,
    )
    background = [
        agent
        for agent in base["agents"]
        if not isinstance(agent, AdaptiveMarketMakerAgent)
    ]
    agents = _reindex_agents(background)
    exchange = agents[0]
    if not isinstance(exchange, ExchangeAgent):
        raise RuntimeError("real ABIDES ExchangeAgent is required")
    routed_book = RoutedOrderBook(exchange, symbol)
    routed_book.last_trade = exchange.order_books[symbol].last_trade
    exchange.order_books[symbol] = routed_book

    rng = np.random.RandomState(seed ^ 0xACE5)
    controlled: List[ControlledMarketMaker] = []
    for _ in range(config.n_agents):
        agent = ControlledMarketMaker(
            id=len(agents),
            symbol=symbol,
            starting_cash=config.starting_cash,
            random_state=np.random.RandomState(
                rng.randint(0, 2**32, dtype="uint32")
            ),
        )
        controlled.append(agent)
        agents.append(agent)
    taker = MarketOrderTaker(
        id=len(agents),
        symbol=symbol,
        starting_cash=config.starting_cash * 10,
        random_state=np.random.RandomState(rng.randint(0, 2**32, dtype="uint32")),
    )
    agents.append(taker)

    path = generate_exogenous_path(config, seed)
    coordinator = EconomicCoordinator(
        id=len(agents),
        symbol=symbol,
        controlled_agent_ids=[agent.id for agent in controlled],
        taker_agent_id=taker.id,
        decision_start=exchange.mkt_open + str_to_ns("1s"),
        config=config,
        path=path,
    )
    agents.append(coordinator)

    base["agents"] = agents
    base["agent_latency_model"] = generate_latency_model(
        len(agents), latency_type="no_latency"
    )
    base["default_computation_delay"] = 1
    base["stdout_log_level"] = "WARNING"
    return base, {
        "exchange": exchange,
        "controlled": controlled,
        "taker": taker,
        "coordinator": coordinator,
        "path": path,
        "symbol": symbol,
    }


class ABIDESDealerMarketEnv:
    """Small reset/step wrapper suitable for fixed policies and host MAPPO."""

    def __init__(self, config: ABIDESDealerMarketConfig):
        self.config = config
        self.kernel: Optional[Kernel] = None
        self.handles: Dict[str, Any] = {}
        self.closed = True

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if not self.closed:
            self.close()
        episode_seed = self.config.seed if seed is None else int(seed)
        kernel_config, self.handles = build_economic_config(
            self.config, episode_seed
        )
        self.kernel = Kernel(
            random_state=kernel_config["random_state_kernel"],
            **subdict(
                kernel_config,
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
        self.kernel.initialize()
        result = self.kernel.runner()
        if result["done"] or result["result"]["backend"] != "abides":
            raise RuntimeError("ABIDES environment failed before first decision")
        self.closed = False
        return result["result"]

    def step(
        self,
        actions: Sequence[int],
        gate_active: bool = False,
        routing_enabled: bool = True,
    ) -> Dict[str, Any]:
        if self.closed or self.kernel is None:
            raise RuntimeError("reset must be called before step")
        result = self.kernel.runner(
            (
                self.handles["coordinator"],
                {
                    "actions": list(actions),
                    "gate_active": gate_active,
                    "routing_enabled": routing_enabled,
                },
            )
        )
        if result["done"]:
            raise RuntimeError("kernel ended before coordinator returned the step")
        return result["result"]

    def close(self) -> None:
        if self.closed or self.kernel is None:
            return
        completion = self.kernel.runner()
        while completion["done"] is False:
            completion = self.kernel.runner()
        self.kernel.terminate()
        self.closed = True
