#!/usr/bin/env python3
"""Step 2/3: simple baselines and prototype gated routing intervention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.independent_q_learning import IndependentQLearners, QLearningConfig
from src.envs.toy_lob_pricing_env import ToyLOBConfig, ToyLOBPricingEnv
from src.interventions.gates import GateConfig, GateMode, InterventionGate
from src.interventions.routing_interventions import RoutingInterventionConfig, eta_boost_from_gate
from src.utils.metrics import enrich_step, summarize_by_risk, summarize_records


def run_gate_mode(
    config: ToyLOBConfig,
    label: str,
    gate_mode: GateMode,
    episodes: int,
    eval_episodes: int,
    seed: int,
    routing_rho: float = 0.75,
) -> Dict[str, object]:
    env = ToyLOBPricingEnv(config)
    agents = IndependentQLearners(QLearningConfig(n_agents=config.n_agents, n_actions=config.n_actions, seed=seed))
    gate = InterventionGate(config, GateConfig(seed=seed + 7), mode=gate_mode)
    routing = RoutingInterventionConfig(rho=routing_rho)

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        gate.reset()
        done = False
        while not done:
            states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
            actions = {i: agents.act(i, states[i], explore=True) for i in range(config.n_agents)}
            spreads = [config.spread_levels[actions[i]] for i in range(config.n_agents)]
            decision = gate.decide(env.risk_state, spreads)
            boost = eta_boost_from_gate(bool(decision["active"]), routing)
            _, rewards, done, _, _ = env.step(
                actions,
                routing_eta_boost=boost,
                gate_active=bool(decision["active"]),
                gate_score=float(decision["score"]),
            )
            for i in range(config.n_agents):
                agents.update(i, states[i], actions[i], rewards[i], env.state_for_agent(i), done)
            agents.step_clock()

    records: List[Dict[str, object]] = []
    for ep in range(eval_episodes):
        env.reset(seed=seed + 20_000 + ep)
        gate.reset()
        done = False
        while not done:
            states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
            actions = {i: agents.act(i, states[i], explore=False) for i in range(config.n_agents)}
            spreads = [config.spread_levels[actions[i]] for i in range(config.n_agents)]
            decision = gate.decide(env.risk_state, spreads)
            boost = eta_boost_from_gate(bool(decision["active"]), routing)
            _, _, done, _, info = env.step(
                actions,
                routing_eta_boost=boost,
                gate_active=bool(decision["active"]),
                gate_score=float(decision["score"]),
            )
            row = enrich_step(config, info)
            row["method"] = label
            records.append(row)
    return {
        "method": label,
        "gate_mode": gate_mode,
        "routing_rho": routing_rho,
        "eval_all": summarize_records(records),
        "eval_by_risk": summarize_by_risk(records),
    }


def run_method(config: ToyLOBConfig, method: str, episodes: int, eval_episodes: int, seed: int) -> Dict[str, object]:
    gate_mode = {
        "no_intervention": "none",
        "always_on": "always",
        "threshold": "threshold",
        "random_gate": "random",
        "gated_intervention": "risk_aware",
    }[method]
    return run_gate_mode(config, method, gate_mode, episodes, eval_episodes, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="revision/experiments/outputs/baselines/summary.json")
    args = parser.parse_args()

    config = ToyLOBConfig(seed=args.seed, risk_mode="mixed")
    methods = ["no_intervention", "always_on", "threshold", "random_gate", "gated_intervention"]
    summary = {
        method: run_method(config, method, args.episodes, args.eval_episodes, args.seed + idx * 1000)
        for idx, method in enumerate(methods)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved baseline summary to {output}")


if __name__ == "__main__":
    main()
