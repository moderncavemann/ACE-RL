#!/usr/bin/env python3
"""Step 2/3: simple baselines and prototype gated routing intervention."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.independent_q_learning import IndependentQLearners, QLearningConfig
from src.envs.toy_lob_pricing_env import ToyLOBConfig, ToyLOBPricingEnv
from src.interventions.gates import GateConfig, GateMode, InterventionGate
from src.interventions.routing_interventions import RoutingInterventionConfig, eta_boost_from_gate
from src.utils.metrics import enrich_step, gate_classification_metrics, summarize_by_risk, summarize_records


def load_gate_config(path: str | None) -> GateConfig | None:
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config_block = raw.get("gate_config", raw)
    valid_fields = {field.name for field in fields(GateConfig)}
    kwargs = {
        key: value
        for key, value in dict(config_block).items()
        if key in valid_fields
    }
    return GateConfig(**kwargs)


def run_gate_mode(
    config: ToyLOBConfig,
    label: str,
    gate_mode: GateMode,
    episodes: int,
    eval_episodes: int,
    seed: int,
    routing_rho: float = 0.75,
    gate_config: GateConfig | None = None,
) -> Dict[str, object]:
    env = ToyLOBPricingEnv(config)
    agents = IndependentQLearners(QLearningConfig(n_agents=config.n_agents, n_actions=config.n_actions, seed=seed))
    gate = InterventionGate(config, gate_config or GateConfig(seed=seed + 7), mode=gate_mode)
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
        "seed": seed,
        "routing_rho": routing_rho,
        "eval_all": summarize_records(records),
        "eval_by_risk": summarize_by_risk(records),
        "gate_metrics": gate_classification_metrics(records),
    }


def run_method(
    config: ToyLOBConfig,
    method: str,
    episodes: int,
    eval_episodes: int,
    seed: int,
    gate_config: GateConfig | None = None,
) -> Dict[str, object]:
    gate_mode = {
        "no_intervention": "none",
        "always_on": "always",
        "threshold": "threshold",
        "oracle_gate": "oracle",
        "random_gate": "random",
        "gated_intervention": "risk_aware",
    }[method]
    return run_gate_mode(config, method, gate_mode, episodes, eval_episodes, seed, gate_config=gate_config)


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _run_method_job(job: Tuple[ToyLOBConfig, str, int, int, int, GateConfig | None]) -> Dict[str, object]:
    config, method, episodes, eval_episodes, seed, gate_config = job
    return run_method(config, method, episodes, eval_episodes, seed, gate_config=gate_config)


def run_method_grid(
    config: ToyLOBConfig,
    method: str,
    episodes: int,
    eval_episodes: int,
    seeds: List[int],
    gate_config: GateConfig | None,
    jobs: int,
) -> List[Dict[str, object]]:
    method_jobs = [
        (config, method, episodes, eval_episodes, seed, gate_config)
        for seed in seeds
    ]
    if jobs <= 1 or len(method_jobs) <= 1:
        return [_run_method_job(job) for job in method_jobs]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(_run_method_job, method_jobs))


def run_methods_grid(
    config: ToyLOBConfig,
    methods: List[str],
    episodes: int,
    eval_episodes: int,
    seeds: List[int],
    gate_config: GateConfig | None,
    jobs: int,
) -> Dict[str, List[Dict[str, object]]]:
    method_jobs = [
        (config, method, episodes, eval_episodes, seed, gate_config)
        for method in methods
        for seed in seeds
    ]
    grouped: Dict[str, List[Dict[str, object]]] = {method: [] for method in methods}
    if jobs <= 1 or len(method_jobs) <= 1:
        results = [_run_method_job(job) for job in method_jobs]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(_run_method_job, method_jobs))
    for result in results:
        grouped[str(result["method"])].append(result)
    return grouped


def mean_se(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "se": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
    }


def aggregate_seed_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"n_seeds": float(len(results))}
    for block_name in ("eval_all", "gate_metrics"):
        keys = sorted({key for result in results for key in dict(result[block_name]).keys()})  # type: ignore[arg-type]
        out[block_name] = {
            key: mean_se(float(dict(result[block_name]).get(key, 0.0)) for result in results)  # type: ignore[arg-type]
            for key in keys
        }
    risk_states = sorted({
        risk
        for result in results
        for risk in dict(result["eval_by_risk"]).keys()  # type: ignore[arg-type]
    })
    eval_by_risk_out: Dict[str, object] = {}
    out["eval_by_risk"] = eval_by_risk_out
    for risk in risk_states:
        risk_rows = [dict(result["eval_by_risk"]).get(risk, {}) for result in results]  # type: ignore[arg-type]
        keys = sorted({key for row in risk_rows for key in dict(row).keys()})
        eval_by_risk_out[risk] = {
            key: mean_se(float(dict(row).get(key, 0.0)) for row in risk_rows)
            for key in keys
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--random-rate", default="auto", help="'auto' matches gated activation rate, or pass a float.")
    parser.add_argument("--gate-config", default=None, help="Optional JSON output from calibrate_toy_lob_gate.py.")
    parser.add_argument("--no-oracle", action="store_true", help="Skip diagnostic oracle_gate baseline.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes across seeds.")
    parser.add_argument("--output", default="revision/experiments/outputs/baselines/summary.json")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    config = ToyLOBConfig(seed=seeds[0] if seeds else 0, risk_mode="mixed")
    gate_config = load_gate_config(args.gate_config)
    summary: Dict[str, object] = {
        "metadata": {
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "seeds": seeds,
            "random_rate": args.random_rate,
            "jobs": args.jobs,
            "parallelism": "method_seed",
            "gate_config_path": args.gate_config,
            "gate_config": vars(gate_config) if gate_config is not None else None,
        },
        "methods": {},
    }
    non_random_methods = ["no_intervention", "always_on", "threshold", "gated_intervention"]
    if not args.no_oracle:
        non_random_methods.append("oracle_gate")
    method_results = run_methods_grid(
        config,
        non_random_methods,
        args.episodes,
        args.eval_episodes,
        seeds,
        gate_config,
        args.jobs,
    )

    gated_rates = [
        float(result["gate_metrics"]["gate_activation_rate"])  # type: ignore[index]
        for result in method_results["gated_intervention"]
    ]
    matched_rate = float(np.mean(gated_rates)) if args.random_rate == "auto" else float(args.random_rate)
    matched_rate = float(np.clip(matched_rate, 0.0, 1.0))
    summary["metadata"]["matched_random_rate"] = matched_rate  # type: ignore[index]
    random_gate_config = gate_config or GateConfig()
    random_gate_config = GateConfig(**{**vars(random_gate_config), "random_rate": matched_rate})
    method_results["random_gate"] = run_method_grid(
        config,
        "random_gate",
        args.episodes,
        args.eval_episodes,
        seeds,
        random_gate_config,
        args.jobs,
    )

    ordered_methods = ["no_intervention", "always_on", "threshold", "random_gate", "gated_intervention"]
    if not args.no_oracle:
        ordered_methods.append("oracle_gate")
    methods_out: Dict[str, object] = {}
    summary["methods"] = methods_out
    for method in ordered_methods:
        methods_out[method] = {
            "seed_results": method_results[method],
            "aggregate": aggregate_seed_results(method_results[method]),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved baseline summary to {output}")


if __name__ == "__main__":
    main()
