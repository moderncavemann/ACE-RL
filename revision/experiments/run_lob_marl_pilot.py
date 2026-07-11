#!/usr/bin/env python3
"""Local matching-engine LOB/MARL pilot for gated routing logic.

This is not the full LOB/MAPPO final experiment. It is a pilot whose job is to
verify that the toy-validated gate/routing logic has a working measurement and
intervention path in a local price-time priority LOB.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.independent_q_learning import IndependentQLearners, QLearningConfig
from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv
from src.interventions.lob_gates import LOBGateConfig, LOBGateMode, LOBInterventionGate


METHODS: Dict[str, Tuple[LOBGateMode, float]] = {
    "no_intervention": ("none", 0.0),
    "always_on": ("always", 0.75),
    "threshold": ("threshold", 0.75),
    "gated_intervention": ("risk_aware", 0.75),
    "random_gate": ("random", 0.75),
}


def load_lob_gate_config(path: str | None) -> LOBGateConfig:
    if not path:
        return LOBGateConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    block = raw.get("gate_config", raw)
    valid = set(LOBGateConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    return LOBGateConfig(**{key: value for key, value in dict(block).items() if key in valid})


def parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in [x.strip() for x in raw.split(",") if x.strip()]:
        if "-" in part:
            lo, hi = [int(x.strip()) for x in part.split("-", 1)]
            step = 1 if hi >= lo else -1
            seeds.extend(range(lo, hi + step, step))
        else:
            seeds.append(int(part))
    return seeds


def mean_se(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "se": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
    }


def summarize_records(records: List[Dict[str, object]]) -> Dict[str, float]:
    if not records:
        return {}
    fields = [
        "avg_agent_spread",
        "best_spread_ticks",
        "total_volume",
        "mean_reward",
        "mean_abs_inventory",
        "gate_active",
        "gate_score",
        "gate_excess",
        "gate_spread_entropy",
        "gate_flow_concentration",
        "gate_volume_suppression",
        "gate_raw_active",
        "gate_controlled_recent_rate",
        "gate_control_budget_open",
        "gate_control_cooldown_open",
        "routing_flow_boost",
        "routing_effective_multiplier",
        "routing_spread_cap_met",
        "routing_targeted_volume",
        "routing_targeted_orders",
        "routing_eligible_agents",
        "routing_contract_severity",
        "routing_contract_alpha",
        "routing_effective_alpha",
    ]
    out: Dict[str, float] = {"n_steps": float(len(records))}
    for field in fields:
        vals = [float(row[field]) for row in records if field in row]
        if vals:
            out[f"{field}_mean"] = float(np.mean(vals))
            out[f"{field}_std"] = float(np.std(vals))
    return out


def summarize_by_risk(records: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for risk in ("low", "high"):
        scoped = [row for row in records if row.get("risk_state") == risk]
        if scoped:
            out[risk] = summarize_records(scoped)
    return out


def gate_metrics(records: List[Dict[str, object]], high_spread_tau: float = 4.0) -> Dict[str, float]:
    active = [float(row.get("gate_active", 0.0)) for row in records]
    high_spread = [
        row for row in records
        if row.get("risk_state") == "low" and float(row.get("avg_agent_spread", 0.0)) > high_spread_tau
    ]
    high_risk_normal = [
        row for row in records
        if row.get("risk_state") == "high" and float(row.get("avg_agent_spread", 0.0)) <= high_spread_tau
    ]

    def rate(rows: List[Dict[str, object]]) -> float:
        return float(np.mean([float(row.get("gate_active", 0.0)) for row in rows])) if rows else 0.0

    return {
        "gate_activation_rate": float(np.mean(active)) if active else 0.0,
        "tpr_low_high_spread": rate(high_spread),
        "tpr_low_high_spread_n": float(len(high_spread)),
        "fpr_high_normal_spread": rate(high_risk_normal),
        "fpr_high_normal_spread_n": float(len(high_risk_normal)),
    }


def aggregate_seed_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"n_seeds": float(len(results))}
    for block_name in ("eval_all", "gate_metrics"):
        keys = sorted({key for result in results for key in dict(result[block_name]).keys()})  # type: ignore[arg-type]
        out[block_name] = {
            key: mean_se(float(dict(result[block_name]).get(key, 0.0)) for result in results)  # type: ignore[arg-type]
            for key in keys
        }
    risks = sorted({risk for result in results for risk in dict(result["eval_by_risk"]).keys()})  # type: ignore[arg-type]
    by_risk: Dict[str, object] = {}
    out["eval_by_risk"] = by_risk
    for risk in risks:
        risk_rows = [dict(result["eval_by_risk"]).get(risk, {}) for result in results]  # type: ignore[arg-type]
        keys = sorted({key for row in risk_rows for key in dict(row).keys()})
        by_risk[risk] = {
            key: mean_se(float(dict(row).get(key, 0.0)) for row in risk_rows)
            for key in keys
        }
    return out


def run_method_seed(
    config: LocalLOBConfig,
    method: str,
    gate_mode: LOBGateMode,
    routing_rho: float,
    episodes: int,
    eval_episodes: int,
    seed: int,
    gate_config: LOBGateConfig,
) -> Dict[str, object]:
    env = LocalLOBMarketEnv(config)
    agents = IndependentQLearners(
        QLearningConfig(
            n_agents=config.n_agents,
            n_actions=config.n_actions,
            seed=seed,
            epsilon_decay_steps=max(1, episodes * config.episode_length),
        )
    )
    gate = LOBInterventionGate(config, gate_config, gate_mode)

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        gate.reset()
        done = False
        while not done:
            states = {i: env.state_for_agent(i) for i in range(config.n_agents)}
            actions = {i: agents.act(i, states[i], explore=True) for i in range(config.n_agents)}
            spreads = [config.half_spread_levels[actions[i]] for i in range(config.n_agents)]
            decision = gate.decide(env.risk_state, spreads, env.last_flow_shares, env.last_total_volume)
            boost = routing_rho if bool(decision["active"]) else 0.0
            _, rewards, done, _, _ = env.step(
                actions,
                routing_flow_boost=boost,
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
            spreads = [config.half_spread_levels[actions[i]] for i in range(config.n_agents)]
            decision = gate.decide(env.risk_state, spreads, env.last_flow_shares, env.last_total_volume)
            boost = routing_rho if bool(decision["active"]) else 0.0
            _, _, done, _, info = env.step(
                actions,
                routing_flow_boost=boost,
                gate_active=bool(decision["active"]),
                gate_score=float(decision["score"]),
            )
            row = {
                key: value for key, value in info.items()
                if isinstance(value, (str, int, float, bool, np.integer, np.floating))
            }
            row["method"] = method
            row["seed"] = seed
            row["gate_excess"] = float(decision["excess"])
            row["gate_spread_entropy"] = float(decision["spread_entropy"])
            row["gate_flow_concentration"] = float(decision["flow_concentration"])
            row["gate_volume_suppression"] = float(decision["volume_suppression"])
            records.append(row)

    return {
        "method": method,
        "gate_mode": gate_mode,
        "seed": seed,
        "routing_rho": routing_rho,
        "eval_all": summarize_records(records),
        "eval_by_risk": summarize_by_risk(records),
        "gate_metrics": gate_metrics(records),
    }


def _run_job(job: Tuple[LocalLOBConfig, str, LOBGateMode, float, int, int, int, LOBGateConfig]) -> Dict[str, object]:
    return run_method_seed(*job)


def run_grid(
    config: LocalLOBConfig,
    methods: List[str],
    episodes: int,
    eval_episodes: int,
    seeds: List[int],
    gate_config: LOBGateConfig,
    jobs: int,
) -> Dict[str, List[Dict[str, object]]]:
    job_items = [
        (
            replace(config, seed=seed),
            method,
            METHODS[method][0],
            METHODS[method][1],
            episodes,
            eval_episodes,
            seed,
            gate_config,
        )
        for method in methods
        for seed in seeds
    ]
    if jobs <= 1 or len(job_items) <= 1:
        results = [_run_job(job) for job in job_items]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(_run_job, job_items))
    grouped: Dict[str, List[Dict[str, object]]] = {method: [] for method in methods}
    for result in results:
        grouped[str(result["method"])].append(result)
    return grouped


def paired_delta(reference: List[Dict[str, object]], target: List[Dict[str, object]], metric: str) -> Dict[str, float]:
    ref_by_seed = {int(row["seed"]): float(dict(row["eval_all"]).get(metric, 0.0)) for row in reference}
    tgt_by_seed = {int(row["seed"]): float(dict(row["eval_all"]).get(metric, 0.0)) for row in target}
    seeds = sorted(set(ref_by_seed) & set(tgt_by_seed))
    if not seeds:
        return {"n": 0.0}
    deltas = np.array([ref_by_seed[seed] - tgt_by_seed[seed] for seed in seeds], dtype=float)
    return {
        "n": float(len(deltas)),
        "positive": float(np.sum(deltas > 0.0)),
        "mean": float(np.mean(deltas)),
        "se": float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0,
    }


def metric_mean(aggregate: Dict[str, object], block: str, metric: str) -> float:
    value = dict(aggregate.get(block, {})).get(metric, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def write_metrics_csv(path: Path, summary: Dict[str, object]) -> None:
    rows: List[Dict[str, object]] = []
    for method, method_block in dict(summary["methods"]).items():
        aggregate = dict(dict(method_block)["aggregate"])
        for block, metric in [
            ("eval_all", "avg_agent_spread_mean"),
            ("eval_all", "best_spread_ticks_mean"),
            ("eval_all", "total_volume_mean"),
            ("eval_all", "mean_reward_mean"),
            ("eval_all", "mean_abs_inventory_mean"),
            ("eval_all", "gate_active_mean"),
            ("gate_metrics", "gate_activation_rate"),
            ("gate_metrics", "tpr_low_high_spread"),
            ("gate_metrics", "fpr_high_normal_spread"),
        ]:
            rows.append({
                "method": method,
                "block": block,
                "metric": metric,
                "mean": metric_mean(aggregate, block, metric),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "block", "metric", "mean"])
        writer.writeheader()
        writer.writerows(rows)


def pass_diagnostics(summary: Dict[str, object]) -> Dict[str, object]:
    methods = dict(summary["methods"])

    def mean(method: str, block: str, metric: str) -> float:
        if method not in methods:
            return 0.0
        return metric_mean(dict(dict(methods[method])["aggregate"]), block, metric)

    no_gap = mean("no_intervention", "eval_all", "avg_agent_spread_mean")
    gated_active = mean("gated_intervention", "gate_metrics", "gate_activation_rate")
    random_active = mean("random_gate", "gate_metrics", "gate_activation_rate")
    always_active = mean("always_on", "gate_metrics", "gate_activation_rate")
    gated_volume = mean("gated_intervention", "eval_all", "total_volume_mean")
    no_volume = mean("no_intervention", "eval_all", "total_volume_mean")
    gated_reward = mean("gated_intervention", "eval_all", "mean_reward_mean")
    no_reward = mean("no_intervention", "eval_all", "mean_reward_mean")

    return {
        "no_intervention_spread_elevated": bool(no_gap > 2.0),
        "gate_in_target_activation_range": bool(0.05 <= gated_active <= 0.40),
        "gate_nonzero": bool(gated_active > 0.0),
        "gate_overactive": bool(gated_active > 0.40),
        "always_on_saturated": bool(always_active >= 0.95),
        "matched_random_runs": bool(random_active > 0.0),
        "routing_changes_volume": bool(abs(gated_volume - no_volume) > 1e-6),
        "reward_not_obviously_collapsed": bool(gated_reward > no_reward - 5.0),
        "observed": {
            "no_intervention_avg_agent_spread": no_gap,
            "gated_activation_rate": gated_active,
            "random_activation_rate": random_active,
            "always_activation_rate": always_active,
            "no_intervention_volume": no_volume,
            "gated_volume": gated_volume,
            "no_intervention_reward": no_reward,
            "gated_reward": gated_reward,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default="no_intervention,always_on,threshold,gated_intervention")
    parser.add_argument("--random-rate", default="auto")
    parser.add_argument("--gate-config", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", default="revision/experiments/outputs/lob_marl_pilot_v1/summary.json")
    parser.add_argument("--metrics-csv", default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    for method in methods:
        if method not in METHODS or method == "random_gate":
            raise SystemExit(f"Unknown or invalid non-random method '{method}'")

    config = LocalLOBConfig(n_agents=args.n_agents, episode_length=args.episode_length, seed=seeds[0] if seeds else 0)
    gate_config = load_lob_gate_config(args.gate_config)
    grouped = run_grid(config, methods, args.episodes, args.eval_episodes, seeds, gate_config, args.jobs)

    gated_rates = [
        float(result["gate_metrics"]["gate_activation_rate"])  # type: ignore[index]
        for result in grouped.get("gated_intervention", [])
    ]
    matched_rate = float(np.mean(gated_rates)) if args.random_rate == "auto" and gated_rates else float(args.random_rate)
    matched_rate = float(np.clip(matched_rate, 0.0, 1.0))
    random_gate_config = LOBGateConfig(**{**vars(gate_config), "random_rate": matched_rate})
    grouped["random_gate"] = run_grid(
        config,
        ["random_gate"],
        args.episodes,
        args.eval_episodes,
        seeds,
        random_gate_config,
        args.jobs,
    )["random_gate"]

    ordered_methods = ["no_intervention", "always_on", "threshold", "random_gate", "gated_intervention"]
    methods_out: Dict[str, object] = {}
    for method in ordered_methods:
        if method not in grouped:
            continue
        methods_out[method] = {
            "seed_results": grouped[method],
            "aggregate": aggregate_seed_results(grouped[method]),
        }
    summary: Dict[str, object] = {
        "metadata": {
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "episode_length": args.episode_length,
            "seeds": seeds,
            "jobs": args.jobs,
            "matched_random_rate": matched_rate,
            "config": asdict(config),
            "gate_config": asdict(gate_config),
            "gate_config_path": args.gate_config,
            "stage": "local_lob_marl_pilot_not_full_mappo_final",
        },
        "methods": methods_out,
        "paired_random_minus_gated_spread": paired_delta(grouped["random_gate"], grouped.get("gated_intervention", []), "avg_agent_spread_mean"),
    }
    summary["pilot_diagnostics"] = pass_diagnostics(summary)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else output.with_name("metrics.csv")
    write_metrics_csv(metrics_csv, summary)
    print(json.dumps({
        "output": str(output),
        "metrics_csv": str(metrics_csv),
        "matched_random_rate": matched_rate,
        "paired_random_minus_gated_spread": summary["paired_random_minus_gated_spread"],
        "pilot_diagnostics": summary["pilot_diagnostics"],
    }, indent=2))


if __name__ == "__main__":
    main()
