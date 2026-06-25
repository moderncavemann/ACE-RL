#!/usr/bin/env python3
"""Controlled toy LOB robustness extensions after confirmation sweep."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_toy_lob_baselines import aggregate_seed_results, load_gate_config, parse_seeds, run_gate_mode
from src.envs.toy_lob_pricing_env import ToyLOBConfig
from src.interventions.gates import GateConfig, GateMode


SCENARIOS: Dict[str, Tuple[str, Dict[str, object]]] = {
    "base_n10": ("Baseline N=10 mixed-risk environment.", {}),
    "n5": ("Smaller market with five agents.", {"n_agents": 5}),
    "n20": ("Larger market with twenty agents.", {"n_agents": 20}),
    "cost_wide": ("Larger risk-cost spread.", {"c_low": 0.25, "c_high": 3.25}),
    "cost_narrow": ("Smaller risk-cost spread.", {"c_low": 0.75, "c_high": 2.00}),
    "risk_fast": ("Faster risk-state transitions.", {"p_low_to_high": 0.12, "p_high_to_low": 0.20}),
    "risk_slow": ("More persistent risk states.", {"p_low_to_high": 0.02, "p_high_to_low": 0.04}),
    "long_horizon": ("Longer episode horizon.", {"episode_length": 300}),
    "demand_steep": ("More price-sensitive demand.", {"demand_alpha": 0.28}),
    "demand_flat": ("Less price-sensitive demand.", {"demand_alpha": 0.14}),
}

METHODS: Dict[str, Tuple[GateMode, float]] = {
    "no_intervention": ("none", 0.75),
    "always_on": ("always", 0.75),
    "threshold": ("threshold", 0.75),
    "gated_intervention": ("risk_aware", 0.75),
    "spread_only_gate": ("spread_only", 0.75),
    "oracle_gate": ("oracle", 0.75),
    "no_risk_adjustment": ("no_risk_adjustment", 0.75),
    "no_routing_intervention": ("risk_aware", 0.0),
}


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def config_for_scenario(name: str, seed: int) -> Tuple[str, ToyLOBConfig]:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Valid scenarios: {', '.join(sorted(SCENARIOS))}")
    _, updates = SCENARIOS[name]
    config = replace(ToyLOBConfig(seed=seed, risk_mode="mixed"), **updates)
    return name, config


def _run_job(
    job: Tuple[str, ToyLOBConfig, str, GateMode, float, int, int, int, GateConfig | None],
) -> Tuple[str, str, Dict[str, object]]:
    scenario, config, method, gate_mode, routing_rho, episodes, eval_episodes, seed, gate_config = job
    result = run_gate_mode(
        config,
        method,
        gate_mode,
        episodes,
        eval_episodes,
        seed,
        routing_rho=routing_rho,
        gate_config=gate_config,
    )
    result["scenario"] = scenario
    return scenario, method, result


def run_jobs(
    jobs: List[Tuple[str, ToyLOBConfig, str, GateMode, float, int, int, int, GateConfig | None]],
    workers: int,
) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    if workers <= 1 or len(jobs) <= 1:
        results = [_run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_job, jobs))
    for scenario, method, result in results:
        grouped.setdefault(scenario, {}).setdefault(method, []).append(result)
    return grouped


def paired_delta(random_results: List[Dict[str, object]], gated_results: List[Dict[str, object]]) -> Dict[str, float]:
    random_by_seed = {
        int(row["seed"]): float(dict(row["eval_all"])["collusion_gap_mean"])
        for row in random_results
    }
    gated_by_seed = {
        int(row["seed"]): float(dict(row["eval_all"])["collusion_gap_mean"])
        for row in gated_results
    }
    seeds = sorted(set(random_by_seed) & set(gated_by_seed))
    deltas = np.array([random_by_seed[seed] - gated_by_seed[seed] for seed in seeds], dtype=float)
    if len(deltas) == 0:
        return {"n": 0.0}
    return {
        "n": float(len(deltas)),
        "positive": float(np.sum(deltas > 0)),
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "se": float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0,
    }


def metric_mean(aggregate: Dict[str, object], block: str, metric: str, risk: str | None = None) -> float:
    source = dict(aggregate.get(block, {}))
    if risk is not None:
        source = dict(source.get(risk, {}))
    value = source.get(metric, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def write_metrics_csv(path: Path, summary: Dict[str, object]) -> None:
    rows: List[Dict[str, object]] = []
    for scenario, scenario_block in dict(summary["scenarios"]).items():
        methods = dict(dict(scenario_block)["methods"])
        for method, method_block in methods.items():
            aggregate = dict(dict(method_block)["aggregate"])
            for metric in (
                "collusion_gap_mean",
                "demand_suppression_mean",
                "avg_spread_mean",
                "alignment_mean",
                "total_demand_mean",
                "mean_reward_mean",
                "gate_active_mean",
            ):
                rows.append({
                    "scenario": scenario,
                    "method": method,
                    "block": "eval_all",
                    "risk": "all",
                    "metric": metric,
                    "mean": metric_mean(aggregate, "eval_all", metric),
                })
            for risk in ("low", "high"):
                rows.append({
                    "scenario": scenario,
                    "method": method,
                    "block": "eval_by_risk",
                    "risk": risk,
                    "metric": "collusion_gap_mean",
                    "mean": metric_mean(aggregate, "eval_by_risk", "collusion_gap_mean", risk),
                })
            for metric in ("gate_activation_rate", "fpr_high_normal", "tpr_low_collusive"):
                rows.append({
                    "scenario": scenario,
                    "method": method,
                    "block": "gate_metrics",
                    "risk": "all",
                    "metric": metric,
                    "mean": metric_mean(aggregate, "gate_metrics", metric),
                })
        paired = dict(dict(scenario_block).get("paired_random_minus_gated_gap", {}))
        if paired:
            rows.append({
                "scenario": scenario,
                "method": "random_minus_gated",
                "block": "paired",
                "risk": "all",
                "metric": "collusion_gap_delta_mean",
                "mean": paired.get("mean", 0.0),
            })
            rows.append({
                "scenario": scenario,
                "method": "random_minus_gated",
                "block": "paired",
                "risk": "all",
                "metric": "positive_seed_count",
                "mean": paired.get("positive", 0.0),
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "method", "block", "risk", "metric", "mean"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seeds", default="35,36,37,38,39,40,41,42,43,44")
    parser.add_argument("--scenarios", default="base_n10,n5,n20,cost_wide,risk_fast,long_horizon")
    parser.add_argument("--methods", default="no_intervention,always_on,threshold,gated_intervention")
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--include-controls", action="store_true", help="Add no_risk_adjustment and no_routing_intervention controls.")
    parser.add_argument("--random-rate", default="auto", help="'auto' matches scenario-specific gated activation rate, or pass a float.")
    parser.add_argument("--gate-config", default="revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", default="revision/experiments/outputs/robustness_sweep_v1/summary.json")
    parser.add_argument("--metrics-csv", default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    scenarios = parse_csv(args.scenarios)
    methods = parse_csv(args.methods)
    if args.include_oracle and "oracle_gate" not in methods:
        methods.append("oracle_gate")
    if args.include_controls:
        for method in ("no_risk_adjustment", "no_routing_intervention"):
            if method not in methods:
                methods.append(method)
    for method in methods:
        if method not in METHODS:
            raise SystemExit(f"Unknown method '{method}'. Valid non-random methods: {', '.join(sorted(METHODS))}")

    gate_config = load_gate_config(args.gate_config)
    scenario_configs = {
        scenario: config_for_scenario(scenario, seeds[0] if seeds else 0)[1]
        for scenario in scenarios
    }

    non_random_jobs = [
        (
            scenario,
            config,
            method,
            METHODS[method][0],
            METHODS[method][1],
            args.episodes,
            args.eval_episodes,
            seed,
            gate_config,
        )
        for scenario, config in scenario_configs.items()
        for method in methods
        for seed in seeds
    ]
    grouped = run_jobs(non_random_jobs, args.jobs)

    random_jobs = []
    matched_rates: Dict[str, float] = {}
    for scenario, scenario_results in grouped.items():
        gated_rates = [
            float(result["gate_metrics"]["gate_activation_rate"])  # type: ignore[index]
            for result in scenario_results.get("gated_intervention", [])
        ]
        matched_rate = float(np.mean(gated_rates)) if args.random_rate == "auto" else float(args.random_rate)
        matched_rate = float(np.clip(matched_rate, 0.0, 1.0))
        matched_rates[scenario] = matched_rate
        random_gate_config = gate_config or GateConfig()
        random_gate_config = GateConfig(**{**vars(random_gate_config), "random_rate": matched_rate})
        for seed in seeds:
            random_jobs.append((
                scenario,
                scenario_configs[scenario],
                "random_gate",
                "random",
                0.75,
                args.episodes,
                args.eval_episodes,
                seed,
                random_gate_config,
            ))
    random_grouped = run_jobs(random_jobs, args.jobs)
    for scenario, scenario_results in random_grouped.items():
        grouped.setdefault(scenario, {}).update(scenario_results)

    summary: Dict[str, object] = {
        "metadata": {
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "seeds": seeds,
            "scenarios": scenarios,
            "methods": methods,
            "random_rate": args.random_rate,
            "jobs": args.jobs,
            "gate_config_path": args.gate_config,
            "gate_config": vars(gate_config) if gate_config is not None else None,
        },
        "scenarios": {},
    }
    scenarios_out: Dict[str, object] = {}
    summary["scenarios"] = scenarios_out
    ordered_methods = ["no_intervention", "always_on", "threshold", "random_gate", "gated_intervention"]
    ordered_methods.extend(method for method in methods if method not in ordered_methods)
    for scenario in scenarios:
        methods_out: Dict[str, object] = {}
        scenario_results = grouped.get(scenario, {})
        for method in ordered_methods:
            if method not in scenario_results:
                continue
            results = scenario_results[method]
            methods_out[method] = {
                "seed_results": results,
                "aggregate": aggregate_seed_results(results),
            }
        scenarios_out[scenario] = {
            "description": SCENARIOS[scenario][0],
            "config": asdict(scenario_configs[scenario]),
            "matched_random_rate": matched_rates.get(scenario, 0.0),
            "methods": methods_out,
            "paired_random_minus_gated_gap": paired_delta(
                scenario_results.get("random_gate", []),
                scenario_results.get("gated_intervention", []),
            ),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else output.with_name("metrics.csv")
    write_metrics_csv(metrics_csv, summary)
    print(json.dumps(summary["metadata"], indent=2))
    print(f"Saved robustness summary to {output}")
    print(f"Saved robustness metrics to {metrics_csv}")


if __name__ == "__main__":
    main()
