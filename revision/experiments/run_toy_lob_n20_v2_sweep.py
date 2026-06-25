#!/usr/bin/env python3
"""Resumable N=20 gate-v2 scale stress test.

N=20 is treated as a stress test for gate representation, not as the primary
toy market size. This runner writes one result file per method/seed so an
expensive sweep can be resumed after interruption without losing completed
jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_toy_lob_baselines import aggregate_seed_results, load_gate_config, run_gate_mode
from src.envs.toy_lob_pricing_env import ToyLOBConfig
from src.interventions.gates import GateConfig, GateMode


METHODS: Dict[str, Tuple[GateMode, float]] = {
    "no_intervention": ("none", 0.75),
    "threshold": ("threshold", 0.75),
    "always_on": ("always", 0.75),
    "gated_intervention": ("risk_aware", 0.75),
    "spread_only_gate": ("spread_only", 0.75),
    "v2_entropy_gate": ("v2_entropy", 0.75),
    "v2_fraction_near_min_gate": ("v2_fraction_near_min", 0.75),
    "v2_two_stage_gate": ("v2_two_stage", 0.75),
    "no_risk_adjustment": ("no_risk_adjustment", 0.75),
    "no_routing_intervention": ("risk_aware", 0.0),
}

DEFAULT_METHODS = (
    "no_intervention,threshold,always_on,gated_intervention,spread_only_gate,"
    "v2_entropy_gate,v2_fraction_near_min_gate,v2_two_stage_gate"
)
DEFAULT_RANDOM_TARGETS = "gated_intervention,v2_entropy_gate,v2_fraction_near_min_gate,v2_two_stage_gate"


def parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in parse_csv(raw):
        if "-" in part:
            lo, hi = [int(x.strip()) for x in part.split("-", 1)]
            step = 1 if hi >= lo else -1
            seeds.extend(range(lo, hi + step, step))
        else:
            seeds.append(int(part))
    return seeds


def result_path(results_dir: Path, method: str, seed: int) -> Path:
    return results_dir / f"{method}_seed{seed}.json"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_result(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_job(
    job: Tuple[ToyLOBConfig, str, GateMode, float, int, int, int, GateConfig | None],
) -> Tuple[str, int, Dict[str, object]]:
    config, method, gate_mode, routing_rho, episodes, eval_episodes, seed, gate_config = job
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
    result["scenario"] = "n20"
    return method, seed, result


def execute_jobs(
    jobs: List[Tuple[ToyLOBConfig, str, GateMode, float, int, int, int, GateConfig | None]],
    results_dir: Path,
    jobs_parallel: int,
) -> None:
    pending = [
        job for job in jobs
        if not result_path(results_dir, job[1], job[6]).exists()
    ]
    if not pending:
        return
    if jobs_parallel <= 1 or len(pending) <= 1:
        for job in pending:
            method, seed, result = run_job(job)
            write_json(result_path(results_dir, method, seed), result)
            print(f"completed {method} seed={seed}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=jobs_parallel) as executor:
        futures = [executor.submit(run_job, job) for job in pending]
        for future in as_completed(futures):
            method, seed, result = future.result()
            write_json(result_path(results_dir, method, seed), result)
            print(f"completed {method} seed={seed}", flush=True)


def load_grouped_results(results_dir: Path) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for path in sorted(results_dir.glob("*.json")):
        result = load_result(path)
        method = str(result["method"])
        grouped.setdefault(method, []).append(result)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["seed"]))
    return grouped


def activation_rate(results: List[Dict[str, object]]) -> float:
    rates = [
        float(dict(result["gate_metrics"]).get("gate_activation_rate", 0.0))
        for result in results
    ]
    return float(np.mean(rates)) if rates else 0.0


def paired_delta(reference_results: List[Dict[str, object]], target_results: List[Dict[str, object]]) -> Dict[str, float]:
    reference_by_seed = {
        int(row["seed"]): float(dict(row["eval_all"])["collusion_gap_mean"])
        for row in reference_results
    }
    target_by_seed = {
        int(row["seed"]): float(dict(row["eval_all"])["collusion_gap_mean"])
        for row in target_results
    }
    seeds = sorted(set(reference_by_seed) & set(target_by_seed))
    if not seeds:
        return {"n": 0.0}
    deltas = np.array([reference_by_seed[seed] - target_by_seed[seed] for seed in seeds], dtype=float)
    return {
        "n": float(len(deltas)),
        "positive": float(np.sum(deltas > 0.0)),
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
    for method, method_block in dict(summary["methods"]).items():
        aggregate = dict(dict(method_block)["aggregate"])
        for metric in (
            "collusion_gap_mean",
            "demand_suppression_mean",
            "avg_spread_mean",
            "alignment_mean",
            "spread_entropy_mean",
            "sparse_near_min_spread_fraction_mean",
            "flow_concentration_mean",
            "v2_entropy_score_mean",
            "v2_fraction_near_min_score_mean",
            "v2_two_stage_score_mean",
            "total_demand_mean",
            "mean_reward_mean",
            "gate_active_mean",
        ):
            rows.append({
                "method": method,
                "block": "eval_all",
                "risk": "all",
                "metric": metric,
                "mean": metric_mean(aggregate, "eval_all", metric),
            })
        for risk in ("low", "high"):
            rows.append({
                "method": method,
                "block": "eval_by_risk",
                "risk": risk,
                "metric": "collusion_gap_mean",
                "mean": metric_mean(aggregate, "eval_by_risk", "collusion_gap_mean", risk),
            })
        for metric in ("gate_activation_rate", "fpr_high_normal", "tpr_low_collusive"):
            rows.append({
                "method": method,
                "block": "gate_metrics",
                "risk": "all",
                "metric": metric,
                "mean": metric_mean(aggregate, "gate_metrics", metric),
            })

    for target, paired in dict(summary.get("paired_random_minus_gate_gap", {})).items():
        for metric in ("mean", "positive", "n", "se"):
            rows.append({
                "method": f"random_minus_{target}",
                "block": "paired",
                "risk": "all",
                "metric": f"collusion_gap_delta_{metric}",
                "mean": dict(paired).get(metric, 0.0),
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "block", "risk", "metric", "mean"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    grouped: Dict[str, List[Dict[str, object]]],
    config: ToyLOBConfig,
    methods: List[str],
    random_targets: List[str],
    matched_random_rates: Dict[str, float],
    metadata: Dict[str, object],
) -> Dict[str, object]:
    methods_out: Dict[str, object] = {}
    for method in sorted(grouped):
        results = grouped[method]
        methods_out[method] = {
            "completed_seeds": [int(row["seed"]) for row in results],
            "seed_results": results,
            "aggregate": aggregate_seed_results(results),
        }

    paired_out: Dict[str, object] = {}
    for target in random_targets:
        random_method = f"random_matched_{target}"
        if random_method in grouped and target in grouped:
            paired_out[target] = paired_delta(grouped[random_method], grouped[target])

    return {
        "metadata": {
            **metadata,
            "scenario": "n20",
            "config": asdict(config),
            "methods": methods,
            "random_match_targets": random_targets,
            "matched_random_rates": matched_random_rates,
            "interpretation": (
                "N=20 is a scale stress test for gate representation, not the "
                "primary toy market size."
            ),
        },
        "methods": methods_out,
        "paired_random_minus_gate_gap": paired_out,
    }


def non_random_jobs(
    config: ToyLOBConfig,
    methods: List[str],
    seeds: List[int],
    episodes: int,
    eval_episodes: int,
    gate_config: GateConfig | None,
) -> List[Tuple[ToyLOBConfig, str, GateMode, float, int, int, int, GateConfig | None]]:
    jobs = []
    for method in methods:
        gate_mode, routing_rho = METHODS[method]
        for seed in seeds:
            seed_config = replace(config, seed=seed)
            jobs.append((seed_config, method, gate_mode, routing_rho, episodes, eval_episodes, seed, gate_config))
    return jobs


def random_jobs(
    config: ToyLOBConfig,
    random_targets: List[str],
    seeds: List[int],
    episodes: int,
    eval_episodes: int,
    gate_config: GateConfig | None,
    matched_random_rates: Dict[str, float],
) -> List[Tuple[ToyLOBConfig, str, GateMode, float, int, int, int, GateConfig | None]]:
    jobs = []
    base_gate_config = gate_config or GateConfig()
    for target in random_targets:
        rate = float(np.clip(matched_random_rates.get(target, 0.0), 0.0, 1.0))
        target_gate_config = GateConfig(**{**vars(base_gate_config), "random_rate": rate})
        method = f"random_matched_{target}"
        for seed in seeds:
            seed_config = replace(config, seed=seed)
            jobs.append((seed_config, method, "random", 0.75, episodes, eval_episodes, seed, target_gate_config))
    return jobs


def validate_methods(methods: Iterable[str]) -> None:
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise SystemExit(f"Unknown method(s): {', '.join(invalid)}. Valid methods: {', '.join(sorted(METHODS))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seeds", default="90-109")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--random-match-targets", default=DEFAULT_RANDOM_TARGETS)
    parser.add_argument("--include-controls", action="store_true", help="Add no_risk_adjustment and no_routing_intervention.")
    parser.add_argument("--gate-config", default="revision/experiments/outputs/n_invariant_gate_diagnostic_v1/v2_gate_config.json")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--output-dir", default="revision/experiments/outputs/n20_v2_formal_sweep_v1")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    methods = parse_csv(args.methods)
    if args.include_controls:
        for method in ("no_risk_adjustment", "no_routing_intervention"):
            if method not in methods:
                methods.append(method)
    random_targets = [
        target for target in parse_csv(args.random_match_targets)
        if target in methods
    ]
    validate_methods(methods)

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "job_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_config = load_gate_config(args.gate_config)
    config = ToyLOBConfig(n_agents=20, risk_mode="mixed", seed=seeds[0] if seeds else 0)
    metadata = {
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "seeds": seeds,
        "jobs": args.jobs,
        "gate_config_path": args.gate_config,
        "gate_config": vars(gate_config) if gate_config is not None else None,
        "aggregate_only": bool(args.aggregate_only),
    }

    if not args.aggregate_only:
        execute_jobs(
            non_random_jobs(config, methods, seeds, args.episodes, args.eval_episodes, gate_config),
            results_dir,
            args.jobs,
        )

    grouped = load_grouped_results(results_dir)
    matched_random_rates = {
        target: activation_rate(grouped.get(target, []))
        for target in random_targets
    }

    if not args.aggregate_only and random_targets:
        execute_jobs(
            random_jobs(config, random_targets, seeds, args.episodes, args.eval_episodes, gate_config, matched_random_rates),
            results_dir,
            args.jobs,
        )
        grouped = load_grouped_results(results_dir)

    summary = build_summary(grouped, config, methods, random_targets, matched_random_rates, metadata)
    write_json(output_dir / "summary.json", summary)
    write_metrics_csv(output_dir / "metrics.csv", summary)
    print(json.dumps({
        "output_dir": str(output_dir),
        "methods": sorted(summary["methods"]),
        "matched_random_rates": matched_random_rates,
        "paired_random_minus_gate_gap": summary["paired_random_minus_gate_gap"],
    }, indent=2))


if __name__ == "__main__":
    main()
