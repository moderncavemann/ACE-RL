#!/usr/bin/env python3
"""Aggregate toy LOB method x seed results with uncertainty intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


DEFAULT_EVAL_METRICS = [
    "collusion_gap_mean",
    "demand_suppression_mean",
    "avg_spread_mean",
    "alignment_mean",
    "total_demand_mean",
    "mean_reward_mean",
    "gate_active_mean",
]
DEFAULT_GATE_METRICS = [
    "gate_activation_rate",
    "fpr_high_normal",
    "tpr_low_collusive",
]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1 or n_boot <= 0:
        mean = float(np.mean(values))
        return mean, mean
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_values(values: Iterable[float], rng: np.random.Generator, n_boot: int) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return {"n": 0.0, "mean": 0.0, "std": 0.0, "se": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    ci_low, ci_high = bootstrap_ci(arr, rng, n_boot)
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def metric_values(seed_results: List[Dict[str, object]], block: str, metric: str, risk: str | None = None) -> List[float]:
    values: List[float] = []
    for result in seed_results:
        source: object
        if risk is None:
            source = result.get(block, {})
        else:
            source = dict(result.get(block, {})).get(risk, {})
        if isinstance(source, dict) and metric in source:
            values.append(float(source[metric]))
    return values


def collect_seed_results(summary: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    methods = summary.get("methods")
    if isinstance(methods, dict):
        return {
            method: list(dict(block).get("seed_results", []))
            for method, block in methods.items()
            if isinstance(block, dict)
        }
    return {
        method: [dict(block)]
        for method, block in summary.items()
        if isinstance(block, dict) and "eval_all" in block
    }


def aggregate(summary: Dict[str, object], n_boot: int, seed: int) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for method, seed_results in collect_seed_results(summary).items():
        for metric in DEFAULT_EVAL_METRICS:
            stats = summarize_values(metric_values(seed_results, "eval_all", metric), rng, n_boot)
            rows.append({"method": method, "block": "eval_all", "risk": "all", "metric": metric, **stats})
        for metric in DEFAULT_GATE_METRICS:
            stats = summarize_values(metric_values(seed_results, "gate_metrics", metric), rng, n_boot)
            rows.append({"method": method, "block": "gate_metrics", "risk": "all", "metric": metric, **stats})
        risks = sorted({
            risk
            for result in seed_results
            for risk in dict(result.get("eval_by_risk", {})).keys()
        })
        for risk in risks:
            for metric in DEFAULT_EVAL_METRICS:
                stats = summarize_values(metric_values(seed_results, "eval_by_risk", metric, risk), rng, n_boot)
                rows.append({"method": method, "block": "eval_by_risk", "risk": risk, "metric": metric, **stats})
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "block", "risk", "metric", "n", "mean", "std", "se", "ci95_low", "ci95_high"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="revision/experiments/outputs/baselines/summary.json")
    parser.add_argument("--output-csv", default="revision/experiments/outputs/aggregated/toy_lob_aggregate.csv")
    parser.add_argument("--output-json", default="revision/experiments/outputs/aggregated/toy_lob_aggregate.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    summary = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = aggregate(summary, args.bootstrap, args.seed)

    csv_path = Path(args.output_csv)
    json_path = Path(args.output_json)
    write_csv(csv_path, rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Aggregated {len(rows)} metric rows from {args.input}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
