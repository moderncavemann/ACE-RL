#!/usr/bin/env python3
"""Plot toy LOB observation, gate calibration, and baseline diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def read_observation_records(observation_dir: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in sorted(observation_dir.glob("*_eval_records.csv")):
        scenario = path.name.replace("_eval_records.csv", "")
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                records.append({
                    "scenario": scenario,
                    "episode": int(_float(row, "episode")),
                    "risk_state": row.get("risk_state", ""),
                    "collusion_gap": _float(row, "collusion_gap"),
                    "passive_risk_score": _float(row, "passive_risk_score"),
                })
    return records


def method_aggregates(summary: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    methods = summary.get("methods", {})
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(methods, dict):
        return out
    for method, block in methods.items():
        aggregate = dict(dict(block).get("aggregate", {}))
        eval_all = dict(aggregate.get("eval_all", {}))
        gate_metrics = dict(aggregate.get("gate_metrics", {}))

        def metric(source: Dict[str, object], key: str) -> float:
            value = source.get(key, {})
            if isinstance(value, dict):
                return float(value.get("mean", 0.0))
            return float(value or 0.0)

        out[method] = {
            "collusion_gap": metric(eval_all, "collusion_gap_mean"),
            "gate_activation_rate": metric(gate_metrics, "gate_activation_rate"),
            "fpr_high_normal": metric(gate_metrics, "fpr_high_normal"),
            "tpr_low_collusive": metric(gate_metrics, "tpr_low_collusive"),
            "mean_reward": metric(eval_all, "mean_reward_mean"),
            "total_demand": metric(eval_all, "total_demand_mean"),
            "demand_suppression": metric(eval_all, "demand_suppression_mean"),
        }
    return out


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=180)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def plot_observation_gap(records: List[Dict[str, object]], output_dir: Path) -> None:
    if not records:
        return
    buckets: Dict[tuple[str, int], List[float]] = defaultdict(list)
    for row in records:
        buckets[(str(row["scenario"]), int(row["episode"]))].append(float(row["collusion_gap"]))
    by_scenario: Dict[str, List[tuple[int, float]]] = defaultdict(list)
    for (scenario, episode), values in buckets.items():
        by_scenario[scenario].append((episode, float(np.mean(values))))

    fig, ax = plt.subplots(figsize=(7, 4))
    for scenario, points in sorted(by_scenario.items()):
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=1.5, label=scenario)
    ax.axhline(0.30, color="black", linestyle="--", linewidth=1, label="collusive tau")
    ax.set_title("No-intervention collusion gap by eval episode")
    ax.set_xlabel("Eval episode")
    ax.set_ylabel("Risk-adjusted collusion gap")
    ax.legend(frameon=False)
    save(fig, output_dir, "observation_collusion_gap")


def plot_passive_score(records: List[Dict[str, object]], output_dir: Path, threshold: float | None) -> None:
    if not records:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for risk_state in ["low", "high"]:
        values = [
            float(row["passive_risk_score"])
            for row in records
            if row.get("risk_state") == risk_state
        ]
        if values:
            ax.hist(values, bins=25, alpha=0.55, density=True, label=risk_state)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.3f}")
    ax.set_title("Passive risk score distribution")
    ax.set_xlabel("Passive risk score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save(fig, output_dir, "passive_risk_score_distribution")


def plot_tradeoff(methods: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, vals in sorted(methods.items()):
        x = vals["gate_activation_rate"]
        y = vals["collusion_gap"]
        ax.scatter(x, y, s=55)
        ax.annotate(method, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Baseline trade-off")
    ax.set_xlabel("Intervention activation rate")
    ax.set_ylabel("Mean collusion gap")
    save(fig, output_dir, "baseline_tradeoff")


def plot_fpr_tradeoff(methods: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, vals in sorted(methods.items()):
        x = vals["fpr_high_normal"]
        y = vals["collusion_gap"]
        ax.scatter(x, y, s=55)
        ax.annotate(method, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("False-positive trade-off")
    ax.set_xlabel("FPR in high-risk normal windows")
    ax.set_ylabel("Mean collusion gap")
    save(fig, output_dir, "fpr_gap_tradeoff")


def plot_reward_tradeoff(methods: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, vals in sorted(methods.items()):
        x = vals["mean_reward"]
        y = vals["collusion_gap"]
        ax.scatter(x, y, s=55)
        ax.annotate(method, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Market-quality trade-off")
    ax.set_xlabel("Mean reward")
    ax.set_ylabel("Mean collusion gap")
    save(fig, output_dir, "reward_gap_tradeoff")


def plot_gate_rates(methods: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not methods:
        return
    names = list(sorted(methods.keys()))
    x = np.arange(len(names))
    width = 0.38
    fpr = [methods[name]["fpr_high_normal"] for name in names]
    tpr = [methods[name]["tpr_low_collusive"] for name in names]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.9), 4))
    ax.bar(x - width / 2, fpr, width, label="FPR high-normal")
    ax.bar(x + width / 2, tpr, width, label="TPR low-collusive")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Gate classification diagnostics")
    ax.set_ylabel("Rate")
    ax.legend(frameon=False)
    save(fig, output_dir, "gate_fpr_tpr")


def plot_paired_delta(summary: Dict[str, object], output_dir: Path) -> None:
    methods = summary.get("methods", {})
    if not isinstance(methods, dict) or "random_gate" not in methods or "gated_intervention" not in methods:
        return

    def seed_gap(method: str) -> Dict[int, float]:
        rows = dict(methods[method]).get("seed_results", [])
        return {
            int(dict(row).get("seed", dict(row).get("base_seed", -1))): float(dict(dict(row).get("eval_all", {})).get("collusion_gap_mean", 0.0))
            for row in rows
            if isinstance(row, dict)
        }

    random_gap = seed_gap("random_gate")
    gated_gap = seed_gap("gated_intervention")
    seeds = sorted(set(random_gap) & set(gated_gap))
    if not seeds:
        return
    deltas = [random_gap[seed] - gated_gap[seed] for seed in seeds]
    colors = ["#2f7d4f" if delta > 0 else "#a94442" for delta in deltas]
    fig, ax = plt.subplots(figsize=(max(7, len(seeds) * 0.35), 4))
    ax.bar([str(seed) for seed in seeds], deltas, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Paired gap delta: random - gated ({sum(delta > 0 for delta in deltas)}/{len(deltas)} positive)")
    ax.set_xlabel("Held-out seed")
    ax.set_ylabel("Collusion-gap reduction")
    save(fig, output_dir, "paired_random_minus_gated_delta")


def plot_ablation_gap(methods: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not methods:
        return
    names = list(sorted(methods.keys()))
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.9), 4))
    ax.bar(x, [methods[name]["collusion_gap"] for name in names])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Ablation collusion gap")
    ax.set_ylabel("Mean collusion gap")
    save(fig, output_dir, "ablation_collusion_gap")


def plot_ablation_gate_rates(methods: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not methods:
        return
    names = list(sorted(methods.keys()))
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.9), 4))
    ax.bar(x - width / 2, [methods[name]["fpr_high_normal"] for name in names], width, label="FPR high-normal")
    ax.bar(x + width / 2, [methods[name]["tpr_low_collusive"] for name in names], width, label="TPR low-collusive")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Ablation gate diagnostics")
    ax.set_ylabel("Rate")
    ax.legend(frameon=False)
    save(fig, output_dir, "ablation_gate_fpr_tpr")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", default="revision/experiments/outputs/observation")
    parser.add_argument("--baseline-summary", default="revision/experiments/outputs/baselines/summary.json")
    parser.add_argument("--ablation-summary", default=None)
    parser.add_argument("--gate-config", default=None)
    parser.add_argument("--output-dir", default="revision/experiments/outputs/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = read_observation_records(Path(args.observation_dir))
    threshold = None
    if args.gate_config:
        gate_config = json.loads(Path(args.gate_config).read_text(encoding="utf-8"))
        threshold = float(gate_config.get("gate_config", gate_config).get("risk_threshold"))

    plot_observation_gap(records, output_dir)
    plot_passive_score(records, output_dir, threshold)

    baseline_path = Path(args.baseline_summary)
    if baseline_path.exists():
        baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
        methods = method_aggregates(baseline_summary)
        plot_tradeoff(methods, output_dir)
        plot_fpr_tradeoff(methods, output_dir)
        plot_reward_tradeoff(methods, output_dir)
        plot_gate_rates(methods, output_dir)
        plot_paired_delta(baseline_summary, output_dir)

    if args.ablation_summary and Path(args.ablation_summary).exists():
        ablation_summary = json.loads(Path(args.ablation_summary).read_text(encoding="utf-8"))
        ablation_methods = method_aggregates(ablation_summary)
        plot_ablation_gap(ablation_methods, output_dir)
        plot_ablation_gate_rates(ablation_methods, output_dir)

    print(f"Saved toy LOB figures to {output_dir}")


if __name__ == "__main__":
    main()
