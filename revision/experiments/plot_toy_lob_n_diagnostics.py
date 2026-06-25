#!/usr/bin/env python3
"""Compare toy LOB diagnostic distributions across N or scenario outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "passive_risk_score",
    "collusion_gap",
    "alignment",
    "demand_suppression",
]


def parse_inputs(raw: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError("--inputs entries must be label=path")
        label, path = part.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def read_records(inputs: List[Tuple[str, Path]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for label, directory in inputs:
        for path in sorted(directory.glob("*_eval_records.csv")):
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    records.append({
                        "label": label,
                        "risk_state": row.get("risk_state", ""),
                        "passive_risk_score": as_float(row, "passive_risk_score"),
                        "collusion_gap": as_float(row, "collusion_gap"),
                        "alignment": as_float(row, "alignment"),
                        "demand_suppression": as_float(row, "demand_suppression"),
                    })
    return records


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return {"n": 0.0}
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def write_distribution_summary(path: Path, records: List[Dict[str, object]]) -> None:
    rows: List[Dict[str, object]] = []
    labels = sorted({str(row["label"]) for row in records})
    risks = ["all", "low", "high"]
    for label in labels:
        for risk in risks:
            scoped = [
                row for row in records
                if row["label"] == label and (risk == "all" or row["risk_state"] == risk)
            ]
            for metric in METRICS:
                rows.append({
                    "label": label,
                    "risk": risk,
                    "metric": metric,
                    **summarize(float(row[metric]) for row in scoped),
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "risk", "metric", "n", "mean", "std", "p10", "p50", "p90"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=180)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def plot_metric_histograms(
    output_dir: Path,
    records: List[Dict[str, object]],
    threshold: float | None,
) -> None:
    labels = sorted({str(row["label"]) for row in records})
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label in labels:
            values = [float(row[metric]) for row in records if row["label"] == label]
            if values:
                ax.hist(values, bins=35, alpha=0.45, density=True, label=label)
        if metric == "passive_risk_score" and threshold is not None:
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.3f}")
        if metric == "collusion_gap":
            ax.axvline(0.30, color="black", linestyle="--", linewidth=1.5, label="collusive tau")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
        save(fig, output_dir, f"{metric}_distribution")


def read_gate_config(path: str | None) -> float | None:
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    block = raw.get("gate_config", raw)
    return float(block.get("risk_threshold"))


def gate_activation_rows(summary_path: str | None, scenarios: List[str]) -> List[Dict[str, object]]:
    if not summary_path:
        return []
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        block = dict(dict(summary.get("scenarios", {})).get(scenario, {}))
        method = dict(dict(block.get("methods", {})).get("gated_intervention", {}))
        by_risk = dict(dict(method.get("aggregate", {})).get("eval_by_risk", {}))
        for risk in ("low", "high"):
            risk_block = dict(by_risk.get(risk, {}))
            value = dict(risk_block.get("gate_active_mean", {})).get("mean", 0.0)
            rows.append({"scenario": scenario, "risk": risk, "gate_activation_rate": float(value)})
    return rows


def write_gate_activation(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "risk", "gate_activation_rate"])
        writer.writeheader()
        writer.writerows(rows)


def plot_gate_activation(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    scenarios = sorted({str(row["scenario"]) for row in rows})
    x = np.arange(len(scenarios))
    width = 0.36
    low = [
        next((float(row["gate_activation_rate"]) for row in rows if row["scenario"] == scenario and row["risk"] == "low"), 0.0)
        for scenario in scenarios
    ]
    high = [
        next((float(row["gate_activation_rate"]) for row in rows if row["scenario"] == scenario and row["risk"] == "high"), 0.0)
        for scenario in scenarios
    ]
    fig, ax = plt.subplots(figsize=(max(7, len(scenarios) * 1.0), 4))
    ax.bar(x - width / 2, low, width, label="low risk")
    ax.bar(x + width / 2, high, width, label="high risk")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("Gate activation rate")
    ax.set_title("Gated activation by risk state")
    ax.legend(frameon=False)
    save(fig, output_dir, "gate_activation_by_risk")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated label=observation_dir entries.")
    parser.add_argument("--gate-config", default=None)
    parser.add_argument("--robustness-summary", default=None)
    parser.add_argument("--robustness-scenarios", default="")
    parser.add_argument("--output-dir", default="revision/experiments/outputs/n_diagnostics")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = read_records(parse_inputs(args.inputs))
    threshold = read_gate_config(args.gate_config)
    write_distribution_summary(output_dir / "distribution_summary.csv", records)
    plot_metric_histograms(output_dir, records, threshold)

    scenarios = [x.strip() for x in args.robustness_scenarios.split(",") if x.strip()]
    gate_rows = gate_activation_rows(args.robustness_summary, scenarios)
    write_gate_activation(output_dir / "gate_activation_by_risk.csv", gate_rows)
    plot_gate_activation(output_dir, gate_rows)
    print(f"Saved N diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
