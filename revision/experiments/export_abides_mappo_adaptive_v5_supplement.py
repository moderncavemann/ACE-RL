#!/usr/bin/env python3
"""Export paper-facing tables and frontiers from the frozen v5 supplement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_SUMMARY = (
    ROOT
    / "experiments/outputs/abides_mappo_adaptive_v5_supplement_v1/summary.json"
)
DEFAULT_FIGURE_DIR = ROOT / "paper/figures"
DEFAULT_CSV = ROOT / "paper/data/abides_mappo_adaptive_v5_supplement_v1_per_seed.csv"

METHOD_ORDER = (
    "no_intervention",
    "risk_aware",
    "matched_random",
    "circular_shifted",
    "volatility_only",
    "return_only",
    "spread_only_threshold",
    "always_on",
    "no_routing",
)
METHOD_LABELS = {
    "no_intervention": "No intervention",
    "risk_aware": "Risk-aware",
    "matched_random": "Matched random",
    "circular_shifted": "Circular shift",
    "volatility_only": "Volatility only",
    "return_only": "Return only",
    "spread_only_threshold": "Spread threshold",
    "always_on": "Always on",
    "no_routing": "No routing",
}
METHOD_COLORS = {
    "no_intervention": "#4D4D4D",
    "risk_aware": "#C62828",
    "matched_random": "#1565C0",
    "circular_shifted": "#00897B",
    "volatility_only": "#7B1FA2",
    "return_only": "#EF6C00",
    "spread_only_threshold": "#6D4C41",
    "always_on": "#000000",
    "no_routing": "#78909C",
}
PER_SEED_METRICS = (
    "low_execution_half_spread",
    "mean_reward",
    "activation_rate",
    "high_risk_activation_rate",
)


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError("paper export requires a passed formal supplement")
    rows = payload.get("rows", [])
    if len(rows) != 20:
        raise RuntimeError("paper export requires all 20 policy seeds")
    return payload


def per_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    exported = []
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        methods = row["final_evaluation"]["methods"]
        record: dict[str, Any] = {"policy_seed": int(row["seed"])}
        for method in METHOD_ORDER:
            for metric in PER_SEED_METRICS:
                record[f"{method}__{metric}"] = float(methods[method][metric])
        risk_spread = float(methods["risk_aware"]["low_execution_half_spread"])
        for comparator in (
            "matched_random",
            "circular_shifted",
            "volatility_only",
            "return_only",
        ):
            record[f"{comparator}__minus_risk_aware_spread"] = (
                float(methods[comparator]["low_execution_half_spread"])
                - risk_spread
            )
        exported.append(record)
    return exported


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_frontiers(summary: Mapping[str, Any], output_dir: Path) -> None:
    methods = summary["method_summary"]
    baseline_reward = float(methods["no_intervention"]["mean_reward"])
    panels = (
        (
            "activation_rate",
            "Activation rate (%)",
            lambda values: 100.0 * float(values["activation_rate"]),
        ),
        (
            "reward_loss",
            "Dealer reward loss vs no intervention (%)",
            lambda values: 100.0
            * (baseline_reward - float(values["mean_reward"]))
            / baseline_reward,
        ),
        (
            "high_risk_activation_rate",
            "High-risk activation rate (%)",
            lambda values: 100.0 * float(values["high_risk_activation_rate"]),
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    handles = []
    labels = []
    for axis, (_, x_label, x_value) in zip(axes, panels):
        for method in METHOD_ORDER:
            marker = "*" if method == "risk_aware" else "o"
            size = 150 if method == "risk_aware" else 55
            point = axis.scatter(
                x_value(methods[method]),
                float(methods[method]["low_execution_half_spread"]),
                color=METHOD_COLORS[method],
                marker=marker,
                s=size,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            if axis is axes[0]:
                handles.append(point)
                labels.append(METHOD_LABELS[method])
        axis.set_xlabel(x_label)
        axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Low-risk execution half-spread (ticks, lower is better)")
    fig.suptitle("Adaptive ABIDES v5 intervention frontiers", fontsize=12)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "abides_mappo_adaptive_v5_supplement_frontiers"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "ACE-RL adaptive-v5 supplement exporter"},
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "ACE-RL adaptive-v5 supplement exporter",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--per-seed-csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    exported = per_seed_rows(summary["rows"])
    write_csv(exported, args.per_seed_csv)
    plot_frontiers(summary, args.figure_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "policy_seeds": len(exported),
                "csv": str(args.per_seed_csv),
                "figure_dir": str(args.figure_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
