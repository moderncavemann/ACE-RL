#!/usr/bin/env python3
"""Calibrate the toy LOB risk-aware gate from no-intervention records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interventions.gates import GateConfig


Record = Dict[str, float | str]


def _parse_target_range(raw: str) -> Tuple[float, float]:
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--target-activation must be formatted as low,high")
    low, high = parts
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("--target-activation values must satisfy 0 <= low <= high <= 1")
    return low, high


def _as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def read_records(observation_dir: Path, glob_pattern: str) -> List[Record]:
    records: List[Record] = []
    for path in sorted(observation_dir.glob(glob_pattern)):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                risk_state = row.get("risk_state")
                if risk_state not in {"low", "high"}:
                    continue
                collusion_gap = _as_float(row, "collusion_gap")
                alignment = _as_float(row, "alignment")
                demand_suppression = _as_float(row, "demand_suppression")
                passive_score = _as_float(
                    row,
                    "passive_risk_score",
                    0.65 * collusion_gap + 0.20 * alignment + 0.15 * demand_suppression,
                )
                records.append({
                    "risk_state": risk_state,
                    "collusion_gap": collusion_gap,
                    "alignment": alignment,
                    "demand_suppression": demand_suppression,
                    "passive_risk_score": passive_score,
                })
    return records


def score_summary(rows: Iterable[Record]) -> Dict[str, float]:
    scores = np.array([float(row["passive_risk_score"]) for row in rows], dtype=float)
    if len(scores) == 0:
        return {"n": 0.0}
    out = {
        "n": float(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }
    for q in (10, 25, 50, 75, 80, 85, 90, 95):
        out[f"p{q}"] = float(np.percentile(scores, q))
    return out


def evaluate_threshold(
    records: List[Record],
    threshold: float,
    positive_tau: float,
    negative_tau: float,
) -> Dict[str, float]:
    active = np.array(
        [float(row["passive_risk_score"]) > threshold for row in records],
        dtype=float,
    )
    low_positive_idx = [
        i for i, row in enumerate(records)
        if row["risk_state"] == "low" and float(row["collusion_gap"]) > positive_tau
    ]
    high_negative_idx = [
        i for i, row in enumerate(records)
        if row["risk_state"] == "high" and float(row["collusion_gap"]) < negative_tau
    ]
    low_all_idx = [i for i, row in enumerate(records) if row["risk_state"] == "low"]
    high_all_idx = [i for i, row in enumerate(records) if row["risk_state"] == "high"]

    def rate(indices: List[int]) -> float:
        if not indices:
            return 0.0
        return float(np.mean(active[indices]))

    return {
        "threshold": float(threshold),
        "activation_rate": float(np.mean(active)) if len(active) else 0.0,
        "low_activation_rate": rate(low_all_idx),
        "high_activation_rate": rate(high_all_idx),
        "tpr_low_collusive": rate(low_positive_idx),
        "tpr_low_collusive_n": float(len(low_positive_idx)),
        "fpr_high_normal": rate(high_negative_idx),
        "fpr_high_normal_n": float(len(high_negative_idx)),
    }


def choose_threshold(
    records: List[Record],
    positive_tau: float,
    negative_tau: float,
    target_activation: Tuple[float, float],
    max_fpr: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    scores = np.array([float(row["passive_risk_score"]) for row in records], dtype=float)
    if len(scores) == 0:
        raise ValueError("No usable records found for gate calibration")

    eps = 1e-9
    candidates = set(float(x) for x in scores)
    candidates.update(float(np.percentile(scores, q)) for q in range(1, 100))
    candidates.add(float(np.min(scores) - eps))
    candidates.add(float(np.max(scores) + eps))
    table = [
        evaluate_threshold(records, threshold, positive_tau, negative_tau)
        for threshold in sorted(candidates)
    ]

    target_low, target_high = target_activation
    target_mid = (target_low + target_high) / 2.0

    def activation_penalty(row: Dict[str, float]) -> float:
        activation = row["activation_rate"]
        if target_low <= activation <= target_high:
            return 0.0
        return min(abs(activation - target_low), abs(activation - target_high))

    for row in table:
        row["in_target_activation"] = float(target_low <= row["activation_rate"] <= target_high)
        row["objective"] = (
            row["tpr_low_collusive"]
            - 1.5 * row["fpr_high_normal"]
            - 0.75 * activation_penalty(row)
            - 0.05 * abs(row["activation_rate"] - target_mid)
        )

    feasible = [
        row for row in table
        if row["in_target_activation"] and row["fpr_high_normal"] <= max_fpr
    ]
    pool = feasible if feasible else table
    selected = max(
        pool,
        key=lambda row: (
            row["objective"],
            row["tpr_low_collusive"],
            -row["fpr_high_normal"],
            -abs(row["activation_rate"] - target_mid),
        ),
    )
    selected["selection_feasible"] = float(bool(feasible))
    return selected, table


def write_threshold_table(path: Path, table: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in table for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", default="revision/experiments/outputs/observation")
    parser.add_argument("--glob", default="*_eval_records.csv")
    parser.add_argument("--positive-tau", type=float, default=0.30)
    parser.add_argument("--negative-tau", type=float, default=0.20)
    parser.add_argument("--target-activation", default="0.20,0.40")
    parser.add_argument("--max-fpr", type=float, default=0.10)
    parser.add_argument("--output", default="revision/experiments/outputs/gate_calibration/gate_config.json")
    parser.add_argument("--table-output", default=None)
    args = parser.parse_args()

    observation_dir = Path(args.observation_dir)
    records = read_records(observation_dir, args.glob)
    if not records:
        raise SystemExit(f"No observation records matched {observation_dir}/{args.glob}")

    target_activation = _parse_target_range(args.target_activation)
    selected, table = choose_threshold(
        records,
        positive_tau=args.positive_tau,
        negative_tau=args.negative_tau,
        target_activation=target_activation,
        max_fpr=args.max_fpr,
    )

    gate_config = GateConfig(
        risk_threshold=float(selected["threshold"]),
        oracle_collusion_gap_tau=args.positive_tau,
    )
    low_collusive = [
        row for row in records
        if row["risk_state"] == "low" and float(row["collusion_gap"]) > args.positive_tau
    ]
    high_normal = [
        row for row in records
        if row["risk_state"] == "high" and float(row["collusion_gap"]) < args.negative_tau
    ]

    output = {
        "metadata": {
            "observation_dir": str(observation_dir),
            "glob": args.glob,
            "n_records": len(records),
            "positive_definition": {
                "risk_state": "low",
                "collusion_gap_gt": args.positive_tau,
            },
            "negative_definition": {
                "risk_state": "high",
                "collusion_gap_lt": args.negative_tau,
            },
            "target_activation": {
                "low": target_activation[0],
                "high": target_activation[1],
            },
            "max_fpr": args.max_fpr,
        },
        "gate_config": asdict(gate_config),
        "selected_threshold": selected,
        "score_distribution": {
            "all": score_summary(records),
            "low_all": score_summary(row for row in records if row["risk_state"] == "low"),
            "high_all": score_summary(row for row in records if row["risk_state"] == "high"),
            "low_collusive": score_summary(low_collusive),
            "high_normal": score_summary(high_normal),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    table_output = Path(args.table_output) if args.table_output else output_path.with_name("threshold_grid.csv")
    write_threshold_table(table_output, table)

    print(json.dumps(output, indent=2))
    print(f"\nSaved gate config to {output_path}")
    print(f"Saved threshold grid to {table_output}")


if __name__ == "__main__":
    main()
