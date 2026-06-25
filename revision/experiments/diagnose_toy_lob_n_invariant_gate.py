#!/usr/bin/env python3
"""Diagnose N-invariant toy LOB gate features and calibrate v2 thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_toy_lob_baselines import load_gate_config
from src.interventions.gates import GateConfig


FEATURE_FIELDS = [
    "collusion_gap",
    "passive_risk_score",
    "alignment",
    "demand_suppression",
    "near_min_spread_fraction",
    "sparse_near_min_spread_fraction",
    "spread_entropy",
    "spread_concentration",
    "spread_iqr_norm",
    "spread_q90_q10_norm",
    "modal_spread_fraction",
    "flow_concentration",
    "top_flow_share",
]

SCORE_FIELDS = [
    "passive_risk_score",
    "collusion_gap",
    "v2_entropy_score",
    "v2_fraction_near_min_score",
    "v2_two_stage_score",
]

V2_THRESHOLD_FIELDS = {
    "v2_entropy_score": "v2_entropy_threshold",
    "v2_fraction_near_min_score": "v2_fraction_near_min_threshold",
    "v2_two_stage_score": "v2_two_stage_threshold",
    "collusion_gap": "v2_base_excess_threshold",
}


Record = Dict[str, float | str]


def parse_inputs(raw: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError("--inputs entries must be formatted as label=path")
        label, path = part.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def parse_target_range(raw: str) -> Tuple[float, float]:
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--target-activation must be formatted as low,high")
    low, high = parts
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("--target-activation values must satisfy 0 <= low <= high <= 1")
    return low, high


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def read_records(inputs: List[Tuple[str, Path]], glob_pattern: str) -> List[Record]:
    records: List[Record] = []
    metric_fields = sorted(set(FEATURE_FIELDS + SCORE_FIELDS))
    for label, directory in inputs:
        for path in sorted(directory.glob(glob_pattern)):
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    risk_state = row.get("risk_state")
                    if risk_state not in {"low", "high"}:
                        continue
                    record: Record = {
                        "label": label,
                        "risk_state": risk_state,
                    }
                    for field in metric_fields:
                        record[field] = as_float(row, field)
                    record["sparse_near_min_spread_fraction"] = (
                        1.0 - float(record["near_min_spread_fraction"])
                    )
                    record["v2_entropy_score"] = (
                        0.65 * float(record["collusion_gap"])
                        + 0.15 * float(record["spread_entropy"])
                        + 0.10 * float(record["flow_concentration"])
                        + 0.15 * float(record["demand_suppression"])
                    )
                    record["v2_fraction_near_min_score"] = (
                        0.65 * float(record["collusion_gap"])
                        + 0.15 * float(record["sparse_near_min_spread_fraction"])
                        + 0.10 * float(record["flow_concentration"])
                        + 0.15 * float(record["demand_suppression"])
                    )
                    record["v2_two_stage_score"] = (
                        0.75 * float(record["collusion_gap"])
                        + 0.10 * float(record["spread_entropy"])
                        + 0.10 * float(record["flow_concentration"])
                        + 0.05 * float(record["demand_suppression"])
                    )
                    records.append(record)
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


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.array(list(x), dtype=float)
    y_arr = np.array(list(y), dtype=float)
    if len(x_arr) < 2 or len(y_arr) < 2:
        return 0.0
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def roc_auc(pos_scores: Iterable[float], neg_scores: Iterable[float]) -> float:
    pos = np.array(list(pos_scores), dtype=float)
    neg = np.array(list(neg_scores), dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    return float((pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def write_feature_summary(path: Path, records: List[Record]) -> None:
    rows: List[Dict[str, object]] = []
    labels = sorted({str(row["label"]) for row in records})
    for label in labels:
        for risk in ("all", "low", "high"):
            scoped = [
                row for row in records
                if row["label"] == label and (risk == "all" or row["risk_state"] == risk)
            ]
            for field in sorted(set(FEATURE_FIELDS + SCORE_FIELDS)):
                rows.append({
                    "label": label,
                    "risk": risk,
                    "metric": field,
                    **summarize(float(row[field]) for row in scoped),
                })
    write_rows(path, rows, ["label", "risk", "metric", "n", "mean", "std", "p10", "p50", "p90"])


def write_predictive_metrics(
    path: Path,
    records: List[Record],
    positive_tau: float,
    negative_tau: float,
) -> None:
    rows: List[Dict[str, object]] = []
    labels = ["all", *sorted({str(row["label"]) for row in records})]
    for label in labels:
        scoped = [row for row in records if label == "all" or row["label"] == label]
        positives = [
            row for row in scoped
            if row["risk_state"] == "low" and float(row["collusion_gap"]) > positive_tau
        ]
        negatives = [
            row for row in scoped
            if row["risk_state"] == "high" and float(row["collusion_gap"]) < negative_tau
        ]
        for field in sorted(set(FEATURE_FIELDS + SCORE_FIELDS)):
            rows.append({
                "label": label,
                "metric": field,
                "positive_n": len(positives),
                "negative_n": len(negatives),
                "auc_low_collusive_vs_high_normal": roc_auc(
                    (float(row[field]) for row in positives),
                    (float(row[field]) for row in negatives),
                ),
                "corr_with_collusion_gap": pearson(
                    (float(row[field]) for row in scoped),
                    (float(row["collusion_gap"]) for row in scoped),
                ),
            })
    write_rows(
        path,
        rows,
        ["label", "metric", "positive_n", "negative_n", "auc_low_collusive_vs_high_normal", "corr_with_collusion_gap"],
    )


def choose_threshold(
    records: List[Record],
    score_field: str,
    positive_tau: float,
    negative_tau: float,
    target_activation: Tuple[float, float],
    max_fpr: float,
    grid_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    scores = np.array([float(row[score_field]) for row in records], dtype=float)
    if len(scores) == 0:
        raise ValueError("No records available for threshold calibration")

    risk_states = np.array([str(row["risk_state"]) for row in records])
    gaps = np.array([float(row["collusion_gap"]) for row in records], dtype=float)
    low_all = risk_states == "low"
    high_all = risk_states == "high"
    low_positive = low_all & (gaps > positive_tau)
    high_negative = high_all & (gaps < negative_tau)

    def mask_rate(active: np.ndarray, mask: np.ndarray) -> float:
        if not bool(np.any(mask)):
            return 0.0
        return float(np.mean(active[mask]))

    def evaluate(threshold: float) -> Dict[str, float]:
        active = scores > threshold
        return {
            "score_field": score_field,
            "threshold": float(threshold),
            "activation_rate": float(np.mean(active)) if len(active) else 0.0,
            "low_activation_rate": mask_rate(active, low_all),
            "high_activation_rate": mask_rate(active, high_all),
            "tpr_low_collusive": mask_rate(active, low_positive),
            "tpr_low_collusive_n": float(np.sum(low_positive)),
            "fpr_high_normal": mask_rate(active, high_negative),
            "fpr_high_normal_n": float(np.sum(high_negative)),
        }

    if grid_size < 2:
        raise ValueError("--grid-size must be at least 2")
    eps = 1e-9
    candidates = set(float(np.percentile(scores, q)) for q in np.linspace(0, 100, grid_size))
    candidates.add(float(np.min(scores) - eps))
    candidates.add(float(np.max(scores) + eps))
    table = [evaluate(threshold) for threshold in sorted(candidates)]

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


def calibrate_v2_config(
    records: List[Record],
    base_gate_config: GateConfig,
    positive_tau: float,
    negative_tau: float,
    target_activation: Tuple[float, float],
    max_fpr: float,
    grid_size: int,
) -> Tuple[GateConfig, Dict[str, object], List[Dict[str, float]]]:
    selected_by_score: Dict[str, Dict[str, float]] = {}
    tables: List[Dict[str, float]] = []
    updates: Dict[str, float] = {}
    for score_field, gate_field in V2_THRESHOLD_FIELDS.items():
        selected, table = choose_threshold(
            records,
            score_field=score_field,
            positive_tau=positive_tau,
            negative_tau=negative_tau,
            target_activation=target_activation,
            max_fpr=max_fpr,
            grid_size=grid_size,
        )
        selected_by_score[score_field] = selected
        for row in table:
            tables.append(row)
        updates[gate_field] = float(selected["threshold"])

    valid_fields = {field.name for field in fields(GateConfig)}
    gate_kwargs = {key: value for key, value in {**vars(base_gate_config), **updates}.items() if key in valid_fields}
    return GateConfig(**gate_kwargs), selected_by_score, tables


def write_rows(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated label=observation_dir entries.")
    parser.add_argument("--glob", default="*_mixed_eval_records.csv")
    parser.add_argument("--calibration-label", default="N10")
    parser.add_argument("--base-gate-config", default="revision/experiments/outputs/gate_calibration_sweep_v1/gate_config.json")
    parser.add_argument("--positive-tau", type=float, default=0.30)
    parser.add_argument("--negative-tau", type=float, default=0.20)
    parser.add_argument("--target-activation", default="0.20,0.40")
    parser.add_argument("--max-fpr", type=float, default=0.10)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--output-dir", default="revision/experiments/outputs/n_invariant_gate_diagnostic_v1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = read_records(parse_inputs(args.inputs), args.glob)
    if not records:
        raise SystemExit("No records matched the provided inputs")

    write_feature_summary(output_dir / "feature_summary.csv", records)
    write_predictive_metrics(output_dir / "predictive_metrics.csv", records, args.positive_tau, args.negative_tau)

    calibration_records = [row for row in records if row["label"] == args.calibration_label]
    if not calibration_records:
        raise SystemExit(f"No calibration records found for label {args.calibration_label}")

    base_gate_config = load_gate_config(args.base_gate_config) or GateConfig()
    target_activation = parse_target_range(args.target_activation)
    v2_gate_config, selected, threshold_table = calibrate_v2_config(
        calibration_records,
        base_gate_config=base_gate_config,
        positive_tau=args.positive_tau,
        negative_tau=args.negative_tau,
        target_activation=target_activation,
        max_fpr=args.max_fpr,
        grid_size=args.grid_size,
    )
    write_rows(
        output_dir / "threshold_grid.csv",
        threshold_table,
        [
            "score_field",
            "threshold",
            "activation_rate",
            "low_activation_rate",
            "high_activation_rate",
            "tpr_low_collusive",
            "tpr_low_collusive_n",
            "fpr_high_normal",
            "fpr_high_normal_n",
            "in_target_activation",
            "objective",
            "selection_feasible",
        ],
    )

    output = {
        "metadata": {
            "inputs": args.inputs,
            "glob": args.glob,
            "calibration_label": args.calibration_label,
            "n_records": len(records),
            "n_calibration_records": len(calibration_records),
            "base_gate_config": args.base_gate_config,
            "positive_definition": {"risk_state": "low", "collusion_gap_gt": args.positive_tau},
            "negative_definition": {"risk_state": "high", "collusion_gap_lt": args.negative_tau},
            "target_activation": {"low": target_activation[0], "high": target_activation[1]},
            "max_fpr": args.max_fpr,
            "grid_size": args.grid_size,
        },
        "selected_thresholds": selected,
        "gate_config": asdict(v2_gate_config),
    }
    (output_dir / "v2_gate_config.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nSaved N-invariant gate diagnostic to {output_dir}")


if __name__ == "__main__":
    main()
