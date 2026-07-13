#!/usr/bin/env python3
"""Fit frozen one-feature ABIDES risk comparators on passive records only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from run_abides_mappo_full_main import atomic_write_json, sha256_file  # noqa: E402
from src.interventions.abides_gate import (  # noqa: E402
    FEATURE_NAMES,
    RISK_FEATURE_NAMES,
    ABIDESPacedSignalController,
    ABIDESPublicRiskProxyModel,
    ABIDESTwoStageGateModel,
)


def load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"empty passive record set: {path}")
    return records


def probabilities(
    records: Sequence[Mapping[str, Any]],
    feature_name: str,
    feature_mean: float,
    feature_scale: float,
    coefficient: float,
    intercept: float,
) -> np.ndarray:
    index = FEATURE_NAMES.index(feature_name)
    values = np.asarray([float(row["features"][index]) for row in records])
    logits = coefficient * ((values - feature_mean) / feature_scale) + intercept
    return 1.0 / (1.0 + np.exp(-logits))


def reference_probabilities(
    records: Sequence[Mapping[str, Any]], model: ABIDESTwoStageGateModel
) -> np.ndarray:
    return np.asarray(
        [model.probability(row["features"]) for row in records], dtype=float
    )


def paced_metrics(
    records: Sequence[Mapping[str, Any]],
    values: Sequence[float],
    probability_threshold: float,
    target_rate: float,
) -> dict[str, float]:
    accepted: list[bool] = []
    high_accepted: list[bool] = []
    low_accepted: list[bool] = []
    proposed: list[bool] = []
    previous_episode: tuple[int, int] | None = None
    controller: ABIDESPacedSignalController | None = None
    episode_steps = 0
    for row, probability in zip(records, values):
        episode = (int(row["seed"]), int(row["episode"]))
        if episode != previous_episode:
            episode_steps = sum(
                1
                for candidate in records
                if (int(candidate["seed"]), int(candidate["episode"])) == episode
            )
            controller = ABIDESPacedSignalController(
                episode_steps, target_rate=target_rate
            )
            previous_episode = episode
        assert controller is not None
        is_proposed = float(probability) >= probability_threshold
        active = controller.decide(float(probability) if is_proposed else 0.0) > 0.0
        accepted.append(active)
        proposed.append(is_proposed)
        if str(row["risk_state"]) == "high":
            high_accepted.append(active)
        else:
            low_accepted.append(active)
    return {
        "activation_rate": float(np.mean(accepted)),
        "proposal_rate": float(np.mean(proposed)),
        "high_risk_activation_rate": float(np.mean(high_accepted)),
        "low_risk_activation_rate": float(np.mean(low_accepted)),
        "episode_steps": float(episode_steps),
    }


def fit_proxy(
    feature_name: str,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    reference_activation: float,
) -> tuple[ABIDESPublicRiskProxyModel, dict[str, Any]]:
    index = FEATURE_NAMES.index(feature_name)
    train_x = np.asarray(
        [[float(row["features"][index])] for row in train_records], dtype=float
    )
    train_y = np.asarray(
        [str(row["risk_state"]) == "low" for row in train_records], dtype=int
    )
    validation_x = np.asarray(
        [[float(row["features"][index])] for row in validation_records], dtype=float
    )
    validation_y = np.asarray(
        [str(row["risk_state"]) == "low" for row in validation_records], dtype=int
    )
    feature_mean = float(np.mean(train_x))
    feature_scale = float(np.std(train_x))
    if feature_scale < 1e-8:
        feature_scale = 1.0
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=42,
        solver="lbfgs",
    )
    classifier.fit((train_x - feature_mean) / feature_scale, train_y)
    coefficient = float(classifier.coef_[0, 0])
    intercept = float(classifier.intercept_[0])
    validation_probabilities = classifier.predict_proba(
        (validation_x - feature_mean) / feature_scale
    )[:, 1]

    probability_threshold = 0.5
    candidates = []
    for target_rate in np.linspace(0.05, 0.35, 301):
        metrics = paced_metrics(
            validation_records,
            validation_probabilities,
            probability_threshold,
            float(target_rate),
        )
        candidates.append(
            (
                abs(metrics["activation_rate"] - reference_activation),
                abs(float(target_rate) - 0.35),
                -float(target_rate),
                float(target_rate),
                metrics,
            )
        )
    _, _, _, pacing_target_rate, validation_metrics = min(candidates)
    model = ABIDESPublicRiskProxyModel(
        feature_name=feature_name,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficient=coefficient,
        intercept=intercept,
        probability_threshold=probability_threshold,
        pacing_target_rate=pacing_target_rate,
    )
    return model, {
        "risk_auc": float(roc_auc_score(validation_y, validation_probabilities)),
        "validation_metrics": validation_metrics,
        "activation_difference_from_reference": float(
            validation_metrics["activation_rate"] - reference_activation
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-records",
        default="revision/experiments/outputs/abides_learned_gate_v1/train_records.json",
    )
    parser.add_argument(
        "--validation-records",
        default="revision/experiments/outputs/abides_learned_gate_v1/validation_records.json",
    )
    parser.add_argument(
        "--reference-gate",
        default="revision/configs/abides_two_stage_gate_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="revision/configs/abides_public_risk_proxies_v1",
    )
    args = parser.parse_args()

    train_path = REPO / args.train_records
    validation_path = REPO / args.validation_records
    reference_path = REPO / args.reference_gate
    output_dir = REPO / args.output_dir
    train_records = load_records(train_path)
    validation_records = load_records(validation_path)
    reference_model = ABIDESTwoStageGateModel.load(reference_path)
    reference_values = reference_probabilities(validation_records, reference_model)
    reference_metrics = paced_metrics(
        validation_records,
        reference_values,
        reference_model.probability_threshold,
        0.35,
    )

    artifacts = {}
    calibration = {}
    for feature_name in RISK_FEATURE_NAMES:
        model, metrics = fit_proxy(
            feature_name,
            train_records,
            validation_records,
            reference_metrics["activation_rate"],
        )
        short_name = (
            "volatility_only"
            if feature_name == "public_volatility"
            else "return_only"
        )
        artifact_path = output_dir / f"{short_name}.json"
        atomic_write_json(artifact_path, model.to_dict())
        artifacts[short_name] = {
            "path": str(artifact_path.relative_to(REPO)),
            "sha256": sha256_file(artifact_path),
        }
        calibration[short_name] = metrics

    summary = {
        "protocol_version": "abides_public_risk_proxies_v1",
        "training_label": "simulator_low_risk",
        "probability_threshold_rule": "fixed_at_0.5",
        "pacing_rule": "match_reference_paced_activation_on_passive_validation",
        "train_records": {
            "path": args.train_records,
            "sha256": sha256_file(train_path),
            "count": len(train_records),
        },
        "validation_records": {
            "path": args.validation_records,
            "sha256": sha256_file(validation_path),
            "count": len(validation_records),
        },
        "reference_gate": {
            "path": args.reference_gate,
            "sha256": sha256_file(reference_path),
            "paced_validation_metrics": reference_metrics,
        },
        "artifacts": artifacts,
        "calibration": calibration,
    }
    atomic_write_json(output_dir / "calibration_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
