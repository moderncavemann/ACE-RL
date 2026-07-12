#!/usr/bin/env python3
"""Diagnose action/risk entanglement after adaptive ABIDES training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_mappo_adaptive_pilot import (  # noqa: E402
    trainer_from_payload,
)
from run_abides_mappo_full_main import (  # noqa: E402
    atomic_write_json,
    has_excess_spread,
    has_modal_undercutter,
    resolve_docker_image_id,
    sha256_file,
)
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.envs.abides_docker_env import ABIDESDockerDealerEnv  # noqa: E402
from src.interventions.abides_gate import (  # noqa: E402
    FEATURE_NAMES,
    ABIDESGateActivationController,
    ABIDESLearnedGateModel,
    ABIDESTwoStageGateModel,
    abides_gate_features,
)


def load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"empty passive record set: {path}")
    return records


def opportunity(features: Sequence[float]) -> bool:
    values = dict(zip(FEATURE_NAMES, features))
    return bool(
        float(values["contract_severity"]) > 0.0
        and float(values["eligible_fraction"]) > 0.0
    )


def target(record: Mapping[str, Any]) -> bool:
    return bool(record["risk_state"] == "low" and opportunity(record["features"]))


def gate_metrics(
    records: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    risk = np.asarray([str(row["risk_state"]) for row in records])
    labels = np.asarray([target(row) for row in records], dtype=bool)
    opportunities = np.asarray(
        [opportunity(row["features"]) for row in records], dtype=bool
    )
    risk_proposed = probabilities >= threshold
    active = risk_proposed & opportunities
    high = risk == "high"
    true_positive = int(np.sum(active & labels))
    return {
        "activation_rate": float(np.mean(active)),
        "proposal_rate": float(np.mean(risk_proposed)),
        "risk_high_fpr": (float(np.mean(risk_proposed[high])) if np.any(high) else 0.0),
        "high_risk_activation_rate": (
            float(np.mean(active[high])) if np.any(high) else 0.0
        ),
        "target_recall": true_positive / max(int(np.sum(labels)), 1),
        "precision": true_positive / max(int(np.sum(active)), 1),
        "opportunity_rate": float(np.mean(opportunities)),
        "low_risk_prevalence": float(np.mean(risk == "low")),
    }


def fit_two_stage_risk_model(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    min_activation: float,
    max_activation: float,
    max_high_risk_activation: float,
) -> tuple[ABIDESTwoStageGateModel, dict[str, Any]]:
    indices = [FEATURE_NAMES.index(name) for name in feature_names]
    train_x = np.asarray(
        [[row["features"][index] for index in indices] for row in train_records],
        dtype=float,
    )
    train_y = np.asarray(
        [row["risk_state"] == "low" for row in train_records], dtype=int
    )
    validation_x = np.asarray(
        [[row["features"][index] for index in indices] for row in validation_records],
        dtype=float,
    )
    validation_y = np.asarray(
        [row["risk_state"] == "low" for row in validation_records], dtype=int
    )
    feature_mean = np.mean(train_x, axis=0)
    feature_scale = np.std(train_x, axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=42,
        solver="lbfgs",
    )
    classifier.fit((train_x - feature_mean) / feature_scale, train_y)
    validation_probabilities = classifier.predict_proba(
        (validation_x - feature_mean) / feature_scale
    )[:, 1]
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 981):
        metrics = gate_metrics(
            validation_records, validation_probabilities, float(threshold)
        )
        if (
            min_activation <= metrics["activation_rate"] <= max_activation
            and metrics["risk_high_fpr"] <= max_high_risk_activation
        ):
            candidates.append((float(threshold), metrics))
    if not candidates:
        raise RuntimeError("no two-stage threshold passes passive constraints")
    threshold, validation_metrics = max(
        candidates,
        key=lambda row: (
            row[1]["target_recall"],
            row[1]["precision"],
            -row[1]["risk_high_fpr"],
        ),
    )
    model = ABIDESTwoStageGateModel(
        feature_names=tuple(feature_names),
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        coefficients=tuple(float(value) for value in classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        probability_threshold=threshold,
    )
    return model, {
        "risk_auc": float(roc_auc_score(validation_y, validation_probabilities)),
        "gate_metrics": validation_metrics,
    }


def load_adaptive_trainer(adaptive_output: Path, seed: int, method: str):
    checkpoint_path = adaptive_output / "seeds" / f"seed_{seed}" / "training_state.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return trainer_from_payload(payload["trainers"][method], "cpu")


def collect_policy_records(
    config: Mapping[str, Any],
    source_model: ABIDESLearnedGateModel,
    risk_model: ABIDESTwoStageGateModel,
) -> list[dict[str, Any]]:
    adaptive_output = REPO / str(config["adaptive_pilot_output"])
    rows = []
    episode_steps = int(config["episode_steps"])
    for training_seed in config["adaptive_training_seeds"]:
        for policy_variant in config["policy_variants"]:
            trainer = load_adaptive_trainer(
                adaptive_output, int(training_seed), str(policy_variant)
            )
            with ABIDESDockerDealerEnv({"episode_steps": episode_steps}) as env:
                for environment_seed in config["diagnostic_environment_seeds"]:
                    current_controller = ABIDESGateActivationController(
                        episode_steps,
                        max_activation_rate=float(config["max_activation_rate"]),
                    )
                    candidate_controller = ABIDESGateActivationController(
                        episode_steps,
                        max_activation_rate=float(config["max_activation_rate"]),
                    )
                    episode_rows = []

                    def diagnostic_decision(state, actions, step):
                        features = abides_gate_features(state, actions)
                        source_probability = source_model.probability(features)
                        risk_probability = risk_model.probability(features)
                        has_opportunity = has_excess_spread(
                            actions
                        ) and has_modal_undercutter(actions)
                        source_proposed = (
                            source_probability >= source_model.probability_threshold
                        )
                        candidate_proposed = bool(
                            risk_probability >= risk_model.probability_threshold
                            and has_opportunity
                        )
                        candidate_risk_proposed = bool(
                            risk_probability >= risk_model.probability_threshold
                        )
                        episode_rows.append(
                            {
                                "training_seed": int(training_seed),
                                "policy_variant": str(policy_variant),
                                "environment_seed": int(environment_seed),
                                "step": int(step),
                                "risk_state": str(
                                    state["info"].get("diagnostic_current_risk")
                                ),
                                "features": features.tolist(),
                                "opportunity": has_opportunity,
                                "source_probability": source_probability,
                                "source_proposed": source_proposed,
                                "source_active": current_controller.decide(
                                    source_proposed
                                ),
                                "candidate_risk_probability": risk_probability,
                                "candidate_risk_proposed": candidate_risk_proposed,
                                "candidate_proposed": candidate_proposed,
                                "candidate_active": candidate_controller.decide(
                                    candidate_proposed
                                ),
                            }
                        )
                        return False

                    rollout_episode(
                        env,
                        trainer,
                        seed=int(environment_seed),
                        episode_steps=episode_steps,
                        train=False,
                        gate_decision=diagnostic_decision,
                        routing_enabled=False,
                        action_namespace=int(config["action_namespace"]),
                    )
                    rows.extend(episode_rows)
    return rows


def rescore_candidate_records(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    risk_model: ABIDESTwoStageGateModel,
) -> list[dict[str, Any]]:
    rescored = []
    controller = None
    previous_episode = None
    for source_row in records:
        row = dict(source_row)
        episode_key = (
            int(row["training_seed"]),
            str(row["policy_variant"]),
            int(row["environment_seed"]),
        )
        if episode_key != previous_episode:
            controller = ABIDESGateActivationController(
                int(config["episode_steps"]),
                max_activation_rate=float(config["max_activation_rate"]),
            )
            previous_episode = episode_key
        assert controller is not None
        probability = risk_model.probability(row["features"])
        risk_proposed = probability >= risk_model.probability_threshold
        proposed = bool(risk_proposed and row["opportunity"])
        row.update(
            {
                "candidate_risk_probability": probability,
                "candidate_risk_proposed": bool(risk_proposed),
                "candidate_proposed": proposed,
                "candidate_active": controller.decide(proposed),
            }
        )
        rescored.append(row)
    return rescored


def adaptive_metrics(
    records: Sequence[Mapping[str, Any]],
    active_key: str,
    proposed_key: str,
    risk_proposed_key: str | None = None,
) -> dict[str, float]:
    risk = np.asarray([row["risk_state"] for row in records])
    active = np.asarray([bool(row[active_key]) for row in records])
    proposed = np.asarray([bool(row[proposed_key]) for row in records])
    opportunity_values = np.asarray([bool(row["opportunity"]) for row in records])
    labels = (risk == "low") & opportunity_values
    high = risk == "high"
    true_positive = int(np.sum(active & labels))
    risk_proposed = (
        np.asarray([bool(row[risk_proposed_key]) for row in records])
        if risk_proposed_key is not None
        else proposed
    )
    return {
        "activation_rate": float(np.mean(active)),
        "proposal_rate": float(np.mean(proposed)),
        "high_risk_activation_rate": (
            float(np.mean(active[high])) if np.any(high) else 0.0
        ),
        "risk_high_fpr": (float(np.mean(risk_proposed[high])) if np.any(high) else 0.0),
        "target_recall": true_positive / max(int(np.sum(labels)), 1),
        "precision": true_positive / max(int(np.sum(active)), 1),
        "opportunity_rate": float(np.mean(opportunity_values)),
    }


def summarize_adaptive_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    policies = {}
    for policy_variant in config["policy_variants"]:
        subset = [row for row in records if row["policy_variant"] == policy_variant]
        policies[str(policy_variant)] = {
            "source_joint_gate": adaptive_metrics(
                subset, "source_active", "source_proposed"
            ),
            "two_stage_gate": adaptive_metrics(
                subset,
                "candidate_active",
                "candidate_proposed",
                "candidate_risk_proposed",
            ),
        }
    learned_candidate = policies["learned_gate"]["two_stage_gate"]
    checks = {
        "activation_in_range": float(config["minimum_activation_rate"])
        <= learned_candidate["activation_rate"]
        <= float(config["max_activation_rate"]),
        "high_risk_activation_controlled": learned_candidate[
            "high_risk_activation_rate"
        ]
        <= float(config["maximum_high_risk_activation_rate"]),
        "risk_classifier_high_fpr_controlled": learned_candidate["risk_high_fpr"]
        <= float(config["maximum_high_risk_activation_rate"]),
        "target_recall_material": learned_candidate["target_recall"]
        >= float(config["minimum_target_recall"]),
        "source_gate_saturation_reproduced": policies["learned_gate"][
            "source_joint_gate"
        ]["proposal_rate"]
        >= 0.70,
    }
    return {"policies": policies, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_adaptive_gate_shift_diagnostic_v1.json",
    )
    args = parser.parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    source_path = REPO / config["source_gate_artifact"]
    source_hash = sha256_file(source_path)
    if source_hash != config["source_gate_artifact_sha256"]:
        raise RuntimeError("source gate artifact hash mismatch")
    docker_image_id = resolve_docker_image_id(str(config["docker_image"]))
    train_records = load_records(REPO / config["passive_train_records"])
    validation_records = load_records(REPO / config["passive_validation_records"])
    risk_model, passive_validation = fit_two_stage_risk_model(
        train_records,
        validation_records,
        config["risk_feature_names"],
        float(config["minimum_activation_rate"]),
        float(config["max_activation_rate"]),
        float(config["maximum_high_risk_activation_rate"]),
    )
    source_model = ABIDESLearnedGateModel.load(source_path)
    output_dir = REPO / config["output_dir"]
    adaptive_records_path = output_dir / "adaptive_records.json"
    if adaptive_records_path.exists():
        adaptive_records = rescore_candidate_records(
            load_records(adaptive_records_path), config, risk_model
        )
    else:
        adaptive_records = collect_policy_records(config, source_model, risk_model)
    adaptive = summarize_adaptive_records(adaptive_records, config)
    summary = {
        "status": "pass" if all(adaptive["checks"].values()) else "fail",
        "protocol_version": config["protocol_version"],
        "docker_image_id": docker_image_id,
        "source_gate_sha256": source_hash,
        "two_stage_risk_model": risk_model.to_dict(),
        "passive_validation": passive_validation,
        "adaptive": adaptive,
        "n_adaptive_records": len(adaptive_records),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "adaptive_records.json", adaptive_records)
    atomic_write_json(
        output_dir / "two_stage_gate_candidate.json", risk_model.to_dict()
    )
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
