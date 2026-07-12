#!/usr/bin/env python3
"""Train a frozen observable gate from passive ABIDES/MAPPO trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.abides_docker_env import ABIDESDockerDealerEnv  # noqa: E402
from src.interventions.abides_gate import (  # noqa: E402
    FEATURE_NAMES,
    ABIDESLearnedGateModel,
    abides_gate_features,
    abides_gate_target,
)


def load_trainer(path: Path, device: str) -> MAPPOTrainer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["trainer_config"])
    config["device"] = device
    trainer = MAPPOTrainer(MAPPOConfig(**config))
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    return trainer


def collect_records(
    env: ABIDESDockerDealerEnv,
    trainer: MAPPOTrainer,
    seeds: Sequence[int],
    episodes_per_seed: int,
    episode_steps: int,
    namespace: int,
) -> list[dict[str, Any]]:
    records = []
    for seed in seeds:
        for episode in range(episodes_per_seed):
            episode_seed = seed * 1_000 + episode
            rollout = rollout_episode(
                env,
                trainer,
                seed=episode_seed,
                episode_steps=episode_steps,
                train=False,
                action_namespace=namespace,
            )
            for row in rollout["records"]:
                features = abides_gate_features(
                    {
                        "info": row["pre_info"],
                        "observations": row["pre_observations"],
                    },
                    row["actions"],
                )
                records.append(
                    {
                        "seed": seed,
                        "episode": episode,
                        "step": row["step"],
                        "features": features.tolist(),
                        "label": int(abides_gate_target(row["info"])),
                        "risk_state": row["info"]["last_step_risk"],
                    }
                )
    return records


def metrics(
    labels: np.ndarray,
    risk: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    active = probabilities >= threshold
    positive = labels == 1
    high = risk == "high"
    tp = int(np.sum(active & positive))
    fp = int(np.sum(active & ~positive))
    precision = tp / max(int(np.sum(active)), 1)
    recall = tp / max(int(np.sum(positive)), 1)
    return {
        "activation_rate": float(np.mean(active)),
        "target_prevalence": float(np.mean(positive)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "negative_fpr": float(fp / max(int(np.sum(~positive)), 1)),
        "high_risk_fpr": float(np.mean(active[high])) if np.any(high) else 0.0,
    }


def select_threshold(
    labels: np.ndarray,
    risk: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 981):
        row = metrics(labels, risk, probabilities, float(threshold))
        if (
            0.05 <= row["activation_rate"] <= 0.35
            and row["high_risk_fpr"] <= 0.15
        ):
            candidates.append((float(threshold), row))
    if not candidates:
        raise RuntimeError("no gate threshold satisfies frozen validation constraints")
    return max(
        candidates,
        key=lambda item: (
            item[1]["f1"],
            -item[1]["high_risk_fpr"],
            -abs(item[0] - 0.5),
        ),
    )


def arrays(records: Sequence[dict[str, Any]]):
    return (
        np.asarray([row["features"] for row in records], dtype=float),
        np.asarray([row["label"] for row in records], dtype=int),
        np.asarray([row["risk_state"] for row in records], dtype=str),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="revision/experiments/outputs/abides_mappo_passive_v1/model.pt",
    )
    parser.add_argument("--train-seeds", default="300000,300001,300002,300003,300004")
    parser.add_argument("--validation-seeds", default="310000,310001,310002,310003,310004")
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--output-dir",
        default="revision/experiments/outputs/abides_learned_gate_v1",
    )
    parser.add_argument(
        "--frozen-artifact",
        default="revision/configs/abides_learned_gate_v1.json",
    )
    args = parser.parse_args()
    train_seeds = [int(value) for value in args.train_seeds.split(",")]
    validation_seeds = [int(value) for value in args.validation_seeds.split(",")]
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("gate train and validation seeds must be disjoint")

    checkpoint = ROOT.parent / args.checkpoint
    trainer = load_trainer(checkpoint, args.device)
    market_config = {"episode_steps": args.episode_steps}
    with ABIDESDockerDealerEnv(market_config) as env:
        train_records = collect_records(
            env,
            trainer,
            train_seeds,
            args.episodes_per_seed,
            args.episode_steps,
            namespace=101_117,
        )
        validation_records = collect_records(
            env,
            trainer,
            validation_seeds,
            args.episodes_per_seed,
            args.episode_steps,
            namespace=101_117,
        )

    train_x, train_y, _ = arrays(train_records)
    validation_x, validation_y, validation_risk = arrays(validation_records)
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale[scale < 1e-8] = 1.0
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=42,
        solver="lbfgs",
    )
    classifier.fit((train_x - mean) / scale, train_y)
    validation_probabilities = classifier.predict_proba(
        (validation_x - mean) / scale
    )[:, 1]
    threshold, validation_metrics = select_threshold(
        validation_y, validation_risk, validation_probabilities
    )
    model = ABIDESLearnedGateModel(
        feature_names=FEATURE_NAMES,
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        probability_threshold=threshold,
    )
    passed = (
        0.05 <= validation_metrics["activation_rate"] <= 0.35
        and validation_metrics["high_risk_fpr"] <= 0.15
        and validation_metrics["f1"] >= 0.50
    )
    output_dir = ROOT.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_artifact = ROOT.parent / args.frozen_artifact
    frozen_artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact_text = json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n"
    frozen_artifact.write_text(artifact_text, encoding="utf-8")
    summary = {
        "status": "pass" if passed else "fail",
        "backend": "abides",
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "validation_metrics": validation_metrics,
        "probability_threshold": threshold,
        "features": list(FEATURE_NAMES),
    }
    (output_dir / "train_records.json").write_text(
        json.dumps(train_records), encoding="utf-8"
    )
    (output_dir / "validation_records.json").write_text(
        json.dumps(validation_records), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
