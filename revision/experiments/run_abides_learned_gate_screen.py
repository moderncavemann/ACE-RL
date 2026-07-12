#!/usr/bin/env python3
"""G5 frozen learned-gate screen against exact-dose timing controls."""

from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Sequence

import numpy as np
import torch
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.abides_docker_env import ABIDESDockerDealerEnv  # noqa: E402
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESGateActivationController,
    ABIDESLearnedGateModel,
)


def load_trainer(path: Path, device: str) -> MAPPOTrainer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["trainer_config"])
    config["device"] = device
    trainer = MAPPOTrainer(MAPPOConfig(**config))
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    return trainer


def shuffled_within_risk(
    schedule: Sequence[bool], risk_path: Sequence[str], seed: int
) -> list[bool]:
    result = np.zeros(len(schedule), dtype=bool)
    rng = np.random.default_rng(seed)
    for risk in ("low", "high"):
        indices = np.asarray(
            [index for index, value in enumerate(risk_path) if value == risk],
            dtype=int,
        )
        values = np.asarray([schedule[index] for index in indices], dtype=bool)
        rng.shuffle(values)
        result[indices] = values
    return result.tolist()


def circular_within_risk(
    schedule: Sequence[bool], risk_path: Sequence[str], seed: int
) -> list[bool]:
    result = np.zeros(len(schedule), dtype=bool)
    rng = np.random.default_rng(seed)
    for risk in ("low", "high"):
        indices = np.asarray(
            [index for index, value in enumerate(risk_path) if value == risk],
            dtype=int,
        )
        values = np.asarray([schedule[index] for index in indices], dtype=bool)
        if len(values) > 1:
            values = np.roll(values, int(rng.integers(1, len(values))))
        result[indices] = values
    return result.tolist()


def exact_dose(
    source: Sequence[bool], comparator: Sequence[bool], risk_path: Sequence[str]
) -> bool:
    return (
        len(source) == len(comparator) == len(risk_path)
        and sum(source) == sum(comparator)
        and all(
            sum(active for active, value in zip(source, risk_path) if value == risk)
            == sum(active for active, value in zip(comparator, risk_path) if value == risk)
            for risk in ("low", "high")
        )
    )


def summarize(rollout: dict) -> dict[str, float]:
    records = rollout["records"]
    low_spreads = [
        float(row["info"]["mean_execution_half_spread"])
        for row in records
        if row["info"]["last_step_risk"] == "low"
    ]
    high_records = [
        row for row in records if row["info"]["last_step_risk"] == "high"
    ]
    return {
        "low_execution_half_spread": float(mean(low_spreads)),
        "mean_reward": float(mean(row["mean_reward_unscaled"] for row in records)),
        "mean_volume": float(
            mean(float(row["info"]["realized_flow_quantity"]) for row in records)
        ),
        "activation_rate": float(mean(bool(row["gate_active"]) for row in records)),
        "high_risk_activation_rate": float(
            mean(bool(row["gate_active"]) for row in high_records)
            if high_records
            else 0.0
        ),
        "targeted_events": float(
            sum(int(row["info"]["targeted_event_count"]) for row in records)
        ),
    }


def paired_summary(delta: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(delta, dtype=float)
    avg = float(np.mean(values))
    se = float(np.std(values, ddof=1) / sqrt(len(values)))
    critical = float(t.ppf(0.975, len(values) - 1))
    return {
        "mean": avg,
        "se": se,
        "ci_low": avg - critical * se,
        "ci_high": avg + critical * se,
        "positive": int(np.sum(values > 1e-9)),
        "ties": int(np.sum(np.abs(values) <= 1e-9)),
        "negative": int(np.sum(values < -1e-9)),
    }


def parse_seeds(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "holdout"), default="screen")
    parser.add_argument("--seeds", default="320000,320001,320002,320003,320004")
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--checkpoint",
        default="revision/experiments/outputs/abides_mappo_passive_v1/model.pt",
    )
    parser.add_argument(
        "--gate-artifact",
        default="revision/configs/abides_learned_gate_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="revision/experiments/outputs/abides_learned_gate_screen_v1",
    )
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if len(seeds) != 5:
        raise ValueError("G5 screen and holdout each require exactly five seeds")
    trainer = load_trainer(ROOT.parent / args.checkpoint, args.device)
    model = ABIDESLearnedGateModel.load(ROOT.parent / args.gate_artifact)
    rows = []

    with ABIDESDockerDealerEnv({"episode_steps": args.episode_steps}) as env:
        for seed in seeds:
            controller = ABIDESGateActivationController(args.episode_steps)

            def learned_decision(state, actions, step):
                proposal = bool(model.decide(state, actions)["active"])
                return controller.decide(proposal)

            learned_rollout = rollout_episode(
                env,
                trainer,
                seed=seed,
                episode_steps=args.episode_steps,
                train=False,
                gate_decision=learned_decision,
                action_namespace=131_117,
            )
            schedule = [
                bool(record["gate_active"])
                for record in learned_rollout["records"]
            ]
            risk_path = [
                str(record["info"]["last_step_risk"])
                for record in learned_rollout["records"]
            ]
            matched = shuffled_within_risk(
                schedule, risk_path, seed=seed ^ 0xA117
            )
            circular = circular_within_risk(
                schedule, risk_path, seed=seed ^ 0xC117
            )
            if not exact_dose(schedule, matched, risk_path):
                raise RuntimeError("matched random dose is not exact")
            if not exact_dose(schedule, circular, risk_path):
                raise RuntimeError("circular shuffled dose is not exact")
            matched_rollout = rollout_episode(
                env,
                trainer,
                seed=seed,
                episode_steps=args.episode_steps,
                train=False,
                gate_schedule=matched,
                action_namespace=131_117,
            )
            circular_rollout = rollout_episode(
                env,
                trainer,
                seed=seed,
                episode_steps=args.episode_steps,
                train=False,
                gate_schedule=circular,
                action_namespace=131_117,
            )
            no_intervention_rollout = rollout_episode(
                env,
                trainer,
                seed=seed,
                episode_steps=args.episode_steps,
                train=False,
                gate_schedule=[False] * args.episode_steps,
                action_namespace=131_117,
            )
            row = {
                "seed": seed,
                "learned": summarize(learned_rollout),
                "matched_random": summarize(matched_rollout),
                "circular_shuffled": summarize(circular_rollout),
                "no_intervention": summarize(no_intervention_rollout),
                "schedule": schedule,
                "risk_path": risk_path,
                "matched_schedule": matched,
                "circular_schedule": circular,
                "raw_proposal_rate": controller.proposal_rate,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "activation": row["learned"]["activation_rate"],
                        "matched_minus_learned": (
                            row["matched_random"]["low_execution_half_spread"]
                            - row["learned"]["low_execution_half_spread"]
                        ),
                        "circular_minus_learned": (
                            row["circular_shuffled"]["low_execution_half_spread"]
                            - row["learned"]["low_execution_half_spread"]
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    matched_delta = [
        row["matched_random"]["low_execution_half_spread"]
        - row["learned"]["low_execution_half_spread"]
        for row in rows
    ]
    circular_delta = [
        row["circular_shuffled"]["low_execution_half_spread"]
        - row["learned"]["low_execution_half_spread"]
        for row in rows
    ]
    matched_summary = paired_summary(matched_delta)
    circular_summary = paired_summary(circular_delta)
    learned_reward = float(mean(row["learned"]["mean_reward"] for row in rows))
    matched_reward = float(
        mean(row["matched_random"]["mean_reward"] for row in rows)
    )
    checks = {
        "activation_in_range": all(
            0.05 <= row["learned"]["activation_rate"] <= 0.35 for row in rows
        ),
        "matched_positive_4_of_5": matched_summary["positive"] >= 4,
        "circular_positive_4_of_5": circular_summary["positive"] >= 4,
        "matched_mean_positive": matched_summary["mean"] > 0.0,
        "circular_mean_positive": circular_summary["mean"] > 0.0,
        "reward_guardrail": learned_reward
        >= matched_reward - 0.10 * max(1.0, abs(matched_reward)),
        "volume_conserved": all(
            abs(row[arm]["mean_volume"] - 20.0) < 1e-9
            for row in rows
            for arm in ("learned", "matched_random", "circular_shuffled")
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "fail",
        "stage": args.stage,
        "backend": "abides",
        "seeds": seeds,
        "checks": checks,
        "matched_random_minus_learned": matched_summary,
        "circular_shuffled_minus_learned": circular_summary,
        "learned_mean_reward": learned_reward,
        "matched_mean_reward": matched_reward,
        "activation_rates": [row["learned"]["activation_rate"] for row in rows],
        "raw_proposal_rates": [row["raw_proposal_rate"] for row in rows],
    }
    output_dir = ROOT.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.stage}_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / f"{args.stage}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
