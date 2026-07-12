#!/usr/bin/env python3
"""Resumable adaptive-policy ABIDES/MAPPO acceptance pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_learned_gate_screen import (  # noqa: E402
    exact_dose,
    paired_summary,
    shuffled_within_risk,
    summarize,
)
from run_abides_mappo_full_main import (  # noqa: E402
    RunLock,
    append_logbook,
    atomic_write_json,
    atomic_write_text,
    current_git_commit,
    has_excess_spread,
    has_modal_undercutter,
    protocol_fingerprint,
    resolve_docker_image_id,
    sha256_file,
    utc_now,
)
from src.agents.abides_mappo import make_abides_trainer, rollout_episode  # noqa: E402
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.abides_docker_env import ABIDESDockerDealerEnv  # noqa: E402
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESGateActivationController,
    ABIDESLearnedGateModel,
)

DEPLOYED_METHODS = (
    "no_intervention",
    "learned_gate",
    "matched_random",
    "spread_only_threshold",
    "always_on",
)
GATE_OFF_METHODS = ("learned_gate_off", "matched_random_gate_off")


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_seed_event(seed_dir: Path, event: Mapping[str, Any]) -> None:
    append_logbook(seed_dir / "training_log.jsonl", dict(event))


def trainer_payload(trainer: MAPPOTrainer) -> dict[str, Any]:
    return {
        "config": asdict(trainer.config),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in trainer.model.state_dict().items()
        },
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "np_rng_state": trainer.np_rng.bit_generator.state,
        "action_call_count": int(trainer.action_call_count),
    }


def trainer_from_payload(payload: Mapping[str, Any], device: str) -> MAPPOTrainer:
    trainer_config = dict(payload["config"])
    trainer_config["device"] = device
    trainer = MAPPOTrainer(MAPPOConfig(**trainer_config))
    trainer.model.load_state_dict(payload["model_state_dict"])
    trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    trainer.np_rng.bit_generator.state = payload["np_rng_state"]
    trainer.action_call_count = int(payload.get("action_call_count", 0))
    return trainer


def model_digest(trainer: MAPPOTrainer) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(trainer.model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_directory(output_dir: Path, seed: int) -> Path:
    return output_dir / "seeds" / f"seed_{seed}"


def seed_result_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "results" / f"seed_{seed}.json"


def training_checkpoint_path(output_dir: Path, seed: int) -> Path:
    return seed_directory(output_dir, seed) / "training_state.pt"


def safe_rollout_summary(rollout: Mapping[str, Any]) -> dict[str, float | None]:
    records = list(rollout["records"])
    low_values = [
        float(row["info"]["mean_execution_half_spread"])
        for row in records
        if row["info"].get("last_step_risk") == "low"
    ]
    high_records = [
        row for row in records if row["info"].get("last_step_risk") == "high"
    ]
    return {
        "low_execution_half_spread": (float(mean(low_values)) if low_values else None),
        "mean_half_spread": float(rollout["mean_half_spread"]),
        "mean_reward": float(rollout["mean_reward"]),
        "mean_volume": float(
            mean(float(row["info"]["realized_flow_quantity"]) for row in records)
        ),
        "activation_rate": float(rollout["activation_rate"]),
        "high_risk_activation_rate": (
            float(mean(bool(row["gate_active"]) for row in high_records))
            if high_records
            else 0.0
        ),
        "targeted_events": float(
            sum(int(row["info"]["targeted_event_count"]) for row in records)
        ),
    }


def training_history_row(
    episode: int,
    environment_seed: int,
    rollout: Mapping[str, Any],
    raw_proposal_rate: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode": episode,
        "environment_seed": environment_seed,
        **safe_rollout_summary(rollout),
    }
    row.update({key: float(value) for key, value in rollout["update"].items()})
    if raw_proposal_rate is not None:
        row["raw_proposal_rate"] = float(raw_proposal_rate)
    if not all(
        math.isfinite(float(value))
        for key, value in row.items()
        if key not in {"episode", "environment_seed"} and value is not None
    ):
        raise FloatingPointError(f"non-finite training history row: {row}")
    return row


def validate_common_path(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    reference_records = reference["records"]
    candidate_records = candidate["records"]
    return len(reference_records) == len(candidate_records) and all(
        str(left["info"].get("path_digest")) == str(right["info"].get("path_digest"))
        and str(left["info"].get("last_step_risk"))
        == str(right["info"].get("last_step_risk"))
        for left, right in zip(reference_records, candidate_records)
    )


def make_trainers(config: Mapping[str, Any], seed: int) -> dict[str, MAPPOTrainer]:
    trainers = {
        method: make_abides_trainer(
            seed=seed,
            episode_steps=int(config["episode_steps"]),
            device=str(config["device"]),
        )
        for method in config["methods"]
    }
    digests = {model_digest(trainer) for trainer in trainers.values()}
    if len(digests) != 1:
        raise RuntimeError("adaptive methods did not start from identical models")
    return trainers


def training_environment_seed(
    config: Mapping[str, Any], seed: int, episode: int
) -> int:
    value = (
        int(config["train_environment_seed_namespace"])
        + (int(seed) % 100_000) * 1_000
        + int(episode)
    )
    if not 0 <= value < 2**32:
        raise ValueError(f"training environment seed outside uint32 range: {value}")
    return value


def save_training_checkpoint(
    output_dir: Path,
    seed: int,
    fingerprint: str,
    completed_episodes: int,
    trainers: Mapping[str, MAPPOTrainer],
    history: Mapping[str, Sequence[Mapping[str, Any]]],
    training_dose_exact: bool,
    training_paths_equal: bool,
) -> None:
    atomic_torch_save(
        training_checkpoint_path(output_dir, seed),
        {
            "protocol_fingerprint": fingerprint,
            "seed": seed,
            "completed_episodes": completed_episodes,
            "trainers": {
                method: trainer_payload(trainer) for method, trainer in trainers.items()
            },
            "history": {method: list(rows) for method, rows in history.items()},
            "training_dose_exact": bool(training_dose_exact),
            "training_paths_equal": bool(training_paths_equal),
            "saved_at": utc_now(),
        },
    )
    atomic_write_json(
        seed_directory(output_dir, seed) / "progress.json",
        {
            "seed": seed,
            "completed_episodes": completed_episodes,
            "status": "training",
            "updated_at": utc_now(),
        },
    )


def load_or_initialize_training(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    fingerprint: str,
) -> tuple[
    dict[str, MAPPOTrainer],
    dict[str, list[dict[str, Any]]],
    int,
    bool,
    bool,
]:
    checkpoint = training_checkpoint_path(output_dir, seed)
    if not checkpoint.exists():
        trainers = make_trainers(config, seed)
        history = {method: [] for method in config["methods"]}
        return trainers, history, 0, True, True

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol_fingerprint") != fingerprint:
        raise RuntimeError(f"training checkpoint fingerprint mismatch: {checkpoint}")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"training checkpoint seed mismatch: {checkpoint}")
    trainers = {
        method: trainer_from_payload(payload["trainers"][method], str(config["device"]))
        for method in config["methods"]
    }
    history = {method: list(payload["history"][method]) for method in config["methods"]}
    completed = int(payload["completed_episodes"])
    if any(len(rows) != completed for rows in history.values()):
        raise RuntimeError("training checkpoint histories are not episode-aligned")
    return (
        trainers,
        history,
        completed,
        bool(payload["training_dose_exact"]),
        bool(payload["training_paths_equal"]),
    )


def train_seed_policies(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    fingerprint: str,
    model: ABIDESLearnedGateModel,
) -> tuple[
    dict[str, MAPPOTrainer],
    dict[str, list[dict[str, Any]]],
    bool,
    bool,
]:
    (
        trainers,
        history,
        completed,
        training_dose_exact,
        training_paths_equal,
    ) = load_or_initialize_training(config, output_dir, seed, fingerprint)
    train_episodes = int(config["train_episodes"])
    episode_steps = int(config["episode_steps"])
    if completed > train_episodes:
        raise RuntimeError("checkpoint is beyond configured training horizon")
    if completed:
        append_seed_event(
            seed_directory(output_dir, seed),
            {"event": "training_resumed", "completed_episodes": completed},
        )

    with ABIDESDockerDealerEnv({"episode_steps": episode_steps}) as env:
        for episode in range(completed, train_episodes):
            environment_seed = training_environment_seed(config, seed, episode)
            controller = ABIDESGateActivationController(
                episode_steps,
                max_activation_rate=float(config["max_activation_rate"]),
            )

            def learned_decision(state, actions, step):
                return controller.decide(bool(model.decide(state, actions)["active"]))

            learned = rollout_episode(
                env,
                trainers["learned_gate"],
                seed=environment_seed,
                episode_steps=episode_steps,
                train=True,
                gate_decision=learned_decision,
                reward_scale=float(config["reward_scale"]),
                action_namespace=int(config["train_action_namespace"]),
            )
            learned_schedule = [bool(row["gate_active"]) for row in learned["records"]]
            risk_path = [
                str(row["info"]["last_step_risk"]) for row in learned["records"]
            ]
            matched_schedule = shuffled_within_risk(
                learned_schedule,
                risk_path,
                seed=environment_seed ^ seed ^ 0xA117,
            )
            episode_dose_exact = exact_dose(
                learned_schedule, matched_schedule, risk_path
            )
            if not episode_dose_exact:
                raise RuntimeError("adaptive training matched dose is not exact")
            matched = rollout_episode(
                env,
                trainers["matched_random"],
                seed=environment_seed,
                episode_steps=episode_steps,
                train=True,
                gate_schedule=matched_schedule,
                reward_scale=float(config["reward_scale"]),
                action_namespace=int(config["train_action_namespace"]),
            )
            no_intervention = rollout_episode(
                env,
                trainers["no_intervention"],
                seed=environment_seed,
                episode_steps=episode_steps,
                train=True,
                gate_schedule=[False] * episode_steps,
                reward_scale=float(config["reward_scale"]),
                action_namespace=int(config["train_action_namespace"]),
            )
            threshold_controller = ABIDESGateActivationController(
                episode_steps,
                max_activation_rate=float(config["max_activation_rate"]),
            )

            def threshold_decision(state, actions, step):
                proposal = has_excess_spread(actions) and has_modal_undercutter(actions)
                return threshold_controller.decide(proposal)

            spread_only = rollout_episode(
                env,
                trainers["spread_only_threshold"],
                seed=environment_seed,
                episode_steps=episode_steps,
                train=True,
                gate_decision=threshold_decision,
                reward_scale=float(config["reward_scale"]),
                action_namespace=int(config["train_action_namespace"]),
            )
            always_on = rollout_episode(
                env,
                trainers["always_on"],
                seed=environment_seed,
                episode_steps=episode_steps,
                train=True,
                gate_schedule=[True] * episode_steps,
                reward_scale=float(config["reward_scale"]),
                action_namespace=int(config["train_action_namespace"]),
            )
            rollouts = {
                "learned_gate": learned,
                "matched_random": matched,
                "no_intervention": no_intervention,
                "spread_only_threshold": spread_only,
                "always_on": always_on,
            }
            episode_paths_equal = all(
                validate_common_path(learned, rollout) for rollout in rollouts.values()
            )
            if not episode_paths_equal:
                raise RuntimeError("adaptive methods did not share the exogenous path")

            history["learned_gate"].append(
                training_history_row(
                    episode,
                    environment_seed,
                    learned,
                    raw_proposal_rate=controller.proposal_rate,
                )
            )
            history["matched_random"].append(
                training_history_row(episode, environment_seed, matched)
            )
            history["no_intervention"].append(
                training_history_row(episode, environment_seed, no_intervention)
            )
            history["spread_only_threshold"].append(
                training_history_row(
                    episode,
                    environment_seed,
                    spread_only,
                    raw_proposal_rate=threshold_controller.proposal_rate,
                )
            )
            history["always_on"].append(
                training_history_row(episode, environment_seed, always_on)
            )
            training_dose_exact = training_dose_exact and episode_dose_exact
            training_paths_equal = training_paths_equal and episode_paths_equal
            save_training_checkpoint(
                output_dir,
                seed,
                fingerprint,
                episode + 1,
                trainers,
                history,
                training_dose_exact,
                training_paths_equal,
            )
            append_seed_event(
                seed_directory(output_dir, seed),
                {
                    "event": "training_episode_completed",
                    "episode": episode + 1,
                    "train_episodes": train_episodes,
                    "learned_activation": learned["activation_rate"],
                    "learned_spread": learned["mean_half_spread"],
                    "matched_spread": matched["mean_half_spread"],
                },
            )
            if (episode + 1) % 10 == 0 or episode + 1 == train_episodes:
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "episode": episode + 1,
                            "train_episodes": train_episodes,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return trainers, history, training_dose_exact, training_paths_equal


def convergence_summary(
    history: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    window = int(config["convergence_window_episodes"])
    methods: dict[str, Any] = {}
    for method, rows in history.items():
        if len(rows) < 2 * window:
            methods[method] = {
                "stable": False,
                "reason": f"requires at least {2 * window} episodes",
            }
            continue
        previous = rows[-2 * window : -window]
        recent = rows[-window:]
        previous_spread = float(
            mean(float(row["mean_half_spread"]) for row in previous)
        )
        recent_spread = float(mean(float(row["mean_half_spread"]) for row in recent))
        previous_reward = float(mean(float(row["mean_reward"]) for row in previous))
        recent_reward = float(mean(float(row["mean_reward"]) for row in recent))
        slope = float(
            np.polyfit(
                np.arange(window, dtype=float),
                np.asarray([float(row["mean_half_spread"]) for row in recent]),
                1,
            )[0]
        )
        reward_shift_fraction = abs(recent_reward - previous_reward) / max(
            1.0, abs(previous_reward)
        )
        stable = (
            abs(slope) <= float(config["maximum_recent_spread_slope"])
            and abs(recent_spread - previous_spread)
            <= float(config["maximum_window_spread_shift"])
            and reward_shift_fraction
            <= float(config["maximum_window_reward_shift_fraction"])
        )
        methods[method] = {
            "stable": stable,
            "recent_spread_slope": slope,
            "previous_spread": previous_spread,
            "recent_spread": recent_spread,
            "spread_window_shift": recent_spread - previous_spread,
            "previous_reward": previous_reward,
            "recent_reward": recent_reward,
            "reward_shift_fraction": reward_shift_fraction,
        }
    return {
        "window_episodes": window,
        "methods": methods,
        "all_methods_stable": all(bool(row.get("stable")) for row in methods.values()),
    }


def evaluate_seed_policies(
    config: Mapping[str, Any],
    seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESLearnedGateModel,
) -> dict[str, Any]:
    episode_steps = int(config["episode_steps"])
    records: dict[str, list[dict[str, Any]]] = {
        method: [] for method in (*DEPLOYED_METHODS, *GATE_OFF_METHODS)
    }
    episode_rows = []
    dose_exact_values = []
    paths_equal_values = []
    raw_proposal_rates = []
    with ABIDESDockerDealerEnv({"episode_steps": episode_steps}) as env:
        for evaluation_seed in config["evaluation_seeds"]:
            controller = ABIDESGateActivationController(
                episode_steps,
                max_activation_rate=float(config["max_activation_rate"]),
            )

            def learned_decision(state, actions, step):
                return controller.decide(bool(model.decide(state, actions)["active"]))

            learned = rollout_episode(
                env,
                trainers["learned_gate"],
                seed=int(evaluation_seed),
                episode_steps=episode_steps,
                train=False,
                gate_decision=learned_decision,
                action_namespace=int(config["eval_action_namespace"]),
            )
            learned_schedule = [bool(row["gate_active"]) for row in learned["records"]]
            risk_path = [
                str(row["info"]["last_step_risk"]) for row in learned["records"]
            ]
            matched_schedule = shuffled_within_risk(
                learned_schedule,
                risk_path,
                seed=int(evaluation_seed) ^ seed ^ 0xE117,
            )
            dose_exact_value = exact_dose(learned_schedule, matched_schedule, risk_path)
            matched = rollout_episode(
                env,
                trainers["matched_random"],
                seed=int(evaluation_seed),
                episode_steps=episode_steps,
                train=False,
                gate_schedule=matched_schedule,
                action_namespace=int(config["eval_action_namespace"]),
            )
            no_intervention = rollout_episode(
                env,
                trainers["no_intervention"],
                seed=int(evaluation_seed),
                episode_steps=episode_steps,
                train=False,
                gate_schedule=[False] * episode_steps,
                action_namespace=int(config["eval_action_namespace"]),
            )
            threshold_controller = ABIDESGateActivationController(
                episode_steps,
                max_activation_rate=float(config["max_activation_rate"]),
            )

            def threshold_decision(state, actions, step):
                proposal = has_excess_spread(actions) and has_modal_undercutter(actions)
                return threshold_controller.decide(proposal)

            spread_only = rollout_episode(
                env,
                trainers["spread_only_threshold"],
                seed=int(evaluation_seed),
                episode_steps=episode_steps,
                train=False,
                gate_decision=threshold_decision,
                action_namespace=int(config["eval_action_namespace"]),
            )
            always_on = rollout_episode(
                env,
                trainers["always_on"],
                seed=int(evaluation_seed),
                episode_steps=episode_steps,
                train=False,
                gate_schedule=[True] * episode_steps,
                action_namespace=int(config["eval_action_namespace"]),
            )
            learned_gate_off = rollout_episode(
                env,
                trainers["learned_gate"],
                seed=int(evaluation_seed),
                episode_steps=episode_steps,
                train=False,
                gate_schedule=[False] * episode_steps,
                action_namespace=int(config["eval_action_namespace"]),
            )
            matched_gate_off = rollout_episode(
                env,
                trainers["matched_random"],
                seed=int(evaluation_seed),
                episode_steps=episode_steps,
                train=False,
                gate_schedule=[False] * episode_steps,
                action_namespace=int(config["eval_action_namespace"]),
            )
            rollouts = {
                "learned_gate": learned,
                "matched_random": matched,
                "no_intervention": no_intervention,
                "spread_only_threshold": spread_only,
                "always_on": always_on,
                "learned_gate_off": learned_gate_off,
                "matched_random_gate_off": matched_gate_off,
            }
            paths_equal = all(
                validate_common_path(learned, rollout) for rollout in rollouts.values()
            )
            if not dose_exact_value or not paths_equal:
                raise RuntimeError("adaptive evaluation CRN or dose invariant failed")
            for method, rollout in rollouts.items():
                records[method].extend(rollout["records"])
            dose_exact_values.append(dose_exact_value)
            paths_equal_values.append(paths_equal)
            raw_proposal_rates.append(controller.proposal_rate)
            episode_rows.append(
                {
                    "evaluation_seed": int(evaluation_seed),
                    "methods": {
                        method: safe_rollout_summary(rollout)
                        for method, rollout in rollouts.items()
                    },
                    "learned_schedule": learned_schedule,
                    "matched_schedule": matched_schedule,
                    "risk_path": risk_path,
                    "dose_exact": dose_exact_value,
                    "paths_equal": paths_equal,
                    "learned_raw_proposal_rate": controller.proposal_rate,
                }
            )

    methods = {
        method: summarize({"records": method_records})
        for method, method_records in records.items()
    }
    return {
        "methods": methods,
        "episodes": episode_rows,
        "dose_exact": bool(all(dose_exact_values)),
        "paths_equal": bool(all(paths_equal_values)),
        "learned_raw_proposal_rate": float(mean(raw_proposal_rates)),
    }


def run_seed(
    seed: int,
    config: dict[str, Any],
    output_dir_raw: str,
    fingerprint: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    output_dir = Path(output_dir_raw)
    model = ABIDESLearnedGateModel.load(REPO / config["gate_artifact"])
    trainers, history, training_dose_exact, training_paths_equal = train_seed_policies(
        config, output_dir, seed, fingerprint, model
    )
    convergence = convergence_summary(history, config)
    evaluation = evaluate_seed_policies(config, seed, trainers, model)
    result = {
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "training_episodes": int(config["train_episodes"]),
        "initial_models_identical": True,
        "training_dose_exact": training_dose_exact,
        "training_paths_equal": training_paths_equal,
        "convergence": convergence,
        "final_model_digests": {
            method: model_digest(trainer) for method, trainer in trainers.items()
        },
        "evaluation": evaluation,
    }
    atomic_write_json(seed_result_path(output_dir, seed), result)
    atomic_write_json(
        seed_directory(output_dir, seed) / "progress.json",
        {
            "seed": seed,
            "completed_episodes": int(config["train_episodes"]),
            "status": "complete",
            "updated_at": utc_now(),
        },
    )
    append_seed_event(
        seed_directory(output_dir, seed),
        {"event": "seed_completed", "elapsed_seconds": result["elapsed_seconds"]},
    )
    return result


def load_completed_results(
    output_dir: Path, seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        path = seed_result_path(output_dir, int(seed))
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if int(row.get("seed", -1)) != int(seed):
                raise RuntimeError(f"invalid adaptive seed result: {path}")
            rows.append(row)
    return rows


def aggregate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    metric_names = (
        "low_execution_half_spread",
        "mean_reward",
        "mean_volume",
        "activation_rate",
        "high_risk_activation_rate",
        "targeted_events",
    )
    all_methods = (*DEPLOYED_METHODS, *GATE_OFF_METHODS)
    method_summary = {
        method: {
            metric: float(
                mean(
                    float(row["evaluation"]["methods"][method][metric]) for row in rows
                )
            )
            for metric in metric_names
        }
        for method in all_methods
    }
    primary_values = [
        float(
            row["evaluation"]["methods"]["matched_random"]["low_execution_half_spread"]
        )
        - float(
            row["evaluation"]["methods"]["learned_gate"]["low_execution_half_spread"]
        )
        for row in rows
    ]
    gate_off_values = [
        float(
            row["evaluation"]["methods"]["matched_random_gate_off"][
                "low_execution_half_spread"
            ]
        )
        - float(
            row["evaluation"]["methods"]["learned_gate_off"][
                "low_execution_half_spread"
            ]
        )
        for row in rows
    ]
    primary = paired_summary(primary_values)
    gate_off = paired_summary(gate_off_values)
    required = int(config["required_training_seed_count"])
    minimum_positive = math.ceil(
        float(config["minimum_positive_seed_rate"]) * required - 1e-12
    )
    learned_reward = method_summary["learned_gate"]["mean_reward"]
    matched_reward = method_summary["matched_random"]["mean_reward"]
    integrity_checks = {
        "required_training_seed_count": len(rows) == required,
        "identical_initial_models": all(
            bool(row["initial_models_identical"]) for row in rows
        ),
        "training_dose_exact": all(bool(row["training_dose_exact"]) for row in rows),
        "training_paths_equal": all(bool(row["training_paths_equal"]) for row in rows),
        "evaluation_dose_exact": all(
            bool(row["evaluation"]["dose_exact"]) for row in rows
        ),
        "evaluation_paths_equal": all(
            bool(row["evaluation"]["paths_equal"]) for row in rows
        ),
        "volume_conserved": all(
            abs(float(row["evaluation"]["methods"][method]["mean_volume"]) - 20.0)
            < 1e-9
            for row in rows
            for method in DEPLOYED_METHODS
        ),
    }
    efficacy_checks = {
        "no_intervention_failure_mode": sum(
            float(
                row["evaluation"]["methods"]["no_intervention"][
                    "low_execution_half_spread"
                ]
            )
            > 4.0
            for row in rows
        )
        >= minimum_positive,
        "learned_activation_in_range": all(
            0.05
            <= float(row["evaluation"]["methods"]["learned_gate"]["activation_rate"])
            <= float(config["max_activation_rate"])
            for row in rows
        ),
        "learned_high_risk_activation_controlled": method_summary["learned_gate"][
            "high_risk_activation_rate"
        ]
        <= float(config["maximum_mean_high_risk_activation_rate"]),
        "learned_beats_matched_mean": float(primary["mean"]) > 0.0,
        "learned_beats_matched_seed_rate": int(primary["positive"]) >= minimum_positive,
        "reward_guardrail": learned_reward
        >= matched_reward
        - float(config["maximum_reward_degradation_fraction"])
        * max(1.0, abs(matched_reward)),
        "training_converged": all(
            bool(row["convergence"]["all_methods_stable"]) for row in rows
        ),
    }
    return {
        "method_summary": method_summary,
        "paired_matched_minus_learned": primary,
        "paired_gate_off_matched_minus_learned": gate_off,
        "integrity_checks": integrity_checks,
        "efficacy_checks": efficacy_checks,
    }


def validate_artifacts(config: Mapping[str, Any]) -> dict[str, str]:
    gate_path = REPO / str(config["gate_artifact"])
    if not gate_path.is_file():
        raise FileNotFoundError(f"missing frozen gate artifact: {gate_path}")
    gate_hash = sha256_file(gate_path)
    if gate_hash != str(config["gate_artifact_sha256"]):
        raise RuntimeError(
            f"gate artifact hash mismatch: expected {config['gate_artifact_sha256']}, "
            f"got {gate_hash}"
        )
    docker_image = str(config["docker_image"])
    return {
        "gate_artifact": gate_hash,
        "docker_image": docker_image,
        "docker_image_id": resolve_docker_image_id(docker_image),
    }


def build_manifest(
    config: Mapping[str, Any], fingerprint: str, artifacts: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "protocol_version": config["protocol_version"],
        "protocol_fingerprint": fingerprint,
        "started_at": utc_now(),
        "git_commit": current_git_commit(),
        "command": sys.argv,
        "training_seeds": [int(seed) for seed in config["training_seeds"]],
        "evaluation_seeds": [int(seed) for seed in config["evaluation_seeds"]],
        "artifact_checks": dict(artifacts),
        "config": dict(config),
    }


def render_logbook(
    manifest: Mapping[str, Any], status: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    complete = {int(row["seed"]): row for row in rows}
    output_dir = Path(str(manifest["config"]["output_dir"]))
    lines = [
        "# ABIDES/MAPPO Adaptive Pilot Logbook",
        "",
        f"- Status: `{status}`",
        f"- Protocol: `{manifest['protocol_version']}`",
        f"- Fingerprint: `{manifest['protocol_fingerprint']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Started: `{manifest['started_at']}`",
        "",
        "| Training seed | Status | Episodes | Runtime (s) |",
        "| ---: | --- | ---: | ---: |",
    ]
    for seed in manifest["training_seeds"]:
        if int(seed) in complete:
            row = complete[int(seed)]
            lines.append(
                f"| {seed} | complete | {row['training_episodes']} | "
                f"{float(row['elapsed_seconds']):.2f} |"
            )
            continue
        progress_path = seed_directory(output_dir, int(seed)) / "progress.json"
        completed_episodes = 0
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            completed_episodes = int(progress.get("completed_episodes", 0))
        lines.append(f"| {seed} | pending | {completed_episodes} |  |")
    lines.extend(
        [
            "",
            "Per-seed progress: `seeds/seed_<seed>/progress.json`  ",
            "Per-episode events: `seeds/seed_<seed>/training_log.jsonl`  ",
            "Atomic trainer state: `seeds/seed_<seed>/training_state.pt`",
            "",
        ]
    )
    return "\n".join(lines)


def persist_state(
    output_dir: Path,
    manifest: Mapping[str, Any],
    status: str,
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]] = (),
) -> None:
    complete = sorted(int(row["seed"]) for row in rows)
    expected = [int(seed) for seed in manifest["training_seeds"]]
    atomic_write_json(
        output_dir / "run_state.json",
        {
            "status": status,
            "updated_at": utc_now(),
            "completed_training_seeds": complete,
            "pending_training_seeds": [
                seed for seed in expected if seed not in complete
            ],
            "failures": list(failures),
        },
    )
    atomic_write_text(output_dir / "LOGBOOK.md", render_logbook(manifest, status, rows))


def run_pending(
    config: dict[str, Any],
    output_dir: Path,
    fingerprint: str,
    rows: list[dict[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete = {int(row["seed"]) for row in rows}
    pending = [
        int(seed) for seed in config["training_seeds"] if int(seed) not in complete
    ]
    failures: list[dict[str, Any]] = []
    if not pending:
        return rows, failures
    if workers == 1:
        for seed in pending:
            try:
                rows.append(run_seed(seed, config, str(output_dir), fingerprint))
            except BaseException as exc:
                failures.append(
                    {
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                break
        return rows, failures
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_seed, seed, config, str(output_dir), fingerprint): seed
            for seed in pending
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                rows.append(future.result())
            except BaseException as exc:
                failures.append(
                    {
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            else:
                print(json.dumps({"seed": seed, "status": "complete"}), flush=True)
    return rows, failures


def configured_run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    if args.training_seeds:
        config["training_seeds"] = [
            int(value) for value in args.training_seeds.split(",") if value
        ]
    if args.evaluation_seeds:
        config["evaluation_seeds"] = [
            int(value) for value in args.evaluation_seeds.split(",") if value
        ]
    if args.train_episodes is not None:
        config["train_episodes"] = args.train_episodes
    if args.episode_steps is not None:
        config["episode_steps"] = args.episode_steps
    if args.workers is not None:
        config["workers"] = args.workers
    if args.device is not None:
        config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.allow_partial or args.engineering_smoke:
        config["required_training_seed_count"] = len(config["training_seeds"])
    if tuple(config["methods"]) != DEPLOYED_METHODS:
        raise ValueError(f"adaptive pilot methods must be {DEPLOYED_METHODS}")
    if not config["training_seeds"] or not config["evaluation_seeds"]:
        raise ValueError("training and evaluation seed lists must be non-empty")
    if set(config["training_seeds"]) & set(config["evaluation_seeds"]):
        raise ValueError("training and evaluation seeds must be disjoint")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_mappo_adaptive_pilot_v1.json",
    )
    parser.add_argument("--training-seeds", default=None)
    parser.add_argument("--evaluation-seeds", default=None)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def execute(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = REPO / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = protocol_fingerprint(config)
    artifacts = validate_artifacts(config)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "refusing adaptive pilot from a dirty worktree; use --allow-dirty "
            "only for an engineering smoke"
        )
    with RunLock(output_dir / ".run.lock", fingerprint, args.force_unlock):
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("protocol_fingerprint") != fingerprint:
                raise RuntimeError("existing adaptive manifest has another protocol")
            for key, expected in manifest.get("artifact_checks", {}).items():
                if artifacts.get(key) != expected:
                    raise RuntimeError(
                        f"resume artifact mismatch for {key}: expected {expected}, "
                        f"got {artifacts.get(key)}"
                    )
        else:
            manifest = build_manifest(config, fingerprint, artifacts)
            atomic_write_json(manifest_path, manifest)
        rows = load_completed_results(output_dir, config["training_seeds"])
        pending_count = len(config["training_seeds"]) - len(rows)
        if pending_count and manifest.get("git_commit") != current_git_commit():
            raise RuntimeError(
                "cannot resume adaptive training from a different git commit; "
                f"expected {manifest.get('git_commit')}"
            )
        persist_state(output_dir, manifest, "running", rows)
        append_logbook(
            output_dir / "logbook.jsonl",
            {
                "event": "run_started" if not rows else "run_resumed",
                "completed_training_seeds": sorted(int(row["seed"]) for row in rows),
                "pending_training_seeds": pending_count,
            },
        )
        workers = min(int(config["workers"]), pending_count) if pending_count else 0
        rows, failures = run_pending(config, output_dir, fingerprint, rows, workers)
        rows.sort(key=lambda row: int(row["seed"]))
        if failures:
            persist_state(output_dir, manifest, "failed", rows, failures)
            append_logbook(
                output_dir / "logbook.jsonl",
                {"event": "run_failed", "failures": failures},
            )
            raise RuntimeError(
                f"{len(failures)} adaptive seed(s) failed; rerun to resume"
            )
        aggregated = aggregate(rows, config)
        all_integrity = all(aggregated["integrity_checks"].values())
        all_efficacy = all(aggregated["efficacy_checks"].values())
        if args.engineering_smoke:
            status = "smoke_complete" if all_integrity else "integrity_fail"
        else:
            status = "pass" if all_integrity and all_efficacy else "fail"
        summary = {
            "status": status,
            "protocol_version": config["protocol_version"],
            "protocol_fingerprint": fingerprint,
            "backend": "abides",
            "training_seeds": [int(seed) for seed in config["training_seeds"]],
            "evaluation_seeds": [int(seed) for seed in config["evaluation_seeds"]],
            "train_episodes": int(config["train_episodes"]),
            "episode_steps": int(config["episode_steps"]),
            "configured_workers": int(config["workers"]),
            "launched_workers_this_invocation": workers,
            **aggregated,
        }
        atomic_write_json(output_dir / "rows.json", rows)
        atomic_write_json(output_dir / "summary.json", summary)
        persist_state(output_dir, manifest, status, rows)
        append_logbook(
            output_dir / "logbook.jsonl",
            {"event": "run_finished", "status": status},
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        if status in {"fail", "integrity_fail"}:
            raise SystemExit(2)
        return summary


def main() -> None:
    args = parse_args()
    config = configured_run(args)
    execute(config, args)


if __name__ == "__main__":
    main()
