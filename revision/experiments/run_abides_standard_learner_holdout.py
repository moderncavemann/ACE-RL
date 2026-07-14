#!/usr/bin/env python3
"""Pinned official MAPPO/IPPO validation and ABIDES holdout runner."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_domain_benchmark import (  # noqa: E402
    audit_step_record,
    brero_mechanism,
    path_metrics,
)
from run_abides_learned_gate_screen import paired_summary  # noqa: E402
from run_abides_mappo_adaptive_pilot import (  # noqa: E402
    atomic_torch_save,
    validate_common_path,
)
from run_abides_mappo_adaptive_v3_learned import (  # noqa: E402
    global_shuffle,
    paced_learned_decision,
)
from run_abides_mappo_full_main import (  # noqa: E402
    RunLock,
    append_logbook,
    atomic_write_json,
    atomic_write_text,
    current_git_commit,
    protocol_fingerprint,
    resolve_docker_image_id,
    sha256_file,
    utc_now,
)
from src.agents.abides_mappo import (  # noqa: E402
    actor_observations,
    canonical_digest,
    centralized_critic_observations,
)
from src.agents.official_on_policy_mappo import (  # noqa: E402
    OFFICIAL_COMMIT,
    OfficialMAPPO,
    OfficialMAPPOConfig,
    verify_official_source,
)
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESPacedSignalController,
    ABIDESTwoStageGateModel,
)


METHODS = (
    "no_intervention",
    "risk_aware",
    "matched_random",
    "brero_threshold_buybox_budget_matched",
)
LEARNER_MODES = {
    "official_full_policy_MAPPO": True,
    "official_IPPO": False,
}
SignalDecision = Callable[[Mapping[str, Any], int], float]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stage_manifest_path(output: Path, stage: str) -> Path:
    if stage not in {"validation", "formal"}:
        raise ValueError(f"unknown official learner stage: {stage}")
    return output / stage / "manifest.json"


def formal_result_paths(config: Mapping[str, Any]) -> list[Path]:
    output = REPO / str(config["output_dir"]) / "formal"
    return [
        output / learner / f"seed_{int(seed)}" / "result.json"
        for learner in config["learners"]
        for seed in config["policy_seeds"]
    ]


def source_root(config: Mapping[str, Any]) -> str:
    return str(Path(str(config["official_source_root"])).expanduser().resolve())


def learner_config(
    config: Mapping[str, Any],
    seed: int,
    learner_mode: str,
    hyperparameters: Mapping[str, Any],
) -> OfficialMAPPOConfig:
    official = config["official_algorithm_parameters"]
    return OfficialMAPPOConfig(
        source_root=source_root(config),
        seed=int(seed),
        episode_steps=int(config["episode_steps"]),
        centralized_value=bool(LEARNER_MODES[learner_mode]),
        hidden_size=int(official["hidden_size"]),
        recurrent_n=int(official["recurrent_n"]),
        data_chunk_length=int(official["data_chunk_length"]),
        learning_rate=float(hyperparameters["learning_rate"]),
        critic_learning_rate=float(hyperparameters["critic_learning_rate"]),
        entropy_coefficient=float(hyperparameters["entropy_coefficient"]),
        ppo_epoch=int(official["ppo_epoch"]),
        num_mini_batch=int(official["num_mini_batch"]),
        gamma=float(official["gamma"]),
        gae_lambda=float(official["gae_lambda"]),
        clip_parameter=float(official["clip_parameter"]),
        value_loss_coefficient=float(official["value_loss_coefficient"]),
        max_gradient_norm=float(official["max_gradient_norm"]),
        reward_scale=float(official["reward_scale"]),
        device=str(config["device"]),
    )


def environment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    market = dict(config["market"])
    market.update(
        {
            "episode_steps": int(config["episode_steps"]),
            "routing_allocation_mode": str(config["routing_allocation_mode"]),
            "routing_target_kappa": float(config["routing_target_kappa"]),
            "routing_routable_fraction": float(
                config["routing_routable_fraction"]
            ),
        }
    )
    return market


def critic_observations(
    learner: OfficialMAPPO,
    state: Mapping[str, Any],
    actor_obs: np.ndarray,
    episode_steps: int,
) -> np.ndarray:
    if learner.config.centralized_value:
        return centralized_critic_observations(state, actor_obs, episode_steps)
    return actor_obs.copy()


def rollout_episode(
    env: ABIDESDockerDealerEnv,
    learner: OfficialMAPPO,
    seed: int,
    episode_steps: int,
    action_namespace: int,
    train: bool,
    update_seed: int,
    announced_signal_schedule: Sequence[float] | None = None,
    announced_signal_decision: SignalDecision | None = None,
    routing_mechanism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if announced_signal_schedule is not None and announced_signal_decision is not None:
        raise ValueError("provide a signal schedule or decision, not both")
    if announced_signal_schedule is not None and len(announced_signal_schedule) != episode_steps:
        raise ValueError("signal schedule length differs from episode horizon")
    state = env.reset(seed=int(seed))
    rnn_actor, rnn_critic, masks = learner.initial_recurrent_states()
    buffer = None
    records = []
    for step in range(episode_steps):
        if announced_signal_schedule is not None:
            signal = float(np.clip(announced_signal_schedule[step], 0.0, 1.0))
        elif announced_signal_decision is not None:
            signal = float(
                np.clip(announced_signal_decision(state, step), 0.0, 1.0)
            )
        else:
            signal = 0.0
        obs = actor_observations(state, signal)
        critic = critic_observations(learner, state, obs, episode_steps)
        if train:
            if buffer is None:
                buffer = learner.new_buffer(obs, critic)
            else:
                buffer.obs[step, 0] = obs
                buffer.share_obs[step, 0] = critic
        uniforms = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(step), int(action_namespace)])
        ).random(learner.config.n_agents)
        pre_state_digest = canonical_digest(
            {"observations": state["observations"], "info": state["info"]}
        )
        actions, logp, values, rnn_actor_next, rnn_critic_next = learner.act(
            obs,
            critic,
            rnn_actor,
            rnn_critic,
            masks,
            uniforms,
        )
        next_state = env.step(
            actions[:, 0],
            gate_active=signal > 0.0,
            routing_enabled=True,
            routing_strength=signal,
            routing_mechanism=routing_mechanism,
        )
        next_masks = np.zeros_like(masks) if bool(next_state["done"]) else np.ones_like(masks)
        placeholder_obs = actor_observations(next_state, 0.0)
        placeholder_critic = critic_observations(
            learner, next_state, placeholder_obs, episode_steps
        )
        rewards = np.asarray(next_state["rewards"], dtype=np.float32) / float(
            learner.config.reward_scale
        )
        if train:
            assert buffer is not None
            learner.insert(
                buffer,
                placeholder_obs,
                placeholder_critic,
                rnn_actor_next,
                rnn_critic_next,
                actions,
                logp,
                values,
                rewards,
                next_masks,
            )
        records.append(
            {
                "step": step,
                "actions": actions[:, 0].tolist(),
                "gate_active": signal > 0.0,
                "gate_signal": signal,
                "pre_state_digest": pre_state_digest,
                "action_uniform_digest": canonical_digest(uniforms.tolist()),
                "pre_info": dict(state["info"]),
                "pre_observations": state["observations"],
                "info": dict(next_state["info"]),
                "rewards_unscaled": list(next_state["rewards"]),
                "post_agents": list(next_state["agents"]),
                "mean_reward_unscaled": float(np.mean(next_state["rewards"])),
            }
        )
        state = next_state
        rnn_actor = rnn_actor_next
        rnn_critic = rnn_critic_next
        masks = next_masks
    if not state["done"]:
        raise RuntimeError("official learner episode did not reach its horizon")
    env.close_episode()
    update = learner.update(buffer, update_seed) if train else {}
    return {
        "records": records,
        "update": update,
        "mean_reward": float(mean(row["mean_reward_unscaled"] for row in records)),
        "mean_half_spread": float(
            mean(float(row["info"]["mean_execution_half_spread"]) for row in records)
        ),
        "activation_rate": float(mean(row["gate_active"] for row in records)),
    }


def write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical_json(row) + "\n")
    temporary.replace(path)


def base_environment_seed(config: Mapping[str, Any], seed: int, episode: int) -> int:
    return int(config["base_environment_seed_namespace"]) + (int(seed) % 100_000) * 1_000 + int(episode)


def response_environment_seed(config: Mapping[str, Any], seed: int, episode: int) -> int:
    return int(config["response_environment_seed_namespace"]) + (int(seed) % 100_000) * 1_000 + int(episode)


def validation_environment_seed(config: Mapping[str, Any], seed: int, episode: int) -> int:
    return int(config["validation_environment_seed_namespace"]) + (int(seed) % 100_000) * 1_000 + int(episode)


def train_base(
    env: ABIDESDockerDealerEnv,
    learner: OfficialMAPPO,
    config: Mapping[str, Any],
    seed: int,
    episodes: int,
    checkpoint: Path,
    fingerprint: str,
    seed_function: Callable[[Mapping[str, Any], int, int], int],
    action_namespace: int,
) -> tuple[OfficialMAPPO, list[dict[str, Any]]]:
    completed = 0
    history: list[dict[str, Any]] = []
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"official base checkpoint mismatch: {checkpoint}")
        learner = OfficialMAPPO.from_payload(payload["learner"])
        completed = int(payload["completed_episodes"])
        history = list(payload["history"])
    while completed < episodes:
        rollout = rollout_episode(
            env,
            learner,
            seed_function(config, seed, completed),
            int(config["episode_steps"]),
            int(action_namespace),
            True,
            int(action_namespace) + (int(seed) % 100_000) * 1_000 + completed,
            announced_signal_schedule=[0.0] * int(config["episode_steps"]),
        )
        completed += 1
        history.append(
            {
                "completed_episodes": completed,
                "mean_reward": rollout["mean_reward"],
                "mean_half_spread": rollout["mean_half_spread"],
                "update": rollout["update"],
            }
        )
        if completed % int(config["checkpoint_interval"]) == 0 or completed == episodes:
            atomic_torch_save(
                checkpoint,
                {
                    "protocol_fingerprint": fingerprint,
                    "completed_episodes": completed,
                    "learner": learner.payload(),
                    "history": history,
                    "saved_at": utc_now(),
                },
            )
    return learner, history


def evaluate_single(
    env: ABIDESDockerDealerEnv,
    learner: OfficialMAPPO,
    config: Mapping[str, Any],
    policy_seed: int,
    evaluation_seeds: Sequence[int],
    method: str,
) -> list[dict[str, Any]]:
    rows = []
    for evaluation_seed in evaluation_seeds:
        rollout = rollout_episode(
            env,
            learner,
            int(evaluation_seed),
            int(config["episode_steps"]),
            int(config["evaluation_action_namespace"]),
            False,
            0,
            announced_signal_schedule=[0.0] * int(config["episode_steps"]),
        )
        rows.append(
            path_metrics(
                str(config["protocol_version"]),
                policy_seed,
                int(evaluation_seed),
                method,
                learner.model_digest(),
                rollout["records"],
            )
        )
    return rows


def validation_task(
    config: dict[str, Any], cell: dict[str, Any], seed: int, fingerprint: str
) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    output = REPO / str(config["output_dir"]) / "validation" / str(cell["cell_id"])
    checkpoint = output / f"seed_{seed}" / "training_state.pt"
    result_path = output / f"seed_{seed}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"validation result fingerprint mismatch: {result_path}")
        return result
    learner = OfficialMAPPO(
        learner_config(config, seed, "official_full_policy_MAPPO", cell)
    )
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(environment_config(config), docker) as env:
        learner, history = train_base(
            env,
            learner,
            config,
            seed,
            int(config["validation_episodes"]),
            checkpoint,
            fingerprint,
            validation_environment_seed,
            int(config["validation_action_namespace"]),
        )
        per_path = evaluate_single(
            env,
            learner,
            config,
            seed,
            config["validation_evaluation_seeds"],
            "no_intervention",
        )
    result = {
        "protocol_fingerprint": fingerprint,
        "cell_id": str(cell["cell_id"]),
        "policy_seed": int(seed),
        "completed_episodes": int(config["validation_episodes"]),
        "finite_training": all(
            math.isfinite(float(value))
            for item in history
            for value in item["update"].values()
        ),
        "mean_dealer_reward": float(
            mean(float(row["dealer_reward_units_per_episode"]) for row in per_path)
        ),
        "mean_low_execution_half_spread": float(
            mean(float(row["low_execution_half_spread"]) for row in per_path)
        ),
        "model_digest": learner.model_digest(),
        "per_path": per_path,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(result_path, result)
    return result


def run_validation(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("selected_hyperparameters") is not None:
        raise RuntimeError("hyperparameter validation cannot run after selection")
    fingerprint = protocol_fingerprint(config)
    tasks = [
        (dict(cell), int(seed))
        for cell in config["hyperparameter_search_space"]
        for seed in config["validation_seeds"]
    ]
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=min(int(config["workers"]), len(tasks))) as pool:
        futures = {
            pool.submit(validation_task, config, cell, seed, fingerprint): (
                cell["cell_id"],
                seed,
            )
            for cell, seed in tasks
        }
        for future in as_completed(futures):
            cell_id, seed = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append(
                    {
                        "cell_id": cell_id,
                        "policy_seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    if failures:
        raise RuntimeError(f"official validation failures: {failures}")
    cells = []
    for cell in config["hyperparameter_search_space"]:
        cell_rows = [row for row in results if row["cell_id"] == cell["cell_id"]]
        cells.append(
            {
                **dict(cell),
                "all_finite": all(row["finite_training"] for row in cell_rows),
                "mean_dealer_reward": float(
                    mean(float(row["mean_dealer_reward"]) for row in cell_rows)
                ),
                "mean_low_execution_half_spread": float(
                    mean(
                        float(row["mean_low_execution_half_spread"])
                        for row in cell_rows
                    )
                ),
                "policy_seed_count": len(cell_rows),
            }
        )
    eligible = [cell for cell in cells if cell["all_finite"]]
    if not eligible:
        raise RuntimeError("no official validation cell remained finite")
    selected = sorted(
        eligible,
        key=lambda cell: (
            -float(cell["mean_dealer_reward"]),
            float(cell["learning_rate"]),
            float(cell["entropy_coefficient"]),
        ),
    )[0]
    artifact = {
        "protocol_version": str(config["protocol_version"]),
        "validation_protocol_fingerprint": fingerprint,
        "official_commit": OFFICIAL_COMMIT,
        "selection_rule": str(config["hyperparameter_selection_rule"]),
        "validation_policy_seeds": list(config["validation_seeds"]),
        "validation_evaluation_seeds": list(config["validation_evaluation_seeds"]),
        "cells": cells,
        "selected_hyperparameters": {
            key: selected[key]
            for key in (
                "cell_id",
                "learning_rate",
                "critic_learning_rate",
                "entropy_coefficient",
            )
        },
        "created_at": utc_now(),
    }
    atomic_write_json(REPO / str(config["selection_artifact"]), artifact)
    summary = {
        "status": "validation_complete",
        "task_count": len(results),
        "cells": cells,
        "selected_hyperparameters": artifact["selected_hyperparameters"],
    }
    atomic_write_json(
        REPO / str(config["output_dir"]) / "validation_summary.json", summary
    )
    return summary


def rollout_response_methods(
    env: ABIDESDockerDealerEnv,
    learners: Mapping[str, OfficialMAPPO],
    config: Mapping[str, Any],
    gate_model: ABIDESTwoStageGateModel,
    environment_seed: int,
    train: bool,
    update_seed: int,
) -> tuple[dict[str, Any], bool, bool, bool]:
    steps = int(config["episode_steps"])
    controller = ABIDESPacedSignalController(
        steps, target_rate=float(config["max_activation_rate"])
    )
    risk = rollout_episode(
        env,
        learners["risk_aware"],
        environment_seed,
        steps,
        int(config["response_action_namespace"] if train else config["evaluation_action_namespace"]),
        train,
        update_seed,
        announced_signal_decision=paced_learned_decision(
            gate_model, controller, include_severity=True
        ),
    )
    strengths = [float(row["gate_signal"]) for row in risk["records"]]
    random_strengths = global_shuffle(strengths, int(environment_seed) ^ 0xA317)
    common = {
        "seed": environment_seed,
        "episode_steps": steps,
        "action_namespace": int(config["response_action_namespace"] if train else config["evaluation_action_namespace"]),
        "train": train,
        "update_seed": update_seed,
    }
    rollouts = {
        "risk_aware": risk,
        "matched_random": rollout_episode(
            env,
            learners["matched_random"],
            announced_signal_schedule=random_strengths,
            **common,
        ),
        "no_intervention": rollout_episode(
            env,
            learners["no_intervention"],
            announced_signal_schedule=[0.0] * steps,
            **common,
        ),
        "brero_threshold_buybox_budget_matched": rollout_episode(
            env,
            learners["brero_threshold_buybox_budget_matched"],
            announced_signal_schedule=[1.0] * steps,
            routing_mechanism=brero_mechanism(config["brero_parameters"]),
            **common,
        ),
    }
    paths_equal = all(
        validate_common_path(risk, rollout) for rollout in rollouts.values()
    )
    random_dose_exact = np.allclose(
        np.sort(strengths), np.sort(random_strengths), atol=1e-12, rtol=0.0
    )
    uniform_schedules = [
        [row["action_uniform_digest"] for row in rollout["records"]]
        for rollout in rollouts.values()
    ]
    action_uniforms_equal = all(
        schedule == uniform_schedules[0] for schedule in uniform_schedules[1:]
    )
    return (
        rollouts,
        bool(paths_equal),
        bool(random_dose_exact),
        bool(action_uniforms_equal),
    )


def evaluate_response(
    env: ABIDESDockerDealerEnv,
    learners: Mapping[str, OfficialMAPPO],
    config: Mapping[str, Any],
    gate_model: ABIDESTwoStageGateModel,
    learner_mode: str,
    policy_seed: int,
    evaluation_seeds: Sequence[int],
    collect_steps: bool,
) -> dict[str, Any]:
    per_path = []
    step_rows = []
    paths = []
    doses = []
    uniforms = []
    for evaluation_seed in evaluation_seeds:
        rollouts, path_equal, dose_exact, uniforms_equal = rollout_response_methods(
            env,
            learners,
            config,
            gate_model,
            int(evaluation_seed),
            False,
            0,
        )
        paths.append(path_equal)
        doses.append(dose_exact)
        uniforms.append(uniforms_equal)
        for method in METHODS:
            records = rollouts[method]["records"]
            row = path_metrics(
                str(config["protocol_version"]),
                policy_seed,
                int(evaluation_seed),
                method,
                learners[method].model_digest(),
                records,
            )
            row["learner"] = learner_mode
            per_path.append(row)
            if collect_steps:
                for record in records:
                    step = audit_step_record(
                        policy_seed, int(evaluation_seed), method, record
                    )
                    step["learner"] = learner_mode
                    step_rows.append(step)
    return {
        "per_path": per_path,
        "step_records": step_rows,
        "paths_equal": bool(all(paths)),
        "random_dose_exact": bool(all(doses)),
        "action_uniforms_equal": bool(all(uniforms)),
    }


def formal_task(
    config: dict[str, Any], learner_mode: str, policy_seed: int, fingerprint: str
) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    output = REPO / str(config["output_dir"])
    task_dir = output / "formal" / learner_mode / f"seed_{policy_seed}"
    result_path = task_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"official formal result mismatch: {result_path}")
        return result
    selected = config["selected_hyperparameters"]
    learner = OfficialMAPPO(
        learner_config(config, policy_seed, learner_mode, selected)
    )
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    checkpoint = task_dir / "training_state.pt"
    response_completed = 0
    training_paths_equal = True
    random_dose_exact = True
    training_action_uniforms_equal = True
    initial_response_digest: str | None = None
    response_history: list[dict[str, Any]] = []
    base_history: list[dict[str, Any]] = []
    learners: dict[str, OfficialMAPPO] | None = None
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"official formal checkpoint mismatch: {checkpoint}")
        learner = OfficialMAPPO.from_payload(payload["base_learner"])
        base_history = list(payload["base_history"])
        response_completed = int(payload["response_completed"])
        response_history = list(payload["response_history"])
        training_paths_equal = bool(payload["training_paths_equal"])
        random_dose_exact = bool(payload["random_dose_exact"])
        training_action_uniforms_equal = bool(
            payload["training_action_uniforms_equal"]
        )
        initial_response_digest = payload.get("initial_response_digest")
        if payload.get("response_learners") is not None:
            learners = {
                method: OfficialMAPPO.from_payload(values)
                for method, values in payload["response_learners"].items()
            }
    gate_model = ABIDESTwoStageGateModel.load(REPO / str(config["gate_artifact"]))
    with ABIDESDockerDealerEnv(environment_config(config), docker) as env:
        learner, base_history = train_base(
            env,
            learner,
            config,
            policy_seed,
            int(config["base_episodes"]),
            task_dir / "base_training_state.pt",
            fingerprint,
            base_environment_seed,
            int(config["base_action_namespace"]),
        )
        base_evaluation = evaluate_single(
            env,
            learner,
            config,
            policy_seed,
            config["base_evaluation_seeds"],
            "no_intervention",
        )
        if learners is None:
            learners = {method: learner.clone() for method in METHODS}
            initial_digests = {item.model_digest() for item in learners.values()}
            if len(initial_digests) != 1:
                raise RuntimeError("official response methods did not start identically")
            initial_response_digest = initial_digests.pop()
        while response_completed < int(config["response_episodes"]):
            environment_seed = response_environment_seed(
                config, policy_seed, response_completed
            )
            rollouts, paths_equal, dose_exact, uniforms_equal = rollout_response_methods(
                env,
                learners,
                config,
                gate_model,
                environment_seed,
                True,
                int(config["response_action_namespace"])
                + (policy_seed % 100_000) * 1_000
                + response_completed,
            )
            response_completed += 1
            training_paths_equal &= paths_equal
            random_dose_exact &= dose_exact
            training_action_uniforms_equal &= uniforms_equal
            response_history.append(
                {
                    "completed_episodes": response_completed,
                    "methods": {
                        method: {
                            "mean_reward": rollout["mean_reward"],
                            "mean_half_spread": rollout["mean_half_spread"],
                            "activation_rate": rollout["activation_rate"],
                            "update": rollout["update"],
                        }
                        for method, rollout in rollouts.items()
                    },
                }
            )
            if response_completed % int(config["checkpoint_interval"]) == 0 or response_completed == int(config["response_episodes"]):
                atomic_torch_save(
                    checkpoint,
                    {
                        "protocol_fingerprint": fingerprint,
                        "base_learner": learner.payload(),
                        "base_history": base_history,
                        "response_completed": response_completed,
                        "response_learners": {
                            method: item.payload() for method, item in learners.items()
                        },
                        "response_history": response_history,
                        "training_paths_equal": training_paths_equal,
                        "random_dose_exact": random_dose_exact,
                        "training_action_uniforms_equal": training_action_uniforms_equal,
                        "initial_response_digest": initial_response_digest,
                        "saved_at": utc_now(),
                    },
                )
        final = evaluate_response(
            env,
            learners,
            config,
            gate_model,
            learner_mode,
            policy_seed,
            config["response_evaluation_seeds"],
            True,
        )
    step_path = task_dir / "evaluation_step_records.jsonl.gz"
    write_jsonl_gz(step_path, final.pop("step_records"))
    result = {
        "protocol_fingerprint": fingerprint,
        "official_commit": OFFICIAL_COMMIT,
        "learner": learner_mode,
        "policy_seed": policy_seed,
        "base_completed_episodes": int(config["base_episodes"]),
        "response_completed_episodes": response_completed,
        "base_model_digest": learner.model_digest(),
        "initial_response_model_digest": initial_response_digest,
        "final_model_digests": {
            method: item.model_digest() for method, item in learners.items()
        },
        "base_evaluation": base_evaluation,
        "training_paths_equal": training_paths_equal,
        "training_random_dose_exact": random_dose_exact,
        "training_action_uniforms_equal": training_action_uniforms_equal,
        "final_evaluation": final,
        "step_records_file": str(step_path.relative_to(REPO)),
        "step_records_sha256": sha256_file(step_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(result_path, result)
    return result


def aggregate_formal(
    config: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_learner = {}
    for learner_mode in config["learners"]:
        learner_results = [row for row in results if row["learner"] == learner_mode]
        per_path = [
            path
            for result in learner_results
            for path in result["final_evaluation"]["per_path"]
        ]
        method_summary = {}
        for method in METHODS:
            rows = [row for row in per_path if row["method"] == method]
            method_summary[method] = {
                key: float(mean(float(row[key]) for row in rows))
                for key in (
                    "low_execution_half_spread",
                    "dealer_reward_units_per_episode",
                    "activation_rate",
                    "high_risk_activation_rate",
                    "mean_routing_total_variation",
                    "mean_volume",
                )
            }
        seed_differences = []
        for policy_seed in config["policy_seeds"]:
            rows = [row for row in per_path if int(row["policy_seed"]) == int(policy_seed)]
            risk = mean(
                float(row["low_execution_half_spread"])
                for row in rows
                if row["method"] == "risk_aware"
            )
            random = mean(
                float(row["low_execution_half_spread"])
                for row in rows
                if row["method"] == "matched_random"
            )
            seed_differences.append(random - risk)
        by_learner[learner_mode] = {
            "method_summary": method_summary,
            "matched_random_minus_risk_aware": paired_summary(seed_differences),
            "per_path": per_path,
        }
    expected = len(config["learners"]) * len(config["policy_seeds"])
    checks = {
        "task_count": len(results) == expected,
        "official_commit_exact": all(
            row["official_commit"] == OFFICIAL_COMMIT for row in results
        ),
        "base_budget_complete": all(
            int(row["base_completed_episodes"]) == int(config["base_episodes"])
            for row in results
        ),
        "response_budget_complete": all(
            int(row["response_completed_episodes"])
            == int(config["response_episodes"])
            for row in results
        ),
        "training_paths_equal": all(row["training_paths_equal"] for row in results),
        "training_random_dose_exact": all(
            row["training_random_dose_exact"] for row in results
        ),
        "training_action_uniforms_equal": all(
            row["training_action_uniforms_equal"] for row in results
        ),
        "initial_response_models_identical": all(
            isinstance(row["initial_response_model_digest"], str)
            and len(row["initial_response_model_digest"]) == 64
            for row in results
        ),
        "evaluation_paths_equal": all(
            row["final_evaluation"]["paths_equal"] for row in results
        ),
        "evaluation_random_dose_exact": all(
            row["final_evaluation"]["random_dose_exact"] for row in results
        ),
        "evaluation_action_uniforms_equal": all(
            row["final_evaluation"]["action_uniforms_equal"] for row in results
        ),
        "per_path_row_count": all(
            len(row["final_evaluation"]["per_path"])
            == len(config["response_evaluation_seeds"]) * len(METHODS)
            for row in results
        ),
        "step_record_hashes_present": all(row["step_records_sha256"] for row in results),
        "volume_conserved": all(
            abs(summary["method_summary"][method]["mean_volume"] - 20.0) < 1e-9
            for summary in by_learner.values()
            for method in METHODS
        ),
    }
    return {
        "implementation_status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "learner_results": by_learner,
        "claim_boundary": (
            "cross-learner timing claim supported only for learners with a favorable "
            "paired interval; failures are retained without additional search"
        ),
    }


def run_formal(config: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    if not bool(config.get("formal_execution_allowed")):
        raise RuntimeError("official formal run is blocked until selection freeze")
    tasks = [
        (str(learner_mode), int(policy_seed))
        for learner_mode in config["learners"]
        for policy_seed in config["policy_seeds"]
    ]
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=min(int(config["workers"]), len(tasks))) as pool:
        futures = {
            pool.submit(formal_task, config, learner, seed, fingerprint): (
                learner,
                seed,
            )
            for learner, seed in tasks
        }
        for future in as_completed(futures):
            learner, seed = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append(
                    {
                        "learner": learner,
                        "policy_seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    output = REPO / str(config["output_dir"])
    atomic_write_json(output / "failures.json", {"failures": failures})
    if failures:
        raise RuntimeError(f"official formal task failures: {len(failures)}")
    results.sort(key=lambda row: (str(row["learner"]), int(row["policy_seed"])))
    summary = aggregate_formal(config, results)
    atomic_write_json(output / "rows.json", results)
    atomic_write_json(output / "summary.json", summary)
    return summary


def validate_config(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    if str(config["official_commit"]) != OFFICIAL_COMMIT:
        raise RuntimeError("standard learner config changed the official commit")
    source = verify_official_source(source_root(config))
    gate_path = REPO / str(config["gate_artifact"])
    if sha256_file(gate_path) != str(config["gate_artifact_sha256"]):
        raise RuntimeError("standard learner gate hash mismatch")
    if tuple(config["methods"]) != METHODS:
        raise RuntimeError(f"standard learner methods must remain {METHODS}")
    if set(config["learners"]) != set(LEARNER_MODES):
        raise RuntimeError("standard learner modes changed")
    seed_sets = [
        set(config["policy_seeds"]),
        set(config["base_evaluation_seeds"]),
        set(config["response_evaluation_seeds"]),
        set(config["validation_seeds"]),
        set(config["validation_evaluation_seeds"]),
    ]
    for index, left in enumerate(seed_sets):
        if any(left & right for right in seed_sets[index + 1 :]):
            raise RuntimeError("standard learner seed roles overlap")
    artifacts = {
        "official_source": source,
        "gate_artifact_sha256": str(config["gate_artifact_sha256"]),
        "docker_image_id": resolve_docker_image_id(str(config["docker_image"])),
    }
    if stage == "formal":
        artifact_path = REPO / str(config["selection_artifact"])
        if sha256_file(artifact_path) != str(config["selection_artifact_sha256"]):
            raise RuntimeError("standard learner selection artifact hash mismatch")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact["selected_hyperparameters"] != config["selected_hyperparameters"]:
            raise RuntimeError("selected official hyperparameters differ from artifact")
        artifacts["selection_artifact_sha256"] = str(
            config["selection_artifact_sha256"]
        )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_standard_learner_holdout_v1.json",
    )
    parser.add_argument("--stage", choices=("validation", "formal"), required=True)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    if args.engineering_smoke:
        config = copy.deepcopy(config)
        config["protocol_version"] += f"_{args.stage}_engineering_smoke_v2"
        config["workers"] = 1
        config["checkpoint_interval"] = 1
        config["output_dir"] = (
            str(config["output_dir"]) + f"_{args.stage}_smoke_v2"
        )
        if args.stage == "validation":
            config["hyperparameter_search_space"] = config[
                "hyperparameter_search_space"
            ][:1]
            config["validation_seeds"] = config["validation_seeds"][:1]
            config["validation_evaluation_seeds"] = config[
                "validation_evaluation_seeds"
            ][:1]
            config["validation_episodes"] = 2
            config["selection_artifact"] = (
                "revision/experiments/outputs/"
                "abides_standard_learner_validation_smoke_selection.json"
            )
        else:
            config["formal_execution_allowed"] = True
            config["selected_hyperparameters"] = dict(
                config["hyperparameter_search_space"][0]
            )
            config["policy_seeds"] = config["policy_seeds"][:1]
            config["base_evaluation_seeds"] = config["base_evaluation_seeds"][:1]
            config["response_evaluation_seeds"] = config[
                "response_evaluation_seeds"
            ][:1]
            config["base_episodes"] = 2
            config["response_episodes"] = 2
    artifacts = validate_config(config, args.stage if not args.engineering_smoke else "validation")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if dirty and not (args.allow_dirty or args.engineering_smoke):
        raise RuntimeError("official learner run requires a clean committed worktree")
    fingerprint = protocol_fingerprint(config)
    output = REPO / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with RunLock(output / ".run.lock", fingerprint, args.force_unlock):
        manifest_path = stage_manifest_path(output, args.stage)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["protocol_fingerprint"] != fingerprint:
                raise RuntimeError("official learner manifest fingerprint mismatch")
        else:
            manifest = {
                "protocol_version": config["protocol_version"],
                "protocol_fingerprint": fingerprint,
                "git_commit": current_git_commit(),
                "started_at": utc_now(),
                "stage": args.stage,
                "engineering_smoke": args.engineering_smoke,
                "artifacts": artifacts,
                "config": config,
            }
            atomic_write_json(manifest_path, manifest)
        append_logbook(
            output / "logbook.jsonl", {"event": f"{args.stage}_started"}
        )
        completed_formal_before = (
            args.stage == "formal"
            and all(path.exists() for path in formal_result_paths(config))
        )
        summary = (
            run_validation(config)
            if args.stage == "validation"
            else run_formal(config, fingerprint)
        )
        append_logbook(
            output / "logbook.jsonl",
            {"event": f"{args.stage}_finished", "status": summary["status"] if "status" in summary else summary["implementation_status"]},
        )
        if completed_formal_before:
            append_logbook(
                output / "logbook.jsonl",
                {
                    "event": "completed_resume_verified",
                    "stage": "formal",
                    "containers_started": 0,
                    "result_count": len(formal_result_paths(config)),
                    "summary_sha256": sha256_file(output / "summary.json"),
                },
            )
        print(canonical_json(summary))


if __name__ == "__main__":
    main()
