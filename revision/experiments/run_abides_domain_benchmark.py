#!/usr/bin/env python3
"""Checkpointed ABIDES domain benchmark and engineering-smoke runner."""

from __future__ import annotations

import argparse
import csv
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

from run_abides_learned_gate_screen import paired_summary  # noqa: E402
from run_abides_mappo_adaptive_pilot import (  # noqa: E402
    atomic_torch_save,
    model_digest,
    trainer_from_payload,
    trainer_payload,
    validate_common_path,
)
from run_abides_mappo_adaptive_v3_full_main import (  # noqa: E402
    global_shuffle,
    spread_only_decision,
)
from run_abides_mappo_adaptive_v3_learned import paced_learned_decision  # noqa: E402
from run_abides_mappo_adaptive_v3_response import (  # noqa: E402
    load_base,
    make_adapter,
    market_config,
    response_environment_seed,
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
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.agents.mappo import MAPPOTrainer  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.interventions.abides_gate import (  # noqa: E402
    FEATURE_NAMES,
    ABIDESPacedSignalController,
    ABIDESTwoStageGateModel,
    abides_gate_features,
    abides_lagged_actions,
)
from src.interventions.abides_literature_baselines import (  # noqa: E402
    spread_only_proposal_strength,
)


NEW_METHODS = (
    "spread_only_budget_matched",
    "brero_threshold_buybox",
    "brero_threshold_buybox_budget_matched",
)
V5_TRAINED_METHODS = (
    "risk_aware",
    "matched_random",
    "spread_only_threshold",
    "always_on",
    "no_routing",
)
EVALUATED_METHODS = (
    "no_intervention",
    *V5_TRAINED_METHODS,
    *NEW_METHODS,
)
SignalDecision = Callable[[Mapping[str, Any], int], float]


def normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result["base_output_dir"] = str(config["source_base_output_dir"])
    result["base_protocol_fingerprint"] = str(config["source_base_fingerprint"])
    return result


def seed_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "domain_seeds" / f"seed_{seed}"


def checkpoint_path(output_dir: Path, seed: int) -> Path:
    return seed_dir(output_dir, seed) / "training_state.pt"


def result_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "domain_results" / f"seed_{seed}.json"


def step_records_path(output_dir: Path, seed: int) -> Path:
    return seed_dir(output_dir, seed) / "evaluation_step_records.jsonl.gz"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical_json(row) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_v5_trainers(
    config: Mapping[str, Any], policy_seed: int
) -> tuple[dict[str, MAPPOTrainer], dict[str, str]]:
    output_dir = REPO / str(config["source_v5_output_dir"])
    checkpoint = (
        output_dir / "response_seeds" / f"seed_{policy_seed}" / "training_state.pt"
    )
    result_file = output_dir / "response_results" / f"seed_{policy_seed}.json"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = json.loads(result_file.read_text(encoding="utf-8"))
    expected = str(config["source_v5_fingerprint"])
    if payload.get("protocol_fingerprint") != expected:
        raise RuntimeError(f"v5 checkpoint fingerprint mismatch: {checkpoint}")
    if result.get("protocol_fingerprint") != expected:
        raise RuntimeError(f"v5 result fingerprint mismatch: {result_file}")
    if int(payload.get("completed_episodes", -1)) != int(
        config["source_v5_response_episodes"]
    ):
        raise RuntimeError(f"v5 response checkpoint incomplete: {checkpoint}")
    trainers = {
        method: trainer_from_payload(payload["trainers"][method], "cpu")
        for method in V5_TRAINED_METHODS
    }
    digests = {method: model_digest(trainer) for method, trainer in trainers.items()}
    if digests != result["final_model_digests"]:
        raise RuntimeError(f"v5 model digest mismatch: {checkpoint}")
    return trainers, digests


def load_mechanism_parameters(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if bool(config.get("engineering_smoke")):
        return {
            method: dict(values)
            for method, values in config["smoke_mechanism_parameters"].items()
        }
    parameters = config.get("mechanism_parameters")
    if not isinstance(parameters, Mapping):
        raise RuntimeError("formal mechanism parameters are not frozen")
    return {method: dict(values) for method, values in parameters.items()}


def spread_budget_decision(
    parameters: Mapping[str, Any], episode_steps: int
) -> SignalDecision:
    controller = ABIDESPacedSignalController(
        episode_steps,
        target_rate=float(parameters["activation_target"]),
    )
    severity_index = FEATURE_NAMES.index("contract_severity")

    def decide(state: Mapping[str, Any], step: int) -> float:
        features = abides_gate_features(state, abides_lagged_actions(state))
        proposal = spread_only_proposal_strength(
            float(features[severity_index]),
            float(parameters["severity_threshold"]),
            float(parameters["signal_scale"]),
        )
        return controller.decide(proposal)

    return decide


def brero_mechanism(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "brero_threshold_buybox",
        "order_statistic_k": int(parameters["order_statistic_k"]),
        "margin_ticks": int(parameters["margin_ticks"]),
        "routable_fraction": float(parameters.get("routable_fraction", 0.5)),
    }


def initial_state(
    config: Mapping[str, Any], output_dir: Path, policy_seed: int, fingerprint: str
) -> tuple[
    dict[str, MAPPOTrainer],
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    bool,
    bool,
    str,
    str,
    dict[str, str],
]:
    path = checkpoint_path(output_dir, policy_seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"domain checkpoint fingerprint mismatch: {path}")
        return (
            {
                method: trainer_from_payload(values, str(config["device"]))
                for method, values in payload["trainers"].items()
            },
            int(payload["completed_episodes"]),
            list(payload["training_history"]),
            list(payload["evaluation_history"]),
            bool(payload["training_paths_equal"]),
            bool(payload["training_action_uniforms_equal"]),
            str(payload["base_model_digest"]),
            str(payload["initial_adapter_digest"]),
            dict(payload["v5_model_digests"]),
        )
    base = load_base(normalized_config(config), policy_seed)
    trainers = {
        method: make_adapter(base, normalized_config(config), policy_seed)
        for method in NEW_METHODS
    }
    initial_digests = {model_digest(trainer) for trainer in trainers.values()}
    if len(initial_digests) != 1:
        raise RuntimeError("new adapters did not start from an identical digest")
    _, v5_digests = load_v5_trainers(config, policy_seed)
    return (
        trainers,
        0,
        [],
        [],
        True,
        True,
        model_digest(base),
        initial_digests.pop(),
        v5_digests,
    )


def save_checkpoint(
    config: Mapping[str, Any],
    output_dir: Path,
    policy_seed: int,
    fingerprint: str,
    trainers: Mapping[str, MAPPOTrainer],
    completed: int,
    training_history: Sequence[Mapping[str, Any]],
    evaluation_history: Sequence[Mapping[str, Any]],
    training_paths_equal: bool,
    training_action_uniforms_equal: bool,
    base_model_digest: str,
    initial_adapter_digest: str,
    v5_model_digests: Mapping[str, str],
) -> None:
    atomic_torch_save(
        checkpoint_path(output_dir, policy_seed),
        {
            "protocol_fingerprint": fingerprint,
            "policy_seed": int(policy_seed),
            "completed_episodes": int(completed),
            "trainers": {
                method: trainer_payload(trainer) for method, trainer in trainers.items()
            },
            "training_history": list(training_history),
            "evaluation_history": list(evaluation_history),
            "training_paths_equal": bool(training_paths_equal),
            "training_action_uniforms_equal": bool(training_action_uniforms_equal),
            "base_model_digest": base_model_digest,
            "initial_adapter_digest": initial_adapter_digest,
            "v5_model_digests": dict(v5_model_digests),
            "saved_at": utc_now(),
        },
    )


def action_uniforms_equal(rollouts: Mapping[str, Mapping[str, Any]]) -> bool:
    schedules = [
        [record["action_uniform_digest"] for record in rollout["records"]]
        for rollout in rollouts.values()
    ]
    return all(schedule == schedules[0] for schedule in schedules[1:])


def rollout_new_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    environment_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    parameters: Mapping[str, Mapping[str, Any]],
    train: bool,
) -> tuple[dict[str, Any], bool, bool]:
    episode_steps = int(config["episode_steps"])
    namespace = (
        int(config["response_action_namespace"])
        if train
        else int(config["evaluation_action_namespace"])
    )
    common = {
        "seed": int(environment_seed),
        "episode_steps": episode_steps,
        "train": train,
        "defer_update": train,
        "reward_scale": float(config["reward_scale"]),
        "action_namespace": namespace,
    }
    spread = rollout_episode(
        env,
        trainers["spread_only_budget_matched"],
        announced_signal_decision=spread_budget_decision(
            parameters["spread_only_budget_matched"], episode_steps
        ),
        **common,
    )
    literature = rollout_episode(
        env,
        trainers["brero_threshold_buybox"],
        announced_signal_schedule=[1.0] * episode_steps,
        routing_mechanism=brero_mechanism(parameters["brero_threshold_buybox"]),
        **common,
    )
    matched = rollout_episode(
        env,
        trainers["brero_threshold_buybox_budget_matched"],
        announced_signal_schedule=[1.0] * episode_steps,
        routing_mechanism=brero_mechanism(
            parameters["brero_threshold_buybox_budget_matched"]
        ),
        **common,
    )
    rollouts = {
        "spread_only_budget_matched": spread,
        "brero_threshold_buybox": literature,
        "brero_threshold_buybox_budget_matched": matched,
    }
    paths_equal = all(
        validate_common_path(spread, rollout) for rollout in rollouts.values()
    )
    return rollouts, paths_equal, action_uniforms_equal(rollouts)


def risk_rollout(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    environment_seed: int,
    trainer: MAPPOTrainer,
    risk_model: ABIDESTwoStageGateModel,
) -> dict[str, Any]:
    controller = ABIDESPacedSignalController(
        int(config["episode_steps"]),
        target_rate=float(config["max_activation_rate"]),
    )
    return rollout_episode(
        env,
        trainer,
        seed=int(environment_seed),
        episode_steps=int(config["episode_steps"]),
        train=False,
        announced_signal_decision=paced_learned_decision(
            risk_model, controller, include_severity=False
        ),
        reward_scale=float(config["reward_scale"]),
        action_namespace=int(config["evaluation_action_namespace"]),
    )


def rollout_all_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    environment_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    v5_trainers: Mapping[str, MAPPOTrainer],
    risk_model: ABIDESTwoStageGateModel,
    parameters: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], bool, bool]:
    risk = risk_rollout(
        env, config, environment_seed, v5_trainers["risk_aware"], risk_model
    )
    strengths = [float(record["gate_signal"]) for record in risk["records"]]
    random_strengths = global_shuffle(strengths, int(environment_seed) ^ 0xD317)
    episode_steps = int(config["episode_steps"])
    common = {
        "seed": int(environment_seed),
        "episode_steps": episode_steps,
        "train": False,
        "reward_scale": float(config["reward_scale"]),
        "action_namespace": int(config["evaluation_action_namespace"]),
    }
    rollouts = {
        "risk_aware": risk,
        "matched_random": rollout_episode(
            env,
            v5_trainers["matched_random"],
            announced_signal_schedule=random_strengths,
            **common,
        ),
        "spread_only_threshold": rollout_episode(
            env,
            v5_trainers["spread_only_threshold"],
            announced_signal_decision=spread_only_decision(
                ABIDESPacedSignalController(
                    episode_steps,
                    target_rate=float(config["max_activation_rate"]),
                )
            ),
            **common,
        ),
        "always_on": rollout_episode(
            env,
            v5_trainers["always_on"],
            announced_signal_schedule=[1.0] * episode_steps,
            **common,
        ),
        "no_routing": rollout_episode(
            env,
            v5_trainers["no_routing"],
            announced_signal_schedule=strengths,
            routing_enabled=False,
            **common,
        ),
    }
    new_rollouts, _, _ = rollout_new_methods(
        env,
        config,
        environment_seed,
        trainers,
        parameters,
        train=False,
    )
    rollouts.update(new_rollouts)
    base = load_base(normalized_config(config), policy_seed)
    rollouts["no_intervention"] = rollout_episode(
        env,
        base,
        announced_signal_schedule=[0.0] * episode_steps,
        **common,
    )
    paths_equal = all(
        validate_common_path(risk, rollout) for rollout in rollouts.values()
    )
    return rollouts, paths_equal, action_uniforms_equal(rollouts)


def path_metrics(
    protocol_version: str,
    policy_seed: int,
    evaluation_seed: int,
    method: str,
    model_digest_value: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    low = [row for row in records if row["info"].get("last_step_risk") == "low"]
    high = [row for row in records if row["info"].get("last_step_risk") == "high"]
    if not low:
        raise RuntimeError("evaluation path contains no low-risk steps")

    def execution(rows: Sequence[Mapping[str, Any]]) -> float:
        return float(
            mean(float(row["info"]["mean_execution_half_spread"]) for row in rows)
        )

    def customer_cost(rows: Sequence[Mapping[str, Any]]) -> float:
        return float(
            sum(
                float(row["info"]["mean_execution_half_spread"])
                * float(row["info"]["realized_flow_quantity"])
                for row in rows
            )
        )

    actual_hhi = []
    inventory_std = []
    for row in records:
        shares = np.asarray(row["info"].get("flow_shares", [0.1] * 10), dtype=float)
        shares = shares / max(float(np.sum(shares)), 1e-12)
        actual_hhi.append(float(np.sum(np.square(shares))))
        inventory = np.asarray(
            [agent["inventory"] for agent in row["post_agents"]], dtype=float
        )
        inventory_std.append(float(np.std(inventory)))
    path_digests = {str(row["info"]["path_digest"]) for row in records}
    if len(path_digests) != 1:
        raise RuntimeError("one evaluation episode produced multiple path digests")
    active = [bool(row["info"].get("active_routing", False)) for row in records]
    return {
        "protocol_version": protocol_version,
        "policy_seed": int(policy_seed),
        "evaluation_seed": int(evaluation_seed),
        "method": method,
        "low_execution_half_spread": execution(low),
        "high_execution_half_spread": execution(high) if high else 0.0,
        "all_state_execution_half_spread": execution(records),
        "mean_reward": float(
            mean(float(row["mean_reward_unscaled"]) for row in records)
        ),
        "dealer_reward_units_per_episode": float(
            sum(float(row["mean_reward_unscaled"]) * 10.0 for row in records)
        ),
        "low_customer_cost_tick_units_per_episode": customer_cost(low),
        "all_customer_cost_tick_units_per_episode": customer_cost(records),
        "activation_rate": float(mean(active)),
        "high_risk_activation_rate": (
            float(mean(bool(row["info"].get("active_routing", False)) for row in high))
            if high
            else 0.0
        ),
        "mean_signal_strength": float(
            mean(float(row["gate_signal"]) for row in records)
        ),
        "mean_routing_total_variation": float(
            mean(float(row["info"]["routing_total_variation"]) for row in records)
        ),
        "mean_flow_hhi": float(mean(actual_hhi)),
        "mean_inventory_std": float(mean(inventory_std)),
        "mean_volume": float(
            mean(float(row["info"]["realized_flow_quantity"]) for row in records)
        ),
        "low_risk_steps": len(low),
        "high_risk_steps": len(high),
        "model_digest": model_digest_value,
        "path_digest": path_digests.pop(),
    }


def audit_step_record(
    policy_seed: int,
    evaluation_seed: int,
    method: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = np.asarray(
        [agent["inventory"] for agent in row["post_agents"]], dtype=float
    )
    info = row["info"]
    return {
        "policy_seed": int(policy_seed),
        "evaluation_seed": int(evaluation_seed),
        "method": method,
        "step": int(row["step"]),
        "pre_state_digest": str(row["pre_state_digest"]),
        "actions": list(row["actions"]),
        "mechanism_signal": float(row["gate_signal"]),
        "mechanism_parameters": dict(info.get("routing_mechanism_parameters", {})),
        "routing_threshold": info.get("routing_threshold"),
        "eligible_dealers": list(info.get("eligible_agent_indices", [])),
        "routing_weights": list(info["routing_weights"]),
        "latent_risk_state": str(info.get("last_step_risk")),
        "execution_half_spread": float(info["mean_execution_half_spread"]),
        "rewards": list(row["rewards_unscaled"]),
        "inventory_summary": {
            "mean": float(np.mean(inventory)),
            "std": float(np.std(inventory)),
            "maximum_absolute": float(np.max(np.abs(inventory))),
        },
        "action_uniform_digest": str(row["action_uniform_digest"]),
        "path_digest": str(info["path_digest"]),
    }


def evaluate(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    v5_trainers: Mapping[str, MAPPOTrainer],
    risk_model: ABIDESTwoStageGateModel,
    parameters: Mapping[str, Mapping[str, Any]],
    evaluation_seeds: Sequence[int],
    collect_steps: bool,
) -> dict[str, Any]:
    per_path = []
    step_records = []
    path_checks = []
    uniform_checks = []
    digests = {
        "no_intervention": model_digest(
            load_base(normalized_config(config), policy_seed)
        ),
        **{method: model_digest(trainer) for method, trainer in v5_trainers.items()},
        **{method: model_digest(trainer) for method, trainer in trainers.items()},
    }
    for evaluation_seed in evaluation_seeds:
        rollouts, paths_equal, uniforms_equal = rollout_all_methods(
            env,
            config,
            policy_seed,
            int(evaluation_seed),
            trainers,
            v5_trainers,
            risk_model,
            parameters,
        )
        path_checks.append(paths_equal)
        uniform_checks.append(uniforms_equal)
        for method in EVALUATED_METHODS:
            records = rollouts[method]["records"]
            per_path.append(
                path_metrics(
                    str(config["protocol_version"]),
                    policy_seed,
                    int(evaluation_seed),
                    method,
                    digests[method],
                    records,
                )
            )
            if collect_steps:
                step_records.extend(
                    audit_step_record(policy_seed, int(evaluation_seed), method, row)
                    for row in records
                )
    return {
        "per_path": per_path,
        "step_records": step_records,
        "paths_equal": bool(all(path_checks)),
        "action_uniforms_equal": bool(all(uniform_checks)),
    }


def convergence(
    history: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    if bool(config.get("engineering_smoke")):
        return {"stable": True, "status": "not_applicable_engineering_smoke"}
    required = int(config["response_required_stable_evaluations"])
    if len(history) < required:
        return {"stable": False, "status": "insufficient_evaluations"}
    recent = list(history[-required:])
    ranges = {}
    for method in NEW_METHODS:
        values = [
            mean(
                float(row["low_execution_half_spread"])
                for row in item["per_path"]
                if row["method"] == method
            )
            for item in recent
        ]
        ranges[method] = max(values) - min(values)
    maximum = float(config["response_maximum_fixed_panel_spread_range"])
    return {
        "stable": all(value <= maximum for value in ranges.values()),
        "status": "evaluated",
        "spread_ranges": ranges,
        "maximum_allowed": maximum,
    }


def gate_off_exact(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
) -> bool:
    base = load_base(normalized_config(config), policy_seed)
    schedule = [0.0] * int(config["episode_steps"])
    common = {
        "seed": int(config["evaluation_seeds"][0]),
        "episode_steps": int(config["episode_steps"]),
        "train": False,
        "announced_signal_schedule": schedule,
        "reward_scale": float(config["reward_scale"]),
        "action_namespace": int(config["evaluation_action_namespace"]),
    }
    rollouts = [
        rollout_episode(env, trainer, **common)
        for trainer in (base, *trainers.values())
    ]
    reference = [row["actions"] for row in rollouts[0]["records"]]
    return all(
        [row["actions"] for row in rollout["records"]] == reference
        for rollout in rollouts[1:]
    )


def train_seed(
    policy_seed: int,
    config: dict[str, Any],
    output_dir_raw: str,
    fingerprint: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    output_dir = Path(output_dir_raw)
    risk_model = ABIDESTwoStageGateModel.load(REPO / str(config["gate_artifact"]))
    parameters = load_mechanism_parameters(config)
    v5_trainers, loaded_v5_digests = load_v5_trainers(config, policy_seed)
    (
        trainers,
        completed,
        training_history,
        evaluation_history,
        training_paths_equal,
        training_uniforms_equal,
        base_digest,
        initial_adapter_digest,
        checkpoint_v5_digests,
    ) = initial_state(config, output_dir, policy_seed, fingerprint)
    if checkpoint_v5_digests != loaded_v5_digests:
        raise RuntimeError("v5 model digests changed across resume")
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(market_config(normalized_config(config)), docker) as env:
        while completed < int(config["response_episodes"]):
            batch = min(
                int(config["response_episodes_per_update"]),
                int(config["response_episodes"]) - completed,
            )
            trajectories = {method: [] for method in NEW_METHODS}
            training_metrics = {method: [] for method in NEW_METHODS}
            for offset in range(batch):
                environment_seed = response_environment_seed(
                    normalized_config(config), policy_seed, completed + offset
                )
                rollouts, paths_equal, uniforms_equal = rollout_new_methods(
                    env,
                    config,
                    environment_seed,
                    trainers,
                    parameters,
                    train=True,
                )
                training_paths_equal = training_paths_equal and paths_equal
                training_uniforms_equal = training_uniforms_equal and uniforms_equal
                for method in NEW_METHODS:
                    trajectories[method].append(rollouts[method]["trajectory"])
                    training_metrics[method].append(
                        {
                            "mean_half_spread": float(
                                rollouts[method]["mean_half_spread"]
                            ),
                            "mean_reward": float(rollouts[method]["mean_reward"]),
                            "announced_activation_rate": float(
                                rollouts[method]["activation_rate"]
                            ),
                        }
                    )
            updates = {
                method: trainers[method].update_episodes(trajectories[method])
                for method in NEW_METHODS
            }
            completed += batch
            training_history.append(
                {
                    "completed_episodes": completed,
                    "methods": {
                        method: {
                            key: float(
                                mean(row[key] for row in training_metrics[method])
                            )
                            for key in training_metrics[method][0]
                        }
                        | {key: float(value) for key, value in updates[method].items()}
                        for method in NEW_METHODS
                    },
                }
            )
            if (
                not bool(config.get("engineering_smoke"))
                and completed >= int(config["response_evaluation_start"])
                and completed % int(config["response_evaluation_interval"]) == 0
            ):
                count = int(config["response_convergence_path_count"])
                evaluation_history.append(
                    evaluate(
                        env,
                        config,
                        policy_seed,
                        trainers,
                        v5_trainers,
                        risk_model,
                        parameters,
                        config["evaluation_seeds"][:count],
                        collect_steps=False,
                    )
                )
            save_checkpoint(
                config,
                output_dir,
                policy_seed,
                fingerprint,
                trainers,
                completed,
                training_history,
                evaluation_history,
                training_paths_equal,
                training_uniforms_equal,
                base_digest,
                initial_adapter_digest,
                loaded_v5_digests,
            )
        final = evaluate(
            env,
            config,
            policy_seed,
            trainers,
            v5_trainers,
            risk_model,
            parameters,
            config["evaluation_seeds"],
            collect_steps=True,
        )
        off_exact = gate_off_exact(env, config, policy_seed, trainers)

    step_path = step_records_path(output_dir, policy_seed)
    write_jsonl_gz(step_path, final.pop("step_records"))
    result = {
        "protocol_fingerprint": fingerprint,
        "policy_seed": int(policy_seed),
        "completed_episodes": int(completed),
        "elapsed_seconds": time.perf_counter() - started,
        "base_model_digest": base_digest,
        "initial_adapter_digest": initial_adapter_digest,
        "v5_model_digests": loaded_v5_digests,
        "training_paths_equal": training_paths_equal,
        "training_action_uniforms_equal": training_uniforms_equal,
        "convergence": convergence(evaluation_history, config),
        "gate_off_exact": off_exact,
        "final_evaluation": final,
        "final_model_digests": {
            method: model_digest(trainer) for method, trainer in trainers.items()
        },
        "step_records_file": str(step_path.relative_to(REPO)),
        "step_records_sha256": sha256_file(step_path),
    }
    atomic_write_json(result_path(output_dir, policy_seed), result)
    return result


def per_seed_rows(per_path: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
    for row in per_path:
        grouped.setdefault(int(row["policy_seed"]), {}).setdefault(
            str(row["method"]), []
        ).append(row)
    rows = []
    metrics = (
        "low_execution_half_spread",
        "mean_reward",
        "dealer_reward_units_per_episode",
        "activation_rate",
        "high_risk_activation_rate",
        "mean_signal_strength",
        "mean_routing_total_variation",
        "mean_flow_hhi",
        "mean_inventory_std",
        "mean_volume",
    )
    for policy_seed in sorted(grouped):
        row: dict[str, Any] = {"policy_seed": policy_seed}
        for method in EVALUATED_METHODS:
            method_rows = grouped[policy_seed][method]
            for metric in metrics:
                row[f"{method}__{metric}"] = float(
                    mean(float(value[metric]) for value in method_rows)
                )
        for comparator in (
            "brero_threshold_buybox_budget_matched",
            "spread_only_budget_matched",
            "matched_random",
        ):
            row[f"{comparator}__minus_risk_aware_spread"] = (
                row[f"{comparator}__low_execution_half_spread"]
                - row["risk_aware__low_execution_half_spread"]
            )
        rows.append(row)
    return rows


def aggregate(
    results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    per_path = [
        row for result in results for row in result["final_evaluation"]["per_path"]
    ]
    seed_rows = per_seed_rows(per_path)
    method_summary = {}
    for method in EVALUATED_METHODS:
        method_rows = [row for row in per_path if row["method"] == method]
        method_summary[method] = {
            metric: float(mean(float(row[metric]) for row in method_rows))
            for metric in (
                "low_execution_half_spread",
                "mean_reward",
                "dealer_reward_units_per_episode",
                "activation_rate",
                "high_risk_activation_rate",
                "mean_signal_strength",
                "mean_routing_total_variation",
                "mean_flow_hhi",
                "mean_inventory_std",
                "mean_volume",
            )
        }
    contrasts = {}
    for comparator in (
        "brero_threshold_buybox_budget_matched",
        "spread_only_budget_matched",
        "matched_random",
    ):
        contrasts[f"{comparator}_minus_risk_aware"] = paired_summary(
            [float(row[f"{comparator}__minus_risk_aware_spread"]) for row in seed_rows]
        )
    expected_path_rows = (
        len(config["policy_seeds"])
        * len(config["evaluation_seeds"])
        * len(EVALUATED_METHODS)
    )
    checks = {
        "required_policy_seed_count": len(results) == len(config["policy_seeds"]),
        "response_episodes_complete": all(
            int(result["completed_episodes"]) == int(config["response_episodes"])
            for result in results
        ),
        "source_v5_digests_present": all(
            set(result["v5_model_digests"]) == set(V5_TRAINED_METHODS)
            for result in results
        ),
        "identical_initial_adapter_digest": all(
            bool(result["initial_adapter_digest"]) for result in results
        ),
        "training_paths_equal": all(
            bool(result["training_paths_equal"]) for result in results
        ),
        "training_action_uniforms_equal": all(
            bool(result["training_action_uniforms_equal"]) for result in results
        ),
        "evaluation_paths_equal": all(
            bool(result["final_evaluation"]["paths_equal"]) for result in results
        ),
        "evaluation_action_uniforms_equal": all(
            bool(result["final_evaluation"]["action_uniforms_equal"])
            for result in results
        ),
        "gate_off_exact": all(bool(result["gate_off_exact"]) for result in results),
        "response_converged": all(
            bool(result["convergence"]["stable"]) for result in results
        ),
        "volume_conserved": all(
            abs(method_summary[method]["mean_volume"] - 20.0) < 1e-9
            for method in EVALUATED_METHODS
        ),
        "per_path_row_count": len(per_path) == expected_path_rows,
        "step_record_hashes_present": all(
            bool(result["step_records_sha256"]) for result in results
        ),
    }
    implementation_status = "pass" if all(checks.values()) else "fail"
    scientific_result = "not_evaluated_engineering_smoke"
    if not bool(config.get("engineering_smoke")) and implementation_status == "pass":
        primary = contrasts["brero_threshold_buybox_budget_matched_minus_risk_aware"]
        selected = config["mechanism_parameters"][
            "brero_threshold_buybox_budget_matched"
        ]
        matched = selected.get("budget_match_status") == "matched"
        favorable = (
            matched
            and float(primary["mean"]) > 0.0
            and float(primary["ci_low"]) > 0.0
            and int(primary["positive"]) >= int(config["minimum_positive_policy_seeds"])
        )
        scientific_result = "risk_aware_better" if favorable else "inconclusive"
    return {
        "implementation_status": implementation_status,
        "scientific_result": scientific_result,
        "checks": checks,
        "method_summary": method_summary,
        "paired_contrasts": contrasts,
        "per_path": per_path,
        "per_seed": seed_rows,
    }


def validate_inputs(
    config: Mapping[str, Any], engineering_smoke: bool
) -> dict[str, Any]:
    if bool(config.get("engineering_smoke")) != engineering_smoke:
        raise ValueError("CLI smoke mode and config engineering_smoke disagree")
    if not engineering_smoke and not bool(config.get("formal_execution_allowed")):
        raise RuntimeError("formal execution is blocked until calibration freeze")
    if tuple(config["new_adapter_methods"]) != NEW_METHODS:
        raise ValueError(f"new adapter methods must be {NEW_METHODS}")
    if int(config["response_episodes"]) != 96 and not engineering_smoke:
        raise ValueError("formal response budget must remain 96 episodes")
    gate_path = REPO / str(config["gate_artifact"])
    gate_hash = sha256_file(gate_path)
    expected_gate_hash = config.get("gate_artifact_sha256")
    if expected_gate_hash is not None and gate_hash != str(expected_gate_hash):
        raise RuntimeError("gate artifact hash mismatch")
    calibration = {
        int(seed) for seed in config.get("calibration_environment_seeds", [])
    }
    evaluation = {int(seed) for seed in config["evaluation_seeds"]}
    if calibration & evaluation:
        raise RuntimeError("calibration and evaluation paths overlap")
    source_output = REPO / str(config["source_v5_output_dir"])
    source_summary = json.loads(
        (source_output / "summary.json").read_text(encoding="utf-8")
    )
    if source_summary.get("status") != "pass":
        raise RuntimeError("domain benchmark requires passed v5 source")
    artifacts: dict[str, Any] = {
        "gate_artifact_sha256": gate_hash,
        "docker_image_id": resolve_docker_image_id(str(config["docker_image"])),
        "source_base_fingerprint": str(config["source_base_fingerprint"]),
        "source_v5_fingerprint": str(config["source_v5_fingerprint"]),
    }
    if not engineering_smoke:
        calibration_path = REPO / str(config["calibration_artifact"])
        calibration_hash = sha256_file(calibration_path)
        if calibration_hash != str(config["calibration_artifact_sha256"]):
            raise RuntimeError("calibration artifact hash mismatch")
        artifacts["calibration_artifact_sha256"] = calibration_hash
    load_mechanism_parameters(config)
    return artifacts


def load_results(
    output_dir: Path, policy_seeds: Sequence[int], fingerprint: str
) -> list[dict[str, Any]]:
    rows = []
    for policy_seed in policy_seeds:
        path = result_path(output_dir, int(policy_seed))
        if not path.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"domain result fingerprint mismatch: {path}")
        step_path = REPO / str(row["step_records_file"])
        if sha256_file(step_path) != row["step_records_sha256"]:
            raise RuntimeError(f"domain step-record hash mismatch: {step_path}")
        rows.append(row)
    return rows


def persist_state(
    output_dir: Path,
    manifest: Mapping[str, Any],
    status: str,
    results: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]] = (),
) -> None:
    policy_seeds = [int(seed) for seed in manifest["policy_seeds"]]
    complete = {int(row["policy_seed"]): row for row in results}
    atomic_write_json(
        output_dir / "run_state.json",
        {
            "status": status,
            "updated_at": utc_now(),
            "completed_policy_seeds": sorted(complete),
            "pending_policy_seeds": [
                seed for seed in policy_seeds if seed not in complete
            ],
            "failures": list(failures),
        },
    )
    lines = [
        "# ABIDES Domain Benchmark v1 Logbook",
        "",
        f"- Status: `{status}`",
        f"- Fingerprint: `{manifest['protocol_fingerprint']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Completed: `{len(complete)}/{len(policy_seeds)}`",
        "",
        "| Policy seed | Status | Episodes | Runtime (s) |",
        "| ---: | --- | ---: | ---: |",
    ]
    for seed in policy_seeds:
        row = complete.get(seed)
        if row:
            lines.append(
                f"| {seed} | complete | {row['completed_episodes']} | "
                f"{float(row['elapsed_seconds']):.2f} |"
            )
        else:
            completed = ""
            checkpoint = checkpoint_path(output_dir, seed)
            if checkpoint.exists():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                completed = str(payload["completed_episodes"])
            lines.append(f"| {seed} | pending | {completed} |  |")
    atomic_write_text(output_dir / "LOGBOOK.md", "\n".join(lines) + "\n")


def execute(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    artifacts = validate_inputs(config, args.engineering_smoke)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError("domain benchmark requires a clean committed worktree")
    fingerprint = protocol_fingerprint(config)
    output_dir = REPO / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(output_dir / ".run.lock", fingerprint, args.force_unlock):
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("protocol_fingerprint") != fingerprint:
                raise RuntimeError("existing domain manifest has another protocol")
            if manifest.get("artifacts") != artifacts:
                raise RuntimeError("domain resume artifact mismatch")
        else:
            manifest = {
                "protocol_version": str(config["protocol_version"]),
                "protocol_fingerprint": fingerprint,
                "git_commit": current_git_commit(),
                "started_at": utc_now(),
                "engineering_smoke": bool(config.get("engineering_smoke")),
                "policy_seeds": [int(seed) for seed in config["policy_seeds"]],
                "evaluation_seeds": [int(seed) for seed in config["evaluation_seeds"]],
                "artifacts": artifacts,
                "config": config,
            }
            atomic_write_json(manifest_path, manifest)
        results = load_results(output_dir, config["policy_seeds"], fingerprint)
        complete = {int(row["policy_seed"]) for row in results}
        pending = [
            int(seed) for seed in config["policy_seeds"] if int(seed) not in complete
        ]
        if pending and manifest.get("git_commit") != current_git_commit():
            raise RuntimeError("cannot resume pending domain run from another commit")
        append_logbook(
            output_dir / "logbook.jsonl",
            {
                "event": "run_resumed" if results else "run_started",
                "completed_policy_seeds": sorted(complete),
                "pending_policy_seeds": pending,
            },
        )
        persist_state(output_dir, manifest, "running", results)
        failures = []
        if pending:
            workers = min(int(config["workers"]), len(pending))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        train_seed,
                        policy_seed,
                        config,
                        str(output_dir),
                        fingerprint,
                    ): policy_seed
                    for policy_seed in pending
                }
                for future in as_completed(futures):
                    policy_seed = futures[future]
                    try:
                        result = future.result()
                    except BaseException as exc:
                        failure = {
                            "policy_seed": policy_seed,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {"event": "seed_failed", **failure},
                        )
                    else:
                        results.append(result)
                        results.sort(key=lambda row: int(row["policy_seed"]))
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {
                                "event": "seed_completed",
                                "policy_seed": policy_seed,
                                "elapsed_seconds": result["elapsed_seconds"],
                            },
                        )
                    persist_state(
                        output_dir,
                        manifest,
                        "failed" if failures else "running",
                        results,
                        failures,
                    )
        if failures:
            raise RuntimeError(f"{len(failures)} domain seed(s) failed; rerun")
        summary = aggregate(results, config)
        summary["status"] = (
            "smoke_complete"
            if args.engineering_smoke and summary["implementation_status"] == "pass"
            else summary["implementation_status"]
        )
        write_csv(output_dir / "per_path.csv", summary.pop("per_path"))
        write_csv(output_dir / "per_seed.csv", summary.pop("per_seed"))
        atomic_write_json(output_dir / "rows.json", results)
        atomic_write_json(output_dir / "summary.json", summary)
        persist_state(output_dir, manifest, summary["status"], results)
        append_logbook(
            output_dir / "logbook.jsonl",
            {"event": "run_finished", "status": summary["status"]},
        )
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "implementation_status": summary["implementation_status"],
                    "checks": summary["checks"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if summary["implementation_status"] != "pass":
            raise SystemExit(2)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_domain_benchmark_smoke_v1.json",
    )
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    execute(config, args)


if __name__ == "__main__":
    main()
