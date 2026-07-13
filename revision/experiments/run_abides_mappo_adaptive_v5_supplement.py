#!/usr/bin/env python3
"""Checkpointed supplemental controls for the passed adaptive-v5 experiment."""

from __future__ import annotations

import argparse
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
    exact_signal_dose,
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
    ABIDESPacedSignalController,
    ABIDESPublicRiskProxyModel,
    ABIDESTwoStageGateModel,
)


NEW_METHODS = ("circular_shifted", "volatility_only", "return_only")
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


def seed_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "supplement_seeds" / f"seed_{seed}"


def checkpoint_path(output_dir: Path, seed: int) -> Path:
    return seed_dir(output_dir, seed) / "training_state.pt"


def result_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "supplement_results" / f"seed_{seed}.json"


def circular_shift(
    strengths: Sequence[float], seed: int
) -> tuple[list[float], int]:
    values = np.asarray(strengths, dtype=float)
    if len(values) < 2:
        raise ValueError("circular comparator requires at least two steps")
    offset = int(np.random.default_rng(seed ^ 0xC1A0).integers(1, len(values)))
    return np.roll(values, offset).tolist(), offset


def circular_structure_exact(
    source: Sequence[float], shifted: Sequence[float], offset: int
) -> bool:
    return bool(
        0 < offset < len(source)
        and np.allclose(
            np.roll(np.asarray(source, dtype=float), offset),
            np.asarray(shifted, dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
    )


def proxy_decision(
    model: ABIDESPublicRiskProxyModel,
    controller: ABIDESPacedSignalController,
) -> SignalDecision:
    def decide(state: Mapping[str, Any], step: int) -> float:
        proposal = model.announced_strength(state)
        return controller.decide(float(proposal["strength"]))

    return decide


def load_v5_trainers(
    config: Mapping[str, Any], seed: int
) -> tuple[dict[str, MAPPOTrainer], dict[str, str]]:
    output_dir = REPO / str(config["v5_output_dir"])
    checkpoint = (
        output_dir / "response_seeds" / f"seed_{seed}" / "training_state.pt"
    )
    result_file = output_dir / "response_results" / f"seed_{seed}.json"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = json.loads(result_file.read_text(encoding="utf-8"))
    expected = str(config["v5_protocol_fingerprint"])
    if payload.get("protocol_fingerprint") != expected:
        raise RuntimeError(f"v5 checkpoint fingerprint mismatch: {checkpoint}")
    if result.get("protocol_fingerprint") != expected:
        raise RuntimeError(f"v5 result fingerprint mismatch: {result_file}")
    if int(payload.get("completed_episodes", -1)) != int(
        config["v5_response_episodes"]
    ):
        raise RuntimeError(f"v5 response checkpoint is incomplete: {checkpoint}")
    trainers = {
        method: trainer_from_payload(payload["trainers"][method], str(config["device"]))
        for method in V5_TRAINED_METHODS
    }
    digests = {method: model_digest(trainer) for method, trainer in trainers.items()}
    if digests != result["final_model_digests"]:
        raise RuntimeError(f"v5 model digest mismatch: {checkpoint}")
    return trainers, digests


def load_or_initialize(
    config: Mapping[str, Any], output_dir: Path, seed: int, fingerprint: str
) -> tuple[
    dict[str, MAPPOTrainer],
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    bool,
    bool,
    bool,
    str,
    dict[str, str],
]:
    path = checkpoint_path(output_dir, seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"supplement checkpoint mismatch: {path}")
        return (
            {
                method: trainer_from_payload(data, str(config["device"]))
                for method, data in payload["trainers"].items()
            },
            int(payload["completed_episodes"]),
            list(payload["training_history"]),
            list(payload["evaluation_history"]),
            bool(payload["training_paths_equal"]),
            bool(payload["training_circular_dose_exact"]),
            bool(payload["training_circular_structure_exact"]),
            str(payload["base_model_digest"]),
            dict(payload["v5_model_digests"]),
        )
    base = load_base(config, seed)
    trainers = {method: make_adapter(base, config, seed) for method in NEW_METHODS}
    if len({model_digest(trainer) for trainer in trainers.values()}) != 1:
        raise RuntimeError("supplement adapters did not start identically")
    _, v5_digests = load_v5_trainers(config, seed)
    return trainers, 0, [], [], True, True, True, model_digest(base), v5_digests


def save_checkpoint(
    output_dir: Path,
    seed: int,
    fingerprint: str,
    completed: int,
    trainers: Mapping[str, MAPPOTrainer],
    training_history: Sequence[Mapping[str, Any]],
    evaluation_history: Sequence[Mapping[str, Any]],
    paths_equal: bool,
    circular_dose_exact: bool,
    structure_exact: bool,
    base_digest: str,
    v5_digests: Mapping[str, str],
) -> None:
    atomic_torch_save(
        checkpoint_path(output_dir, seed),
        {
            "protocol_fingerprint": fingerprint,
            "seed": seed,
            "completed_episodes": completed,
            "trainers": {
                method: trainer_payload(trainer) for method, trainer in trainers.items()
            },
            "training_history": list(training_history),
            "evaluation_history": list(evaluation_history),
            "training_paths_equal": paths_equal,
            "training_circular_dose_exact": circular_dose_exact,
            "training_circular_structure_exact": structure_exact,
            "base_model_digest": base_digest,
            "v5_model_digests": dict(v5_digests),
            "saved_at": utc_now(),
        },
    )


def risk_source_rollout(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    seed: int,
    trainer: MAPPOTrainer,
    risk_model: ABIDESTwoStageGateModel,
    action_namespace: int,
) -> dict[str, Any]:
    controller = ABIDESPacedSignalController(
        int(config["episode_steps"]),
        target_rate=float(config["max_activation_rate"]),
    )
    return rollout_episode(
        env,
        trainer,
        seed=seed,
        episode_steps=int(config["episode_steps"]),
        train=False,
        announced_signal_decision=paced_learned_decision(
            risk_model, controller, include_severity=False
        ),
        reward_scale=float(config["reward_scale"]),
        action_namespace=action_namespace,
    )


def rollout_new_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    v5_trainers: Mapping[str, MAPPOTrainer],
    risk_model: ABIDESTwoStageGateModel,
    proxy_models: Mapping[str, ABIDESPublicRiskProxyModel],
    train: bool,
) -> tuple[dict[str, Any], bool, bool, bool]:
    namespace = (
        int(config["response_action_namespace"])
        if train
        else int(config["evaluation_action_namespace"])
    )
    source = risk_source_rollout(
        env,
        config,
        seed,
        v5_trainers["risk_aware"],
        risk_model,
        namespace,
    )
    source_strengths = [float(row["gate_signal"]) for row in source["records"]]
    circular_schedule, offset = circular_shift(source_strengths, seed)
    common = {
        "seed": seed,
        "episode_steps": int(config["episode_steps"]),
        "train": train,
        "defer_update": train,
        "reward_scale": float(config["reward_scale"]),
        "action_namespace": namespace,
    }
    circular = rollout_episode(
        env,
        trainers["circular_shifted"],
        announced_signal_schedule=circular_schedule,
        **common,
    )
    proxies = {}
    for method in ("volatility_only", "return_only"):
        proxy = proxy_models[method]
        controller = ABIDESPacedSignalController(
            int(config["episode_steps"]),
            target_rate=proxy.pacing_target_rate,
        )
        proxies[method] = rollout_episode(
            env,
            trainers[method],
            announced_signal_decision=proxy_decision(proxy, controller),
            **common,
        )
    rollouts = {"risk_aware": source, "circular_shifted": circular, **proxies}
    paths_equal = all(
        validate_common_path(source, rollout) for rollout in rollouts.values()
    )
    return (
        rollouts,
        paths_equal,
        exact_signal_dose(source_strengths, circular_schedule),
        circular_structure_exact(source_strengths, circular_schedule, offset),
    )


def rollout_all_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    environment_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    v5_trainers: Mapping[str, MAPPOTrainer],
    risk_model: ABIDESTwoStageGateModel,
    proxy_models: Mapping[str, ABIDESPublicRiskProxyModel],
) -> tuple[dict[str, Any], bool, bool, bool]:
    episode_steps = int(config["episode_steps"])
    namespace = int(config["evaluation_action_namespace"])
    risk = risk_source_rollout(
        env,
        config,
        environment_seed,
        v5_trainers["risk_aware"],
        risk_model,
        namespace,
    )
    strengths = [float(row["gate_signal"]) for row in risk["records"]]
    random_schedule = global_shuffle(strengths, environment_seed ^ 0xD317)
    circular_schedule, offset = circular_shift(strengths, environment_seed)
    common = {
        "seed": environment_seed,
        "episode_steps": episode_steps,
        "train": False,
        "reward_scale": float(config["reward_scale"]),
        "action_namespace": namespace,
    }
    rollouts = {
        "risk_aware": risk,
        "matched_random": rollout_episode(
            env,
            v5_trainers["matched_random"],
            announced_signal_schedule=random_schedule,
            **common,
        ),
        "spread_only_threshold": rollout_episode(
            env,
            v5_trainers["spread_only_threshold"],
            announced_signal_decision=spread_only_decision(
                ABIDESPacedSignalController(
                    episode_steps, target_rate=float(config["max_activation_rate"])
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
        "circular_shifted": rollout_episode(
            env,
            trainers["circular_shifted"],
            announced_signal_schedule=circular_schedule,
            **common,
        ),
    }
    for method in ("volatility_only", "return_only"):
        proxy = proxy_models[method]
        rollouts[method] = rollout_episode(
            env,
            trainers[method],
            announced_signal_decision=proxy_decision(
                proxy,
                ABIDESPacedSignalController(
                    episode_steps, target_rate=proxy.pacing_target_rate
                ),
            ),
            **common,
        )
    base = load_base(config, policy_seed)
    rollouts["no_intervention"] = rollout_episode(
        env,
        base,
        announced_signal_schedule=[0.0] * episode_steps,
        **common,
    )
    paths_equal = all(
        validate_common_path(risk, rollout) for rollout in rollouts.values()
    )
    circular_dose = exact_signal_dose(strengths, circular_schedule)
    structure = circular_structure_exact(strengths, circular_schedule, offset)
    return rollouts, paths_equal, circular_dose, structure


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    low = [row for row in records if row["info"].get("last_step_risk") == "low"]
    high = [row for row in records if row["info"].get("last_step_risk") == "high"]
    episodes = max(sum(int(row["step"]) == 0 for row in records), 1)
    flow_hhi = []
    inventory_std = []
    for row in records:
        shares = np.asarray(row["info"].get("flow_shares", [0.1] * 10), dtype=float)
        total = float(np.sum(shares))
        flow_hhi.append(float(np.sum(np.square(shares / total))) if total else 0.0)
        observations = np.asarray(row["pre_observations"], dtype=float)
        inventory_std.append(float(np.std(observations[:, 0] * 100.0)))
    low_customer_cost = sum(
        float(row["info"]["mean_execution_half_spread"])
        * float(row["info"]["realized_flow_quantity"])
        for row in low
    )
    return {
        "low_execution_half_spread": float(
            mean(float(row["info"]["mean_execution_half_spread"]) for row in low)
        ),
        "high_execution_half_spread": float(
            mean(float(row["info"]["mean_execution_half_spread"]) for row in high)
            if high
            else 0.0
        ),
        "mean_reward": float(mean(float(row["mean_reward_unscaled"]) for row in records)),
        "dealer_reward_units_per_episode": float(
            sum(float(row["mean_reward_unscaled"]) * 10.0 for row in records)
            / episodes
        ),
        "mean_volume": float(
            mean(float(row["info"]["realized_flow_quantity"]) for row in records)
        ),
        "activation_rate": float(mean(bool(row["gate_active"]) for row in records)),
        "high_risk_activation_rate": float(
            mean(bool(row["gate_active"]) for row in high) if high else 0.0
        ),
        "mean_signal_strength": float(
            mean(float(row["gate_signal"]) for row in records)
        ),
        "targeted_events_per_episode": float(
            sum(int(row["info"]["targeted_event_count"]) for row in records)
            / episodes
        ),
        "low_customer_cost_tick_units_per_episode": float(
            low_customer_cost / episodes
        ),
        "low_risk_steps_per_episode": float(len(low) / episodes),
        "mean_spread_revenue": float(
            mean(float(row["info"]["mean_spread_revenue"]) for row in records)
        ),
        "mean_adverse_selection_component": float(
            mean(
                float(row["info"]["mean_adverse_selection_component"])
                for row in records
            )
        ),
        "mean_inventory_penalty": float(
            mean(float(row["info"]["mean_inventory_penalty"]) for row in records)
        ),
        "mean_flow_hhi": float(mean(flow_hhi)),
        "mean_inventory_std": float(mean(inventory_std)),
    }


def evaluate_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    v5_trainers: Mapping[str, MAPPOTrainer],
    risk_model: ABIDESTwoStageGateModel,
    proxy_models: Mapping[str, ABIDESPublicRiskProxyModel],
    evaluation_seeds: Sequence[int],
    all_methods: bool,
) -> dict[str, Any]:
    selected = EVALUATED_METHODS if all_methods else ("risk_aware", *NEW_METHODS)
    records = {method: [] for method in selected}
    path_checks = []
    dose_checks = []
    structure_checks = []
    for evaluation_seed in evaluation_seeds:
        if all_methods:
            rollouts, paths, dose, structure = rollout_all_methods(
                env,
                config,
                policy_seed,
                int(evaluation_seed),
                trainers,
                v5_trainers,
                risk_model,
                proxy_models,
            )
        else:
            rollouts, paths, dose, structure = rollout_new_methods(
                env,
                config,
                int(evaluation_seed),
                trainers,
                v5_trainers,
                risk_model,
                proxy_models,
                train=False,
            )
        for method in selected:
            records[method].extend(rollouts[method]["records"])
        path_checks.append(paths)
        dose_checks.append(dose)
        structure_checks.append(structure)
    return {
        "methods": {
            method: summarize_records(method_records)
            for method, method_records in records.items()
        },
        "paths_equal": bool(all(path_checks)),
        "circular_dose_exact": bool(all(dose_checks)),
        "circular_structure_exact": bool(all(structure_checks)),
    }


def convergence(
    history: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    required = int(config["response_required_stable_evaluations"])
    if len(history) < required:
        return {"stable": False, "reason": "insufficient evaluations"}
    recent = list(history[-required:])
    ranges = {
        method: max(
            float(row["evaluation"]["methods"][method]["low_execution_half_spread"])
            for row in recent
        )
        - min(
            float(row["evaluation"]["methods"][method]["low_execution_half_spread"])
            for row in recent
        )
        for method in NEW_METHODS
    }
    maximum = float(config["response_maximum_fixed_panel_spread_range"])
    return {
        "stable": all(value <= maximum for value in ranges.values()),
        "spread_ranges": ranges,
        "method_stability": {
            method: value <= maximum for method, value in ranges.items()
        },
        "recent_evaluations": recent,
    }


def gate_off_exact(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
) -> bool:
    base = load_base(config, policy_seed)
    schedule = [0.0] * int(config["episode_steps"])
    rollouts = [
        rollout_episode(
            env,
            trainer,
            seed=int(config["evaluation_seeds"][0]),
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_schedule=schedule,
            reward_scale=float(config["reward_scale"]),
            action_namespace=int(config["evaluation_action_namespace"]),
        )
        for trainer in (base, *[trainers[method] for method in NEW_METHODS])
    ]
    reference = [row["actions"] for row in rollouts[0]["records"]]
    return all(
        [row["actions"] for row in rollout["records"]] == reference
        for rollout in rollouts[1:]
    )


def load_proxy_models(
    config: Mapping[str, Any]
) -> dict[str, ABIDESPublicRiskProxyModel]:
    return {
        method: ABIDESPublicRiskProxyModel.load(REPO / config["proxy_artifacts"][method])
        for method in ("volatility_only", "return_only")
    }


def train_seed(
    seed: int, config: dict[str, Any], output_dir_raw: str, fingerprint: str
) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    output_dir = Path(output_dir_raw)
    risk_model = ABIDESTwoStageGateModel.load(REPO / str(config["gate_artifact"]))
    proxy_models = load_proxy_models(config)
    v5_trainers, loaded_v5_digests = load_v5_trainers(config, seed)
    (
        trainers,
        completed,
        training_history,
        evaluation_history,
        paths_equal,
        circular_dose_exact,
        structure_exact,
        base_digest,
        checkpoint_v5_digests,
    ) = load_or_initialize(config, output_dir, seed, fingerprint)
    if checkpoint_v5_digests != loaded_v5_digests:
        raise RuntimeError("v5 source model changed across supplement resume")
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(market_config(config), docker) as env:
        while completed < int(config["response_episodes"]):
            batch = min(
                int(config["response_episodes_per_update"]),
                int(config["response_episodes"]) - completed,
            )
            trajectories = {method: [] for method in NEW_METHODS}
            spreads = {method: [] for method in NEW_METHODS}
            rewards = {method: [] for method in NEW_METHODS}
            activations = {method: [] for method in NEW_METHODS}
            for offset in range(batch):
                episode_seed = response_environment_seed(
                    config, seed, completed + offset
                )
                rollouts, paths, dose, structure = rollout_new_methods(
                    env,
                    config,
                    episode_seed,
                    trainers,
                    v5_trainers,
                    risk_model,
                    proxy_models,
                    train=True,
                )
                paths_equal = paths_equal and paths
                circular_dose_exact = circular_dose_exact and dose
                structure_exact = structure_exact and structure
                for method in NEW_METHODS:
                    trajectories[method].append(rollouts[method]["trajectory"])
                    spreads[method].append(float(rollouts[method]["mean_half_spread"]))
                    rewards[method].append(float(rollouts[method]["mean_reward"]))
                    activations[method].append(float(rollouts[method]["activation_rate"]))
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
                            "mean_half_spread": mean(spreads[method]),
                            "mean_reward": mean(rewards[method]),
                            "activation_rate": mean(activations[method]),
                            **{key: float(value) for key, value in updates[method].items()},
                        }
                        for method in NEW_METHODS
                    },
                }
            )
            if (
                completed >= int(config["response_evaluation_start"])
                and completed % int(config["response_evaluation_interval"]) == 0
            ):
                count = int(config["response_convergence_path_count"])
                evaluation_history.append(
                    {
                        "completed_episodes": completed,
                        "evaluation": evaluate_methods(
                            env,
                            config,
                            seed,
                            trainers,
                            v5_trainers,
                            risk_model,
                            proxy_models,
                            config["evaluation_seeds"][:count],
                            all_methods=False,
                        ),
                    }
                )
            save_checkpoint(
                output_dir,
                seed,
                fingerprint,
                completed,
                trainers,
                training_history,
                evaluation_history,
                paths_equal,
                circular_dose_exact,
                structure_exact,
                base_digest,
                loaded_v5_digests,
            )
        final = evaluate_methods(
            env,
            config,
            seed,
            trainers,
            v5_trainers,
            risk_model,
            proxy_models,
            config["evaluation_seeds"],
            all_methods=True,
        )
        off_exact = gate_off_exact(env, config, seed, trainers)
    result = {
        "protocol_fingerprint": fingerprint,
        "seed": seed,
        "completed_episodes": completed,
        "elapsed_seconds": time.perf_counter() - started,
        "base_model_digest": base_digest,
        "v5_model_digests": loaded_v5_digests,
        "training_paths_equal": paths_equal,
        "training_circular_dose_exact": circular_dose_exact,
        "training_circular_structure_exact": structure_exact,
        "convergence": convergence(evaluation_history, config),
        "gate_off_exact": off_exact,
        "final_evaluation": final,
        "final_model_digests": {
            method: model_digest(trainer) for method, trainer in trainers.items()
        },
    }
    atomic_write_json(result_path(output_dir, seed), result)
    return result


def aggregate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    metric_names = tuple(
        rows[0]["final_evaluation"]["methods"]["risk_aware"].keys()
    )
    methods = {
        method: {
            metric: mean(
                float(row["final_evaluation"]["methods"][method][metric])
                for row in rows
            )
            for metric in metric_names
        }
        for method in EVALUATED_METHODS
    }

    def contrast(comparator: str) -> dict[str, Any]:
        return paired_summary(
            [
                float(
                    row["final_evaluation"]["methods"][comparator][
                        "low_execution_half_spread"
                    ]
                )
                - float(
                    row["final_evaluation"]["methods"]["risk_aware"][
                        "low_execution_half_spread"
                    ]
                )
                for row in rows
            ]
        )

    contrasts = {
        f"{method}_minus_risk_aware": contrast(method)
        for method in EVALUATED_METHODS
        if method != "risk_aware"
    }
    circular = contrasts["circular_shifted_minus_risk_aware"]
    required = math.ceil(
        float(config["minimum_positive_seed_rate"]) * len(rows) - 1e-12
    )
    activation_tolerance = float(config["proxy_activation_tolerance"])
    checks = {
        "required_policy_seed_count": len(rows)
        == int(config["required_policy_seed_count"]),
        "v5_reference_digests_present": all(
            set(row["v5_model_digests"]) == set(V5_TRAINED_METHODS) for row in rows
        ),
        "training_paths_equal": all(bool(row["training_paths_equal"]) for row in rows),
        "training_circular_dose_exact": all(
            bool(row["training_circular_dose_exact"]) for row in rows
        ),
        "training_circular_structure_exact": all(
            bool(row["training_circular_structure_exact"]) for row in rows
        ),
        "evaluation_paths_equal": all(
            bool(row["final_evaluation"]["paths_equal"]) for row in rows
        ),
        "evaluation_circular_dose_exact": all(
            bool(row["final_evaluation"]["circular_dose_exact"]) for row in rows
        ),
        "evaluation_circular_structure_exact": all(
            bool(row["final_evaluation"]["circular_structure_exact"])
            for row in rows
        ),
        "gate_off_exact": all(bool(row["gate_off_exact"]) for row in rows),
        "response_converged": all(bool(row["convergence"]["stable"]) for row in rows),
        "circular_timing_positive_mean": float(circular["mean"]) > 0.0,
        "circular_timing_positive_seed_rate": int(circular["positive"]) >= required,
        "circular_timing_positive_ci": float(circular["ci_low"]) > 0.0,
        "volatility_activation_comparable": abs(
            methods["volatility_only"]["activation_rate"]
            - methods["risk_aware"]["activation_rate"]
        )
        <= activation_tolerance,
        "return_activation_comparable": abs(
            methods["return_only"]["activation_rate"]
            - methods["risk_aware"]["activation_rate"]
        )
        <= activation_tolerance,
        "volume_conserved": all(
            abs(methods[method]["mean_volume"] - 20.0) < 1e-9
            for method in EVALUATED_METHODS
        ),
    }
    random_delta = contrasts["matched_random_minus_risk_aware"]
    economic = {
        "risk_aware_spread_reduction_vs_no_intervention_fraction": (
            methods["no_intervention"]["low_execution_half_spread"]
            - methods["risk_aware"]["low_execution_half_spread"]
        )
        / methods["no_intervention"]["low_execution_half_spread"],
        "risk_aware_spread_reduction_vs_random_fraction": float(random_delta["mean"])
        / methods["matched_random"]["low_execution_half_spread"],
        "customer_tick_units_saved_vs_random_per_episode": (
            methods["matched_random"]["low_customer_cost_tick_units_per_episode"]
            - methods["risk_aware"]["low_customer_cost_tick_units_per_episode"]
        ),
        "dealer_reward_units_change_vs_random_per_episode": (
            methods["risk_aware"]["dealer_reward_units_per_episode"]
            - methods["matched_random"]["dealer_reward_units_per_episode"]
        ),
    }
    proxy_diagnostics = {
        method: {
            "minus_risk_aware": contrasts[f"{method}_minus_risk_aware"],
            "activation_difference": methods[method]["activation_rate"]
            - methods["risk_aware"]["activation_rate"],
            "high_risk_activation_difference": methods[method][
                "high_risk_activation_rate"
            ]
            - methods["risk_aware"]["high_risk_activation_rate"],
        }
        for method in ("volatility_only", "return_only")
    }
    return {
        "method_summary": methods,
        "paired_contrasts": contrasts,
        "economic_interpretation": economic,
        "proxy_diagnostics": proxy_diagnostics,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def validate_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    if tuple(config["new_methods"]) != NEW_METHODS:
        raise ValueError(f"supplement methods must be {NEW_METHODS}")
    if int(config["v5_response_episodes"]) != 96:
        raise ValueError("supplement source must be the 96-episode v5 response")
    if int(config["response_episodes"]) != 96 and not bool(
        config.get("engineering_smoke")
    ):
        raise ValueError("supplement must match the frozen v5 response budget")
    artifacts = {}
    for key, expected_hash in (
        ("gate_artifact", config["gate_artifact_sha256"]),
        (
            "proxy_calibration_summary",
            config["proxy_calibration_summary_sha256"],
        ),
    ):
        path = REPO / str(config[key])
        actual = sha256_file(path)
        if actual != str(expected_hash):
            raise RuntimeError(f"supplement artifact hash mismatch: {path}")
        artifacts[key] = actual
    for method, path_raw in config["proxy_artifacts"].items():
        path = REPO / str(path_raw)
        actual = sha256_file(path)
        expected = str(config["proxy_artifact_sha256"][method])
        if actual != expected:
            raise RuntimeError(f"proxy artifact hash mismatch: {path}")
        artifacts[f"proxy_{method}"] = actual

    v5_output = REPO / str(config["v5_output_dir"])
    v5_manifest = json.loads((v5_output / "manifest.json").read_text(encoding="utf-8"))
    v5_summary = json.loads((v5_output / "summary.json").read_text(encoding="utf-8"))
    if (
        v5_manifest.get("protocol_fingerprint")
        != str(config["v5_protocol_fingerprint"])
        or v5_summary.get("status") != "pass"
        or not all(v5_summary.get("checks", {}).values())
    ):
        raise RuntimeError("supplement requires the passed frozen v5 source")
    if {int(seed) for seed in config["policy_seeds"]} != {
        int(seed) for seed in v5_manifest["policy_seeds"]
    }:
        if not bool(config.get("engineering_smoke")):
            raise RuntimeError("supplement policy seeds must equal the v5 policy seeds")
    prior_eval = {int(seed) for seed in v5_manifest["evaluation_seeds"]}
    current_eval = {int(seed) for seed in config["evaluation_seeds"]}
    if prior_eval & current_eval:
        raise RuntimeError("supplement evaluation paths must be fresh")
    artifacts["docker_image_id"] = resolve_docker_image_id(str(config["docker_image"]))
    artifacts["v5_protocol_fingerprint"] = str(config["v5_protocol_fingerprint"])
    return artifacts


def load_results(
    output_dir: Path, seeds: Sequence[int], fingerprint: str
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        path = result_path(output_dir, int(seed))
        if not path.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"supplement result fingerprint mismatch: {path}")
        rows.append(row)
    return rows


def persist_state(
    output_dir: Path,
    manifest: Mapping[str, Any],
    status: str,
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]] = (),
) -> None:
    seeds = [int(seed) for seed in manifest["policy_seeds"]]
    complete = {int(row["seed"]): row for row in rows}
    atomic_write_json(
        output_dir / "run_state.json",
        {
            "status": status,
            "updated_at": utc_now(),
            "completed_policy_seeds": sorted(complete),
            "pending_policy_seeds": [seed for seed in seeds if seed not in complete],
            "failures": list(failures),
        },
    )
    lines = [
        "# ABIDES/MAPPO Adaptive-v5 Supplement Logbook",
        "",
        f"- Status: `{status}`",
        f"- Fingerprint: `{manifest['protocol_fingerprint']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Completed: `{len(complete)}/{len(seeds)}`",
        "",
        "| Policy seed | Status | Episodes | Runtime (s) |",
        "| ---: | --- | ---: | ---: |",
    ]
    for seed in seeds:
        row = complete.get(seed)
        if row:
            lines.append(
                f"| {seed} | complete | {int(row['completed_episodes'])} | "
                f"{float(row['elapsed_seconds']):.2f} |"
            )
        else:
            completed = ""
            path = checkpoint_path(output_dir, seed)
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                completed = str(int(payload["completed_episodes"]))
            lines.append(f"| {seed} | pending | {completed} |  |")
    atomic_write_text(output_dir / "LOGBOOK.md", "\n".join(lines) + "\n")


def execute(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    artifacts = validate_inputs(config)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError("formal supplement requires a clean committed worktree")
    fingerprint = protocol_fingerprint(config)
    output_dir = REPO / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(output_dir / ".run.lock", fingerprint, args.force_unlock):
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("protocol_fingerprint") != fingerprint:
                raise RuntimeError("existing supplement manifest has another protocol")
            if manifest.get("artifacts") != artifacts:
                raise RuntimeError("supplement resume artifact mismatch")
        else:
            manifest = {
                "protocol_version": config["protocol_version"],
                "protocol_fingerprint": fingerprint,
                "git_commit": current_git_commit(),
                "started_at": utc_now(),
                "policy_seeds": [int(seed) for seed in config["policy_seeds"]],
                "evaluation_seeds": [int(seed) for seed in config["evaluation_seeds"]],
                "artifacts": artifacts,
                "config": config,
            }
            atomic_write_json(manifest_path, manifest)
        rows = load_results(output_dir, config["policy_seeds"], fingerprint)
        complete = {int(row["seed"]) for row in rows}
        pending = [
            int(seed) for seed in config["policy_seeds"] if int(seed) not in complete
        ]
        if pending and manifest.get("git_commit") != current_git_commit():
            raise RuntimeError("cannot resume pending supplement from another commit")
        append_logbook(
            output_dir / "logbook.jsonl",
            {
                "event": "run_resumed" if rows else "run_started",
                "completed_policy_seeds": sorted(complete),
                "pending_policy_seeds": pending,
            },
        )
        persist_state(output_dir, manifest, "running", rows)
        failures = []
        if pending:
            workers = min(int(config["workers"]), len(pending))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        train_seed, seed, config, str(output_dir), fingerprint
                    ): seed
                    for seed in pending
                }
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        row = future.result()
                    except BaseException as exc:
                        failure = {
                            "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {"event": "seed_failed", **failure},
                        )
                    else:
                        rows.append(row)
                        rows.sort(key=lambda item: int(item["seed"]))
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {
                                "event": "seed_completed",
                                "seed": seed,
                                "elapsed_seconds": float(row["elapsed_seconds"]),
                            },
                        )
                    persist_state(
                        output_dir,
                        manifest,
                        "failed" if failures else "running",
                        rows,
                        failures,
                    )
        if failures:
            raise RuntimeError(f"{len(failures)} supplement seed(s) failed; rerun")
        rows.sort(key=lambda row: int(row["seed"]))
        result = {"rows": rows, **aggregate(rows, config)}
        if args.engineering_smoke:
            integrity = (
                "training_paths_equal",
                "training_circular_dose_exact",
                "training_circular_structure_exact",
                "evaluation_paths_equal",
                "evaluation_circular_dose_exact",
                "evaluation_circular_structure_exact",
                "gate_off_exact",
                "volume_conserved",
            )
            result["status"] = (
                "smoke_complete"
                if all(result["checks"][key] for key in integrity)
                else "fail"
            )
        atomic_write_json(output_dir / "rows.json", rows)
        atomic_write_json(output_dir / "summary.json", result)
        persist_state(output_dir, manifest, result["status"], rows)
        append_logbook(
            output_dir / "logbook.jsonl",
            {"event": "run_finished", "status": result["status"]},
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "checks": result["checks"],
                    "circular": result["paired_contrasts"][
                        "circular_shifted_minus_risk_aware"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if result["status"] == "fail":
            raise SystemExit(2)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_mappo_adaptive_v5_supplement_smoke_v1.json",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    execute(config, args)


if __name__ == "__main__":
    main()
