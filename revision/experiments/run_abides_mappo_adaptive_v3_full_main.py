#!/usr/bin/env python3
"""Resumable multi-method adaptive-v3 ABIDES/MAPPO full-scale runner."""

from __future__ import annotations

import argparse
import copy
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

from run_abides_learned_gate_screen import paired_summary, summarize  # noqa: E402
from run_abides_mappo_adaptive_pilot import (  # noqa: E402
    atomic_torch_save,
    model_digest,
    trainer_from_payload,
    trainer_payload,
    validate_common_path,
)
from run_abides_mappo_adaptive_v3_learned import (  # noqa: E402
    global_shuffle,
    paced_learned_decision,
)
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
    FEATURE_NAMES,
    ABIDESPacedSignalController,
    ABIDESTwoStageGateModel,
    abides_gate_features,
    abides_lagged_actions,
)


TRAINED_METHODS = (
    "risk_aware",
    "matched_random",
    "spread_only_threshold",
    "always_on",
    "no_routing",
)
EVALUATED_METHODS = ("no_intervention", *TRAINED_METHODS)
SignalDecision = Callable[[Mapping[str, Any], int], float]


def seed_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "response_seeds" / f"seed_{seed}"


def checkpoint_path(output_dir: Path, seed: int) -> Path:
    return seed_dir(output_dir, seed) / "training_state.pt"


def result_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "response_results" / f"seed_{seed}.json"


def spread_only_decision(
    controller: ABIDESPacedSignalController,
) -> SignalDecision:
    severity_index = FEATURE_NAMES.index("contract_severity")

    def decide(state: Mapping[str, Any], step: int) -> float:
        features = abides_gate_features(state, abides_lagged_actions(state))
        return controller.decide(float(features[severity_index]))

    return decide


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
]:
    path = checkpoint_path(output_dir, seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"full-scale checkpoint mismatch: {path}")
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"full-scale checkpoint seed mismatch: {path}")
        return (
            {
                method: trainer_from_payload(data, str(config["device"]))
                for method, data in payload["trainers"].items()
            },
            int(payload["completed_episodes"]),
            list(payload["training_history"]),
            list(payload["evaluation_history"]),
            bool(payload["training_paths_equal"]),
            bool(payload["training_dose_exact"]),
            bool(payload["training_no_routing_dose_exact"]),
            str(payload["base_model_digest"]),
        )
    base = load_base(config, seed)
    trainers = {method: make_adapter(base, config, seed) for method in TRAINED_METHODS}
    if len({model_digest(trainer) for trainer in trainers.values()}) != 1:
        raise RuntimeError("full-scale adapters did not start identically")
    return trainers, 0, [], [], True, True, True, model_digest(base)


def save_checkpoint(
    output_dir: Path,
    seed: int,
    fingerprint: str,
    completed: int,
    trainers: Mapping[str, MAPPOTrainer],
    training_history: Sequence[Mapping[str, Any]],
    evaluation_history: Sequence[Mapping[str, Any]],
    paths_equal: bool,
    dose_exact: bool,
    no_routing_dose_exact: bool,
    base_digest: str,
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
            "training_dose_exact": dose_exact,
            "training_no_routing_dose_exact": no_routing_dose_exact,
            "base_model_digest": base_digest,
            "saved_at": utc_now(),
        },
    )


def rollout_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESTwoStageGateModel,
    train: bool,
    include_base: bool,
) -> tuple[dict[str, Any], bool, bool, bool]:
    episode_steps = int(config["episode_steps"])
    namespace = (
        int(config["response_action_namespace"])
        if train
        else int(config["evaluation_action_namespace"])
    )
    common = {
        "seed": seed,
        "episode_steps": episode_steps,
        "train": train,
        "defer_update": train,
        "reward_scale": float(config["reward_scale"]),
        "action_namespace": namespace,
    }
    risk_controller = ABIDESPacedSignalController(
        episode_steps, target_rate=float(config["max_activation_rate"])
    )
    risk_aware = rollout_episode(
        env,
        trainers["risk_aware"],
        announced_signal_decision=paced_learned_decision(
            model, risk_controller, include_severity=False
        ),
        **common,
    )
    risk_strengths = [float(row["gate_signal"]) for row in risk_aware["records"]]
    random_strengths = global_shuffle(risk_strengths, seed ^ 0xD317)
    matched_random = rollout_episode(
        env,
        trainers["matched_random"],
        announced_signal_schedule=random_strengths,
        **common,
    )
    threshold_controller = ABIDESPacedSignalController(
        episode_steps, target_rate=float(config["max_activation_rate"])
    )
    threshold = rollout_episode(
        env,
        trainers["spread_only_threshold"],
        announced_signal_decision=spread_only_decision(threshold_controller),
        **common,
    )
    always_on = rollout_episode(
        env,
        trainers["always_on"],
        announced_signal_schedule=[1.0] * episode_steps,
        **common,
    )
    no_routing = rollout_episode(
        env,
        trainers["no_routing"],
        announced_signal_schedule=risk_strengths,
        routing_enabled=False,
        **common,
    )
    rollouts = {
        "risk_aware": risk_aware,
        "matched_random": matched_random,
        "spread_only_threshold": threshold,
        "always_on": always_on,
        "no_routing": no_routing,
    }
    if include_base:
        base = load_base(config, int(config.get("active_policy_seed", seed)))
        rollouts["no_intervention"] = rollout_episode(
            env,
            base,
            seed=seed,
            episode_steps=episode_steps,
            train=False,
            announced_signal_schedule=[0.0] * episode_steps,
            reward_scale=float(config["reward_scale"]),
            action_namespace=namespace,
        )
    paths_equal = all(
        validate_common_path(risk_aware, rollout) for rollout in rollouts.values()
    )
    dose_exact = exact_signal_dose(risk_strengths, random_strengths)
    no_routing_dose_exact = exact_signal_dose(
        risk_strengths,
        [float(row["gate_signal"]) for row in no_routing["records"]],
    )
    return rollouts, paths_equal, dose_exact, no_routing_dose_exact


def evaluate_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESTwoStageGateModel,
    evaluation_seeds: Sequence[int],
) -> dict[str, Any]:
    records = {method: [] for method in EVALUATED_METHODS}
    path_checks = []
    dose_checks = []
    no_routing_checks = []
    scoped = dict(config)
    scoped["active_policy_seed"] = policy_seed
    for evaluation_seed in evaluation_seeds:
        rollouts, paths, dose, no_routing_dose = rollout_methods(
            env,
            scoped,
            int(evaluation_seed),
            trainers,
            model,
            train=False,
            include_base=True,
        )
        for method, rollout in rollouts.items():
            records[method].extend(rollout["records"])
        path_checks.append(paths)
        dose_checks.append(dose)
        no_routing_checks.append(no_routing_dose)
    return {
        "methods": {
            method: summarize({"records": rows}) for method, rows in records.items()
        },
        "paths_equal": bool(all(path_checks)),
        "dose_exact": bool(all(dose_checks)),
        "no_routing_dose_exact": bool(all(no_routing_checks)),
    }


def gate_off_exact(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
) -> bool:
    base = load_base(config, policy_seed)
    evaluation_seed = int(config["evaluation_seeds"][0])
    schedule = [0.0] * int(config["episode_steps"])
    rollouts = [
        rollout_episode(
            env,
            trainer,
            seed=evaluation_seed,
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_schedule=schedule,
            reward_scale=float(config["reward_scale"]),
            action_namespace=int(config["evaluation_action_namespace"]),
        )
        for trainer in (base, *[trainers[method] for method in TRAINED_METHODS])
    ]
    reference = [row["actions"] for row in rollouts[0]["records"]]
    return all(
        [row["actions"] for row in rollout["records"]] == reference
        for rollout in rollouts[1:]
    )


def convergence(
    history: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    required = int(config["response_required_stable_evaluations"])
    if len(history) < required:
        return {"stable": False, "reason": "insufficient response evaluations"}
    recent = list(history[-required:])
    ranges = {}
    for method in EVALUATED_METHODS:
        values = [
            float(row["evaluation"]["methods"][method]["low_execution_half_spread"])
            for row in recent
        ]
        ranges[method] = max(values) - min(values)
    return {
        "stable": all(
            value <= float(config["response_maximum_fixed_panel_spread_range"])
            for value in ranges.values()
        ),
        "spread_ranges": ranges,
        "recent_evaluations": recent,
    }


def train_seed(
    seed: int, config: dict[str, Any], output_dir_raw: str, fingerprint: str
) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    output_dir = Path(output_dir_raw)
    model = ABIDESTwoStageGateModel.load(REPO / str(config["gate_artifact"]))
    (
        trainers,
        completed,
        training_history,
        evaluation_history,
        paths_equal,
        dose_exact,
        no_routing_dose_exact,
        base_digest,
    ) = load_or_initialize(config, output_dir, seed, fingerprint)
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(market_config(config), docker) as env:
        while completed < int(config["response_episodes"]):
            batch = min(
                int(config["response_episodes_per_update"]),
                int(config["response_episodes"]) - completed,
            )
            trajectories = {method: [] for method in TRAINED_METHODS}
            spreads = {method: [] for method in TRAINED_METHODS}
            rewards = {method: [] for method in TRAINED_METHODS}
            for offset in range(batch):
                episode_seed = response_environment_seed(
                    config, seed, completed + offset
                )
                rollouts, paths, dose, no_routing_dose = rollout_methods(
                    env,
                    config,
                    episode_seed,
                    trainers,
                    model,
                    train=True,
                    include_base=False,
                )
                paths_equal = paths_equal and paths
                dose_exact = dose_exact and dose
                no_routing_dose_exact = no_routing_dose_exact and no_routing_dose
                for method in TRAINED_METHODS:
                    trajectories[method].append(rollouts[method]["trajectory"])
                    spreads[method].append(float(rollouts[method]["mean_half_spread"]))
                    rewards[method].append(float(rollouts[method]["mean_reward"]))
            updates = {
                method: trainers[method].update_episodes(trajectories[method])
                for method in TRAINED_METHODS
            }
            completed += batch
            training_history.append(
                {
                    "completed_episodes": completed,
                    "methods": {
                        method: {
                            "mean_half_spread": mean(spreads[method]),
                            "mean_reward": mean(rewards[method]),
                            **{
                                key: float(value)
                                for key, value in updates[method].items()
                            },
                        }
                        for method in TRAINED_METHODS
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
                            model,
                            config["evaluation_seeds"][:count],
                        ),
                    }
                )
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "episodes": completed,
                            "stable": convergence(evaluation_history, config)["stable"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
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
                dose_exact,
                no_routing_dose_exact,
                base_digest,
            )
        final = evaluate_methods(
            env,
            config,
            seed,
            trainers,
            model,
            config["evaluation_seeds"],
        )
        off_exact = gate_off_exact(env, config, seed, trainers)
    result = {
        "protocol_fingerprint": fingerprint,
        "seed": seed,
        "completed_episodes": completed,
        "elapsed_seconds": time.perf_counter() - started,
        "base_model_digest": base_digest,
        "training_paths_equal": paths_equal,
        "training_dose_exact": dose_exact,
        "training_no_routing_dose_exact": no_routing_dose_exact,
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
    metrics = (
        "low_execution_half_spread",
        "mean_reward",
        "mean_volume",
        "activation_rate",
        "high_risk_activation_rate",
        "targeted_events",
    )
    methods = {
        method: {
            metric: mean(
                float(row["final_evaluation"]["methods"][method][metric])
                for row in rows
            )
            for metric in metrics
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
        for method in (
            "matched_random",
            "no_intervention",
            "spread_only_threshold",
            "always_on",
            "no_routing",
        )
    }
    primary = contrasts["matched_random_minus_risk_aware"]
    routing = contrasts["no_routing_minus_risk_aware"]
    required = math.ceil(
        float(config["minimum_positive_seed_rate"]) * len(rows) - 1e-12
    )
    risk_reward = methods["risk_aware"]["mean_reward"]
    random_reward = methods["matched_random"]["mean_reward"]
    base_reward = methods["no_intervention"]["mean_reward"]
    maximum_degradation = float(config["maximum_reward_degradation_fraction"])
    checks = {
        "required_policy_seed_count": len(rows)
        == int(config["required_policy_seed_count"]),
        "training_paths_equal": all(bool(row["training_paths_equal"]) for row in rows),
        "training_dose_exact": all(bool(row["training_dose_exact"]) for row in rows),
        "training_no_routing_dose_exact": all(
            bool(row["training_no_routing_dose_exact"]) for row in rows
        ),
        "evaluation_paths_equal": all(
            bool(row["final_evaluation"]["paths_equal"]) for row in rows
        ),
        "evaluation_dose_exact": all(
            bool(row["final_evaluation"]["dose_exact"]) for row in rows
        ),
        "evaluation_no_routing_dose_exact": all(
            bool(row["final_evaluation"]["no_routing_dose_exact"]) for row in rows
        ),
        "gate_off_exact": all(bool(row["gate_off_exact"]) for row in rows),
        "response_converged": all(bool(row["convergence"]["stable"]) for row in rows),
        "no_intervention_failure_mode": sum(
            float(
                row["final_evaluation"]["methods"]["no_intervention"][
                    "low_execution_half_spread"
                ]
            )
            > float(config["minimum_failure_spread"])
            for row in rows
        )
        >= required,
        "risk_aware_beats_random_mean": float(primary["mean"]) > 0.0,
        "risk_aware_beats_random_seed_rate": int(primary["positive"]) >= required,
        "risk_aware_positive_ci": float(primary["ci_low"]) > 0.0,
        "routing_channel_positive_mean": float(routing["mean"]) > 0.0,
        "routing_channel_positive_seed_rate": int(routing["positive"]) >= required,
        "routing_channel_positive_ci": float(routing["ci_low"]) > 0.0,
        "risk_activation_in_range": 0.05
        <= methods["risk_aware"]["activation_rate"]
        <= float(config["max_activation_rate"]),
        "risk_high_risk_activation_controlled": methods["risk_aware"][
            "high_risk_activation_rate"
        ]
        <= float(config["maximum_high_risk_activation_rate"]),
        "threshold_has_more_high_risk_activation": methods["spread_only_threshold"][
            "high_risk_activation_rate"
        ]
        - methods["risk_aware"]["high_risk_activation_rate"]
        >= float(config["minimum_threshold_high_risk_activation_gap"]),
        "risk_activates_less_than_always_on": methods["risk_aware"]["activation_rate"]
        < methods["always_on"]["activation_rate"],
        "risk_reward_guardrail_vs_random": risk_reward
        >= random_reward - maximum_degradation * max(1.0, abs(random_reward)),
        "risk_reward_guardrail_vs_no_intervention": risk_reward
        >= base_reward - maximum_degradation * max(1.0, abs(base_reward)),
        "volume_conserved": all(
            abs(methods[method]["mean_volume"] - 20.0) < 1e-9
            for method in EVALUATED_METHODS
        ),
    }
    return {
        "method_summary": methods,
        "paired_contrasts": contrasts,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def validate_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    if tuple(config["methods"]) != EVALUATED_METHODS:
        raise ValueError(f"full-scale methods must be {EVALUATED_METHODS}")
    policy_seeds = {int(seed) for seed in config["policy_seeds"]}
    evaluation_seeds = {int(seed) for seed in config["evaluation_seeds"]}
    if policy_seeds & evaluation_seeds:
        raise ValueError("policy and evaluation seeds overlap")
    gate_path = REPO / str(config["gate_artifact"])
    gate_hash = sha256_file(gate_path)
    if gate_hash != str(config["gate_artifact_sha256"]):
        raise RuntimeError("full-scale gate artifact hash mismatch")
    base_output = REPO / str(config["base_output_dir"])
    base_manifest = json.loads((base_output / "manifest.json").read_text())
    base_summary = json.loads((base_output / "base_summary.json").read_text())
    if (
        base_manifest.get("protocol_fingerprint") != config["base_protocol_fingerprint"]
        or base_summary.get("status") != "pass"
    ):
        raise RuntimeError("full-scale run requires a passing frozen base")
    base_config = base_manifest["config"]
    if policy_seeds != {int(seed) for seed in base_config["base_training_seeds"]}:
        if not bool(config.get("engineering_smoke")) or not policy_seeds.issubset(
            {int(seed) for seed in base_config["base_training_seeds"]}
        ):
            raise RuntimeError("full-scale policy seeds do not match base policies")
    if dict(config.get("market", {})) != dict(base_config.get("market", {})):
        raise RuntimeError("response ecology differs from frozen base ecology")
    base_eval = {int(seed) for seed in base_config["base_fixed_evaluation_seeds"]}
    if evaluation_seeds & base_eval:
        raise RuntimeError("response and base-convergence seeds overlap")
    holdout_output = REPO / str(config["validated_holdout_output_dir"])
    holdout_manifest = json.loads((holdout_output / "manifest.json").read_text())
    holdout_summary = json.loads((holdout_output / "summary.json").read_text())
    if (
        holdout_manifest.get("protocol_fingerprint")
        != config["validated_holdout_protocol_fingerprint"]
        or holdout_summary.get("status") != "pass"
        or not all(holdout_summary.get("checks", {}).values())
    ):
        raise RuntimeError("validated adaptive-v3 holdout prerequisite failed")
    frozen = holdout_manifest["config"]
    frozen_fields = [
        "docker_image",
        "episode_steps",
        "response_episodes_per_update",
        "response_maximum_fixed_panel_spread_range",
        "learning_rate",
        "entropy_coefficient",
        "reward_scale",
        "max_activation_rate",
        "routing_allocation_mode",
        "routing_target_kappa",
        "routing_routable_fraction",
        "gate_artifact",
        "gate_artifact_sha256",
    ]
    if not bool(config.get("engineering_smoke")):
        frozen_fields.extend(
            [
                "response_episodes",
                "response_evaluation_start",
                "response_evaluation_interval",
                "response_required_stable_evaluations",
                "response_convergence_path_count",
            ]
        )
    for field in frozen_fields:
        if config.get(field) != frozen.get(field):
            raise RuntimeError(f"full-scale run changed validated field: {field}")
    return {
        "gate_artifact": gate_hash,
        "docker_image_id": resolve_docker_image_id(str(config["docker_image"])),
        "base_protocol_fingerprint": str(config["base_protocol_fingerprint"]),
        "validated_holdout_protocol_fingerprint": str(
            config["validated_holdout_protocol_fingerprint"]
        ),
    }


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
            raise RuntimeError(f"full-scale result fingerprint mismatch: {path}")
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
    state = {
        "status": status,
        "updated_at": utc_now(),
        "completed_policy_seeds": sorted(complete),
        "pending_policy_seeds": [seed for seed in seeds if seed not in complete],
        "failures": list(failures),
    }
    atomic_write_json(output_dir / "run_state.json", state)
    lines = [
        "# ABIDES/MAPPO Adaptive-v3 Full-Scale Logbook",
        "",
        f"- Status: `{status}`",
        f"- Protocol: `{manifest['protocol_version']}`",
        f"- Fingerprint: `{manifest['protocol_fingerprint']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Completed: `{len(complete)}/{len(seeds)}`",
        "",
        "| Policy seed | Status | Episodes | Runtime (s) |",
        "| ---: | --- | ---: | ---: |",
    ]
    for seed in seeds:
        row = complete.get(seed)
        completed = ""
        if row is None:
            path = checkpoint_path(output_dir, seed)
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                completed = str(int(payload["completed_episodes"]))
        if row:
            lines.append(
                f"| {seed} | complete | {int(row['completed_episodes'])} | "
                f"{float(row['elapsed_seconds']):.2f} |"
            )
        else:
            lines.append(f"| {seed} | pending | {completed} |  |")
    atomic_write_text(output_dir / "LOGBOOK.md", "\n".join(lines) + "\n")


def execute(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    artifacts = validate_inputs(config)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "refusing formal full-scale run from a dirty worktree; use "
            "--allow-dirty only for engineering smoke"
        )
    fingerprint = protocol_fingerprint(config)
    output_dir = REPO / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(output_dir / ".run.lock", fingerprint, args.force_unlock):
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("protocol_fingerprint") != fingerprint:
                raise RuntimeError("existing full-scale manifest has another protocol")
            if manifest.get("artifacts") != artifacts:
                raise RuntimeError("full-scale resume artifact mismatch")
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
            raise RuntimeError(
                "cannot resume pending full-scale seeds from a different git commit"
            )
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
                        print(
                            json.dumps({"seed": seed, "status": "complete"}),
                            flush=True,
                        )
                    persist_state(
                        output_dir,
                        manifest,
                        "failed" if failures else "running",
                        rows,
                        failures,
                    )
        if failures:
            raise RuntimeError(
                f"{len(failures)} full-scale seed(s) failed; rerun to resume"
            )
        rows.sort(key=lambda row: int(row["seed"]))
        result = {"rows": rows, **aggregate(rows, config)}
        status = (
            "smoke_complete"
            if args.engineering_smoke
            and all(
                result["checks"][key]
                for key in (
                    "training_paths_equal",
                    "training_dose_exact",
                    "training_no_routing_dose_exact",
                    "evaluation_paths_equal",
                    "evaluation_dose_exact",
                    "evaluation_no_routing_dose_exact",
                    "gate_off_exact",
                    "volume_conserved",
                )
            )
            else result["status"]
        )
        result["status"] = status
        atomic_write_json(output_dir / "rows.json", rows)
        atomic_write_json(output_dir / "summary.json", result)
        persist_state(output_dir, manifest, status, rows)
        append_logbook(
            output_dir / "logbook.jsonl",
            {"event": "run_finished", "status": status},
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "checks": result["checks"],
                    "primary": result["paired_contrasts"][
                        "matched_random_minus_risk_aware"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if status == "fail":
            raise SystemExit(2)
        return result


def configured_run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    if args.policy_seeds:
        config["policy_seeds"] = [
            int(value) for value in args.policy_seeds.split(",") if value
        ]
    if args.evaluation_seeds:
        config["evaluation_seeds"] = [
            int(value) for value in args.evaluation_seeds.split(",") if value
        ]
    if args.response_episodes is not None:
        config["response_episodes"] = int(args.response_episodes)
    if args.workers is not None:
        config["workers"] = int(args.workers)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.allow_partial or args.engineering_smoke:
        config["engineering_smoke"] = bool(args.engineering_smoke)
        config["required_policy_seed_count"] = len(config["policy_seeds"])
        config["minimum_positive_seed_rate"] = 0.0
        config["response_evaluation_start"] = int(config["response_episodes"])
        config["response_evaluation_interval"] = int(config["response_episodes"])
        config["response_required_stable_evaluations"] = 1
        config["response_convergence_path_count"] = 1
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_mappo_adaptive_v3_full_main_primary_v1.json",
    )
    parser.add_argument("--policy-seeds", default=None)
    parser.add_argument("--evaluation-seeds", default=None)
    parser.add_argument("--response-episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = configured_run(args)
    execute(config, args)


if __name__ == "__main__":
    main()
