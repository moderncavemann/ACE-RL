#!/usr/bin/env python3
"""Learned announced-contract development stage for ABIDES adaptive v3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
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
from run_abides_mappo_adaptive_v3_response import (  # noqa: E402
    block_shuffle_within_risk,
    exact_signal_dose,
    load_base,
    make_adapter,
    market_config,
    response_environment_seed,
)
from run_abides_mappo_full_main import (  # noqa: E402
    atomic_write_json,
    current_git_commit,
    protocol_fingerprint,
    resolve_docker_image_id,
    sha256_file,
)
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.agents.mappo import MAPPOTrainer  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESPacedSignalController,
    ABIDESTwoStageGateModel,
)


METHODS = (
    "learned_full",
    "overall_random",
    "risk_block_shuffled",
    "risk_only",
)
SignalDecision = Callable[[Mapping[str, Any], int], float]


def seed_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "response_seeds" / f"seed_{seed}"


def checkpoint_path(output_dir: Path, seed: int) -> Path:
    return seed_dir(output_dir, seed) / "training_state.pt"


def result_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "response_results" / f"seed_{seed}.json"


def paced_learned_decision(
    model: ABIDESTwoStageGateModel,
    controller: ABIDESPacedSignalController,
    include_severity: bool,
) -> SignalDecision:
    def decide(state: Mapping[str, Any], step: int) -> float:
        proposal = model.announced_strength(
            state,
            include_spread_severity=include_severity,
        )
        return controller.decide(float(proposal["strength"]))

    return decide


def global_shuffle(strengths: Sequence[float], seed: int) -> list[float]:
    values = np.asarray(strengths, dtype=float).copy()
    np.random.default_rng(seed).shuffle(values)
    return values.tolist()


def response_checkpoint_payload(
    fingerprint: str,
    seed: int,
    completed: int,
    trainers: Mapping[str, MAPPOTrainer],
    training_history: Sequence[Mapping[str, Any]],
    evaluation_history: Sequence[Mapping[str, Any]],
    paths_equal: bool,
    dose_exact: bool,
    base_digest: str,
) -> dict[str, Any]:
    return {
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
        "base_model_digest": base_digest,
    }


def load_or_initialize(
    config: Mapping[str, Any], output_dir: Path, seed: int, fingerprint: str
):
    path = checkpoint_path(output_dir, seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"learned response checkpoint mismatch: {path}")
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
            str(payload["base_model_digest"]),
        )
    base = load_base(config, seed)
    trainers = {
        method: make_adapter(base, config, seed) for method in METHODS
    }
    if len({model_digest(trainer) for trainer in trainers.values()}) != 1:
        raise RuntimeError("learned response adapters did not start identically")
    return trainers, 0, [], [], True, True, model_digest(base)


def generated_schedules(
    rollout: Mapping[str, Any], seed: int
) -> tuple[list[float], list[float], list[float], list[str]]:
    full = [float(row["gate_signal"]) for row in rollout["records"]]
    risks = [
        str(row["pre_info"].get("diagnostic_current_risk"))
        for row in rollout["records"]
    ]
    overall = global_shuffle(full, seed ^ 0xA317)
    block = block_shuffle_within_risk(full, risks, seed ^ 0xB317)
    return full, overall, block, risks


def rollout_online_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESTwoStageGateModel,
    train: bool,
) -> tuple[dict[str, Any], bool, bool]:
    full_controller = ABIDESPacedSignalController(
        int(config["episode_steps"]),
        target_rate=float(config["max_activation_rate"]),
    )
    full = rollout_episode(
        env,
        trainers["learned_full"],
        seed=seed,
        episode_steps=int(config["episode_steps"]),
        train=train,
        defer_update=train,
        announced_signal_decision=paced_learned_decision(
            model, full_controller, include_severity=True
        ),
        reward_scale=float(config["reward_scale"]),
        action_namespace=(
            int(config["response_action_namespace"])
            if train
            else int(config["development_evaluation_action_namespace"])
        ),
    )
    full_strengths, overall_schedule, block_schedule, _ = generated_schedules(
        full, seed
    )
    overall = rollout_episode(
        env,
        trainers["overall_random"],
        seed=seed,
        episode_steps=int(config["episode_steps"]),
        train=train,
        defer_update=train,
        announced_signal_schedule=overall_schedule,
        reward_scale=float(config["reward_scale"]),
        action_namespace=(
            int(config["response_action_namespace"])
            if train
            else int(config["development_evaluation_action_namespace"])
        ),
    )
    risk_block = rollout_episode(
        env,
        trainers["risk_block_shuffled"],
        seed=seed,
        episode_steps=int(config["episode_steps"]),
        train=train,
        defer_update=train,
        announced_signal_schedule=block_schedule,
        reward_scale=float(config["reward_scale"]),
        action_namespace=(
            int(config["response_action_namespace"])
            if train
            else int(config["development_evaluation_action_namespace"])
        ),
    )
    risk_controller = ABIDESPacedSignalController(
        int(config["episode_steps"]),
        target_rate=float(config["max_activation_rate"]),
    )
    risk_only = rollout_episode(
        env,
        trainers["risk_only"],
        seed=seed,
        episode_steps=int(config["episode_steps"]),
        train=train,
        defer_update=train,
        announced_signal_decision=paced_learned_decision(
            model, risk_controller, include_severity=False
        ),
        reward_scale=float(config["reward_scale"]),
        action_namespace=(
            int(config["response_action_namespace"])
            if train
            else int(config["development_evaluation_action_namespace"])
        ),
    )
    rollouts = {
        "learned_full": full,
        "overall_random": overall,
        "risk_block_shuffled": risk_block,
        "risk_only": risk_only,
    }
    paths_equal = all(
        validate_common_path(full, rollout) for rollout in rollouts.values()
    )
    dose_exact = exact_signal_dose(full_strengths, overall_schedule) and exact_signal_dose(
        full_strengths, block_schedule
    )
    return rollouts, paths_equal, dose_exact


def evaluate_online(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESTwoStageGateModel,
    evaluation_seeds: Sequence[int],
) -> dict[str, Any]:
    records = {method: [] for method in METHODS}
    paths_equal = []
    dose_exact = []
    for evaluation_seed in evaluation_seeds:
        rollouts, paths, dose = rollout_online_methods(
            env,
            config,
            int(evaluation_seed),
            trainers,
            model,
            train=False,
        )
        for method, rollout in rollouts.items():
            records[method].extend(rollout["records"])
        paths_equal.append(paths)
        dose_exact.append(dose)
    return {
        "methods": {
            method: summarize({"records": rows}) for method, rows in records.items()
        },
        "paths_equal": bool(all(paths_equal)),
        "dose_exact": bool(all(dose_exact)),
    }


def gate_off_exact(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
) -> bool:
    base = load_base(config, policy_seed)
    evaluation_seed = int(config["development_evaluation_seeds"][0])
    namespace = int(config["development_evaluation_action_namespace"])
    base_rollout = rollout_episode(
        env,
        base,
        seed=evaluation_seed,
        episode_steps=int(config["episode_steps"]),
        train=False,
        announced_signal_schedule=[0.0] * int(config["episode_steps"]),
        reward_scale=float(config["reward_scale"]),
        action_namespace=namespace,
    )
    for trainer in trainers.values():
        adapter_rollout = rollout_episode(
            env,
            trainer,
            seed=evaluation_seed,
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_schedule=[0.0] * int(config["episode_steps"]),
            reward_scale=float(config["reward_scale"]),
            action_namespace=namespace,
        )
        if any(
            left["actions"] != right["actions"]
            for left, right in zip(base_rollout["records"], adapter_rollout["records"])
        ):
            return False
    return True


def convergence(
    evaluation_history: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    required = int(config["response_required_stable_evaluations"])
    if len(evaluation_history) < required:
        return {"stable": False, "reason": "insufficient response evaluations"}
    recent = list(evaluation_history[-required:])
    ranges = {}
    for method in METHODS:
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
        base_digest,
    ) = load_or_initialize(config, output_dir, seed, fingerprint)
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(market_config(config), docker) as env:
        while completed < int(config["response_episodes"]):
            batch = min(
                int(config["response_episodes_per_update"]),
                int(config["response_episodes"]) - completed,
            )
            trajectories = {method: [] for method in METHODS}
            spreads = {method: [] for method in METHODS}
            rewards = {method: [] for method in METHODS}
            for offset in range(batch):
                episode_seed = response_environment_seed(config, seed, completed + offset)
                rollouts, paths, dose = rollout_online_methods(
                    env, config, episode_seed, trainers, model, train=True
                )
                paths_equal = paths_equal and paths
                dose_exact = dose_exact and dose
                for method, rollout in rollouts.items():
                    trajectories[method].append(rollout["trajectory"])
                    spreads[method].append(float(rollout["mean_half_spread"]))
                    rewards[method].append(float(rollout["mean_reward"]))
            updates = {
                method: trainers[method].update_episodes(trajectories[method])
                for method in METHODS
            }
            completed += batch
            training_history.append(
                {
                    "completed_episodes": completed,
                    "methods": {
                        method: {
                            "mean_half_spread": mean(spreads[method]),
                            "mean_reward": mean(rewards[method]),
                            **{key: float(value) for key, value in updates[method].items()},
                        }
                        for method in METHODS
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
                        "evaluation": evaluate_online(
                            env,
                            config,
                            seed,
                            trainers,
                            model,
                            config["development_evaluation_seeds"][:count],
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
            atomic_torch_save(
                checkpoint_path(output_dir, seed),
                response_checkpoint_payload(
                    fingerprint,
                    seed,
                    completed,
                    trainers,
                    training_history,
                    evaluation_history,
                    paths_equal,
                    dose_exact,
                    base_digest,
                ),
            )
        final = evaluate_online(
            env,
            config,
            seed,
            trainers,
            model,
            config["development_evaluation_seeds"],
        )
        off_exact = gate_off_exact(env, config, seed, trainers)
    result = {
        "seed": seed,
        "completed_episodes": completed,
        "base_model_digest": base_digest,
        "training_paths_equal": paths_equal,
        "training_dose_exact": dose_exact,
        "convergence": convergence(evaluation_history, config),
        "gate_off_exact": off_exact,
        "final_evaluation": final,
        "final_model_digests": {
            method: model_digest(trainer) for method, trainer in trainers.items()
        },
    }
    atomic_write_json(result_path(output_dir, seed), result)
    return result


def contrast(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, float | int]:
    return paired_summary([
        float(row["final_evaluation"]["methods"][left]["low_execution_half_spread"])
        - float(row["final_evaluation"]["methods"][right]["low_execution_half_spread"])
        for row in rows
    ])


def aggregate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    methods = {
        method: {
            metric: mean(
                float(row["final_evaluation"]["methods"][method][metric])
                for row in rows
            )
            for metric in (
                "low_execution_half_spread",
                "mean_reward",
                "mean_volume",
                "activation_rate",
                "high_risk_activation_rate",
                "targeted_events",
            )
        }
        for method in METHODS
    }
    overall = contrast(rows, "overall_random", "learned_full")
    risk_block = contrast(rows, "risk_block_shuffled", "learned_full")
    risk_only_minus_full = contrast(rows, "risk_only", "learned_full")
    required = math.ceil(
        float(config["minimum_positive_seed_rate"]) * len(rows) - 1e-12
    )
    activation_reduction = (
        methods["risk_only"]["activation_rate"]
        - methods["learned_full"]["activation_rate"]
    )
    spread_degradation = (
        methods["learned_full"]["low_execution_half_spread"]
        - methods["risk_only"]["low_execution_half_spread"]
    )
    severity_supported = (
        activation_reduction
        >= float(config["minimum_severity_activation_reduction"])
        and spread_degradation
        <= float(config["maximum_severity_spread_degradation"])
    ) or (
        spread_degradation < 0.0
        and methods["learned_full"]["activation_rate"]
        <= methods["risk_only"]["activation_rate"]
    )
    full_reward = methods["learned_full"]["mean_reward"]
    random_reward = methods["overall_random"]["mean_reward"]
    checks = {
        "training_paths_equal": all(bool(row["training_paths_equal"]) for row in rows),
        "training_dose_exact": all(bool(row["training_dose_exact"]) for row in rows),
        "evaluation_paths_equal": all(
            bool(row["final_evaluation"]["paths_equal"]) for row in rows
        ),
        "evaluation_dose_exact": all(
            bool(row["final_evaluation"]["dose_exact"]) for row in rows
        ),
        "gate_off_exact": all(bool(row["gate_off_exact"]) for row in rows),
        "response_converged": all(bool(row["convergence"]["stable"]) for row in rows),
        "learned_beats_overall_mean": float(overall["mean"]) > 0.0,
        "learned_beats_overall_seed_rate": int(overall["positive"]) >= required,
        "learned_activation_in_range": 0.05
        <= methods["learned_full"]["activation_rate"]
        <= float(config["max_activation_rate"]),
        "learned_high_risk_controlled": methods["learned_full"][
            "high_risk_activation_rate"
        ]
        <= float(config["maximum_high_risk_activation_rate"]),
        "volume_conserved": all(
            abs(methods[method]["mean_volume"] - 20.0) < 1e-9
            for method in METHODS
        ),
        "reward_guardrail": full_reward
        >= random_reward
        - float(config["maximum_reward_degradation_fraction"])
        * max(1.0, abs(random_reward)),
    }
    return {
        "method_summary": methods,
        "paired_overall_minus_learned": overall,
        "paired_risk_block_minus_learned": risk_block,
        "paired_risk_only_minus_full": risk_only_minus_full,
        "severity_module": {
            "activation_reduction": activation_reduction,
            "spread_degradation": spread_degradation,
            "supported": severity_supported,
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def validate_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    gate_path = REPO / str(config["gate_artifact"])
    gate_hash = sha256_file(gate_path)
    if gate_hash != str(config["gate_artifact_sha256"]):
        raise RuntimeError("learned response gate artifact hash mismatch")
    base_output = REPO / str(config["base_output_dir"])
    base_manifest = json.loads(
        (base_output / "manifest.json").read_text(encoding="utf-8")
    )
    base_summary = json.loads(
        (base_output / "base_summary.json").read_text(encoding="utf-8")
    )
    if base_manifest.get("protocol_fingerprint") != config.get(
        "base_protocol_fingerprint"
    ) or base_summary.get("status") != "pass":
        raise RuntimeError("learned response requires the frozen passing base")
    oracle_summary = json.loads(
        (REPO / str(config["oracle_output_dir"]) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if oracle_summary.get("status") != "pass":
        raise RuntimeError("learned response requires a passing oracle gate")
    return {
        "gate_artifact": gate_hash,
        "docker_image_id": resolve_docker_image_id(str(config["docker_image"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_mappo_adaptive_v3_learned_dev_v1.json",
    )
    parser.add_argument("--smoke-episodes", type=int, default=0)
    args = parser.parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    if args.smoke_episodes:
        config = copy.deepcopy(config)
        config["protocol_version"] += f"_smoke_{args.smoke_episodes}"
        config["policy_seeds"] = config["policy_seeds"][:1]
        config["development_evaluation_seeds"] = config[
            "development_evaluation_seeds"
        ][:2]
        config["response_episodes"] = int(args.smoke_episodes)
        config["response_evaluation_start"] = int(args.smoke_episodes)
        config["response_evaluation_interval"] = int(args.smoke_episodes)
        config["response_required_stable_evaluations"] = 1
        config["response_convergence_path_count"] = 1
        config["output_dir"] = (
            "revision/experiments/outputs/"
            f"abides_mappo_adaptive_v3_learned_smoke_{args.smoke_episodes}"
        )
    artifacts = validate_inputs(config)
    fingerprint = protocol_fingerprint(config)
    output_dir = REPO / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "protocol_version": config["protocol_version"],
            "protocol_fingerprint": fingerprint,
            "git_commit": current_git_commit(),
            "config": config,
            "artifacts": artifacts,
        },
    )
    seeds = [int(seed) for seed in config["policy_seeds"]]
    rows = []
    with ProcessPoolExecutor(max_workers=min(int(config["workers"]), len(seeds))) as pool:
        futures = {
            pool.submit(train_seed, seed, config, str(output_dir), fingerprint): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({"seed": row["seed"], "status": "complete"}), flush=True)
    rows.sort(key=lambda row: int(row["seed"]))
    result = {"rows": rows, **aggregate(rows, config)}
    atomic_write_json(output_dir / "summary.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": result["checks"],
                "primary": result["paired_overall_minus_learned"],
                "severity_module": result["severity_module"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
