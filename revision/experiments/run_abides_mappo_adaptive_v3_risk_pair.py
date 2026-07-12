#!/usr/bin/env python3
"""Targeted risk-aware versus exact-dose random response experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

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


METHODS = ("risk_aware", "matched_random")


def seed_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "response_seeds" / f"seed_{seed}"


def checkpoint_path(output_dir: Path, seed: int) -> Path:
    return seed_dir(output_dir, seed) / "training_state.pt"


def result_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "response_results" / f"seed_{seed}.json"


def load_or_initialize(
    config: Mapping[str, Any], output_dir: Path, seed: int, fingerprint: str
):
    path = checkpoint_path(output_dir, seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"risk-pair checkpoint mismatch: {path}")
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
        raise RuntimeError("risk-pair adapters did not start identically")
    return trainers, 0, [], [], True, True, model_digest(base)


def rollout_pair(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESTwoStageGateModel,
    train: bool,
) -> tuple[dict[str, Any], bool, bool]:
    controller = ABIDESPacedSignalController(
        int(config["episode_steps"]),
        target_rate=float(config["max_activation_rate"]),
    )
    namespace = (
        int(config["response_action_namespace"])
        if train
        else int(config["development_evaluation_action_namespace"])
    )
    risk_aware = rollout_episode(
        env,
        trainers["risk_aware"],
        seed=seed,
        episode_steps=int(config["episode_steps"]),
        train=train,
        defer_update=train,
        announced_signal_decision=paced_learned_decision(
            model, controller, include_severity=False
        ),
        reward_scale=float(config["reward_scale"]),
        action_namespace=namespace,
    )
    strengths = [float(row["gate_signal"]) for row in risk_aware["records"]]
    random_schedule = global_shuffle(strengths, seed ^ 0xD317)
    matched = rollout_episode(
        env,
        trainers["matched_random"],
        seed=seed,
        episode_steps=int(config["episode_steps"]),
        train=train,
        defer_update=train,
        announced_signal_schedule=random_schedule,
        reward_scale=float(config["reward_scale"]),
        action_namespace=namespace,
    )
    return (
        {"risk_aware": risk_aware, "matched_random": matched},
        validate_common_path(risk_aware, matched),
        exact_signal_dose(strengths, random_schedule),
    )


def evaluate_pair(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESTwoStageGateModel,
    evaluation_seeds: Sequence[int],
) -> dict[str, Any]:
    records = {method: [] for method in METHODS}
    path_checks = []
    dose_checks = []
    for evaluation_seed in evaluation_seeds:
        rollouts, paths, dose = rollout_pair(
            env, config, int(evaluation_seed), trainers, model, train=False
        )
        for method, rollout in rollouts.items():
            records[method].extend(rollout["records"])
        path_checks.append(paths)
        dose_checks.append(dose)
    return {
        "methods": {
            method: summarize({"records": rows}) for method, rows in records.items()
        },
        "paths_equal": bool(all(path_checks)),
        "dose_exact": bool(all(dose_checks)),
    }


def gate_off_exact(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    seed: int,
    trainers: Mapping[str, MAPPOTrainer],
) -> bool:
    base = load_base(config, seed)
    evaluation_seed = int(config["development_evaluation_seeds"][0])
    namespace = int(config["development_evaluation_action_namespace"])
    rollouts = [
        rollout_episode(
            env,
            trainer,
            seed=evaluation_seed,
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_schedule=[0.0] * int(config["episode_steps"]),
            reward_scale=float(config["reward_scale"]),
            action_namespace=namespace,
        )
        for trainer in (base, trainers["risk_aware"], trainers["matched_random"])
    ]
    return all(
        left["actions"] == middle["actions"] == right["actions"]
        for left, middle, right in zip(
            rollouts[0]["records"],
            rollouts[1]["records"],
            rollouts[2]["records"],
        )
    )


def convergence(
    history: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    required = int(config["response_required_stable_evaluations"])
    if len(history) < required:
        return {"stable": False, "reason": "insufficient response evaluations"}
    recent = list(history[-required:])
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
    base_digest: str,
) -> None:
    atomic_torch_save(
        checkpoint_path(output_dir, seed),
        {
            "protocol_fingerprint": fingerprint,
            "seed": seed,
            "completed_episodes": completed,
            "trainers": {
                method: trainer_payload(trainer)
                for method, trainer in trainers.items()
            },
            "training_history": list(training_history),
            "evaluation_history": list(evaluation_history),
            "training_paths_equal": paths_equal,
            "training_dose_exact": dose_exact,
            "base_model_digest": base_digest,
        },
    )


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
                rollouts, paths, dose = rollout_pair(
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
                        "evaluation": evaluate_pair(
                            env,
                            config,
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
                base_digest,
            )
        final = evaluate_pair(
            env,
            config,
            trainers,
            model,
            config["development_evaluation_seeds"],
        )
        off_exact = gate_off_exact(env, config, seed, trainers)
    result = {
        "seed": seed,
        "completed_episodes": completed,
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


def aggregate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    deltas = [
        float(row["final_evaluation"]["methods"]["matched_random"]["low_execution_half_spread"])
        - float(row["final_evaluation"]["methods"]["risk_aware"]["low_execution_half_spread"])
        for row in rows
    ]
    paired = paired_summary(deltas)
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
    required = math.ceil(
        float(config["minimum_positive_seed_rate"]) * len(rows) - 1e-12
    )
    risk_reward = methods["risk_aware"]["mean_reward"]
    random_reward = methods["matched_random"]["mean_reward"]
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
        "risk_aware_beats_random_mean": float(paired["mean"]) > 0.0,
        "risk_aware_beats_random_seed_rate": int(paired["positive"]) >= required,
        "risk_aware_positive_ci": (
            not bool(config.get("require_positive_ci_lower_bound", False))
            or float(paired["ci_low"]) > 0.0
        ),
        "activation_in_range": 0.05
        <= methods["risk_aware"]["activation_rate"]
        <= float(config["max_activation_rate"]),
        "high_risk_activation_controlled": methods["risk_aware"][
            "high_risk_activation_rate"
        ]
        <= float(config["maximum_high_risk_activation_rate"]),
        "volume_conserved": all(
            abs(methods[method]["mean_volume"] - 20.0) < 1e-9
            for method in METHODS
        ),
        "reward_guardrail": risk_reward
        >= random_reward
        - float(config["maximum_reward_degradation_fraction"])
        * max(1.0, abs(random_reward)),
    }
    return {
        "method_summary": methods,
        "paired_random_minus_risk_aware": paired,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def validate_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    gate_path = REPO / str(config["gate_artifact"])
    gate_hash = sha256_file(gate_path)
    if gate_hash != str(config["gate_artifact_sha256"]):
        raise RuntimeError("risk-pair gate artifact hash mismatch")
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
        raise RuntimeError("risk-pair requires the frozen passing base")
    learned_summary = json.loads(
        (
            REPO
            / str(config["learned_development_output_dir"])
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    if learned_summary.get("status") != "pass" or learned_summary.get(
        "severity_module", {}
    ).get("supported") is not False:
        raise RuntimeError("risk-pair requires passed development and failed severity")
    return {
        "gate_artifact": gate_hash,
        "docker_image_id": resolve_docker_image_id(str(config["docker_image"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_mappo_adaptive_v3_risk_pair_dev_v1.json",
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
            f"abides_mappo_adaptive_v3_risk_pair_smoke_{args.smoke_episodes}"
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
                "paired": result["paired_random_minus_risk_aware"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
