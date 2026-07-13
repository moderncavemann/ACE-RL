#!/usr/bin/env python3
"""Checkpointed announced-contract response training for ABIDES adaptive v3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
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
from run_abides_mappo_full_main import (  # noqa: E402
    atomic_write_json,
    current_git_commit,
    protocol_fingerprint,
    resolve_docker_image_id,
    sha256_file,
)
from src.agents.abides_mappo import make_abides_trainer, rollout_episode  # noqa: E402
from src.agents.mappo import MAPPOTrainer  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESPacedSignalController,
    ABIDESTwoStageGateModel,
)


SignalDecision = Callable[[Mapping[str, Any], int], float]


def response_seed_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "response_seeds" / f"seed_{seed}"


def response_checkpoint(output_dir: Path, seed: int) -> Path:
    return response_seed_dir(output_dir, seed) / "training_state.pt"


def response_result(output_dir: Path, seed: int) -> Path:
    return output_dir / "response_results" / f"seed_{seed}.json"


def base_checkpoint(config: Mapping[str, Any], seed: int) -> Path:
    return (
        REPO
        / str(config["base_output_dir"])
        / "base_seeds"
        / f"seed_{seed}"
        / "training_state.pt"
    )


def response_environment_seed(
    config: Mapping[str, Any], seed: int, episode: int
) -> int:
    return (
        int(config["response_environment_seed_namespace"])
        + (int(seed) % 100_000) * 1_000
        + int(episode)
    )


def paced_oracle_decision(
    model: ABIDESTwoStageGateModel,
    controller: ABIDESPacedSignalController,
) -> SignalDecision:
    def decide(state: Mapping[str, Any], step: int) -> float:
        diagnostic = model.announced_strength(state)
        current_risk = state["info"].get("diagnostic_current_risk")
        proposed = float(diagnostic["severity"]) if current_risk == "low" else 0.0
        return controller.decide(proposed)

    return decide


def block_shuffle_within_risk(
    strengths: Sequence[float],
    risk_path: Sequence[str],
    seed: int,
    block_size: int = 10,
) -> list[float]:
    result = np.asarray(strengths, dtype=float).copy()
    rng = np.random.default_rng(seed)
    for block_start in range(0, len(result), block_size):
        block_end = min(block_start + block_size, len(result))
        for risk in ("low", "high"):
            indices = np.asarray(
                [
                    index
                    for index in range(block_start, block_end)
                    if risk_path[index] == risk
                ],
                dtype=int,
            )
            if len(indices) > 1:
                values = result[indices].copy()
                rng.shuffle(values)
                result[indices] = values
    return result.tolist()


def exact_signal_dose(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(
        np.allclose(
            np.sort(np.asarray(left, dtype=float)),
            np.sort(np.asarray(right, dtype=float)),
            atol=1e-12,
            rtol=0.0,
        )
    )


def load_base(config: Mapping[str, Any], seed: int) -> MAPPOTrainer:
    path = base_checkpoint(config, seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol_fingerprint") != str(config["base_protocol_fingerprint"]):
        raise RuntimeError(f"base checkpoint fingerprint mismatch: {path}")
    return trainer_from_payload(payload["trainer"], str(config["device"]))


def make_adapter(
    base: MAPPOTrainer,
    config: Mapping[str, Any],
    seed: int,
) -> MAPPOTrainer:
    adapter_config = replace(
        base.config,
        policy_mode="intervention_adapter",
        policy_advantage_mode="gae",
        learning_rate=float(config["learning_rate"]),
        entropy_coef=float(config["entropy_coefficient"]),
        adapter_risk_feature_index=None,
        seed=seed,
        device=str(config["device"]),
    )
    trainer = MAPPOTrainer(adapter_config)
    trainer.initialize_intervention_adapter(base)
    return trainer


def load_or_initialize(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    fingerprint: str,
):
    path = response_checkpoint(output_dir, seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"response checkpoint fingerprint mismatch: {path}")
        trainers = {
            method: trainer_from_payload(data, str(config["device"]))
            for method, data in payload["trainers"].items()
        }
        return (
            trainers,
            int(payload["completed_episodes"]),
            list(payload["training_history"]),
            list(payload["evaluation_history"]),
            bool(payload["training_paths_equal"]),
            bool(payload["training_dose_exact"]),
            str(payload["base_model_digest"]),
        )
    base = load_base(config, seed)
    base_digest = model_digest(base)
    trainers = {
        "oracle_timed": make_adapter(base, config, seed),
        "block_shuffled": make_adapter(base, config, seed),
    }
    initial = {model_digest(trainer) for trainer in trainers.values()}
    if len(initial) != 1:
        raise RuntimeError("response adapters did not start identically")
    return trainers, 0, [], [], True, True, base_digest


def save_response_checkpoint(
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
        response_checkpoint(output_dir, seed),
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
            "base_model_digest": base_digest,
        },
    )


def market_config(config: Mapping[str, Any]) -> dict[str, Any]:
    market = dict(config.get("market", {}))
    market.update(
        {
            "episode_steps": int(config["episode_steps"]),
            "routing_allocation_mode": str(config["routing_allocation_mode"]),
            "routing_target_kappa": float(config["routing_target_kappa"]),
            "routing_routable_fraction": float(config["routing_routable_fraction"]),
        }
    )
    return market


def evaluate_pair(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESTwoStageGateModel,
    evaluation_seeds: Sequence[int],
) -> dict[str, Any]:
    records = {method: [] for method in ("oracle_timed", "block_shuffled")}
    base = load_base(config, seed)
    gate_off_records = {method: [] for method in ("base", "oracle", "shuffled")}
    dose_exact = []
    paths_equal = []
    for evaluation_seed in evaluation_seeds:
        controller = ABIDESPacedSignalController(
            int(config["episode_steps"]),
            target_rate=float(config["max_activation_rate"]),
        )
        oracle = rollout_episode(
            env,
            trainers["oracle_timed"],
            seed=int(evaluation_seed),
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_decision=paced_oracle_decision(model, controller),
            reward_scale=float(config["reward_scale"]),
            action_namespace=int(config["development_evaluation_action_namespace"]),
        )
        strengths = [float(row["gate_signal"]) for row in oracle["records"]]
        risks = [
            str(row["pre_info"].get("diagnostic_current_risk"))
            for row in oracle["records"]
        ]
        shuffled_strengths = block_shuffle_within_risk(
            strengths,
            risks,
            seed=int(evaluation_seed) ^ seed ^ 0xB117,
        )
        shuffled = rollout_episode(
            env,
            trainers["block_shuffled"],
            seed=int(evaluation_seed),
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_schedule=shuffled_strengths,
            reward_scale=float(config["reward_scale"]),
            action_namespace=int(config["development_evaluation_action_namespace"]),
        )
        base_off = rollout_episode(
            env,
            base,
            seed=int(evaluation_seed),
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_schedule=[0.0] * int(config["episode_steps"]),
            reward_scale=float(config["reward_scale"]),
            action_namespace=int(config["development_evaluation_action_namespace"]),
        )
        oracle_off = rollout_episode(
            env,
            trainers["oracle_timed"],
            seed=int(evaluation_seed),
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_schedule=[0.0] * int(config["episode_steps"]),
            reward_scale=float(config["reward_scale"]),
            action_namespace=int(config["development_evaluation_action_namespace"]),
        )
        shuffled_off = rollout_episode(
            env,
            trainers["block_shuffled"],
            seed=int(evaluation_seed),
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_schedule=[0.0] * int(config["episode_steps"]),
            reward_scale=float(config["reward_scale"]),
            action_namespace=int(config["development_evaluation_action_namespace"]),
        )
        for method, rollout in (
            ("oracle_timed", oracle),
            ("block_shuffled", shuffled),
        ):
            records[method].extend(rollout["records"])
        for method, rollout in (
            ("base", base_off),
            ("oracle", oracle_off),
            ("shuffled", shuffled_off),
        ):
            gate_off_records[method].extend(rollout["records"])
        dose_exact.append(exact_signal_dose(strengths, shuffled_strengths))
        paths_equal.append(
            validate_common_path(oracle, shuffled)
            and validate_common_path(oracle, base_off)
        )
    methods = {method: summarize({"records": rows}) for method, rows in records.items()}
    gate_off = {
        method: summarize({"records": rows})
        for method, rows in gate_off_records.items()
    }
    gate_off_exact = all(
        base_row["actions"] == oracle_row["actions"] == shuffled_row["actions"]
        for base_row, oracle_row, shuffled_row in zip(
            gate_off_records["base"],
            gate_off_records["oracle"],
            gate_off_records["shuffled"],
        )
    )
    return {
        "methods": methods,
        "gate_off": gate_off,
        "gate_off_exact": gate_off_exact,
        "dose_exact": bool(all(dose_exact)),
        "paths_equal": bool(all(paths_equal)),
    }


def response_convergence(
    evaluation_history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    required = int(config["response_required_stable_evaluations"])
    if len(evaluation_history) < required:
        return {"stable": False, "reason": "insufficient response evaluations"}
    recent = list(evaluation_history[-required:])
    ranges = {}
    for method in ("oracle_timed", "block_shuffled"):
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
    seed: int,
    config: dict[str, Any],
    output_dir_raw: str,
    fingerprint: str,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    output_dir = Path(output_dir_raw)
    model = ABIDESTwoStageGateModel.load(REPO / str(config["gate_artifact"]))
    (
        trainers,
        completed,
        training_history,
        evaluation_history,
        training_paths_equal,
        training_dose_exact,
        base_digest,
    ) = load_or_initialize(config, output_dir, seed, fingerprint)
    max_episodes = int(config["response_episodes"])
    batch_size = int(config["response_episodes_per_update"])
    env_config = market_config(config)
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(env_config, docker) as env:
        while completed < max_episodes:
            trajectories = {method: [] for method in trainers}
            batch_spreads = {method: [] for method in trainers}
            batch_rewards = {method: [] for method in trainers}
            current_batch = min(batch_size, max_episodes - completed)
            for offset in range(current_batch):
                episode = completed + offset
                episode_seed = response_environment_seed(config, seed, episode)
                controller = ABIDESPacedSignalController(
                    int(config["episode_steps"]),
                    target_rate=float(config["max_activation_rate"]),
                )
                oracle = rollout_episode(
                    env,
                    trainers["oracle_timed"],
                    seed=episode_seed,
                    episode_steps=int(config["episode_steps"]),
                    train=True,
                    defer_update=True,
                    announced_signal_decision=paced_oracle_decision(model, controller),
                    reward_scale=float(config["reward_scale"]),
                    action_namespace=int(config["response_action_namespace"]),
                )
                strengths = [float(row["gate_signal"]) for row in oracle["records"]]
                risks = [
                    str(row["pre_info"].get("diagnostic_current_risk"))
                    for row in oracle["records"]
                ]
                shuffled_strengths = block_shuffle_within_risk(
                    strengths,
                    risks,
                    seed=episode_seed ^ seed ^ 0xB117,
                )
                shuffled = rollout_episode(
                    env,
                    trainers["block_shuffled"],
                    seed=episode_seed,
                    episode_steps=int(config["episode_steps"]),
                    train=True,
                    defer_update=True,
                    announced_signal_schedule=shuffled_strengths,
                    reward_scale=float(config["reward_scale"]),
                    action_namespace=int(config["response_action_namespace"]),
                )
                training_paths_equal = training_paths_equal and validate_common_path(
                    oracle, shuffled
                )
                training_dose_exact = training_dose_exact and exact_signal_dose(
                    strengths, shuffled_strengths
                )
                for method, rollout in (
                    ("oracle_timed", oracle),
                    ("block_shuffled", shuffled),
                ):
                    trajectories[method].append(rollout["trajectory"])
                    batch_spreads[method].append(float(rollout["mean_half_spread"]))
                    batch_rewards[method].append(float(rollout["mean_reward"]))
            updates = {
                method: trainers[method].update_episodes(trajectories[method])
                for method in trainers
            }
            completed += current_batch
            training_history.append(
                {
                    "completed_episodes": completed,
                    "methods": {
                        method: {
                            "mean_half_spread": mean(batch_spreads[method]),
                            "mean_reward": mean(batch_rewards[method]),
                            **{
                                key: float(value)
                                for key, value in updates[method].items()
                            },
                        }
                        for method in trainers
                    },
                }
            )
            if (
                completed >= int(config["response_evaluation_start"])
                and completed % int(config["response_evaluation_interval"]) == 0
            ):
                panel_seeds = config["development_evaluation_seeds"][:5]
                evaluation_history.append(
                    {
                        "completed_episodes": completed,
                        "evaluation": evaluate_pair(
                            env,
                            config,
                            seed,
                            trainers,
                            model,
                            panel_seeds,
                        ),
                    }
                )
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "episodes": completed,
                            "stable": response_convergence(evaluation_history, config)[
                                "stable"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            save_response_checkpoint(
                output_dir,
                seed,
                fingerprint,
                completed,
                trainers,
                training_history,
                evaluation_history,
                training_paths_equal,
                training_dose_exact,
                base_digest,
            )

        final_evaluation = evaluate_pair(
            env,
            config,
            seed,
            trainers,
            model,
            config["development_evaluation_seeds"],
        )
    result = {
        "seed": seed,
        "completed_episodes": completed,
        "base_model_digest": base_digest,
        "final_model_digests": {
            method: model_digest(trainer) for method, trainer in trainers.items()
        },
        "training_paths_equal": training_paths_equal,
        "training_dose_exact": training_dose_exact,
        "convergence": response_convergence(evaluation_history, config),
        "evaluation_history": evaluation_history,
        "final_evaluation": final_evaluation,
    }
    atomic_write_json(response_result(output_dir, seed), result)
    return result


def aggregate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    deltas = [
        float(
            row["final_evaluation"]["methods"]["block_shuffled"][
                "low_execution_half_spread"
            ]
        )
        - float(
            row["final_evaluation"]["methods"]["oracle_timed"][
                "low_execution_half_spread"
            ]
        )
        for row in rows
    ]
    paired = paired_summary(deltas)
    method_summary = {
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
        for method in ("oracle_timed", "block_shuffled")
    }
    required_positive = math.ceil(
        float(config["minimum_positive_seed_rate"]) * len(rows) - 1e-12
    )
    oracle_reward = method_summary["oracle_timed"]["mean_reward"]
    shuffled_reward = method_summary["block_shuffled"]["mean_reward"]
    checks = {
        "training_paths_equal": all(bool(row["training_paths_equal"]) for row in rows),
        "training_dose_exact": all(bool(row["training_dose_exact"]) for row in rows),
        "evaluation_paths_equal": all(
            bool(row["final_evaluation"]["paths_equal"]) for row in rows
        ),
        "evaluation_dose_exact": all(
            bool(row["final_evaluation"]["dose_exact"]) for row in rows
        ),
        "gate_off_exact": all(
            bool(row["final_evaluation"]["gate_off_exact"]) for row in rows
        ),
        "response_converged": all(bool(row["convergence"]["stable"]) for row in rows),
        "oracle_mean_positive": float(paired["mean"]) > 0.0,
        "oracle_seed_rate": int(paired["positive"]) >= required_positive,
        "activation_in_range": 0.05
        <= method_summary["oracle_timed"]["activation_rate"]
        <= float(config["max_activation_rate"]),
        "oracle_high_risk_zero": method_summary["oracle_timed"][
            "high_risk_activation_rate"
        ]
        == 0.0,
        "volume_conserved": all(
            abs(method_summary[method]["mean_volume"] - 20.0) < 1e-9
            for method in method_summary
        ),
        "reward_guardrail": oracle_reward
        >= shuffled_reward
        - float(config["maximum_reward_degradation_fraction"])
        * max(1.0, abs(shuffled_reward)),
    }
    return {
        "method_summary": method_summary,
        "paired_block_minus_oracle": paired,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def validate_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    if str(config.get("stage")) != "oracle":
        raise RuntimeError(
            "this frozen response runner currently requires oracle stage"
        )
    gate_path = REPO / str(config["gate_artifact"])
    gate_hash = sha256_file(gate_path)
    if gate_hash != str(config["gate_artifact_sha256"]):
        raise RuntimeError("oracle response gate artifact hash mismatch")
    base_output = REPO / str(config["base_output_dir"])
    base_manifest = json.loads(
        (base_output / "manifest.json").read_text(encoding="utf-8")
    )
    base_summary = json.loads(
        (base_output / "base_summary.json").read_text(encoding="utf-8")
    )
    if base_manifest.get("protocol_fingerprint") != config.get(
        "base_protocol_fingerprint"
    ):
        raise RuntimeError("oracle response base fingerprint mismatch")
    if base_summary.get("status") != "pass":
        raise RuntimeError("oracle response requires a passing base policy gate")
    image = str(config["docker_image"])
    return {
        "gate_artifact": gate_hash,
        "docker_image_id": resolve_docker_image_id(image),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_mappo_adaptive_v3_oracle_dev_v1.json",
    )
    parser.add_argument("--smoke-episodes", type=int, default=0)
    args = parser.parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    if args.smoke_episodes:
        config = copy.deepcopy(config)
        config["protocol_version"] += f"_smoke_{args.smoke_episodes}"
        config["policy_seeds"] = config["policy_seeds"][:1]
        config["development_evaluation_seeds"] = config["development_evaluation_seeds"][
            :2
        ]
        config["response_episodes"] = int(args.smoke_episodes)
        config["response_evaluation_start"] = int(args.smoke_episodes)
        config["response_evaluation_interval"] = int(args.smoke_episodes)
        config["response_required_stable_evaluations"] = 1
        config["output_dir"] = (
            "revision/experiments/outputs/"
            f"abides_mappo_adaptive_v3_oracle_smoke_{args.smoke_episodes}"
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
    with ProcessPoolExecutor(
        max_workers=min(int(config["workers"]), len(seeds))
    ) as pool:
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
                "paired": result["paired_block_minus_oracle"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
