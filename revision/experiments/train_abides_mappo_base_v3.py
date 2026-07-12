#!/usr/bin/env python3
"""Resumable no-intervention GAE-MAPPO base training for adaptive v3."""

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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_mappo_adaptive_pilot import (  # noqa: E402
    atomic_torch_save,
    model_digest,
    trainer_from_payload,
    trainer_payload,
)
from run_abides_mappo_full_main import (  # noqa: E402
    atomic_write_json,
    current_git_commit,
    protocol_fingerprint,
    resolve_docker_image_id,
    sha256_file,
)
from src.agents.abides_mappo import make_abides_trainer, rollout_episode  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)


def seed_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "base_seeds" / f"seed_{seed}"


def checkpoint_path(output_dir: Path, seed: int) -> Path:
    return seed_dir(output_dir, seed) / "training_state.pt"


def result_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "base_results" / f"seed_{seed}.json"


def environment_seed(config: Mapping[str, Any], seed: int, episode: int) -> int:
    return (
        int(config["base_environment_seed_namespace"])
        + (int(seed) % 100_000) * 1_000
        + int(episode)
    )


def fixed_panel_summary(
    env: ABIDESDockerDealerEnv,
    trainer,
    config: Mapping[str, Any],
) -> dict[str, float]:
    spreads = []
    rewards = []
    for eval_seed in config["base_fixed_evaluation_seeds"]:
        rollout = rollout_episode(
            env,
            trainer,
            seed=int(eval_seed),
            episode_steps=int(config["episode_steps"]),
            train=False,
            announced_signal_schedule=[0.0] * int(config["episode_steps"]),
            reward_scale=float(config["reward_scale"]),
            action_namespace=int(config["base_evaluation_action_namespace"]),
        )
        spreads.append(float(rollout["mean_half_spread"]))
        rewards.append(float(rollout["mean_reward"]))
    return {
        "mean_half_spread": mean(spreads),
        "mean_reward": mean(rewards),
        "spread_path_sd": float(np.std(spreads, ddof=1)) if len(spreads) > 1 else 0.0,
        "reward_path_sd": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
    }


def convergence(
    evaluation_history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    required = int(config["base_required_stable_evaluations"])
    if len(evaluation_history) < required:
        return {"stable": False, "reason": "insufficient fixed-panel evaluations"}
    recent = list(evaluation_history[-required:])
    spreads = [float(row["mean_half_spread"]) for row in recent]
    rewards = [float(row["mean_reward"]) for row in recent]
    spread_range = max(spreads) - min(spreads)
    reward_range_fraction = (max(rewards) - min(rewards)) / max(
        1.0, abs(mean(rewards))
    )
    failure_mode = min(spreads) >= float(config["base_minimum_failure_spread"])
    stable = (
        spread_range <= float(config["base_maximum_fixed_panel_spread_range"])
        and reward_range_fraction
        <= float(config["base_maximum_fixed_panel_reward_range_fraction"])
        and failure_mode
    )
    return {
        "stable": stable,
        "spread_range": spread_range,
        "reward_range_fraction": reward_range_fraction,
        "failure_mode": failure_mode,
        "recent_evaluations": recent,
    }


def save_checkpoint(
    output_dir: Path,
    seed: int,
    fingerprint: str,
    completed_episodes: int,
    trainer,
    training_history: Sequence[Mapping[str, Any]],
    evaluation_history: Sequence[Mapping[str, Any]],
) -> None:
    atomic_torch_save(
        checkpoint_path(output_dir, seed),
        {
            "protocol_fingerprint": fingerprint,
            "seed": seed,
            "completed_episodes": completed_episodes,
            "trainer": trainer_payload(trainer),
            "training_history": list(training_history),
            "evaluation_history": list(evaluation_history),
        },
    )


def load_or_initialize(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    fingerprint: str,
):
    path = checkpoint_path(output_dir, seed)
    if not path.exists():
        trainer = make_abides_trainer(
            seed=seed,
            episode_steps=int(config["episode_steps"]),
            device=str(config["device"]),
            entropy_coef=float(config["entropy_coefficient"]),
            policy_mode="multihead",
            policy_advantage_mode="gae",
            learning_rate=float(config["learning_rate"]),
            adapter_risk_feature_index=None,
        )
        return trainer, 0, [], []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol_fingerprint") != fingerprint:
        raise RuntimeError(f"base checkpoint fingerprint mismatch: {path}")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"base checkpoint seed mismatch: {path}")
    trainer = trainer_from_payload(payload["trainer"], str(config["device"]))
    return (
        trainer,
        int(payload["completed_episodes"]),
        list(payload["training_history"]),
        list(payload["evaluation_history"]),
    )


def train_seed(
    seed: int,
    config: dict[str, Any],
    output_dir_raw: str,
    fingerprint: str,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    output_dir = Path(output_dir_raw)
    trainer, completed, training_history, evaluation_history = load_or_initialize(
        config, output_dir, seed, fingerprint
    )
    max_episodes = int(config["base_max_episodes"])
    episodes_per_update = int(config["base_episodes_per_update"])
    evaluation_interval = int(config["base_evaluation_interval"])
    market = {
        "episode_steps": int(config["episode_steps"]),
        "routing_allocation_mode": str(config["routing_allocation_mode"]),
        "routing_target_kappa": float(config["routing_target_kappa"]),
        "routing_routable_fraction": float(config["routing_routable_fraction"]),
    }
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    status = "running"
    with ABIDESDockerDealerEnv(market, docker) as env:
        while completed < max_episodes:
            batch_size = min(episodes_per_update, max_episodes - completed)
            trajectories = []
            spreads = []
            rewards = []
            for offset in range(batch_size):
                episode = completed + offset
                rollout = rollout_episode(
                    env,
                    trainer,
                    seed=environment_seed(config, seed, episode),
                    episode_steps=int(config["episode_steps"]),
                    train=True,
                    defer_update=True,
                    announced_signal_schedule=[0.0] * int(config["episode_steps"]),
                    reward_scale=float(config["reward_scale"]),
                    action_namespace=int(config["base_action_namespace"]),
                )
                trajectories.append(rollout["trajectory"])
                spreads.append(float(rollout["mean_half_spread"]))
                rewards.append(float(rollout["mean_reward"]))
            update = trainer.update_episodes(trajectories)
            completed += batch_size
            training_history.append(
                {
                    "completed_episodes": completed,
                    "batch_episodes": batch_size,
                    "mean_half_spread": mean(spreads),
                    "mean_reward": mean(rewards),
                    **{key: float(value) for key, value in update.items()},
                }
            )
            if completed % evaluation_interval == 0 or completed == max_episodes:
                evaluation_history.append(
                    {
                        "completed_episodes": completed,
                        **fixed_panel_summary(env, trainer, config),
                    }
                )
                current = convergence(evaluation_history, config)
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "episodes": completed,
                            "spread": evaluation_history[-1]["mean_half_spread"],
                            "stable": current["stable"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if bool(current["stable"]):
                    status = "pass"
            save_checkpoint(
                output_dir,
                seed,
                fingerprint,
                completed,
                trainer,
                training_history,
                evaluation_history,
            )
            if status == "pass":
                break

    final_convergence = convergence(evaluation_history, config)
    status = "pass" if bool(final_convergence["stable"]) else "fail"
    result = {
        "seed": seed,
        "status": status,
        "completed_episodes": completed,
        "model_digest": model_digest(trainer),
        "convergence": final_convergence,
        "training_history": training_history,
        "evaluation_history": evaluation_history,
    }
    atomic_write_json(result_path(output_dir, seed), result)
    return result


def validate_config(config: Mapping[str, Any]) -> dict[str, str]:
    gate_path = REPO / str(config["gate_artifact"])
    if sha256_file(gate_path) != str(config["gate_artifact_sha256"]):
        raise RuntimeError("adaptive-v3 gate artifact hash mismatch")
    image = str(config["docker_image"])
    return {
        "gate_artifact": str(config["gate_artifact_sha256"]),
        "docker_image_id": resolve_docker_image_id(image),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_mappo_adaptive_v3_development_v1.json",
    )
    parser.add_argument("--smoke-episodes", type=int, default=0)
    args = parser.parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    if args.smoke_episodes:
        config = copy.deepcopy(config)
        config["protocol_version"] += f"_smoke_{args.smoke_episodes}"
        config["base_training_seeds"] = config["base_training_seeds"][:1]
        config["base_fixed_evaluation_seeds"] = config[
            "base_fixed_evaluation_seeds"
        ][:2]
        config["base_max_episodes"] = int(args.smoke_episodes)
        config["base_required_stable_evaluations"] = max(
            1,
            math.floor(args.smoke_episodes / int(config["base_evaluation_interval"])),
        )
        config["output_dir"] = (
            "revision/experiments/outputs/"
            f"abides_mappo_adaptive_v3_base_smoke_{args.smoke_episodes}"
        )
    artifacts = validate_config(config)
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
    seeds = [int(seed) for seed in config["base_training_seeds"]]
    rows = []
    with ProcessPoolExecutor(max_workers=min(int(config["workers"]), len(seeds))) as pool:
        futures = {
            pool.submit(train_seed, seed, config, str(output_dir), fingerprint): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({"seed": row["seed"], "status": row["status"]}), flush=True)
    rows.sort(key=lambda row: int(row["seed"]))
    summary = {
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "rows": rows,
    }
    atomic_write_json(output_dir / "base_summary.json", summary)
    print(json.dumps({"status": summary["status"], "seeds": len(rows)}))
    raise SystemExit(0 if summary["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
