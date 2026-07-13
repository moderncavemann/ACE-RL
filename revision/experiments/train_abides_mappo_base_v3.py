#!/usr/bin/env python3
"""Resumable no-intervention GAE-MAPPO base training for adaptive v3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
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
    reward_range_fraction = (max(rewards) - min(rewards)) / max(1.0, abs(mean(rewards)))
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


def fixed_horizon_diagnostic(
    evaluation_history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    required = int(config["base_fixed_horizon_window_evaluations"])
    if len(evaluation_history) < required:
        return {"stable": False, "reason": "insufficient fixed-horizon panels"}
    recent = list(evaluation_history[-required:])
    spreads = [float(row["mean_half_spread"]) for row in recent]
    rewards = [float(row["mean_reward"]) for row in recent]
    spread_range = max(spreads) - min(spreads)
    reward_range_fraction = (max(rewards) - min(rewards)) / max(1.0, abs(mean(rewards)))
    failure_mode = min(spreads) >= float(config["base_minimum_failure_spread"])
    spread_stable = spread_range <= float(
        config["base_fixed_horizon_maximum_spread_range"]
    )
    reward_stable = reward_range_fraction <= float(
        config["base_fixed_horizon_maximum_reward_range_fraction"]
    )
    return {
        "stable": spread_stable and reward_stable and failure_mode,
        "spread_stable": spread_stable,
        "reward_stable": reward_stable,
        "failure_mode": failure_mode,
        "spread_range": spread_range,
        "reward_range_fraction": reward_range_fraction,
        "recent_evaluations": recent,
    }


def summarize_base_rows(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    mode = str(config.get("base_training_mode", "convergence"))
    expected = len(config["base_training_seeds"])
    if mode == "convergence":
        checks = {
            "required_seed_count": len(rows) == expected,
            "all_seeds_converged": all(row["status"] == "pass" for row in rows),
        }
        return {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "rows": list(rows),
        }
    if mode != "fixed_horizon":
        raise ValueError(f"unknown base training mode: {mode}")

    required_spread_stable = math.ceil(
        float(config["base_fixed_horizon_minimum_spread_stable_rate"]) * expected
        - 1e-12
    )
    required_reward_stable = math.ceil(
        float(config["base_fixed_horizon_minimum_reward_stable_rate"]) * expected
        - 1e-12
    )
    panel_count = int(config["base_fixed_horizon_window_evaluations"])
    population_panel_means = (
        [
            mean(
                float(
                    row["fixed_horizon"]["recent_evaluations"][panel_index][
                        "mean_half_spread"
                    ]
                )
                for row in rows
            )
            for panel_index in range(panel_count)
        ]
        if len(rows) == expected
        else []
    )
    population_spread_range = (
        max(population_panel_means) - min(population_panel_means)
        if population_panel_means
        else math.inf
    )
    checks = {
        "required_seed_count": len(rows) == expected,
        "all_reached_fixed_horizon": len(rows) == expected
        and all(
            int(row["completed_episodes"]) == int(config["base_max_episodes"])
            for row in rows
        ),
        "all_failure_modes_present": len(rows) == expected
        and all(bool(row["fixed_horizon"]["failure_mode"]) for row in rows),
        "spread_stable_seed_rate": sum(
            bool(row["fixed_horizon"]["spread_stable"]) for row in rows
        )
        >= required_spread_stable,
        "reward_stable_seed_rate": sum(
            bool(row["fixed_horizon"]["reward_stable"]) for row in rows
        )
        >= required_reward_stable,
        "population_mean_spread_stable": population_spread_range
        <= float(config["base_population_maximum_mean_spread_range"]),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "population_panel_mean_spreads": population_panel_means,
        "population_mean_spread_range": population_spread_range,
        "required_spread_stable_seed_count": required_spread_stable,
        "required_reward_stable_seed_count": required_reward_stable,
        "rows": list(rows),
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
    expected_fingerprint = fingerprint
    if not path.exists() and config.get("parent_output_dir"):
        parent_output = REPO / str(config["parent_output_dir"])
        path = checkpoint_path(parent_output, seed)
        expected_fingerprint = str(config["parent_protocol_fingerprint"])
        if not path.exists():
            raise FileNotFoundError(f"missing parent base checkpoint: {path}")
    elif not path.exists():
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
    if payload.get("protocol_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"base checkpoint fingerprint mismatch: {path}")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"base checkpoint seed mismatch: {path}")
    completed = int(payload["completed_episodes"])
    if config.get("parent_output_dir") and expected_fingerprint != fingerprint:
        if completed != int(config["parent_completed_episodes"]):
            raise RuntimeError(
                f"parent base checkpoint expected {config['parent_completed_episodes']} "
                f"episodes, got {completed}"
            )
    trainer = trainer_from_payload(payload["trainer"], str(config["device"]))
    return (
        trainer,
        completed,
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
    mode = str(config.get("base_training_mode", "convergence"))
    if mode not in {"convergence", "fixed_horizon"}:
        raise ValueError(f"unknown base training mode: {mode}")
    max_episodes = int(config["base_max_episodes"])
    initial_convergence = convergence(evaluation_history, config)
    initial_fixed = (
        fixed_horizon_diagnostic(evaluation_history, config)
        if mode == "fixed_horizon"
        else None
    )
    if (mode == "convergence" and bool(initial_convergence.get("stable"))) or (
        mode == "fixed_horizon" and completed == max_episodes
    ):
        result = {
            "protocol_fingerprint": fingerprint,
            "seed": seed,
            "status": "pass" if mode == "convergence" else "complete",
            "completed_episodes": completed,
            "model_digest": model_digest(trainer),
            "convergence": initial_convergence,
            "fixed_horizon": initial_fixed,
            "training_history": training_history,
            "evaluation_history": evaluation_history,
        }
        atomic_write_json(result_path(output_dir, seed), result)
        return result
    episodes_per_update = int(config["base_episodes_per_update"])
    evaluation_interval = int(config["base_evaluation_interval"])
    market = dict(config.get("market", {}))
    market.update(
        {
            "episode_steps": int(config["episode_steps"]),
            "routing_allocation_mode": str(config["routing_allocation_mode"]),
            "routing_target_kappa": float(config["routing_target_kappa"]),
            "routing_routable_fraction": float(config["routing_routable_fraction"]),
        }
    )
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
                if mode == "convergence" and bool(current["stable"]):
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
            if mode == "convergence" and status == "pass":
                break

    final_convergence = convergence(evaluation_history, config)
    final_fixed = (
        fixed_horizon_diagnostic(evaluation_history, config)
        if mode == "fixed_horizon"
        else None
    )
    status = (
        "complete"
        if mode == "fixed_horizon" and completed == max_episodes
        else "pass" if bool(final_convergence["stable"]) else "fail"
    )
    result = {
        "protocol_fingerprint": fingerprint,
        "seed": seed,
        "status": status,
        "completed_episodes": completed,
        "model_digest": model_digest(trainer),
        "convergence": final_convergence,
        "fixed_horizon": final_fixed,
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


def validate_parent_extension(config: Mapping[str, Any]) -> None:
    if not config.get("parent_output_dir"):
        return
    parent_output = REPO / str(config["parent_output_dir"])
    parent_manifest = json.loads(
        (parent_output / "manifest.json").read_text(encoding="utf-8")
    )
    parent_summary = json.loads(
        (parent_output / "base_summary.json").read_text(encoding="utf-8")
    )
    if parent_manifest.get("protocol_fingerprint") != config.get(
        "parent_protocol_fingerprint"
    ):
        raise RuntimeError("parent base protocol fingerprint mismatch")
    if parent_summary.get("status") != "fail":
        raise RuntimeError("base extension requires a failed parent convergence gate")
    if not all(
        bool(row.get("convergence", {}).get("failure_mode"))
        for row in parent_summary.get("rows", [])
    ):
        raise RuntimeError("parent base run failed the market failure-mode gate")
    parent_config = parent_manifest["config"]
    frozen_fields = (
        "docker_image",
        "episode_steps",
        "base_training_seeds",
        "base_fixed_evaluation_seeds",
        "base_episodes_per_update",
        "base_evaluation_interval",
        "base_required_stable_evaluations",
        "base_maximum_fixed_panel_spread_range",
        "base_maximum_fixed_panel_reward_range_fraction",
        "base_minimum_failure_spread",
        "learning_rate",
        "entropy_coefficient",
        "reward_scale",
        "routing_allocation_mode",
        "routing_target_kappa",
        "routing_routable_fraction",
        "base_environment_seed_namespace",
        "base_action_namespace",
        "base_evaluation_action_namespace",
    )
    for field in frozen_fields:
        if config.get(field) != parent_config.get(field):
            raise RuntimeError(f"base extension changed frozen field: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_mappo_adaptive_v3_development_v1.json",
    )
    parser.add_argument("--smoke-episodes", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    args = parser.parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    if args.smoke_episodes:
        config = copy.deepcopy(config)
        config["protocol_version"] += f"_smoke_{args.smoke_episodes}"
        config["base_training_seeds"] = config["base_training_seeds"][:1]
        config["base_fixed_evaluation_seeds"] = config["base_fixed_evaluation_seeds"][
            :2
        ]
        config["base_max_episodes"] = int(args.smoke_episodes)
        config["base_required_stable_evaluations"] = max(
            1,
            math.floor(args.smoke_episodes / int(config["base_evaluation_interval"])),
        )
        if str(config.get("base_training_mode")) == "fixed_horizon":
            config["base_fixed_horizon_window_evaluations"] = max(
                1,
                math.floor(
                    args.smoke_episodes / int(config["base_evaluation_interval"])
                ),
            )
            config["base_fixed_horizon_maximum_spread_range"] = 999.0
            config["base_fixed_horizon_maximum_reward_range_fraction"] = 999.0
            config["base_fixed_horizon_minimum_spread_stable_rate"] = 0.0
            config["base_fixed_horizon_minimum_reward_stable_rate"] = 0.0
            config["base_population_maximum_mean_spread_range"] = 999.0
        config["output_dir"] = (
            "revision/experiments/outputs/"
            f"abides_mappo_adaptive_v3_base_smoke_{args.smoke_episodes}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "refusing formal base training from a dirty worktree; use "
            "--allow-dirty only for a smoke test"
        )
    artifacts = validate_config(config)
    validate_parent_extension(config)
    fingerprint = protocol_fingerprint(config)
    output_dir = REPO / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(output_dir / ".run.lock", fingerprint, args.force_unlock):
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("protocol_fingerprint") != fingerprint:
                raise RuntimeError("existing base manifest has another protocol")
            if manifest.get("artifacts") != artifacts:
                raise RuntimeError("base resume artifact mismatch")
        else:
            manifest = {
                "protocol_version": config["protocol_version"],
                "protocol_fingerprint": fingerprint,
                "git_commit": current_git_commit(),
                "started_at": utc_now(),
                "config": config,
                "artifacts": artifacts,
            }
            atomic_write_json(manifest_path, manifest)

        seeds = [int(seed) for seed in config["base_training_seeds"]]
        rows = []
        for seed in seeds:
            path = result_path(output_dir, seed)
            if not path.exists():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            row_fingerprint = row.get("protocol_fingerprint")
            if row_fingerprint is None:
                checkpoint = torch.load(
                    checkpoint_path(output_dir, seed),
                    map_location="cpu",
                    weights_only=False,
                )
                row_fingerprint = checkpoint.get("protocol_fingerprint")
            if row_fingerprint != fingerprint:
                raise RuntimeError(f"base result fingerprint mismatch: {path}")
            row["protocol_fingerprint"] = fingerprint
            rows.append(row)
        completed = {
            int(row["seed"]) for row in rows if row["status"] in {"pass", "complete"}
        }
        pending = [seed for seed in seeds if seed not in completed]
        if pending and manifest.get("git_commit") != current_git_commit():
            raise RuntimeError(
                "cannot resume pending base seeds from a different git commit"
            )

        def persist(status: str) -> None:
            complete = sorted(int(row["seed"]) for row in rows)
            state = {
                "status": status,
                "updated_at": utc_now(),
                "completed_seeds": complete,
                "pending_seeds": [seed for seed in seeds if seed not in complete],
            }
            atomic_write_json(output_dir / "run_state.json", state)
            lines = [
                "# ABIDES/MAPPO Adaptive-v3 Base Logbook",
                "",
                f"- Status: `{status}`",
                f"- Protocol: `{config['protocol_version']}`",
                f"- Fingerprint: `{fingerprint}`",
                f"- Completed: `{len(complete)}/{len(seeds)}`",
                "",
                "| Seed | Status | Episodes |",
                "| ---: | --- | ---: |",
            ]
            by_seed = {int(row["seed"]): row for row in rows}
            for seed in seeds:
                row = by_seed.get(seed)
                lines.append(
                    f"| {seed} | {'complete' if row else 'pending'} | "
                    f"{int(row['completed_episodes']) if row else ''} |"
                )
            atomic_write_text(output_dir / "LOGBOOK.md", "\n".join(lines) + "\n")

        append_logbook(
            output_dir / "logbook.jsonl",
            {
                "event": "run_resumed" if rows else "run_started",
                "completed_seeds": sorted(completed),
                "pending_seeds": pending,
            },
        )
        persist("running")
        failures = []
        if pending:
            with ProcessPoolExecutor(
                max_workers=min(int(config["workers"]), len(pending))
            ) as pool:
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
                        failures.append(
                            {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                        )
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {"event": "seed_failed", **failures[-1]},
                        )
                    else:
                        rows.append(row)
                        rows.sort(key=lambda item: int(item["seed"]))
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {
                                "event": "seed_completed",
                                "seed": seed,
                                "episodes": int(row["completed_episodes"]),
                            },
                        )
                        print(
                            json.dumps({"seed": row["seed"], "status": row["status"]}),
                            flush=True,
                        )
                    persist("failed" if failures else "running")
        rows.sort(key=lambda row: int(row["seed"]))
        summary = summarize_base_rows(rows, config)
        if failures:
            summary["status"] = "fail"
        summary["failures"] = failures
        status = str(summary["status"])
        atomic_write_json(output_dir / "base_summary.json", summary)
        persist(status)
        append_logbook(
            output_dir / "logbook.jsonl",
            {"event": "run_finished", "status": status},
        )
        print(json.dumps({"status": status, "seeds": len(rows)}))
        raise SystemExit(0 if status == "pass" else 2)


if __name__ == "__main__":
    main()
