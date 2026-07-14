#!/usr/bin/env python3
"""Frozen-policy forced-deviation event study for the ABIDES assay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
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

from run_abides_mappo_adaptive_pilot import model_digest  # noqa: E402
from run_abides_mappo_adaptive_v3_response import (  # noqa: E402
    load_base,
    market_config,
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
from src.agents.mappo import MAPPOTrainer  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.utils.abides_audit import exact_sign_test, t_interval  # noqa: E402


ARMS = ("sham", "forced_undercut", "upward_deviation")
PRIMARY_METRICS = (
    "rival_spread_compression_after_undercut",
    "deviator_continuation_flow_loss_after_undercut",
    "deviator_continuation_reward_loss_after_undercut",
    "high_spread_recovery_rate_after_undercut",
    "flow_loss_asymmetry",
    "reward_loss_asymmetry",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def seed_result_path(output_dir: Path, policy_seed: int) -> Path:
    return output_dir / "seed_results" / f"seed_{policy_seed}.json"


def source_manifest_path(config: Mapping[str, Any]) -> Path:
    return REPO / str(config["source_base_output_dir"]) / "manifest.json"


def source_checkpoint_path(config: Mapping[str, Any], policy_seed: int) -> Path:
    return (
        REPO
        / str(config["source_base_output_dir"])
        / "base_seeds"
        / f"seed_{policy_seed}"
        / "training_state.pt"
    )


def source_result_path(config: Mapping[str, Any], policy_seed: int) -> Path:
    return (
        REPO
        / str(config["source_base_output_dir"])
        / "base_results"
        / f"seed_{policy_seed}.json"
    )


def base_loader_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_output_dir": str(config["source_base_output_dir"]),
        "base_protocol_fingerprint": str(config["source_base_fingerprint"]),
        "device": str(config["device"]),
    }


def event_dealer_index(
    policy_seed: int, evaluation_seed: int, event_step: int, n_agents: int
) -> int:
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    payload = f"{int(policy_seed)}:{int(evaluation_seed)}:{int(event_step)}".encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % n_agents


def event_is_eligible(
    *,
    risk_state: Any,
    proposed_half_spreads: Sequence[float],
    step: int,
    episode_steps: int,
    required_risk_state: str,
    minimum_mean_half_spread: float,
    minimum_remaining_steps: int,
) -> bool:
    spreads = np.asarray(proposed_half_spreads, dtype=float)
    return bool(
        str(risk_state) == required_risk_state
        and spreads.size > 0
        and np.all(np.isfinite(spreads))
        and float(np.mean(spreads)) >= minimum_mean_half_spread
        and episode_steps - int(step) >= minimum_remaining_steps
    )


def action_for_arm(original_action: int, arm: str, n_actions: int) -> tuple[int, bool]:
    if arm == "sham":
        return int(original_action), True
    if arm == "forced_undercut":
        changed = max(int(original_action) - 1, 0)
    elif arm == "upward_deviation":
        changed = min(int(original_action) + 1, n_actions - 1)
    else:
        raise ValueError(f"unknown diagnostic arm: {arm}")
    return changed, changed != int(original_action)


def action_uniforms(seed: int, step: int, namespace: int, n_agents: int) -> np.ndarray:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(step), int(namespace)])
    ).random(n_agents)


def _policy_actions(
    trainer: MAPPOTrainer,
    state: Mapping[str, Any],
    episode_steps: int,
    uniforms: np.ndarray,
) -> np.ndarray:
    observations = actor_observations(state)
    critic = centralized_critic_observations(state, observations, episode_steps)
    actions, _, _ = trainer.act(
        observations,
        critic,
        deterministic=False,
        action_uniforms=uniforms,
    )
    return actions.astype(np.int64)


def rollout_arm(
    env: ABIDESDockerDealerEnv,
    trainer: MAPPOTrainer,
    config: Mapping[str, Any],
    policy_seed: int,
    evaluation_seed: int,
    arm: str,
    frozen_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if arm != "sham" and frozen_event is None:
        raise ValueError("deviation arms require a frozen sham event")
    episode_steps = int(config["episode_steps"])
    horizon = int(config["event_horizon"])
    levels = np.asarray(config["market"]["half_spread_levels"], dtype=float)
    n_agents = int(config["market"]["n_agents"])
    expected_volume = float(
        int(config["market"]["flow_events_per_step"])
        * int(config["market"]["flow_quantity"])
    )
    state = env.reset(seed=int(evaluation_seed))
    path_digest = str(state["info"]["path_digest"])
    selected_event = None if frozen_event is None else dict(frozen_event)
    trajectory: list[dict[str, Any]] = []
    prefix_records: list[dict[str, Any]] = []
    uniform_digests: list[str] = []
    risk_path: list[str] = []
    volumes: list[float] = []
    routing_tvs: list[float] = []
    gate_flags: list[bool] = []

    for step in range(episode_steps):
        uniforms = action_uniforms(
            evaluation_seed, step, int(config["action_namespace"]), n_agents
        )
        uniform_digest = canonical_digest(uniforms.tolist())
        original_actions = _policy_actions(trainer, state, episode_steps, uniforms)
        proposed_spreads = levels[original_actions]
        pre_state_digest = canonical_digest(
            {"observations": state["observations"], "info": state["info"]}
        )
        current_risk = str(state["info"].get("diagnostic_current_risk"))
        if selected_event is None and event_is_eligible(
            risk_state=current_risk,
            proposed_half_spreads=proposed_spreads,
            step=step,
            episode_steps=episode_steps,
            required_risk_state=str(config["event_risk_state"]),
            minimum_mean_half_spread=float(config["minimum_mean_half_spread"]),
            minimum_remaining_steps=int(config["minimum_remaining_steps"]),
        ):
            dealer_index = event_dealer_index(
                policy_seed, evaluation_seed, step, n_agents
            )
            undercut_action, undercut_feasible = action_for_arm(
                int(original_actions[dealer_index]),
                "forced_undercut",
                len(levels),
            )
            upward_action, upward_feasible = action_for_arm(
                int(original_actions[dealer_index]),
                "upward_deviation",
                len(levels),
            )
            selected_event = {
                "event_step": int(step),
                "dealer_index": int(dealer_index),
                "pre_event_mean_half_spread": float(np.mean(proposed_spreads)),
                "original_dealer_action": int(original_actions[dealer_index]),
                "original_dealer_half_spread": float(proposed_spreads[dealer_index]),
                "forced_undercut_action": int(undercut_action),
                "forced_undercut_feasible": bool(undercut_feasible),
                "upward_deviation_action": int(upward_action),
                "upward_deviation_feasible": bool(upward_feasible),
                "event_pre_state_digest": pre_state_digest,
                "event_action_uniform_digest": uniform_digest,
                "event_original_actions": original_actions.tolist(),
                "event_risk_state": current_risk,
            }

        actual_actions = original_actions.copy()
        if selected_event is not None and step == int(selected_event["event_step"]):
            dealer_index = int(selected_event["dealer_index"])
            changed, _ = action_for_arm(
                int(original_actions[dealer_index]), arm, len(levels)
            )
            actual_actions[dealer_index] = changed

        if selected_event is None or step <= int(selected_event["event_step"]):
            prefix_records.append(
                {
                    "step": int(step),
                    "pre_state_digest": pre_state_digest,
                    "original_actions": original_actions.tolist(),
                    "action_uniform_digest": uniform_digest,
                }
            )

        next_state = env.step(
            actual_actions,
            gate_active=False,
            routing_enabled=False,
        )
        info = dict(next_state["info"])
        rewards = np.asarray(next_state["rewards"], dtype=float)
        actual_spreads = levels[actual_actions]
        uniform_digests.append(uniform_digest)
        risk_path.append(str(info.get("last_step_risk")))
        volumes.append(float(info["realized_flow_quantity"]))
        routing_tvs.append(float(info["routing_total_variation"]))
        gate_flags.append(bool(info["gate_active"]))

        if selected_event is not None:
            relative_step = step - int(selected_event["event_step"])
            if 0 <= relative_step <= horizon:
                dealer_index = int(selected_event["dealer_index"])
                rival_mask = np.arange(n_agents) != dealer_index
                flow_shares = np.asarray(info["flow_shares"], dtype=float)
                trajectory.append(
                    {
                        "policy_seed": int(policy_seed),
                        "evaluation_seed": int(evaluation_seed),
                        "arm": arm,
                        "event_step": int(selected_event["event_step"]),
                        "relative_step": int(relative_step),
                        "dealer_index": dealer_index,
                        "risk_state": str(info.get("last_step_risk")),
                        "pre_state_digest": pre_state_digest,
                        "action_uniform_digest": uniform_digest,
                        "path_digest": path_digest,
                        "original_actions": original_actions.tolist(),
                        "actual_actions": actual_actions.tolist(),
                        "rivals_mean_half_spread": float(
                            np.mean(actual_spreads[rival_mask])
                        ),
                        "deviator_half_spread": float(actual_spreads[dealer_index]),
                        "joint_mean_half_spread": float(np.mean(actual_spreads)),
                        "deviator_flow_share": float(flow_shares[dealer_index]),
                        "deviator_reward": float(rewards[dealer_index]),
                        "aggregate_dealer_reward": float(np.sum(rewards)),
                        "customer_execution_half_spread": float(
                            info["mean_execution_half_spread"]
                        ),
                        "realized_flow_quantity": float(
                            info["realized_flow_quantity"]
                        ),
                        "routing_total_variation": float(
                            info["routing_total_variation"]
                        ),
                        "gate_active": bool(info["gate_active"]),
                    }
                )
        state = next_state

    if not bool(state["done"]):
        raise RuntimeError("diagnostic episode did not finish at configured horizon")
    env.close_episode()
    expected_trajectory_length = horizon + 1 if selected_event is not None else 0
    return {
        "arm": arm,
        "event": selected_event,
        "trajectory": trajectory,
        "path_digest": path_digest,
        "prefix_digest": digest(prefix_records),
        "uniform_schedule_digest": digest(uniform_digests),
        "risk_path_digest": digest(risk_path),
        "integrity": {
            "trajectory_length_exact": len(trajectory) == expected_trajectory_length,
            "volume_conserved": all(
                abs(value - expected_volume) <= 1e-12 for value in volumes
            ),
            "routing_off_exact": all(abs(value) <= 1e-12 for value in routing_tvs),
            "gate_off_exact": not any(gate_flags),
        },
    }


def arm_summary(
    trajectory: Sequence[Mapping[str, Any]], pre_event_mean_half_spread: float
) -> dict[str, Any]:
    rows = sorted(trajectory, key=lambda row: int(row["relative_step"]))
    if not rows or int(rows[0]["relative_step"]) != 0:
        raise ValueError("event trajectory must begin at relative step zero")
    continuation = [row for row in rows if int(row["relative_step"]) > 0]
    return_steps = [
        int(row["relative_step"])
        for row in continuation
        if float(row["joint_mean_half_spread"])
        >= float(pre_event_mean_half_spread) - 1e-12
    ]
    return {
        "deviator_immediate_reward": float(rows[0]["deviator_reward"]),
        "deviator_cumulative_continuation_reward": float(
            sum(float(row["deviator_reward"]) for row in continuation)
        ),
        "deviator_cumulative_continuation_flow_share": float(
            sum(float(row["deviator_flow_share"]) for row in continuation)
        ),
        "aggregate_dealer_reward_t0_to_h": float(
            sum(float(row["aggregate_dealer_reward"]) for row in rows)
        ),
        "mean_customer_execution_half_spread_t0_to_h": float(
            mean(float(row["customer_execution_half_spread"]) for row in rows)
        ),
        "mean_rivals_continuation_half_spread": float(
            mean(float(row["rivals_mean_half_spread"]) for row in continuation)
        ),
        "mean_joint_continuation_half_spread": float(
            mean(float(row["joint_mean_half_spread"]) for row in continuation)
        ),
        "return_to_pre_event_spread_time": (
            min(return_steps) if return_steps else None
        ),
        "high_spread_recovered_by_h": int(bool(return_steps)),
    }


def event_result(
    policy_seed: int,
    evaluation_seed: int,
    model_sha: str,
    sham: Mapping[str, Any],
    forced: Mapping[str, Any] | None,
    upward: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event = sham.get("event")
    base = {
        "policy_seed": int(policy_seed),
        "evaluation_seed": int(evaluation_seed),
        "model_digest": model_sha,
        "path_digest": str(sham["path_digest"]),
        "event_found": event is not None,
    }
    if event is None:
        base["paired_path_exact"] = True
        base["arm_integrity_pass"] = all(
            bool(value) for value in sham["integrity"].values()
        )
        return base, []
    assert forced is not None and upward is not None
    arms = {"sham": sham, "forced_undercut": forced, "upward_deviation": upward}
    pre_mean = float(event["pre_event_mean_half_spread"])
    summaries = {
        arm: arm_summary(result["trajectory"], pre_mean)
        for arm, result in arms.items()
    }
    row = {**base, **dict(event)}
    for arm, values in summaries.items():
        for key, value in values.items():
            row[f"{arm}__{key}"] = value

    row.update(
        {
            "rival_spread_compression_after_undercut": (
                summaries["sham"]["mean_rivals_continuation_half_spread"]
                - summaries["forced_undercut"][
                    "mean_rivals_continuation_half_spread"
                ]
            ),
            "deviator_continuation_flow_loss_after_undercut": (
                summaries["sham"][
                    "deviator_cumulative_continuation_flow_share"
                ]
                - summaries["forced_undercut"][
                    "deviator_cumulative_continuation_flow_share"
                ]
            ),
            "deviator_continuation_reward_loss_after_undercut": (
                summaries["sham"][
                    "deviator_cumulative_continuation_reward"
                ]
                - summaries["forced_undercut"][
                    "deviator_cumulative_continuation_reward"
                ]
            ),
            "high_spread_recovery_rate_after_undercut": float(
                summaries["forced_undercut"]["high_spread_recovered_by_h"]
            ),
            "rival_spread_compression_after_upward": (
                summaries["sham"]["mean_rivals_continuation_half_spread"]
                - summaries["upward_deviation"][
                    "mean_rivals_continuation_half_spread"
                ]
            ),
            "deviator_continuation_flow_loss_after_upward": (
                summaries["sham"][
                    "deviator_cumulative_continuation_flow_share"
                ]
                - summaries["upward_deviation"][
                    "deviator_cumulative_continuation_flow_share"
                ]
            ),
            "deviator_continuation_reward_loss_after_upward": (
                summaries["sham"][
                    "deviator_cumulative_continuation_reward"
                ]
                - summaries["upward_deviation"][
                    "deviator_cumulative_continuation_reward"
                ]
            ),
        }
    )
    row["flow_loss_asymmetry"] = (
        row["deviator_continuation_flow_loss_after_undercut"]
        - row["deviator_continuation_flow_loss_after_upward"]
    )
    row["reward_loss_asymmetry"] = (
        row["deviator_continuation_reward_loss_after_undercut"]
        - row["deviator_continuation_reward_loss_after_upward"]
    )
    row["paired_path_exact"] = all(
        str(result["path_digest"]) == str(sham["path_digest"])
        and str(result["uniform_schedule_digest"])
        == str(sham["uniform_schedule_digest"])
        and str(result["risk_path_digest"]) == str(sham["risk_path_digest"])
        and str(result["prefix_digest"]) == str(sham["prefix_digest"])
        for result in (forced, upward)
    )
    row["arm_integrity_pass"] = all(
        all(bool(value) for value in result["integrity"].values())
        for result in arms.values()
    )
    trajectories = [dict(item) for result in arms.values() for item in result["trajectory"]]
    return row, trajectories


def validate_source(config: Mapping[str, Any], policy_seed: int) -> tuple[str, str]:
    manifest = source_manifest_path(config)
    if sha256_file(manifest) != str(config["source_base_manifest_sha256"]):
        raise RuntimeError("source base manifest digest changed")
    checkpoint = source_checkpoint_path(config, policy_seed)
    result_path = source_result_path(config, policy_seed)
    if not checkpoint.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"source policy artifacts missing for {policy_seed}")
    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    if source_result.get("protocol_fingerprint") != str(
        config["source_base_fingerprint"]
    ):
        raise RuntimeError(f"source result fingerprint mismatch for {policy_seed}")
    return sha256_file(checkpoint), str(source_result["model_digest"])


def run_policy_seed(
    config: Mapping[str, Any], policy_seed: int, fingerprint: str
) -> dict[str, Any]:
    checkpoint_sha, expected_model_sha = validate_source(config, policy_seed)
    trainer = load_base(base_loader_config(config), policy_seed)
    actual_model_sha = model_digest(trainer)
    if actual_model_sha != expected_model_sha:
        raise RuntimeError(f"source model digest mismatch for {policy_seed}")
    event_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(market_config(config), docker) as env:
        for evaluation_seed in config["evaluation_seeds"]:
            sham = rollout_arm(
                env,
                trainer,
                config,
                policy_seed,
                int(evaluation_seed),
                "sham",
            )
            if sham["event"] is None:
                row, event_trajectories = event_result(
                    policy_seed,
                    int(evaluation_seed),
                    actual_model_sha,
                    sham,
                    None,
                    None,
                )
            else:
                forced = rollout_arm(
                    env,
                    trainer,
                    config,
                    policy_seed,
                    int(evaluation_seed),
                    "forced_undercut",
                    sham["event"],
                )
                upward = rollout_arm(
                    env,
                    trainer,
                    config,
                    policy_seed,
                    int(evaluation_seed),
                    "upward_deviation",
                    sham["event"],
                )
                row, event_trajectories = event_result(
                    policy_seed,
                    int(evaluation_seed),
                    actual_model_sha,
                    sham,
                    forced,
                    upward,
                )
            event_rows.append(row)
            trajectories.extend(event_trajectories)
    found = [row for row in event_rows if bool(row["event_found"])]
    result = {
        "protocol_version": str(config["protocol_version"]),
        "protocol_fingerprint": fingerprint,
        "policy_seed": int(policy_seed),
        "source_checkpoint_sha256": checkpoint_sha,
        "model_digest": actual_model_sha,
        "status": "complete",
        "eligible_event_count": len(found),
        "event_rows": event_rows,
        "trajectory_rows": trajectories,
        "integrity": {
            "evaluation_path_count_exact": len(event_rows)
            == len(config["evaluation_seeds"]),
            "paired_paths_exact": all(
                bool(row.get("paired_path_exact", True)) for row in event_rows
            ),
            "arm_integrity_pass": all(
                bool(row.get("arm_integrity_pass", True)) for row in event_rows
            ),
            "trajectory_row_count_exact": len(trajectories)
            == len(found) * len(ARMS) * (int(config["event_horizon"]) + 1),
        },
    }
    atomic_write_json(
        seed_result_path(REPO / str(config["output_dir"]), policy_seed), result
    )
    return result


def _metric_is_feasible(row: Mapping[str, Any], metric: str) -> bool:
    if metric in {
        "flow_loss_asymmetry",
        "reward_loss_asymmetry",
    }:
        return bool(row["forced_undercut_feasible"]) and bool(
            row["upward_deviation_feasible"]
        )
    if "after_undercut" in metric:
        return bool(row["forced_undercut_feasible"])
    if "after_upward" in metric:
        return bool(row["upward_deviation_feasible"])
    return True


def aggregate_policy_seeds(
    event_rows: Sequence[Mapping[str, Any]], policy_seeds: Sequence[int]
) -> list[dict[str, Any]]:
    output = []
    all_metrics = PRIMARY_METRICS + (
        "rival_spread_compression_after_upward",
        "deviator_continuation_flow_loss_after_upward",
        "deviator_continuation_reward_loss_after_upward",
    )
    for policy_seed in policy_seeds:
        rows = [
            row
            for row in event_rows
            if int(row["policy_seed"]) == int(policy_seed) and bool(row["event_found"])
        ]
        result: dict[str, Any] = {
            "policy_seed": int(policy_seed),
            "eligible_event_count": len(rows),
            "forced_undercut_feasible_count": sum(
                bool(row["forced_undercut_feasible"]) for row in rows
            ),
            "upward_deviation_feasible_count": sum(
                bool(row["upward_deviation_feasible"]) for row in rows
            ),
        }
        for metric in all_metrics:
            values = [
                float(row[metric])
                for row in rows
                if _metric_is_feasible(row, metric)
            ]
            result[metric] = float(mean(values)) if values else None
            result[f"{metric}__path_count"] = len(values)
        output.append(result)
    return output


def two_way_bootstrap(
    event_rows: Sequence[Mapping[str, Any]],
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    grouped: dict[int, np.ndarray] = {}
    for policy_seed in sorted({int(row["policy_seed"]) for row in event_rows}):
        values = np.asarray(
            [
                float(row[metric])
                for row in event_rows
                if int(row["policy_seed"]) == policy_seed
                and bool(row["event_found"])
                and _metric_is_feasible(row, metric)
            ],
            dtype=float,
        )
        if values.size:
            grouped[policy_seed] = values
    if not grouped:
        raise ValueError(f"no bootstrap observations for {metric}")
    keys = np.asarray(sorted(grouped), dtype=int)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        seed_means = []
        for sampled_key in sampled_keys:
            values = grouped[int(sampled_key)]
            sampled_values = rng.choice(values, size=len(values), replace=True)
            seed_means.append(float(np.mean(sampled_values)))
        estimates[replicate] = float(np.mean(seed_means))
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "mean": float(np.mean(estimates)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def summarize(
    config: Mapping[str, Any], results: Sequence[Mapping[str, Any]], failures: Sequence[Any]
) -> dict[str, Any]:
    event_rows = [dict(row) for result in results for row in result["event_rows"]]
    trajectory_rows = [
        dict(row) for result in results for row in result["trajectory_rows"]
    ]
    seed_rows = aggregate_policy_seeds(event_rows, config["source_policy_seeds"])
    statistics: dict[str, Any] = {}
    complete_seed_metrics = True
    for index, metric in enumerate(PRIMARY_METRICS):
        values = [row[metric] for row in seed_rows if row[metric] is not None]
        if len(values) != len(config["source_policy_seeds"]):
            complete_seed_metrics = False
        if values:
            statistics[metric] = {
                "policy_seed_t_interval": t_interval(values),
                "policy_seed_exact_sign_test": exact_sign_test(values),
                "two_way_bootstrap": two_way_bootstrap(
                    event_rows,
                    metric,
                    int(config["bootstrap_replicates"]),
                    int(config["bootstrap_seed"]) + index,
                ),
            }

    integrity_checks = {
        "required_policy_seed_count": len(results)
        == int(config["required_policy_seed_count"]),
        "no_worker_failures": not failures,
        "all_seed_integrity_checks_pass": all(
            all(bool(value) for value in result["integrity"].values())
            for result in results
        ),
        "one_or_more_events_per_policy_seed": all(
            int(result["eligible_event_count"]) > 0 for result in results
        ),
        "complete_policy_seed_estimands": complete_seed_metrics,
    }
    integrity_pass = all(integrity_checks.values())
    requirements = {
        "rival_spread_compression_after_undercut": bool(
            statistics.get("rival_spread_compression_after_undercut", {})
            .get("policy_seed_t_interval", {})
            .get("ci_low", -math.inf)
            > 0.0
        ),
        "deviator_continuation_loss": bool(
            statistics.get(
                "deviator_continuation_flow_loss_after_undercut", {}
            )
            .get("policy_seed_t_interval", {})
            .get("ci_low", -math.inf)
            > 0.0
            and statistics.get(
                "deviator_continuation_reward_loss_after_undercut", {}
            )
            .get("policy_seed_t_interval", {})
            .get("ci_low", -math.inf)
            > 0.0
        ),
        "high_spread_state_recovery": bool(
            statistics.get("high_spread_recovery_rate_after_undercut", {})
            .get("policy_seed_t_interval", {})
            .get("ci_low", -math.inf)
            > float(
                config["estimand_rules"][
                    "minimum_recovery_rate_ci_lower_bound"
                ]
            )
        ),
        "upward_deviation_asymmetry": bool(
            statistics.get("flow_loss_asymmetry", {})
            .get("policy_seed_t_interval", {})
            .get("ci_low", -math.inf)
            > 0.0
            and statistics.get("reward_loss_asymmetry", {})
            .get("policy_seed_t_interval", {})
            .get("ci_low", -math.inf)
            > 0.0
        ),
        "policy_seed_ci_support": complete_seed_metrics,
    }
    if bool(config.get("engineering_smoke")):
        scientific_result = "not_evaluated_engineering_smoke"
    elif not integrity_pass:
        scientific_result = "not_evaluated_integrity_fail"
    elif all(requirements.values()):
        scientific_result = "punishment_supported_collusion_pattern_supported"
    else:
        scientific_result = "high_spread_coordination_only"
    return {
        "protocol_version": str(config["protocol_version"]),
        "engineering_smoke": bool(config.get("engineering_smoke")),
        "policy_seed_count": len(results),
        "evaluation_path_count": len(event_rows),
        "eligible_event_count": sum(bool(row["event_found"]) for row in event_rows),
        "trajectory_row_count": len(trajectory_rows),
        "integrity_checks": integrity_checks,
        "implementation_status": "pass" if integrity_pass else "fail",
        "strong_collusion_requirements": requirements,
        "scientific_result": scientific_result,
        "required_wording": (
            "punishment-supported collusion"
            if scientific_result == "punishment_supported_collusion_pattern_supported"
            else str(config["fallback_wording"])
        ),
        "statistics": statistics,
        "per_seed": seed_rows,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: canonical_json(value)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {summary['protocol_version']} Summary",
        "",
        f"- Implementation status: `{summary['implementation_status']}`",
        f"- Scientific result: `{summary['scientific_result']}`",
        f"- Required wording: `{summary['required_wording']}`",
        f"- Policy seeds: `{summary['policy_seed_count']}`",
        f"- Evaluation paths: `{summary['evaluation_path_count']}`",
        f"- Eligible events: `{summary['eligible_event_count']}`",
        "",
        "## Frozen Requirements",
        "",
    ]
    for key, value in summary["strong_collusion_requirements"].items():
        lines.append(f"- `{key}`: `{str(bool(value)).lower()}`")
    lines.extend(["", "## Policy-seed Contrasts", ""])
    for metric, payload in summary["statistics"].items():
        interval = payload["policy_seed_t_interval"]
        lines.append(
            f"- `{metric}`: mean `{interval['mean']:.6f}`, "
            f"95% CI `[{interval['ci_low']:.6f}, {interval['ci_high']:.6f}]`, "
            f"positive `{interval['positive']}/{interval['n']}`"
        )
    lines.extend(
        [
            "",
            "Strong collusion wording is fail-closed: any failed frozen requirement "
            "forces the fallback wording.",
            "",
        ]
    )
    return "\n".join(lines)


def load_completed_results(
    config: Mapping[str, Any], output_dir: Path, fingerprint: str
) -> list[dict[str, Any]]:
    results = []
    for policy_seed in config["source_policy_seeds"]:
        path = seed_result_path(output_dir, int(policy_seed))
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"diagnostic result fingerprint mismatch: {path}")
        results.append(result)
    return sorted(results, key=lambda row: int(row["policy_seed"]))


def validate_config(config: Mapping[str, Any]) -> None:
    if tuple(config["arms"]) != ARMS:
        raise RuntimeError("diagnostic arms differ from the frozen protocol")
    if int(config["maximum_events_per_episode"]) != 1:
        raise RuntimeError("only the frozen first-event design is implemented")
    if str(config["event_selection"]) != "first_eligible_state":
        raise RuntimeError("event selection differs from the frozen protocol")
    if str(config["dealer_selection"]) != "sha256_policy_path_step_mod_n_agents":
        raise RuntimeError("dealer selection differs from the frozen protocol")
    if int(config["minimum_remaining_steps"]) < int(config["event_horizon"]) + 1:
        raise RuntimeError("minimum remaining steps cannot cover t0 through t0+H")
    required_estimand_rules = {
        "event_spread_source": "current_sham_policy_joint_proposal",
        "continuation_window": "t1_through_tH_excluding_t0",
        "recovery_definition": "first_t1_to_tH_joint_mean_ge_pre_event_mean",
        "continuation_loss_requires": (
            "both_flow_and_reward_policy_seed_ci_low_gt_zero"
        ),
        "asymmetry_requires": (
            "both_flow_and_reward_loss_asymmetry_policy_seed_ci_low_gt_zero"
        ),
        "complete_policy_seed_coverage_required": True,
    }
    for key, expected in required_estimand_rules.items():
        if config["estimand_rules"].get(key) != expected:
            raise RuntimeError(f"estimand rule changed: {key}")
    recovery_bound = float(
        config["estimand_rules"]["minimum_recovery_rate_ci_lower_bound"]
    )
    if not 0.0 <= recovery_bound <= 1.0:
        raise RuntimeError("recovery CI lower-bound requirement is invalid")
    if len(config["source_policy_seeds"]) != int(config["required_policy_seed_count"]):
        raise RuntimeError("source policy seed count differs from the frozen count")
    if not bool(config.get("engineering_smoke")) and not bool(
        config.get("formal_execution_allowed")
    ):
        raise RuntimeError("formal execution is not allowed by this config")
    source_manifest = source_manifest_path(config)
    if sha256_file(source_manifest) != str(config["source_base_manifest_sha256"]):
        raise RuntimeError("source base manifest digest changed")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("protocol_fingerprint") != str(config["source_base_fingerprint"]):
        raise RuntimeError("source base protocol fingerprint changed")
    source_config = source["config"]
    for key in (
        "episode_steps",
        "routing_allocation_mode",
        "routing_target_kappa",
        "routing_routable_fraction",
    ):
        if source_config[key] != config[key]:
            raise RuntimeError(f"diagnostic environment drifted on {key}")
    if source_config["market"] != config["market"]:
        raise RuntimeError("diagnostic market differs from source policy market")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_collusion_diagnostic_v1.json",
    )
    parser.add_argument("--force-unlock", action="store_true")
    args = parser.parse_args()
    config_path = REPO / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    fingerprint = protocol_fingerprint(config)
    output_dir = REPO / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failures.jsonl"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("output directory belongs to another protocol")
    else:
        manifest = {
            "protocol_version": str(config["protocol_version"]),
            "protocol_fingerprint": fingerprint,
            "config": config,
            "config_sha256": sha256_file(config_path),
            "source_base_manifest_sha256": sha256_file(source_manifest_path(config)),
            "docker_image_id": resolve_docker_image_id(str(config["docker_image"])),
            "execution_commit": current_git_commit(),
            "started_at": utc_now(),
        }
        atomic_write_json(manifest_path, manifest)

    with RunLock(output_dir / "run.lock", fingerprint, args.force_unlock):
        started = time.monotonic()
        prior_aggregate_sha = (
            sha256_file(output_dir / "aggregate.json")
            if (output_dir / "aggregate.json").is_file()
            else None
        )
        completed = load_completed_results(config, output_dir, fingerprint)
        completed_seeds = {int(row["policy_seed"]) for row in completed}
        pending = [
            int(seed)
            for seed in config["source_policy_seeds"]
            if int(seed) not in completed_seeds
        ]
        append_logbook(
            output_dir / "logbook.jsonl",
            {
                "event": "run_started",
                "completed_policy_seeds": sorted(completed_seeds),
                "pending_policy_seeds": pending,
                "protocol_fingerprint": fingerprint,
            },
        )
        failures: list[dict[str, Any]] = []
        if pending:
            with ProcessPoolExecutor(max_workers=int(config["workers"])) as pool:
                futures = {
                    pool.submit(run_policy_seed, config, seed, fingerprint): seed
                    for seed in pending
                }
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        result = future.result()
                        completed.append(result)
                        completed.sort(key=lambda row: int(row["policy_seed"]))
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {
                                "event": "policy_seed_completed",
                                "policy_seed": seed,
                                "eligible_event_count": result[
                                    "eligible_event_count"
                                ],
                            },
                        )
                    except Exception as exc:  # pragma: no cover - worker boundary
                        failure = {
                            "timestamp": utc_now(),
                            "policy_seed": seed,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        with failures_path.open("a", encoding="utf-8") as handle:
                            handle.write(canonical_json(failure) + "\n")
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {"event": "policy_seed_failed", **failure},
                        )

        summary = summarize(config, completed, failures)
        summary["protocol_fingerprint"] = fingerprint
        event_rows = [dict(row) for result in completed for row in result["event_rows"]]
        trajectory_rows = [
            dict(row) for result in completed for row in result["trajectory_rows"]
        ]
        atomic_write_json(output_dir / "aggregate.json", summary)
        write_csv(output_dir / "per_event.csv", event_rows)
        write_csv(output_dir / "per_seed.csv", summary["per_seed"])
        write_csv(output_dir / "trajectory.csv", trajectory_rows)
        atomic_write_text(output_dir / "SUMMARY.md", summary_markdown(summary))
        final_aggregate_sha = sha256_file(output_dir / "aggregate.json")
        completed_resume = not pending
        completion = {
            "protocol_fingerprint": fingerprint,
            "completed_policy_seeds": [
                int(row["policy_seed"]) for row in completed
            ],
            "pending_policy_seeds": [
                int(seed)
                for seed in config["source_policy_seeds"]
                if int(seed) not in {int(row["policy_seed"]) for row in completed}
            ],
            "failed_policy_seeds_this_invocation": [
                int(row["policy_seed"]) for row in failures
            ],
            "containers_started": len(pending),
            "completed_resume_verification": completed_resume,
            "aggregate_sha256_before": prior_aggregate_sha,
            "aggregate_sha256_after": final_aggregate_sha,
            "aggregate_unchanged_on_completed_resume": (
                completed_resume and prior_aggregate_sha == final_aggregate_sha
            ),
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": utc_now(),
        }
        atomic_write_json(output_dir / "completion.json", completion)
        append_logbook(
            output_dir / "logbook.jsonl",
            {
                "event": "run_finished",
                "implementation_status": summary["implementation_status"],
                "scientific_result": summary["scientific_result"],
                "aggregate_sha256": final_aggregate_sha,
            },
        )
        print(canonical_json({"summary": summary, "completion": completion}))
        if failures:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
