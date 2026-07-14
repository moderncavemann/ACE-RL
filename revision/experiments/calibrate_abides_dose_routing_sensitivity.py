#!/usr/bin/env python3
"""Calibrate frozen ABIDES dose controllers on passive, disjoint paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_mappo_adaptive_pilot import model_digest  # noqa: E402
from run_abides_mappo_adaptive_v3_response import load_base, market_config  # noqa: E402
from run_abides_mappo_full_main import (  # noqa: E402
    atomic_write_json,
    current_git_commit,
    protocol_fingerprint,
    resolve_docker_image_id,
    sha256_file,
    utc_now,
)
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.interventions.abides_gate import ABIDESTwoStageGateModel  # noqa: E402
from src.interventions.abides_literature_baselines import (  # noqa: E402
    pace_signal_strengths,
    routing_total_variation,
    soft_rank_routing_weights,
    spread_only_proposal_strength,
)


def base_loader_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_output_dir": str(config["source_base_output_dir"]),
        "base_protocol_fingerprint": str(config["source_base_fingerprint"]),
        "device": str(config["device"]),
    }


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    if bool(config["formal_execution_allowed"]):
        raise RuntimeError("calibration must run before formal execution is enabled")
    if str(config["freeze_status"]) != "calibration_pending":
        raise RuntimeError("calibration config is not in the pending state")
    source_manifest = REPO / str(config["source_base_output_dir"]) / "manifest.json"
    if sha256_file(source_manifest) != str(config["source_base_manifest_sha256"]):
        raise RuntimeError("source base manifest changed")
    if sha256_file(REPO / str(config["gate_artifact"])) != str(
        config["gate_artifact_sha256"]
    ):
        raise RuntimeError("gate artifact changed")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source["protocol_fingerprint"] != str(config["source_base_fingerprint"]):
        raise RuntimeError("source base fingerprint changed")
    for key in (
        "episode_steps",
        "routing_allocation_mode",
        "routing_target_kappa",
        "routing_routable_fraction",
    ):
        if source["config"][key] != config[key]:
            raise RuntimeError(f"source market drifted on {key}")
    if source["config"]["market"] != config["market"]:
        raise RuntimeError("source market parameters changed")


def collect_policy_seed(
    config: Mapping[str, Any], policy_seed: int
) -> dict[str, Any]:
    trainer = load_base(base_loader_config(config), policy_seed)
    model = ABIDESTwoStageGateModel.load(REPO / str(config["gate_artifact"]))
    rows = []
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(market_config(config), docker) as env:
        for evaluation_seed in config["passive_calibration_seeds"]:
            rollout = rollout_episode(
                env,
                trainer,
                seed=int(evaluation_seed),
                episode_steps=int(config["episode_steps"]),
                train=False,
                routing_enabled=False,
                reward_scale=float(config["reward_scale"]),
                action_namespace=int(config["evaluation_action_namespace"]),
            )
            for record in rollout["records"]:
                state = {
                    "observations": record["pre_observations"],
                    "info": record["pre_info"],
                }
                diagnostic = model.announced_strength(
                    state, include_spread_severity=True
                )
                rows.append(
                    {
                        "policy_seed": int(policy_seed),
                        "evaluation_seed": int(evaluation_seed),
                        "step": int(record["step"]),
                        "actions": list(record["actions"]),
                        "risk_probability": float(diagnostic["risk_probability"]),
                        "spread_severity": float(diagnostic["severity"]),
                        "risk_state": str(
                            record["pre_info"].get("diagnostic_current_risk")
                        ),
                        "path_digest": str(record["pre_info"]["path_digest"]),
                        "action_uniform_digest": str(
                            record["action_uniform_digest"]
                        ),
                    }
                )
    return {
        "policy_seed": int(policy_seed),
        "model_digest": model_digest(trainer),
        "rows": rows,
    }


def episodes(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["policy_seed"]), int(row["evaluation_seed"])), []
        ).append(row)
    return [
        sorted(group, key=lambda row: int(row["step"]))
        for _, group in sorted(grouped.items())
    ]


def budget_metrics(
    episode_rows: Sequence[Sequence[Mapping[str, Any]]],
    strengths: Sequence[Sequence[float]],
    kappa: float,
    routable_fraction: float,
) -> dict[str, float | int]:
    flat_strengths = [float(value) for episode in strengths for value in episode]
    flat_rows = [row for episode in episode_rows for row in episode]
    if len(flat_rows) != len(flat_strengths):
        raise RuntimeError("calibration strength and state counts differ")
    tv = []
    high_active = []
    for row, strength in zip(flat_rows, flat_strengths):
        half_spreads = np.asarray(row["actions"], dtype=float) + 1.0
        weights = soft_rank_routing_weights(
            half_spreads, strength, kappa, routable_fraction
        )
        tv.append(routing_total_variation(weights))
        if row["risk_state"] == "high":
            high_active.append(strength > 0.0)
    active = [value > 0.0 for value in flat_strengths]
    active_strengths = [value for value in flat_strengths if value > 0.0]
    return {
        "record_count": len(flat_rows),
        "activation_rate": float(np.mean(active)),
        "high_risk_activation_rate": (
            float(np.mean(high_active)) if high_active else 0.0
        ),
        "mean_signal_strength": float(np.mean(flat_strengths)),
        "mean_active_signal_strength": (
            float(np.mean(active_strengths)) if active_strengths else 0.0
        ),
        "mean_routing_total_variation": float(np.mean(tv)),
    }


def risk_candidate(
    episode_rows: Sequence[Sequence[Mapping[str, Any]]],
    target: float,
    cutoff: float,
    kappa: float,
    routable_fraction: float,
) -> dict[str, Any]:
    strengths = []
    for episode in episode_rows:
        proposed = [
            float(row["risk_probability"])
            if float(row["risk_probability"]) >= cutoff
            else 0.0
            for row in episode
        ]
        strengths.append(pace_signal_strengths(proposed, target))
    return {
        "probability_cutoff": float(cutoff),
        "controller_target_rate": float(target),
        **budget_metrics(episode_rows, strengths, kappa, routable_fraction),
    }


def spread_candidate(
    episode_rows: Sequence[Sequence[Mapping[str, Any]]],
    target: float,
    threshold: float,
    signal_scale: float,
    kappa: float,
    routable_fraction: float,
) -> dict[str, Any]:
    strengths = []
    for episode in episode_rows:
        proposed = [
            spread_only_proposal_strength(
                float(row["spread_severity"]), threshold, signal_scale
            )
            for row in episode
        ]
        strengths.append(pace_signal_strengths(proposed, target))
    return {
        "severity_threshold": float(threshold),
        "signal_scale": float(signal_scale),
        "controller_target_rate": float(target),
        **budget_metrics(episode_rows, strengths, kappa, routable_fraction),
    }


def calibrate(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = episodes(rows)
    calibration = config["calibration"]
    kappa = float(config["routing_target_kappa"])
    fraction = float(config["routing_routable_fraction"])
    selections = {}
    all_candidates = {}
    calibration_pass = True
    for target_value in config["dose_targets"]:
        target = float(target_value)
        risk_candidates = [
            risk_candidate(grouped, target, float(cutoff), kappa, fraction)
            for cutoff in calibration["risk_probability_cutoff_grid"]
        ]
        risk_candidates.sort(
            key=lambda row: (
                abs(float(row["activation_rate"]) - target),
                -float(row["probability_cutoff"]),
            )
        )
        selected_risk = risk_candidates[0]
        risk_tv = float(selected_risk["mean_routing_total_variation"])
        spread_candidates = []
        for threshold in calibration["spread_severity_threshold_grid"]:
            for scale in calibration["spread_signal_scale_grid"]:
                candidate = spread_candidate(
                    grouped,
                    target,
                    float(threshold),
                    float(scale),
                    kappa,
                    fraction,
                )
                candidate["tv_ratio_to_risk"] = (
                    float(candidate["mean_routing_total_variation"]) / risk_tv
                    if risk_tv > 0.0
                    else 0.0
                )
                spread_candidates.append(candidate)
        spread_candidates.sort(
            key=lambda row: (
                abs(float(row["activation_rate"]) - target),
                abs(float(row["tv_ratio_to_risk"]) - 1.0),
                -float(row["severity_threshold"]),
                float(row["signal_scale"]),
            )
        )
        selected_spread = spread_candidates[0]
        risk_activation_error = abs(
            float(selected_risk["activation_rate"]) - target
        )
        spread_activation_error = abs(
            float(selected_spread["activation_rate"]) - target
        )
        tv_ratio = float(selected_spread["tv_ratio_to_risk"])
        passed = bool(
            risk_activation_error
            <= float(calibration["maximum_activation_error"])
            and spread_activation_error
            <= float(calibration["maximum_activation_error"])
            and float(calibration["minimum_tv_ratio"])
            <= tv_ratio
            <= float(calibration["maximum_tv_ratio"])
        )
        calibration_pass &= passed
        key = f"{target:.2f}"
        selections[key] = {
            "target": target,
            "risk_aware": selected_risk,
            "spread_only_budget_matched": selected_spread,
            "risk_activation_error": risk_activation_error,
            "spread_activation_error": spread_activation_error,
            "budget_match_status": "matched" if passed else "unmatched",
        }
        all_candidates[key] = {
            "risk_aware": risk_candidates,
            "spread_only_budget_matched": spread_candidates,
        }
    return {
        "calibration_status": "pass" if calibration_pass else "fail",
        "selected": selections,
        "candidates": all_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_dose_routing_sensitivity_v1.json",
    )
    args = parser.parse_args()
    config_path = REPO / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    results = []
    with ProcessPoolExecutor(max_workers=int(config["workers"])) as pool:
        futures = {
            pool.submit(collect_policy_seed, config, int(seed)): int(seed)
            for seed in config["source_policy_seeds"]
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: int(row["policy_seed"]))
    rows = [row for result in results for row in result["rows"]]
    expected = (
        len(config["source_policy_seeds"])
        * len(config["passive_calibration_seeds"])
        * int(config["episode_steps"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} passive records, got {len(rows)}")
    calibrated = calibrate(config, rows)
    artifact = {
        "protocol_version": str(config["protocol_version"]),
        "calibration_protocol_fingerprint": protocol_fingerprint(config),
        "calibration_config_sha256": sha256_file(config_path),
        "source_base_manifest_sha256": str(config["source_base_manifest_sha256"]),
        "gate_artifact_sha256": str(config["gate_artifact_sha256"]),
        "docker_image_id": resolve_docker_image_id(str(config["docker_image"])),
        "execution_commit": current_git_commit(),
        "created_at": utc_now(),
        "policy_seeds": list(config["source_policy_seeds"]),
        "passive_calibration_seeds": list(config["passive_calibration_seeds"]),
        "passive_record_count": len(rows),
        "passive_record_digest": digest(rows),
        "source_model_digests": {
            str(result["policy_seed"]): result["model_digest"] for result in results
        },
        **calibrated,
    }
    output = REPO / str(config["calibration"]["artifact"])
    atomic_write_json(output, artifact)
    print(
        json.dumps(
            {
                "artifact": str(output.relative_to(REPO)),
                "status": artifact["calibration_status"],
                "selected": artifact["selected"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if artifact["calibration_status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
