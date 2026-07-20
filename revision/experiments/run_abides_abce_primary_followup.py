#!/usr/bin/env python3
"""Checkpointed ABCE primary follow-up on frozen adaptive-v5 populations."""

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

from run_abides_mappo_adaptive_pilot import (  # noqa: E402
    atomic_torch_save,
    model_digest,
    trainer_from_payload,
    trainer_payload,
    validate_common_path,
)
from run_abides_mappo_adaptive_v3_response import (  # noqa: E402
    load_base,
    make_adapter,
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
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.agents.mappo import MAPPOTrainer  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.interventions.abides_abce import (  # noqa: E402
    ScheduleTransform,
    aligned_schedule,
    circular_schedule,
    exact_random_schedule,
    frozen_identity_permutation,
    identity_permutation_schedule,
    lagged_dealer_schedule,
    periodic_schedule,
    signal_multiset_digest,
)
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESPacedSignalController,
    ABIDESTwoStageGateModel,
)
from src.utils.abides_abce_audit import (  # noqa: E402
    aggregate_per_seed,
    audit_step_record,
    build_statistics,
    path_metrics,
)


METHODS = (
    "aligned_soft_rank",
    "exact_random_soft_rank",
    "circular_shifted_soft_rank",
    "periodic_exact_multiset_soft_rank",
    "lagged_revelation_soft_rank",
    "identity_permuted_soft_rank",
    "hard_best_aligned",
    "hard_best_exact_random",
    "no_routing",
)
FORMAL_NEW_METHODS = (
    "periodic_exact_multiset_soft_rank",
    "lagged_revelation_soft_rank",
    "identity_permuted_soft_rank",
    "hard_best_aligned",
    "hard_best_exact_random",
)
REUSED_METHODS = (
    "aligned_soft_rank",
    "exact_random_soft_rank",
    "circular_shifted_soft_rank",
    "no_routing",
)
CONTRASTS = {
    "A_exact_random_minus_aligned": (
        "exact_random_soft_rank",
        "aligned_soft_rank",
    ),
    "A_circular_minus_aligned": (
        "circular_shifted_soft_rank",
        "aligned_soft_rank",
    ),
    "A_periodic_minus_aligned": (
        "periodic_exact_multiset_soft_rank",
        "aligned_soft_rank",
    ),
    "B_lagged_minus_current": (
        "lagged_revelation_soft_rank",
        "aligned_soft_rank",
    ),
    "C1_identity_permuted_minus_quote_aligned": (
        "identity_permuted_soft_rank",
        "aligned_soft_rank",
    ),
    "C2_hard_best_random_minus_aligned": (
        "hard_best_exact_random",
        "hard_best_aligned",
    ),
}
HOLM_FAMILY = (
    "B_lagged_minus_current",
    "C1_identity_permuted_minus_quote_aligned",
    "C2_hard_best_random_minus_aligned",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_seed(namespace: int, environment_seed: int, label: str) -> int:
    payload = f"{int(namespace)}:{int(environment_seed)}:{label}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def source_bundle_sha256() -> str:
    paths = (
        "revision/src/interventions/abides_literature_baselines.py",
        "revision/infra/abides/ace_abides/adapter.py",
        "revision/infra/abides/ace_abides/economic_env.py",
        "revision/infra/abides/ace_abides/routing.py",
        "revision/infra/abides/worker.py",
    )
    digest = hashlib.sha256()
    for relative in paths:
        path = REPO / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def seed_dir(output: Path, seed: int) -> Path:
    return output / "policy_seeds" / f"seed_{int(seed)}"


def checkpoint_path(output: Path, seed: int) -> Path:
    return seed_dir(output, seed) / "training_state.pt"


def result_path(output: Path, seed: int) -> Path:
    return seed_dir(output, seed) / "result.json"


def step_path(output: Path, seed: int) -> Path:
    return seed_dir(output, seed) / "final_step_records.jsonl.gz"


def progress_path(output: Path, seed: int) -> Path:
    return seed_dir(output, seed) / "progress.jsonl"


def write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical_json(row) + "\n")
    temporary.replace(path)


def append_seed_progress(output: Path, seed: int, event: Mapping[str, Any]) -> None:
    append_logbook(
        progress_path(output, seed),
        {"policy_seed": int(seed), **dict(event)},
    )


def response_environment_seed(
    config: Mapping[str, Any], policy_seed: int, episode: int
) -> int:
    return (
        int(config["response_environment_seed_namespace"])
        + (int(policy_seed) % 100_000) * 1_000
        + int(episode)
    )


def audited_risk_decision(
    model: ABIDESTwoStageGateModel,
    controller: ABIDESPacedSignalController,
    gate_artifact_sha256: str,
) -> Callable[[Mapping[str, Any], int], Mapping[str, Any]]:
    def decide(state: Mapping[str, Any], step: int) -> Mapping[str, Any]:
        proposal = model.announced_strength(
            state,
            include_spread_severity=False,
        )
        before = float(controller.tokens)
        applied = float(controller.decide(float(proposal["strength"])))
        return {
            "applied_signal": applied,
            "raw_proposal": float(proposal["strength"]),
            "risk_probability": float(proposal["risk_probability"]),
            "probability_threshold": float(model.probability_threshold),
            "public_features": list(proposal["features"]),
            "normalization_artifact": gate_artifact_sha256,
            "token_balance_before": before,
            "token_balance_after": float(controller.tokens),
            "computed_before_action": True,
            "proposal_step": int(step),
        }

    return decide


def schedule_transforms(
    strengths: Sequence[float],
    environment_seed: int,
    config: Mapping[str, Any],
) -> dict[str, ScheduleTransform]:
    aligned = aligned_schedule(strengths)
    random = exact_random_schedule(
        strengths,
        stable_seed(
            int(config["signal_permutation_namespace"]),
            environment_seed,
            "exact_random",
        ),
    )
    circular = circular_schedule(
        strengths,
        stable_seed(
            int(config["signal_permutation_namespace"]),
            environment_seed,
            "circular",
        ),
    )
    periodic = periodic_schedule(strengths)
    return {
        "aligned_soft_rank": aligned,
        "exact_random_soft_rank": random,
        "circular_shifted_soft_rank": circular,
        "periodic_exact_multiset_soft_rank": periodic,
        "lagged_revelation_soft_rank": aligned,
        "identity_permuted_soft_rank": aligned,
        "hard_best_aligned": aligned,
        "hard_best_exact_random": random,
        "no_routing": aligned,
    }


def signal_audits(
    method: str,
    transform: ScheduleTransform,
    source_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    multiset_hash = signal_multiset_digest(transform.values)
    result = []
    for step, source_step in enumerate(transform.source_steps):
        current = dict(source_records[step].get("signal_audit", {}))
        result.append(
            {
                **current,
                "schedule_source": method,
                "scheduled_value_source_step": int(source_step),
                "signal_permutation_id": transform.transform_id,
                "circular_shift_id": (
                    transform.transform_id
                    if transform.transform_id.startswith("circular_shift:")
                    else None
                ),
                "signal_multiset_hash": multiset_hash,
                "computed_before_action": True,
            }
        )
    return result


def load_reused_trainers(
    config: Mapping[str, Any], policy_seed: int
) -> tuple[dict[str, MAPPOTrainer], dict[str, str]]:
    if not bool(config.get("reuse_existing_controls", False)):
        return {}, {}
    v5_output = REPO / str(config["v5_output_dir"])
    v5_checkpoint = (
        v5_output
        / "response_seeds"
        / f"seed_{int(policy_seed)}"
        / "training_state.pt"
    )
    v5_result_path = (
        v5_output / "response_results" / f"seed_{int(policy_seed)}.json"
    )
    v5_payload = torch.load(v5_checkpoint, map_location="cpu", weights_only=False)
    v5_result = json.loads(v5_result_path.read_text(encoding="utf-8"))
    if (
        v5_payload.get("protocol_fingerprint")
        != str(config["v5_protocol_fingerprint"])
        or v5_result.get("protocol_fingerprint")
        != str(config["v5_protocol_fingerprint"])
    ):
        raise RuntimeError(f"adaptive-v5 source mismatch for seed {policy_seed}")
    if int(v5_payload.get("completed_episodes", -1)) != int(
        config["source_response_episodes"]
    ):
        raise RuntimeError(f"adaptive-v5 source is incomplete for seed {policy_seed}")
    v5_keys = {
        "aligned_soft_rank": "risk_aware",
        "exact_random_soft_rank": "matched_random",
        "no_routing": "no_routing",
    }
    trainers = {
        method: trainer_from_payload(
            v5_payload["trainers"][payload_key], str(config["device"])
        )
        for method, payload_key in v5_keys.items()
    }
    expected_v5 = v5_result["final_model_digests"]
    for method, payload_key in v5_keys.items():
        if model_digest(trainers[method]) != str(expected_v5[payload_key]):
            raise RuntimeError(f"adaptive-v5 model digest mismatch: {method}")

    supplement_output = REPO / str(config["supplement_output_dir"])
    supplement_checkpoint = (
        supplement_output
        / "supplement_seeds"
        / f"seed_{int(policy_seed)}"
        / "training_state.pt"
    )
    supplement_result_path = (
        supplement_output
        / "supplement_results"
        / f"seed_{int(policy_seed)}.json"
    )
    supplement_payload = torch.load(
        supplement_checkpoint, map_location="cpu", weights_only=False
    )
    supplement_result = json.loads(
        supplement_result_path.read_text(encoding="utf-8")
    )
    if (
        supplement_payload.get("protocol_fingerprint")
        != str(config["supplement_protocol_fingerprint"])
        or supplement_result.get("protocol_fingerprint")
        != str(config["supplement_protocol_fingerprint"])
    ):
        raise RuntimeError(f"v5 supplement source mismatch for seed {policy_seed}")
    circular = trainer_from_payload(
        supplement_payload["trainers"]["circular_shifted"],
        str(config["device"]),
    )
    if model_digest(circular) != str(
        supplement_result["final_model_digests"]["circular_shifted"]
    ):
        raise RuntimeError("v5 circular model digest mismatch")
    trainers["circular_shifted_soft_rank"] = circular
    digests = {method: model_digest(trainer) for method, trainer in trainers.items()}
    return trainers, digests


def load_or_initialize(
    config: Mapping[str, Any],
    output: Path,
    policy_seed: int,
    fingerprint: str,
) -> tuple[
    dict[str, MAPPOTrainer],
    dict[str, MAPPOTrainer],
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
    str,
    dict[str, str],
    float,
]:
    reused, reused_digests = load_reused_trainers(config, policy_seed)
    checkpoint = checkpoint_path(output, policy_seed)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"ABCE checkpoint fingerprint mismatch: {checkpoint}")
        if dict(payload.get("reused_model_digests", {})) != reused_digests:
            raise RuntimeError("reused control models changed across ABCE resume")
        trained = {
            method: trainer_from_payload(data, str(config["device"]))
            for method, data in payload["trained_models"].items()
        }
        return (
            trained,
            reused,
            int(payload["completed_episodes"]),
            list(payload["training_history"]),
            list(payload["evaluation_history"]),
            dict(payload["integrity"]),
            str(payload["base_model_digest"]),
            reused_digests,
            float(payload.get("elapsed_seconds", 0.0)),
        )
    base = load_base(config, policy_seed)
    trained = {
        method: make_adapter(base, config, policy_seed)
        for method in config["trained_methods"]
    }
    if len({model_digest(trainer) for trainer in trained.values()}) != 1:
        raise RuntimeError("ABCE response adapters did not start identically")
    return (
        trained,
        reused,
        0,
        [],
        [],
        {
            "training_paths_equal": True,
            "training_action_uniforms_equal": True,
            "training_signal_multisets_exact": True,
            "training_identity_weight_multisets_exact": True,
            "training_lagged_semantics_exact": True,
            "all_updates_finite": True,
        },
        model_digest(base),
        reused_digests,
        0.0,
    )


def save_checkpoint(
    config: Mapping[str, Any],
    output: Path,
    policy_seed: int,
    fingerprint: str,
    trained: Mapping[str, MAPPOTrainer],
    completed: int,
    training_history: Sequence[Mapping[str, Any]],
    evaluation_history: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, bool],
    base_digest: str,
    reused_digests: Mapping[str, str],
    elapsed_seconds: float,
) -> None:
    atomic_torch_save(
        checkpoint_path(output, policy_seed),
        {
            "protocol_fingerprint": fingerprint,
            "policy_seed": int(policy_seed),
            "completed_episodes": int(completed),
            "trained_models": {
                method: trainer_payload(trainer)
                for method, trainer in trained.items()
            },
            "training_history": list(training_history),
            "evaluation_history": list(evaluation_history),
            "integrity": dict(integrity),
            "base_model_digest": base_digest,
            "reused_model_digests": dict(reused_digests),
            "elapsed_seconds": float(elapsed_seconds),
            "saved_at": utc_now(),
        },
    )


def all_trainers(
    trained: Mapping[str, MAPPOTrainer], reused: Mapping[str, MAPPOTrainer]
) -> dict[str, MAPPOTrainer]:
    overlap = set(trained).intersection(reused)
    if overlap:
        raise RuntimeError(f"ABCE trained/reused methods overlap: {sorted(overlap)}")
    result = {**dict(reused), **dict(trained)}
    if set(result) != set(METHODS):
        raise RuntimeError(
            f"ABCE model grid mismatch: expected {METHODS}, got {sorted(result)}"
        )
    return result


def arm_rollout(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    environment_seed: int,
    method: str,
    trainer: MAPPOTrainer,
    transform: ScheduleTransform,
    audits: Sequence[Mapping[str, Any]],
    train: bool,
) -> dict[str, Any]:
    routing_enabled = method != "no_routing"
    dealer_schedule: Sequence[float] | None = None
    if method == "lagged_revelation_soft_rank":
        dealer_schedule = lagged_dealer_schedule(transform.values)
    mechanism: Mapping[str, Any] | None = {"kind": "soft_rank"}
    mechanism_schedule: Sequence[Mapping[str, Any]] | None = None
    if method == "identity_permuted_soft_rank":
        permutations = identity_permutation_schedule(
            int(config["market"]["n_agents"]),
            environment_seed,
            int(config["episode_steps"]),
            int(config["allocation_permutation_namespace"]),
        )
        mechanism = None
        mechanism_schedule = [
            {
                "kind": "identity_permuted_soft_rank",
                "identity_permutation": list(permutation),
            }
            for permutation in permutations
        ]
    elif method.startswith("hard_best_"):
        mechanism = {"kind": "hard_best"}
    return rollout_episode(
        env,
        trainer,
        seed=int(environment_seed),
        episode_steps=int(config["episode_steps"]),
        train=train,
        defer_update=train,
        announced_signal_schedule=transform.values,
        dealer_signal_schedule=dealer_schedule,
        signal_audit_schedule=audits,
        routing_enabled=routing_enabled,
        routing_mechanism=mechanism,
        routing_mechanism_schedule=mechanism_schedule,
        reward_scale=float(config["reward_scale"]),
        action_namespace=int(
            config[
                "response_action_namespace"
                if train
                else "evaluation_action_namespace"
            ]
        ),
    )


def action_uniforms_equal(rollouts: Mapping[str, Mapping[str, Any]]) -> bool:
    schedules = [
        [str(row["action_uniform_digest"]) for row in rollout["records"]]
        for rollout in rollouts.values()
    ]
    return all(schedule == schedules[0] for schedule in schedules[1:])


def rollout_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    environment_seed: int,
    trained: Mapping[str, MAPPOTrainer],
    reused: Mapping[str, MAPPOTrainer],
    gate_model: ABIDESTwoStageGateModel,
    train: bool,
    selected_methods: Sequence[str],
) -> tuple[dict[str, Any], dict[str, bool]]:
    trainers = all_trainers(trained, reused)
    controller = ABIDESPacedSignalController(
        int(config["episode_steps"]),
        target_rate=float(config["max_activation_rate"]),
    )
    source_is_trained = train and "aligned_soft_rank" in trained
    source = rollout_episode(
        env,
        trainers["aligned_soft_rank"],
        seed=int(environment_seed),
        episode_steps=int(config["episode_steps"]),
        train=source_is_trained,
        defer_update=source_is_trained,
        announced_signal_decision=audited_risk_decision(
            gate_model,
            controller,
            str(config["gate_artifact_sha256"]),
        ),
        routing_mechanism={"kind": "soft_rank"},
        reward_scale=float(config["reward_scale"]),
        action_namespace=int(
            config[
                "response_action_namespace"
                if train
                else "evaluation_action_namespace"
            ]
        ),
    )
    strengths = [float(row["platform_applied_signal"]) for row in source["records"]]
    transforms = schedule_transforms(strengths, environment_seed, config)
    aligned_audits = signal_audits(
        "aligned_soft_rank", transforms["aligned_soft_rank"], source["records"]
    )
    for row, audit in zip(source["records"], aligned_audits):
        row["signal_audit"] = audit
    rollouts: dict[str, Any] = {}
    for method in selected_methods:
        if method == "aligned_soft_rank":
            rollouts[method] = source
            continue
        rollouts[method] = arm_rollout(
            env,
            config,
            environment_seed,
            method,
            trainers[method],
            transforms[method],
            signal_audits(method, transforms[method], source["records"]),
            train=train and method in trained,
        )
    paths_equal = all(
        validate_common_path(source, rollout) for rollout in rollouts.values()
    )
    multisets_exact = all(
        transforms[method].exact_multiset_of(strengths)
        for method in selected_methods
    )
    lagged_exact = True
    if "lagged_revelation_soft_rank" in rollouts:
        records = rollouts["lagged_revelation_soft_rank"]["records"]
        applied = [float(row["platform_applied_signal"]) for row in records]
        observed = [float(row["dealer_observed_signal"]) for row in records]
        lagged_exact = observed == list(lagged_dealer_schedule(applied))
    identity_exact = True
    if "identity_permuted_soft_rank" in rollouts:
        identity_exact = all(
            bool(
                np.allclose(
                    np.sort(row["info"]["pre_permutation_routing_weights"]),
                    np.sort(row["info"]["routing_weights"]),
                    atol=1e-12,
                    rtol=0.0,
                )
            )
            for row in rollouts["identity_permuted_soft_rank"]["records"]
        )
    return rollouts, {
        "paths_equal": bool(paths_equal),
        "action_uniforms_equal": bool(
            action_uniforms_equal({"__source": source, **rollouts})
        ),
        "signal_multisets_exact": bool(multisets_exact),
        "identity_weight_multisets_exact": bool(identity_exact),
        "lagged_semantics_exact": bool(lagged_exact),
    }


def model_digests(
    trained: Mapping[str, MAPPOTrainer], reused: Mapping[str, MAPPOTrainer]
) -> dict[str, str]:
    return {
        method: model_digest(trainer)
        for method, trainer in all_trainers(trained, reused).items()
    }


def evaluate_methods(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
    trained: Mapping[str, MAPPOTrainer],
    reused: Mapping[str, MAPPOTrainer],
    gate_model: ABIDESTwoStageGateModel,
    evaluation_seeds: Sequence[int],
    selected_methods: Sequence[str],
    protocol_fingerprint_value: str,
    runtime: Mapping[str, str],
    base_digest: str,
    collect_steps: bool,
) -> dict[str, Any]:
    digests = model_digests(trained, reused)
    per_path = []
    step_rows = []
    checks = []
    for evaluation_seed in evaluation_seeds:
        rollouts, path_checks = rollout_methods(
            env,
            config,
            int(evaluation_seed),
            trained,
            reused,
            gate_model,
            train=False,
            selected_methods=selected_methods,
        )
        checks.append(path_checks)
        for method in selected_methods:
            records = rollouts[method]["records"]
            per_path.append(
                path_metrics(
                    str(config["protocol_version"]),
                    policy_seed,
                    int(evaluation_seed),
                    method,
                    digests[method],
                    base_digest,
                    records,
                )
            )
            if collect_steps:
                step_rows.extend(
                    audit_step_record(
                        str(config["protocol_version"]),
                        protocol_fingerprint_value,
                        str(runtime["git_commit"]),
                        str(runtime["docker_image_id"]),
                        policy_seed,
                        int(evaluation_seed),
                        method,
                        digests[method],
                        base_digest,
                        row,
                    )
                    for row in records
                )
    return {
        "per_path": per_path,
        "step_records": step_rows,
        "paths_equal": all(row["paths_equal"] for row in checks),
        "action_uniforms_equal": all(
            row["action_uniforms_equal"] for row in checks
        ),
        "signal_multisets_exact": all(
            row["signal_multisets_exact"] for row in checks
        ),
        "identity_weight_multisets_exact": all(
            row["identity_weight_multisets_exact"] for row in checks
        ),
        "lagged_semantics_exact": all(
            row["lagged_semantics_exact"] for row in checks
        ),
    }


def convergence(
    history: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    required = int(config["response_required_stable_evaluations"])
    if len(history) < required:
        return {"stable": False, "reason": "insufficient fixed-panel evaluations"}
    recent = list(history[-required:])
    ranges = {}
    for method in config["trained_methods"]:
        values = []
        for item in recent:
            rows = [
                row
                for row in item["evaluation"]["per_path"]
                if row["method"] == method
            ]
            values.append(mean(float(row["low_execution_half_spread"]) for row in rows))
        ranges[method] = max(values) - min(values)
    maximum = float(config["response_maximum_fixed_panel_spread_range"])
    return {
        "stable": all(value <= maximum for value in ranges.values()),
        "spread_ranges": ranges,
        "method_stability": {
            method: value <= maximum for method, value in ranges.items()
        },
        "evaluation_episodes": [
            int(item["completed_episodes"]) for item in recent
        ],
    }


def gate_off_recovery(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    policy_seed: int,
) -> bool:
    base = load_base(config, policy_seed)
    evaluation_seed = int(config["convergence_evaluation_seeds"][0])
    schedule = [0.0] * int(config["episode_steps"])
    common = {
        "seed": evaluation_seed,
        "episode_steps": int(config["episode_steps"]),
        "train": False,
        "announced_signal_schedule": schedule,
        "reward_scale": float(config["reward_scale"]),
        "action_namespace": int(config["evaluation_action_namespace"]),
    }
    base_rollout = rollout_episode(
        env,
        base,
        routing_enabled=True,
        routing_mechanism={"kind": "soft_rank"},
        **common,
    )
    mechanisms: list[tuple[bool, Mapping[str, Any]]] = [
        (True, {"kind": "soft_rank"}),
        (True, {"kind": "hard_best"}),
        (
            True,
            {
                "kind": "identity_permuted_soft_rank",
                "identity_permutation": list(
                    frozen_identity_permutation(
                        int(config["market"]["n_agents"]),
                        evaluation_seed,
                        0,
                        int(config["allocation_permutation_namespace"]),
                    )
                ),
            },
        ),
        (False, {"kind": "soft_rank"}),
    ]

    def economic_signature(rollout: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "actions": row["actions"],
                "rewards": row["rewards_unscaled"],
                "post_observations": row["post_observations"],
                "executions": row["info"]["execution_records"],
                "flow_shares": row["info"]["flow_shares"],
                "customer_cost": row["info"][
                    "customer_execution_cost_tick_units"
                ],
            }
            for row in rollout["records"]
        ]

    expected = economic_signature(base_rollout)
    for routing_enabled, mechanism in mechanisms:
        rollout = rollout_episode(
            env,
            base,
            routing_enabled=routing_enabled,
            routing_mechanism=mechanism,
            **common,
        )
        if economic_signature(rollout) != expected:
            return False
    return True


def train_seed(
    policy_seed: int,
    config: dict[str, Any],
    output_raw: str,
    fingerprint: str,
    runtime: dict[str, str],
) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    output = Path(output_raw)
    existing_result = result_path(output, policy_seed)
    if existing_result.exists():
        payload = json.loads(existing_result.read_text(encoding="utf-8"))
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"ABCE result fingerprint mismatch: {existing_result}")
        return payload
    gate_model = ABIDESTwoStageGateModel.load(REPO / str(config["gate_artifact"]))
    (
        trained,
        reused,
        completed,
        training_history,
        evaluation_history,
        integrity,
        base_digest,
        reused_digests,
        prior_elapsed_seconds,
    ) = load_or_initialize(config, output, policy_seed, fingerprint)
    env_config = market_config(config)
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(env_config, docker) as env:
        while completed < int(config["response_episodes"]):
            batch = min(
                int(config["response_episodes_per_update"]),
                int(config["response_episodes"]) - completed,
            )
            trajectories = {method: [] for method in config["trained_methods"]}
            batch_spreads = {method: [] for method in config["trained_methods"]}
            batch_rewards = {method: [] for method in config["trained_methods"]}
            for offset in range(batch):
                episode = completed + offset
                environment_seed = response_environment_seed(
                    config, policy_seed, episode
                )
                rollouts, checks = rollout_methods(
                    env,
                    config,
                    environment_seed,
                    trained,
                    reused,
                    gate_model,
                    train=True,
                    selected_methods=config["trained_methods"],
                )
                for key, value in checks.items():
                    integrity[f"training_{key}"] = bool(
                        integrity.get(f"training_{key}", True) and value
                    )
                for method in config["trained_methods"]:
                    trajectories[method].append(rollouts[method]["trajectory"])
                    batch_spreads[method].append(
                        float(rollouts[method]["mean_half_spread"])
                    )
                    batch_rewards[method].append(float(rollouts[method]["mean_reward"]))
            updates = {
                method: trained[method].update_episodes(trajectories[method])
                for method in config["trained_methods"]
            }
            finite_updates = all(
                math.isfinite(float(value))
                for update in updates.values()
                for value in update.values()
            )
            integrity["all_updates_finite"] = bool(
                integrity.get("all_updates_finite", True) and finite_updates
            )
            completed += batch
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
                        for method in config["trained_methods"]
                    },
                }
            )
            if (
                completed >= int(config["response_evaluation_start"])
                and completed % int(config["response_evaluation_interval"]) == 0
            ):
                evaluation = evaluate_methods(
                    env,
                    config,
                    policy_seed,
                    trained,
                    reused,
                    gate_model,
                    config["convergence_evaluation_seeds"],
                    config["trained_methods"],
                    fingerprint,
                    runtime,
                    base_digest,
                    collect_steps=False,
                )
                evaluation_history.append(
                    {"completed_episodes": completed, "evaluation": evaluation}
                )
            save_checkpoint(
                config,
                output,
                policy_seed,
                fingerprint,
                trained,
                completed,
                training_history,
                evaluation_history,
                integrity,
                base_digest,
                reused_digests,
                prior_elapsed_seconds + time.perf_counter() - started,
            )
            append_seed_progress(
                output,
                policy_seed,
                {
                    "event": "response_checkpoint",
                    "completed_episodes": completed,
                    "stable": convergence(evaluation_history, config)["stable"],
                    "elapsed_seconds": prior_elapsed_seconds
                    + time.perf_counter()
                    - started,
                },
            )
        final = evaluate_methods(
            env,
            config,
            policy_seed,
            trained,
            reused,
            gate_model,
            config["evaluation_seeds"],
            METHODS,
            fingerprint,
            runtime,
            base_digest,
            collect_steps=True,
        )
        gate_off_exact = gate_off_recovery(env, config, policy_seed)
    write_jsonl_gz(step_path(output, policy_seed), final.pop("step_records"))
    result = {
        "protocol_fingerprint": fingerprint,
        "policy_seed": int(policy_seed),
        "completed_episodes": int(completed),
        "base_model_digest": base_digest,
        "reused_model_digests": reused_digests,
        "final_model_digests": model_digests(trained, reused),
        "training_integrity": integrity,
        "convergence": convergence(evaluation_history, config),
        "evaluation_history": evaluation_history,
        "gate_off_recovery_exact": bool(gate_off_exact),
        "final_evaluation": final,
        "step_records_file": str(step_path(output, policy_seed).relative_to(REPO)),
        "step_records_sha256": sha256_file(step_path(output, policy_seed)),
        "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
    }
    atomic_write_json(existing_result, result)
    append_seed_progress(
        output,
        policy_seed,
        {"event": "policy_seed_complete", "elapsed_seconds": result["elapsed_seconds"]},
    )
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        canonical_json(value)
                        if isinstance(value, (list, dict, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def aggregate_results(
    config: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    per_path = [
        row
        for result in results
        for row in result["final_evaluation"]["per_path"]
    ]
    per_seed = aggregate_per_seed(per_path, METHODS)
    statistics = build_statistics(
        per_path,
        METHODS,
        CONTRASTS,
        HOLM_FAMILY,
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]),
    )
    expected = len(config["policy_seeds"])
    converged = sum(bool(row["convergence"]["stable"]) for row in results)
    checks = {
        "exact_policy_seed_grid": len(results) == expected
        and {int(row["policy_seed"]) for row in results}
        == {int(seed) for seed in config["policy_seeds"]},
        "fixed_response_budget": all(
            int(row["completed_episodes"]) == int(config["response_episodes"])
            for row in results
        ),
        "minimum_common_converged_seeds": converged
        >= int(config["minimum_converged_policy_seeds"]),
        "training_integrity": all(
            all(bool(value) for value in row["training_integrity"].values())
            for row in results
        ),
        "evaluation_paths_equal": all(
            bool(row["final_evaluation"]["paths_equal"]) for row in results
        ),
        "evaluation_action_uniforms_equal": all(
            bool(row["final_evaluation"]["action_uniforms_equal"])
            for row in results
        ),
        "exact_signal_multisets": all(
            bool(row["final_evaluation"]["signal_multisets_exact"])
            for row in results
        ),
        "exact_identity_weight_multisets": all(
            bool(row["final_evaluation"]["identity_weight_multisets_exact"])
            for row in results
        ),
        "lagged_revelation_semantics": all(
            bool(row["final_evaluation"]["lagged_semantics_exact"])
            for row in results
        ),
        "gate_off_recovery": all(
            bool(row["gate_off_recovery_exact"]) for row in results
        ),
        "finite_metrics": all(bool(row["finite"]) for row in per_path),
        "volume_conserved": all(bool(row["volume_conserved"]) for row in per_path),
        "step_record_hashes_present": all(
            len(str(row["step_records_sha256"])) == 64 for row in results
        ),
        "per_path_grid": len(per_path)
        == expected * len(config["evaluation_seeds"]) * len(METHODS),
    }
    a_primary = statistics["contrasts"]["A_exact_random_minus_aligned"][
        "t_interval"
    ]
    a_supported = bool(
        float(a_primary["ci_low"]) > 0.0
        and int(a_primary["positive"])
        >= int(config["minimum_A_positive_policy_seeds"])
    )
    classifications = {"A": "supported" if a_supported else "not_supported"}
    for axis, contrast in (
        ("B", "B_lagged_minus_current"),
        ("C1", "C1_identity_permuted_minus_quote_aligned"),
        ("C2", "C2_hard_best_random_minus_aligned"),
    ):
        interval = statistics["contrasts"][contrast]["t_interval"]
        if float(interval["ci_low"]) > 0.0:
            classification = "supported"
        elif float(interval["ci_high"]) < 0.0:
            classification = "reversed"
        elif float(interval["mean"]) > 0.0:
            classification = "directional_inconclusive"
        else:
            classification = "null_or_inconclusive"
        classifications[axis] = classification
    if not all(checks.values()):
        classifications = {
            key: "descriptive_only" for key in classifications
        }
    if bool(config.get("engineering_smoke", False)):
        classifications = {key: "smoke_only" for key in classifications}
    return {
        "implementation_status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "converged_policy_seeds": converged,
        "classifications": classifications,
        "statistics": statistics,
        "per_path": per_path,
        "per_seed": per_seed,
        "claim_boundary": (
            "ABCE effects are confirmatory only when every integrity gate passes; "
            "null and reversed formal cells narrow the claim without further search"
        ),
    }


def logbook_markdown(
    config: Mapping[str, Any], results: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]
) -> str:
    completed = {int(row["policy_seed"]): row for row in results}
    failed = {int(row["policy_seed"]): row for row in failures}
    lines = [
        "# ABCE Primary Follow-up Logbook",
        "",
        f"Protocol: `{config['protocol_version']}`",
        "",
        "| Policy seed | Status | Episodes | Stable | Seconds |",
        "| ---: | --- | ---: | --- | ---: |",
    ]
    for seed_raw in config["policy_seeds"]:
        seed = int(seed_raw)
        if seed in completed:
            row = completed[seed]
            lines.append(
                f"| {seed} | complete | {int(row['completed_episodes'])} | "
                f"{bool(row['convergence']['stable'])} | "
                f"{float(row['elapsed_seconds']):.2f} |"
            )
        elif seed in failed:
            lines.append(f"| {seed} | failed | - | - | - |")
        else:
            checkpoint = checkpoint_path(REPO / str(config["output_dir"]), seed)
            episodes = 0
            if checkpoint.exists():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                episodes = int(payload.get("completed_episodes", 0))
            lines.append(f"| {seed} | pending | {episodes} | - | - |")
    return "\n".join(lines) + "\n"


def docker_label(image_id: str, name: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--type",
            "image",
            "--format",
            f"{{{{ index .Config.Labels {json.dumps(name)} }}}}",
            image_id,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def disallowed_worktree_entries(
    status_lines: Sequence[str], allowed_untracked_root: str | None
) -> list[str]:
    allowed = None if allowed_untracked_root is None else allowed_untracked_root.rstrip("/")
    result = []
    for line in status_lines:
        if not line:
            continue
        path = line[3:]
        allowed_untracked = (
            allowed is not None
            and line.startswith("?? ")
            and (path == allowed or path.startswith(f"{allowed}/"))
        )
        if not allowed_untracked:
            result.append(line)
    return result


def formal_worktree_violations(allowed_untracked_root: str | None) -> list[str]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return disallowed_worktree_entries(status, allowed_untracked_root)


def validate_inputs(config: dict[str, Any], config_path: Path) -> dict[str, str]:
    if tuple(config["methods"]) != METHODS:
        raise ValueError(f"ABCE method order must be {METHODS}")
    trained = tuple(config["trained_methods"])
    if bool(config.get("formal", False)) and bool(
        config.get("reuse_existing_controls", False)
    ):
        raise ValueError("formal ABCE must retrain every response arm from base")
    if bool(config.get("reuse_existing_controls", False)):
        if trained != FORMAL_NEW_METHODS:
            raise ValueError(f"reused-control ABCE new methods must be {FORMAL_NEW_METHODS}")
    elif set(trained) != set(METHODS):
        raise ValueError("ABCE without reused controls must train the complete grid")
    if int(config["episode_steps"]) != 100:
        raise ValueError("ABCE primary horizon must remain 100 decisions")
    if int(config["response_episodes"]) != 96 and not bool(
        config.get("engineering_smoke", False)
    ):
        raise ValueError("formal ABCE response budget must remain 96 episodes")
    if bool(config.get("formal", False)):
        if str(config.get("freeze_status")) != "frozen_pre_outcome":
            raise RuntimeError("formal ABCE config is not frozen pre-outcome")
        if list(config["policy_seeds"]) != list(range(530000, 530020)):
            raise ValueError("formal ABCE policy seed namespace changed")
        if list(config["evaluation_seeds"]) != list(range(534000, 534010)):
            raise ValueError("formal ABCE final path namespace changed")
        violations = formal_worktree_violations(
            str(config.get("managed_artifact_output", "")) or None
        )
        if violations:
            raise RuntimeError("formal ABCE refuses to run from a dirty worktree")
        for tracked in (
            config_path,
            REPO / str(config["protocol_file"]),
            REPO / str(config["execution_amendment_file"]),
        ):
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(tracked.relative_to(REPO))],
                cwd=REPO,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    artifacts = {}
    for path_key, hash_key in (
        ("gate_artifact", "gate_artifact_sha256"),
        ("protocol_file", "protocol_file_sha256"),
        ("execution_amendment_file", "execution_amendment_file_sha256"),
    ):
        path = REPO / str(config[path_key])
        actual = sha256_file(path)
        if actual != str(config[hash_key]):
            raise RuntimeError(f"ABCE frozen artifact mismatch: {path}")
        artifacts[hash_key] = actual
    bundle_sha = source_bundle_sha256()
    if bundle_sha != str(config["adapter_source_bundle_sha256"]):
        raise RuntimeError("ABCE host adapter source bundle changed")
    image_id = resolve_docker_image_id(str(config["docker_image"]))
    expected_image_id = str(config["docker_image_id"])
    if image_id != expected_image_id:
        raise RuntimeError(
            f"ABCE Docker image changed: expected {expected_image_id}, got {image_id}"
        )
    if docker_label(
        image_id, "org.opencontainers.image.ace_rl_abce_source_sha256"
    ) != bundle_sha:
        raise RuntimeError("ABCE Docker adapter source label mismatch")
    parent_id = resolve_docker_image_id(str(config["parent_docker_image"]))
    if parent_id != str(config["parent_docker_image_id"]):
        raise RuntimeError("ABCE parent adaptive-v3 image changed")
    artifacts.update(
        {
            "adapter_source_bundle_sha256": bundle_sha,
            "docker_image_id": image_id,
            "parent_docker_image_id": parent_id,
        }
    )
    return artifacts


def load_completed(
    config: Mapping[str, Any], output: Path, fingerprint: str
) -> list[dict[str, Any]]:
    rows = []
    for seed in config["policy_seeds"]:
        path = result_path(output, int(seed))
        if not path.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"completed ABCE result fingerprint mismatch: {path}")
        rows.append(row)
    return rows


def pending_policy_seeds(
    configured_seeds: Sequence[int], completed_results: Sequence[Mapping[str, Any]]
) -> list[int]:
    completed = {int(row["policy_seed"]) for row in completed_results}
    seeds = [int(seed) for seed in configured_seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("ABCE policy seed list contains duplicates")
    return [seed for seed in seeds if seed not in completed]


def persist_state(
    config: Mapping[str, Any],
    output: Path,
    results: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    status: str,
) -> None:
    completed = {int(row["policy_seed"]) for row in results}
    atomic_write_json(
        output / "run_state.json",
        {
            "status": status,
            "completed_policy_seeds": sorted(completed),
            "pending_policy_seeds": [
                int(seed) for seed in config["policy_seeds"] if int(seed) not in completed
            ],
            "failures": list(failures),
            "updated_at": utc_now(),
        },
    )
    atomic_write_text(
        output / "LOGBOOK.md",
        logbook_markdown(config, results, failures),
    )


def execute(config: dict[str, Any], config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    output = REPO / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = protocol_fingerprint(config)
    artifacts = validate_inputs(config, config_path)
    runtime = {
        "git_commit": current_git_commit(),
        "docker_image_id": artifacts["docker_image_id"],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol_fingerprint") != fingerprint
            or manifest.get("git_commit") != runtime["git_commit"]
            or manifest.get("artifacts") != artifacts
        ):
            raise RuntimeError("ABCE resume manifest mismatch")
    else:
        atomic_write_json(
            manifest_path,
            {
                "protocol_version": str(config["protocol_version"]),
                "protocol_fingerprint": fingerprint,
                "config": config,
                "git_commit": runtime["git_commit"],
                "artifacts": artifacts,
                "formal": bool(config.get("formal", False)),
                "started_at": utc_now(),
            },
        )
    with RunLock(output / "run.lock", fingerprint, args.force_unlock):
        results = load_completed(config, output, fingerprint)
        failures: list[dict[str, Any]] = []
        pending = pending_policy_seeds(config["policy_seeds"], results)
        append_logbook(
            output / "logbook.jsonl",
            {
                "event": "run_resumed" if results else "run_started",
                "completed": len(results),
                "pending": len(pending),
                "protocol_fingerprint": fingerprint,
            },
        )
        persist_state(config, output, results, failures, "running")
        if args.preflight_only:
            return {
                "status": "preflight_pass",
                "protocol_fingerprint": fingerprint,
                "artifacts": artifacts,
                "pending_policy_seeds": pending,
            }
        workers = min(int(args.workers or config["workers"]), max(len(pending), 1))
        if pending:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        train_seed,
                        seed,
                        config,
                        str(output),
                        fingerprint,
                        runtime,
                    ): seed
                    for seed in pending
                }
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        row = future.result()
                        results.append(row)
                        append_logbook(
                            output / "logbook.jsonl",
                            {
                                "event": "policy_seed_complete",
                                "policy_seed": seed,
                                "completed": len(results),
                                "total": len(config["policy_seeds"]),
                                "elapsed_seconds": row["elapsed_seconds"],
                            },
                        )
                    except BaseException as exc:
                        failure = {
                            "policy_seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        append_logbook(
                            output / "logbook.jsonl",
                            {"event": "policy_seed_failed", **failure},
                        )
                    persist_state(config, output, results, failures, "running")
        results = sorted(results, key=lambda row: int(row["policy_seed"]))
        if failures or len(results) != len(config["policy_seeds"]):
            persist_state(config, output, results, failures, "incomplete")
            atomic_write_json(output / "failures.json", failures)
            raise RuntimeError(
                f"ABCE run incomplete: {len(results)} results, {len(failures)} failures"
            )
        aggregate = aggregate_results(config, results)
        write_csv(output / "per_path.csv", aggregate.pop("per_path"))
        write_csv(output / "per_seed.csv", aggregate.pop("per_seed"))
        atomic_write_json(output / "statistics.json", aggregate["statistics"])
        atomic_write_json(output / "summary.json", aggregate)
        status = str(aggregate["implementation_status"])
        persist_state(config, output, results, [], status)
        append_logbook(
            output / "logbook.jsonl",
            {"event": "run_finished", "status": status},
        )
        return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="revision/configs/abides_abce_primary_followup_smoke_v1.json",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (REPO / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = execute(config, config_path, args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
