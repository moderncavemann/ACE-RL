#!/usr/bin/env python3
"""Checkpointed ABIDES dose-frontier and routing-sensitivity runner."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
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

from run_abides_mappo_adaptive_pilot import (  # noqa: E402
    atomic_torch_save,
    model_digest,
    trainer_from_payload,
    trainer_payload,
    validate_common_path,
)
from run_abides_mappo_adaptive_v3_full_main import global_shuffle  # noqa: E402
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
from run_abides_domain_benchmark import audit_step_record, path_metrics  # noqa: E402
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.agents.mappo import MAPPOTrainer  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESPacedSignalController,
    ABIDESTwoStageGateModel,
    FEATURE_NAMES,
    abides_gate_features,
    abides_lagged_actions,
)
from src.interventions.abides_literature_baselines import (  # noqa: E402
    spread_only_proposal_strength,
)
from src.utils.abides_audit import exact_sign_test, t_interval  # noqa: E402


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dose_key(value: float) -> str:
    return f"{float(value):.2f}"


def dose_group_id(value: float) -> str:
    return f"dose_{int(round(float(value) * 100)):03d}"


def routing_group_id(kappa: float, fraction: float) -> str:
    return f"routing_k{float(kappa):g}_f{int(round(float(fraction) * 100)):03d}"


def group_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = []
    calibrated = config["calibrated_dose_parameters"]
    for target_value in config["dose_targets"]:
        target = float(target_value)
        specs.append(
            {
                "group_id": dose_group_id(target),
                "study": "dose",
                "dose_target": target,
                "routing_target_kappa": float(config["routing_target_kappa"]),
                "routing_routable_fraction": float(
                    config["routing_routable_fraction"]
                ),
                "methods": list(config["dose_methods"]),
                "risk_parameters": dict(calibrated[dose_key(target)]["risk_aware"]),
                "spread_parameters": dict(
                    calibrated[dose_key(target)][
                        "spread_only_budget_matched"
                    ]
                ),
            }
        )
    routing_target = float(config["routing_activation_target"])
    for setting in config["routing_settings"]:
        kappa = float(setting["routing_target_kappa"])
        fraction = float(setting["routing_routable_fraction"])
        specs.append(
            {
                "group_id": routing_group_id(kappa, fraction),
                "study": "routing",
                "dose_target": routing_target,
                "routing_target_kappa": kappa,
                "routing_routable_fraction": fraction,
                "methods": list(config["routing_methods"]),
                "risk_parameters": dict(
                    calibrated[dose_key(routing_target)]["risk_aware"]
                ),
                "spread_parameters": None,
            }
        )
    identifiers = [spec["group_id"] for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("sensitivity group identifiers are not unique")
    return specs


def group_dir(output_dir: Path, group_id: str, policy_seed: int) -> Path:
    return output_dir / "groups" / group_id / f"seed_{policy_seed}"


def checkpoint_path(output_dir: Path, group_id: str, policy_seed: int) -> Path:
    return group_dir(output_dir, group_id, policy_seed) / "training_state.pt"


def result_path(output_dir: Path, group_id: str, policy_seed: int) -> Path:
    return group_dir(output_dir, group_id, policy_seed) / "result.json"


def step_records_path(output_dir: Path, group_id: str, policy_seed: int) -> Path:
    return group_dir(output_dir, group_id, policy_seed) / "evaluation_steps.jsonl.gz"


def base_loader_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_output_dir": str(config["source_base_output_dir"]),
        "base_protocol_fingerprint": str(config["source_base_fingerprint"]),
        "device": str(config["device"]),
    }


def source_result(config: Mapping[str, Any], policy_seed: int) -> dict[str, Any]:
    path = (
        REPO
        / str(config["source_base_output_dir"])
        / "base_results"
        / f"seed_{policy_seed}.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["protocol_fingerprint"] != str(config["source_base_fingerprint"]):
        raise RuntimeError(f"source base result fingerprint mismatch: {path}")
    return result


def source_checkpoint(config: Mapping[str, Any], policy_seed: int) -> Path:
    return (
        REPO
        / str(config["source_base_output_dir"])
        / "base_seeds"
        / f"seed_{policy_seed}"
        / "training_state.pt"
    )


def response_environment_seed(
    config: Mapping[str, Any], policy_seed: int, episode: int
) -> int:
    return (
        int(config["response_environment_seed_namespace"])
        + (int(policy_seed) % 100_000) * 1_000
        + int(episode)
    )


def action_uniforms_equal(rollouts: Mapping[str, Mapping[str, Any]]) -> bool:
    schedules = [
        [record["action_uniform_digest"] for record in rollout["records"]]
        for rollout in rollouts.values()
    ]
    return all(schedule == schedules[0] for schedule in schedules[1:])


def exact_signal_multiset(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(
        np.allclose(
            np.sort(np.asarray(left, dtype=float)),
            np.sort(np.asarray(right, dtype=float)),
            atol=1e-12,
            rtol=0.0,
        )
    )


def risk_signal_decision(
    model: ABIDESTwoStageGateModel,
    episode_steps: int,
    parameters: Mapping[str, Any],
):
    controller = ABIDESPacedSignalController(
        episode_steps,
        target_rate=float(parameters["controller_target_rate"]),
    )
    cutoff = float(parameters["probability_cutoff"])

    def decide(state: Mapping[str, Any], step: int) -> float:
        diagnostic = model.announced_strength(
            state, include_spread_severity=False
        )
        probability = float(diagnostic["risk_probability"])
        proposal = probability if probability >= cutoff else 0.0
        return controller.decide(proposal)

    return decide


def spread_signal_decision(episode_steps: int, parameters: Mapping[str, Any]):
    controller = ABIDESPacedSignalController(
        episode_steps,
        target_rate=float(parameters["controller_target_rate"]),
    )

    def decide(state: Mapping[str, Any], step: int) -> float:
        features = abides_gate_features(state, abides_lagged_actions(state))
        severity = float(features[FEATURE_NAMES.index("contract_severity")])
        proposal = spread_only_proposal_strength(
            severity,
            float(parameters["severity_threshold"]),
            float(parameters["signal_scale"]),
        )
        return controller.decide(proposal)

    return decide


def market_for_group(config: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    values = market_config(config)
    values["routing_target_kappa"] = float(spec["routing_target_kappa"])
    values["routing_routable_fraction"] = float(spec["routing_routable_fraction"])
    return values


def rollout_group(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESTwoStageGateModel,
    environment_seed: int,
    train: bool,
) -> tuple[dict[str, Any], bool, bool, bool]:
    episode_steps = int(config["episode_steps"])
    namespace = (
        int(config["response_action_namespace"])
        if train
        else int(config["evaluation_action_namespace"])
    )
    common = {
        "seed": int(environment_seed),
        "episode_steps": episode_steps,
        "train": train,
        "defer_update": train,
        "reward_scale": float(config["reward_scale"]),
        "action_namespace": namespace,
    }
    risk = rollout_episode(
        env,
        trainers["risk_aware"],
        announced_signal_decision=risk_signal_decision(
            model, episode_steps, spec["risk_parameters"]
        ),
        **common,
    )
    risk_strengths = [float(row["gate_signal"]) for row in risk["records"]]
    random_strengths = global_shuffle(risk_strengths, int(environment_seed) ^ 0xD317)
    rollouts = {
        "risk_aware": risk,
        "matched_random": rollout_episode(
            env,
            trainers["matched_random"],
            announced_signal_schedule=random_strengths,
            **common,
        ),
    }
    if "spread_only_budget_matched" in spec["methods"]:
        rollouts["spread_only_budget_matched"] = rollout_episode(
            env,
            trainers["spread_only_budget_matched"],
            announced_signal_decision=spread_signal_decision(
                episode_steps, spec["spread_parameters"]
            ),
            **common,
        )
    paths_equal = all(
        validate_common_path(risk, rollout) for rollout in rollouts.values()
    )
    uniforms_equal = action_uniforms_equal(rollouts)
    dose_exact = exact_signal_multiset(risk_strengths, random_strengths)
    return rollouts, paths_equal, uniforms_equal, dose_exact


def load_or_initialize(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    output_dir: Path,
    policy_seed: int,
    fingerprint: str,
) -> tuple[dict[str, MAPPOTrainer], int, list[Any], bool, bool, bool, str, str]:
    path = checkpoint_path(output_dir, str(spec["group_id"]), policy_seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["protocol_fingerprint"] != fingerprint:
            raise RuntimeError(f"sensitivity checkpoint fingerprint mismatch: {path}")
        if payload["group_spec_digest"] != digest(spec):
            raise RuntimeError(f"sensitivity group spec changed: {path}")
        return (
            {
                method: trainer_from_payload(values, str(config["device"]))
                for method, values in payload["trainers"].items()
            },
            int(payload["completed_episodes"]),
            list(payload["training_history"]),
            bool(payload["training_paths_equal"]),
            bool(payload["training_action_uniforms_equal"]),
            bool(payload["training_dose_exact"]),
            str(payload["base_model_digest"]),
            str(payload["initial_adapter_digest"]),
        )
    base = load_base(base_loader_config(config), policy_seed)
    expected = str(source_result(config, policy_seed)["model_digest"])
    if model_digest(base) != expected:
        raise RuntimeError(f"source base model digest mismatch for {policy_seed}")
    trainers = {
        method: make_adapter(base, config, policy_seed)
        for method in spec["methods"]
    }
    initial = {model_digest(trainer) for trainer in trainers.values()}
    if len(initial) != 1:
        raise RuntimeError("sensitivity adapters did not initialize identically")
    return trainers, 0, [], True, True, True, expected, initial.pop()


def save_checkpoint(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    output_dir: Path,
    policy_seed: int,
    fingerprint: str,
    trainers: Mapping[str, MAPPOTrainer],
    completed: int,
    history: Sequence[Mapping[str, Any]],
    paths_equal: bool,
    uniforms_equal: bool,
    dose_exact: bool,
    base_digest: str,
    initial_digest: str,
) -> None:
    atomic_torch_save(
        checkpoint_path(output_dir, str(spec["group_id"]), policy_seed),
        {
            "protocol_fingerprint": fingerprint,
            "group_spec_digest": digest(spec),
            "policy_seed": int(policy_seed),
            "completed_episodes": int(completed),
            "trainers": {
                method: trainer_payload(trainer)
                for method, trainer in trainers.items()
            },
            "training_history": list(history),
            "training_paths_equal": bool(paths_equal),
            "training_action_uniforms_equal": bool(uniforms_equal),
            "training_dose_exact": bool(dose_exact),
            "base_model_digest": base_digest,
            "initial_adapter_digest": initial_digest,
            "saved_at": utc_now(),
        },
    )


def write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical_json(row) + "\n")
    temporary.replace(path)


def evaluate_group(
    env: ABIDESDockerDealerEnv,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    policy_seed: int,
    trainers: Mapping[str, MAPPOTrainer],
    model: ABIDESTwoStageGateModel,
) -> dict[str, Any]:
    base = load_base(base_loader_config(config), policy_seed)
    digests = {
        "no_intervention": model_digest(base),
        **{method: model_digest(trainer) for method, trainer in trainers.items()},
    }
    per_path = []
    step_records = []
    path_checks = []
    uniform_checks = []
    dose_checks = []
    for evaluation_seed in config["evaluation_seeds"]:
        rollouts, paths_equal, uniforms_equal, dose_exact = rollout_group(
            env,
            config,
            spec,
            trainers,
            model,
            int(evaluation_seed),
            train=False,
        )
        rollouts["no_intervention"] = rollout_episode(
            env,
            base,
            seed=int(evaluation_seed),
            episode_steps=int(config["episode_steps"]),
            train=False,
            routing_enabled=False,
            reward_scale=float(config["reward_scale"]),
            action_namespace=int(config["evaluation_action_namespace"]),
        )
        reference = rollouts["risk_aware"]
        paths_equal &= validate_common_path(reference, rollouts["no_intervention"])
        uniforms_equal &= action_uniforms_equal(rollouts)
        path_checks.append(paths_equal)
        uniform_checks.append(uniforms_equal)
        dose_checks.append(dose_exact)
        for method, rollout in rollouts.items():
            row = path_metrics(
                str(config["protocol_version"]),
                policy_seed,
                int(evaluation_seed),
                method,
                digests[method],
                rollout["records"],
            )
            row.update(
                {
                    "group_id": str(spec["group_id"]),
                    "study": str(spec["study"]),
                    "dose_target": float(spec["dose_target"]),
                    "routing_target_kappa": float(spec["routing_target_kappa"]),
                    "routing_routable_fraction": float(
                        spec["routing_routable_fraction"]
                    ),
                }
            )
            per_path.append(row)
            for record in rollout["records"]:
                step = audit_step_record(
                    policy_seed, int(evaluation_seed), method, record
                )
                step.update(
                    {
                        "group_id": str(spec["group_id"]),
                        "study": str(spec["study"]),
                    }
                )
                step_records.append(step)
    expected_methods = len(spec["methods"]) + 1
    expected_steps = (
        len(config["evaluation_seeds"])
        * expected_methods
        * int(config["episode_steps"])
    )
    return {
        "per_path": per_path,
        "step_records": step_records,
        "paths_equal": all(path_checks),
        "action_uniforms_equal": all(uniform_checks),
        "dose_exact": all(dose_checks),
        "step_count_exact": len(step_records) == expected_steps,
        "volume_conserved_every_step": all(
            abs(float(row["realized_flow_quantity"]) - 20.0) <= 1e-12
            for row in step_records
        ),
        "routing_weights_valid_every_step": all(
            all(float(value) >= 0.0 for value in row["routing_weights"])
            and abs(sum(float(value) for value in row["routing_weights"]) - 1.0)
            <= 1e-12
            for row in step_records
        ),
    }


def run_group_task(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    policy_seed: int,
    fingerprint: str,
) -> dict[str, Any]:
    output_dir = REPO / str(config["output_dir"])
    existing = result_path(output_dir, str(spec["group_id"]), policy_seed)
    if existing.exists():
        result = json.loads(existing.read_text(encoding="utf-8"))
        if result["protocol_fingerprint"] != fingerprint:
            raise RuntimeError(f"existing result fingerprint mismatch: {existing}")
        return result
    (
        trainers,
        completed,
        history,
        training_paths_equal,
        training_uniforms_equal,
        training_dose_exact,
        base_digest,
        initial_digest,
    ) = load_or_initialize(config, spec, output_dir, policy_seed, fingerprint)
    model = ABIDESTwoStageGateModel.load(REPO / str(config["gate_artifact"]))
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(market_for_group(config, spec), docker) as env:
        while completed < int(config["response_episodes"]):
            batch_size = min(
                int(config["response_episodes_per_update"]),
                int(config["response_episodes"]) - completed,
            )
            trajectories: dict[str, list[dict[str, np.ndarray]]] = {
                method: [] for method in spec["methods"]
            }
            batch_metrics: dict[str, list[dict[str, float]]] = {
                method: [] for method in spec["methods"]
            }
            for offset in range(batch_size):
                episode = completed + offset
                environment_seed = response_environment_seed(
                    config, policy_seed, episode
                )
                rollouts, paths_equal, uniforms_equal, dose_exact = rollout_group(
                    env,
                    config,
                    spec,
                    trainers,
                    model,
                    environment_seed,
                    train=True,
                )
                training_paths_equal &= paths_equal
                training_uniforms_equal &= uniforms_equal
                training_dose_exact &= dose_exact
                for method, rollout in rollouts.items():
                    trajectories[method].append(rollout["trajectory"])
                    batch_metrics[method].append(
                        {
                            "mean_half_spread": float(rollout["mean_half_spread"]),
                            "mean_reward": float(rollout["mean_reward"]),
                            "signal_activation_rate": float(
                                rollout["activation_rate"]
                            ),
                            "realized_activation_rate": float(
                                mean(
                                    bool(row["info"]["active_routing"])
                                    for row in rollout["records"]
                                )
                            ),
                            "mean_routing_total_variation": float(
                                mean(
                                    float(row["info"]["routing_total_variation"])
                                    for row in rollout["records"]
                                )
                            ),
                        }
                    )
            updates = {}
            for method, trainer in trainers.items():
                update = trainer.update_episodes(trajectories[method])
                if not all(math.isfinite(float(value)) for value in update.values()):
                    raise FloatingPointError(
                        f"non-finite sensitivity update: {method} {update}"
                    )
                updates[method] = update
            completed += batch_size
            history.append(
                {
                    "completed_episodes": completed,
                    "methods": {
                        method: {
                            key: float(mean(row[key] for row in rows))
                            for key in rows[0]
                        }
                        for method, rows in batch_metrics.items()
                    },
                    "updates": updates,
                }
            )
            save_checkpoint(
                config,
                spec,
                output_dir,
                policy_seed,
                fingerprint,
                trainers,
                completed,
                history,
                training_paths_equal,
                training_uniforms_equal,
                training_dose_exact,
                base_digest,
                initial_digest,
            )
        evaluation = evaluate_group(
            env, config, spec, policy_seed, trainers, model
        )
    step_path = step_records_path(output_dir, str(spec["group_id"]), policy_seed)
    write_jsonl_gz(step_path, evaluation.pop("step_records"))
    result = {
        "protocol_version": str(config["protocol_version"]),
        "protocol_fingerprint": fingerprint,
        "group_spec": dict(spec),
        "group_spec_digest": digest(spec),
        "policy_seed": int(policy_seed),
        "source_checkpoint_sha256": sha256_file(
            source_checkpoint(config, policy_seed)
        ),
        "base_model_digest": base_digest,
        "initial_adapter_digest": initial_digest,
        "final_model_digests": {
            method: model_digest(trainer) for method, trainer in trainers.items()
        },
        "completed_episodes": completed,
        "training_history": history,
        "training_paths_equal": training_paths_equal,
        "training_action_uniforms_equal": training_uniforms_equal,
        "training_dose_exact": training_dose_exact,
        "evaluation": evaluation,
        "step_records_sha256": sha256_file(step_path),
        "status": "complete",
    }
    atomic_write_json(existing, result)
    return result


def load_results(
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    fingerprint: str,
) -> list[dict[str, Any]]:
    results = []
    for spec in specs:
        for policy_seed in config["source_policy_seeds"]:
            path = result_path(output_dir, str(spec["group_id"]), int(policy_seed))
            if not path.exists():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["protocol_fingerprint"] != fingerprint:
                raise RuntimeError(f"result fingerprint mismatch: {path}")
            if result["group_spec_digest"] != digest(spec):
                raise RuntimeError(f"result group spec mismatch: {path}")
            results.append(result)
    return sorted(
        results, key=lambda row: (str(row["group_spec"]["group_id"]), int(row["policy_seed"]))
    )


def aggregate_per_seed(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["group_id"]), int(row["policy_seed"]), str(row["method"])),
            [],
        ).append(row)
    output = []
    metrics = (
        "low_execution_half_spread",
        "all_state_execution_half_spread",
        "mean_reward",
        "dealer_reward_units_per_episode",
        "activation_rate",
        "high_risk_activation_rate",
        "mean_signal_strength",
        "mean_routing_total_variation",
        "mean_volume",
    )
    for (group_id, policy_seed, method), method_rows in sorted(grouped.items()):
        first = method_rows[0]
        row: dict[str, Any] = {
            "group_id": group_id,
            "study": first["study"],
            "dose_target": float(first["dose_target"]),
            "routing_target_kappa": float(first["routing_target_kappa"]),
            "routing_routable_fraction": float(
                first["routing_routable_fraction"]
            ),
            "policy_seed": policy_seed,
            "method": method,
            "evaluation_path_count": len(method_rows),
        }
        for metric in metrics:
            row[metric] = float(mean(float(item[metric]) for item in method_rows))
        output.append(row)
    return output


def contrast_statistics(
    seed_rows: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lookup = {
        (str(row["group_id"]), int(row["policy_seed"]), str(row["method"])): row
        for row in seed_rows
    }
    results = {}
    for spec in specs:
        group_id = str(spec["group_id"])
        values = []
        spread_values = []
        for policy_seed in sorted(
            {int(row["policy_seed"]) for row in seed_rows if row["group_id"] == group_id}
        ):
            risk = lookup[(group_id, policy_seed, "risk_aware")]
            random = lookup[(group_id, policy_seed, "matched_random")]
            values.append(
                float(random["low_execution_half_spread"])
                - float(risk["low_execution_half_spread"])
            )
            if "spread_only_budget_matched" in spec["methods"]:
                spread = lookup[
                    (group_id, policy_seed, "spread_only_budget_matched")
                ]
                spread_values.append(
                    float(spread["low_execution_half_spread"])
                    - float(risk["low_execution_half_spread"])
                )
        if not values:
            results[group_id] = {"status": "missing_group_results"}
            continue
        results[group_id] = {
            "matched_random_minus_risk_aware_low_spread": {
                "policy_seed_t_interval": t_interval(values),
                "policy_seed_exact_sign_test": exact_sign_test(values),
            }
        }
        if spread_values:
            results[group_id]["spread_only_minus_risk_aware_low_spread"] = {
                "policy_seed_t_interval": t_interval(spread_values),
                "policy_seed_exact_sign_test": exact_sign_test(spread_values),
            }
    return results


def routing_classification(
    specs: Sequence[Mapping[str, Any]], statistics: Mapping[str, Any]
) -> dict[str, Any]:
    routing = [spec for spec in specs if spec["study"] == "routing"]
    intervals = {
        str(spec["group_id"]): statistics[str(spec["group_id"])][
            "matched_random_minus_risk_aware_low_spread"
        ]["policy_seed_t_interval"]
        for spec in routing
    }
    favorable = sum(float(row["mean"]) > 0.0 for row in intervals.values())
    adverse = [group for group, row in intervals.items() if float(row["ci_high"]) < 0.0]
    center = routing_group_id(4.0, 0.5)
    center_supported = float(intervals[center]["ci_low"]) > 0.0
    if favorable >= 3 and not adverse:
        label = "broadly_robust"
    elif center_supported and not adverse:
        label = "locally_robust"
    else:
        label = "parameter_specific"
    return {
        "classification": label,
        "favorable_mean_settings": favorable,
        "total_settings": len(routing),
        "significantly_adverse_settings": adverse,
        "center_supported": center_supported,
        "nonsignificance_is_not_equivalence": True,
    }


def summarize(
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    per_path = [
        dict(row) for result in results for row in result["evaluation"]["per_path"]
    ]
    per_seed = aggregate_per_seed(per_path)
    statistics = contrast_statistics(per_seed, specs) if per_seed else {}
    required_tasks = len(specs) * len(config["source_policy_seeds"])
    source_initials: dict[int, set[str]] = {}
    for result in results:
        source_initials.setdefault(int(result["policy_seed"]), set()).add(
            str(result["initial_adapter_digest"])
        )
    checks = {
        "required_task_count": len(results) == required_tasks,
        "no_worker_failures": not failures,
        "fixed_response_budget_complete": all(
            int(result["completed_episodes"]) == int(config["response_episodes"])
            for result in results
        ),
        "training_paths_equal": all(
            bool(result["training_paths_equal"]) for result in results
        ),
        "training_action_uniforms_equal": all(
            bool(result["training_action_uniforms_equal"]) for result in results
        ),
        "training_random_dose_exact": all(
            bool(result["training_dose_exact"]) for result in results
        ),
        "evaluation_paths_equal": all(
            bool(result["evaluation"]["paths_equal"]) for result in results
        ),
        "evaluation_action_uniforms_equal": all(
            bool(result["evaluation"]["action_uniforms_equal"])
            for result in results
        ),
        "evaluation_random_dose_exact": all(
            bool(result["evaluation"]["dose_exact"]) for result in results
        ),
        "evaluation_step_count_exact": all(
            bool(result["evaluation"]["step_count_exact"]) for result in results
        ),
        "volume_conserved_every_step": all(
            bool(result["evaluation"]["volume_conserved_every_step"])
            for result in results
        ),
        "routing_weights_valid_every_step": all(
            bool(result["evaluation"]["routing_weights_valid_every_step"])
            for result in results
        ),
        "same_initial_adapter_within_policy_seed": all(
            len(values) == 1 for values in source_initials.values()
        )
        and len(source_initials) == len(config["source_policy_seeds"]),
    }
    implementation_pass = all(checks.values())
    engineering_smoke = bool(config.get("engineering_smoke"))
    routing_result = (
        routing_classification(specs, statistics)
        if implementation_pass and not engineering_smoke
        else {"classification": "not_evaluated_integrity_fail"}
    )
    if engineering_smoke:
        routing_result = {"classification": "not_evaluated_engineering_smoke"}
    summary = {
        "protocol_version": str(config["protocol_version"]),
        "implementation_status": "pass" if implementation_pass else "fail",
        "scientific_result": (
            "not_evaluated_engineering_smoke"
            if engineering_smoke
            else (
                "secondary_sensitivity_evaluated"
                if implementation_pass
                else "not_evaluated_integrity_fail"
            )
        ),
        "policy_seed_count": len(config["source_policy_seeds"]),
        "group_count": len(specs),
        "task_count": len(results),
        "per_path_row_count": len(per_path),
        "checks": checks,
        "contrast_statistics": statistics,
        "routing_robustness": routing_result,
        "claim_boundary": (
            "secondary sensitivity only; no setting replaces v5 and "
            "nonsignificance is not equivalence"
        ),
    }
    return summary, per_path, per_seed


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty sensitivity CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# ABIDES Dose And Routing Sensitivity v1",
        "",
        f"- Implementation: `{summary['implementation_status']}`",
        f"- Scientific result: `{summary['scientific_result']}`",
        f"- Policy seeds: `{summary['policy_seed_count']}`",
        f"- Independent groups: `{summary['group_count']}`",
        f"- Completed tasks: `{summary['task_count']}`",
        f"- Routing classification: `{summary['routing_robustness']['classification']}`",
        "",
        "## Paired Random Minus Risk-aware",
        "",
    ]
    for group_id, values in summary["contrast_statistics"].items():
        if values.get("status") == "missing_group_results":
            lines.append(f"- `{group_id}`: missing due to retained task failure")
            continue
        interval = values["matched_random_minus_risk_aware_low_spread"][
            "policy_seed_t_interval"
        ]
        lines.append(
            f"- `{group_id}`: mean `{interval['mean']:.6f}`, "
            f"95% CI `[{interval['ci_low']:.6f}, {interval['ci_high']:.6f}]`, "
            f"positive `{interval['positive']}/{interval['n']}`"
        )
    lines.extend(["", summary["claim_boundary"], ""])
    return "\n".join(lines)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(config["formal_execution_allowed"]):
        raise RuntimeError("formal sensitivity execution is not allowed")
    if str(config["freeze_status"]) != "frozen_after_passive_calibration":
        raise RuntimeError("sensitivity config is not frozen after calibration")
    artifact_path = REPO / str(config["calibration"]["artifact"])
    if sha256_file(artifact_path) != str(config["calibration_artifact_sha256"]):
        raise RuntimeError("dose calibration artifact changed")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact["calibration_status"] != "pass":
        raise RuntimeError("dose calibration did not pass")
    if artifact["calibration_protocol_fingerprint"] != str(
        config["calibration_protocol_fingerprint"]
    ):
        raise RuntimeError("dose calibration fingerprint changed")
    if sha256_file(REPO / str(config["gate_artifact"])) != str(
        config["gate_artifact_sha256"]
    ):
        raise RuntimeError("gate artifact changed")
    for target, parameters in config["calibrated_dose_parameters"].items():
        selected = artifact["selected"][target]
        if selected["budget_match_status"] != "matched":
            raise RuntimeError(f"dose {target} is not budget matched")
        for method in ("risk_aware", "spread_only_budget_matched"):
            for key, value in parameters[method].items():
                if selected[method][key] != value:
                    raise RuntimeError(
                        f"frozen dose parameter differs from calibration: "
                        f"{target} {method} {key}"
                    )
    source_manifest = REPO / str(config["source_base_output_dir"]) / "manifest.json"
    if sha256_file(source_manifest) != str(config["source_base_manifest_sha256"]):
        raise RuntimeError("source base manifest changed")
    specs = group_specs(config)
    expected_groups = 2 if bool(config.get("engineering_smoke")) else 8
    if len(specs) != expected_groups:
        raise RuntimeError(
            f"sensitivity design must contain {expected_groups} groups"
        )
    return artifact


def engineering_smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    smoke = json.loads(json.dumps(config))
    smoke.update(
        {
            "protocol_version": "abides_dose_routing_sensitivity_smoke_v1",
            "engineering_smoke": True,
            "source_policy_seeds": [int(config["source_policy_seeds"][0])],
            "evaluation_seeds": [538900],
            "response_episodes": 4,
            "response_episodes_per_update": 2,
            "dose_targets": [0.1],
            "routing_settings": [
                {"routing_target_kappa": 4.0, "routing_routable_fraction": 0.5}
            ],
            "required_policy_seed_count": 1,
            "response_environment_seed_namespace": 639000000,
            "response_action_namespace": 579117,
            "evaluation_action_namespace": 589117,
            "workers": 2,
            "output_dir": (
                "revision/experiments/outputs/"
                "abides_dose_routing_sensitivity_smoke_v1"
            ),
        }
    )
    smoke["calibrated_dose_parameters"] = {
        key: dict(config["calibrated_dose_parameters"][key])
        for key in ("0.10", "0.20")
    }
    return smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_dose_routing_sensitivity_v1.json",
    )
    parser.add_argument("--force-unlock", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    config_path = REPO / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.engineering_smoke:
        config = engineering_smoke_config(config)
    validate_config(config)
    specs = group_specs(config)
    fingerprint = protocol_fingerprint(config)
    output_dir = REPO / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol_fingerprint"] != fingerprint:
            raise RuntimeError("sensitivity output belongs to another protocol")
    else:
        manifest = {
            "protocol_version": str(config["protocol_version"]),
            "protocol_fingerprint": fingerprint,
            "config": config,
            "config_sha256": sha256_file(config_path),
            "calibration_artifact_sha256": str(
                config["calibration_artifact_sha256"]
            ),
            "docker_image_id": resolve_docker_image_id(str(config["docker_image"])),
            "execution_commit": current_git_commit(),
            "group_specs": specs,
            "started_at": utc_now(),
        }
        atomic_write_json(manifest_path, manifest)
    with RunLock(output_dir / "run.lock", fingerprint, args.force_unlock):
        started = time.monotonic()
        prior_sha = (
            sha256_file(output_dir / "aggregate.json")
            if (output_dir / "aggregate.json").exists()
            else None
        )
        results = load_results(config, specs, output_dir, fingerprint)
        complete = {
            (str(row["group_spec"]["group_id"]), int(row["policy_seed"]))
            for row in results
        }
        tasks = [
            (spec, int(policy_seed))
            for spec in specs
            for policy_seed in config["source_policy_seeds"]
            if (str(spec["group_id"]), int(policy_seed)) not in complete
        ]
        append_logbook(
            output_dir / "logbook.jsonl",
            {
                "event": "run_started",
                "completed_tasks": len(complete),
                "pending_tasks": len(tasks),
                "protocol_fingerprint": fingerprint,
            },
        )
        failures = []
        if tasks:
            with ProcessPoolExecutor(max_workers=int(config["workers"])) as pool:
                futures = {
                    pool.submit(
                        run_group_task, config, spec, policy_seed, fingerprint
                    ): (str(spec["group_id"]), policy_seed)
                    for spec, policy_seed in tasks
                }
                for future in as_completed(futures):
                    group_id, policy_seed = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {
                                "event": "task_completed",
                                "group_id": group_id,
                                "policy_seed": policy_seed,
                            },
                        )
                    except Exception as exc:  # pragma: no cover - worker boundary
                        failure = {
                            "timestamp": utc_now(),
                            "group_id": group_id,
                            "policy_seed": policy_seed,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        with (output_dir / "failures.jsonl").open(
                            "a", encoding="utf-8"
                        ) as handle:
                            handle.write(canonical_json(failure) + "\n")
                        append_logbook(
                            output_dir / "logbook.jsonl",
                            {"event": "task_failed", **failure},
                        )
        results.sort(
            key=lambda row: (
                str(row["group_spec"]["group_id"]), int(row["policy_seed"])
            )
        )
        summary, per_path, per_seed = summarize(config, specs, results, failures)
        summary["protocol_fingerprint"] = fingerprint
        atomic_write_json(output_dir / "aggregate.json", summary)
        write_csv(output_dir / "per_path.csv", per_path)
        write_csv(output_dir / "per_seed.csv", per_seed)
        atomic_write_text(output_dir / "SUMMARY.md", summary_markdown(summary))
        aggregate_sha = sha256_file(output_dir / "aggregate.json")
        completed_resume = not tasks
        completion = {
            "protocol_fingerprint": fingerprint,
            "completed_tasks": len(results),
            "required_tasks": len(specs) * len(config["source_policy_seeds"]),
            "failed_tasks_this_invocation": [
                [row["group_id"], row["policy_seed"]] for row in failures
            ],
            "containers_started": len(tasks),
            "completed_resume_verification": completed_resume,
            "aggregate_sha256_before": prior_sha,
            "aggregate_sha256_after": aggregate_sha,
            "aggregate_unchanged_on_completed_resume": (
                completed_resume and prior_sha == aggregate_sha
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
                "aggregate_sha256": aggregate_sha,
            },
        )
        print(canonical_json({"summary": summary, "completion": completion}))
        if failures:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
