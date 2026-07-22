#!/usr/bin/env python3
"""Frozen non-outcome microbenchmark and evidence exporter for ABCE."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.special import expit

from src.interventions.abides_gate import (
    FEATURE_NAMES,
    ABIDESPacedSignalController,
    ABIDESPublicRiskProxyModel,
    ABIDESTwoStageGateModel,
    abides_gate_features,
)
from src.interventions.abides_literature_baselines import (
    eligible_routing_weights,
    soft_rank_routing_weights,
    uniform_routing_weights,
)

REPO = Path(__file__).resolve().parents[2]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_source(source: Mapping[str, Any]) -> Path:
    path = REPO / str(source["path"])
    observed = sha256_file(path)
    expected = str(source["sha256"])
    if observed != expected:
        raise RuntimeError(f"source hash mismatch for {path}: {observed} != {expected}")
    return path


def percentile_summary(samples_ns: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(samples_ns, dtype=float) / 1000.0
    return {
        "sample_batches": int(values.size),
        "median_microseconds": float(np.median(values)),
        "p95_microseconds": float(np.percentile(values, 95)),
        "p99_microseconds": float(np.percentile(values, 99)),
        "mean_microseconds": float(np.mean(values)),
        "sample_sd_microseconds": float(np.std(values, ddof=1)),
    }


def benchmark(
    callable_: Callable[[], Any],
    warmup_iterations: int,
    measured_iterations: int,
    batch_size: int,
) -> dict[str, float | int]:
    if measured_iterations % batch_size:
        raise ValueError("measured_iterations must be divisible by batch_size")
    for _ in range(warmup_iterations):
        callable_()
    samples = []
    for _ in range(measured_iterations // batch_size):
        start = time.perf_counter_ns()
        for _ in range(batch_size):
            callable_()
        samples.append((time.perf_counter_ns() - start) / batch_size)
    return percentile_summary(samples)


def tie_aware_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    if y.shape != values.shape or y.ndim != 1:
        raise ValueError("labels and scores must be matching one-dimensional arrays")
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0 or positives + negatives != y.size:
        raise ValueError("AUC requires binary labels with both classes")
    order = np.argsort(values, kind="mergesort")
    sorted_scores = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(np.sum(ranks[y == 1]))
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def linear_parameters(input_dim: int, output_dim: int) -> int:
    return input_dim * output_dim + output_dim


def parameter_counts(dimensions: Mapping[str, int]) -> dict[str, int]:
    obs = int(dimensions["actor_observation"])
    critic_obs = int(dimensions["critic_observation"])
    hidden = int(dimensions["hidden"])
    actions = int(dimensions["actions"])
    agents = int(dimensions["agents"])
    actor_trunk = linear_parameters(obs, hidden) + linear_parameters(hidden, hidden)
    actor_heads = agents * linear_parameters(hidden, actions)
    critic = (
        linear_parameters(critic_obs, hidden)
        + linear_parameters(hidden, hidden)
        + linear_parameters(hidden, 1)
    )
    actor_adapter = agents * linear_parameters(obs, actions)
    base_actor = actor_trunk + actor_heads
    base_total = base_actor + critic
    return {
        "base_actor_trunk": actor_trunk,
        "base_actor_heads": actor_heads,
        "base_actor_total": base_actor,
        "centralized_critic": critic,
        "base_actor_critic_total": base_total,
        "actor_adapter": actor_adapter,
        "adapter_actor_critic_total": base_total + actor_adapter,
        "response_trainable_including_critic": actor_adapter + critic,
        "response_frozen_base_actor": base_actor,
    }


def fixed_state(config: Mapping[str, Any]) -> dict[str, Any]:
    actions = np.asarray(config["fixed_lagged_actions"], dtype=float)
    observations = np.zeros((int(config["n_agents"]), 6), dtype=float)
    observations[:, 1] = actions / 6.0
    return {
        "observations": observations.tolist(),
        "info": {
            "flow_shares": [1.0 / int(config["n_agents"])] * int(config["n_agents"]),
            "public_volatility": 0.0,
        },
    }


def hard_best_weights(
    half_spreads: Sequence[float], signal_strength: float, routable_fraction: float
) -> np.ndarray:
    spreads = np.asarray(half_spreads, dtype=float)
    minimum = float(np.min(spreads))
    indices = np.flatnonzero(spreads == minimum)
    return eligible_routing_weights(
        spreads.size,
        indices,
        float(routable_fraction) * float(signal_strength),
    )


@dataclass
class PacingCallable:
    episode_steps: int
    target_rate: float
    risk_probability: float
    probability_threshold: float
    severity: float
    controller: ABIDESPacedSignalController | None = None

    def __call__(self) -> float:
        if self.controller is None or self.controller.steps >= self.episode_steps:
            self.controller = ABIDESPacedSignalController(
                episode_steps=self.episode_steps,
                target_rate=self.target_rate,
            )
        proposed_strength = (
            self.risk_probability * self.severity
            if self.risk_probability >= self.probability_threshold
            and self.severity > 0.0
            else 0.0
        )
        return self.controller.decide(proposed_strength)


@dataclass
class ContractCallable:
    model: ABIDESTwoStageGateModel
    state: Mapping[str, Any]
    half_spreads: np.ndarray
    episode_steps: int
    target_rate: float
    kappa: float
    routable_fraction: float
    controller: ABIDESPacedSignalController | None = None

    def __call__(self) -> np.ndarray:
        if self.controller is None or self.controller.steps >= self.episode_steps:
            self.controller = ABIDESPacedSignalController(
                episode_steps=self.episode_steps,
                target_rate=self.target_rate,
            )
        proposal = self.model.announced_strength(self.state)
        strength = self.controller.decide(float(proposal["strength"]))
        return soft_rank_routing_weights(
            self.half_spreads,
            strength,
            self.kappa,
            self.routable_fraction,
        )


@dataclass
class AnnouncedNoRoutingCallable:
    model: ABIDESTwoStageGateModel
    state: Mapping[str, Any]
    n_agents: int
    episode_steps: int
    target_rate: float
    controller: ABIDESPacedSignalController | None = None

    def __call__(self) -> np.ndarray:
        if self.controller is None or self.controller.steps >= self.episode_steps:
            self.controller = ABIDESPacedSignalController(
                episode_steps=self.episode_steps,
                target_rate=self.target_rate,
            )
        proposal = self.model.announced_strength(self.state)
        self.controller.decide(float(proposal["strength"]))
        return uniform_routing_weights(self.n_agents)


def get_cgroup_memory() -> dict[str, int | None]:
    result: dict[str, int | None] = {"current_bytes": None, "peak_bytes": None}
    for key, filename in (
        ("current_bytes", "memory.current"),
        ("peak_bytes", "memory.peak"),
    ):
        path = Path("/sys/fs/cgroup") / filename
        if path.exists():
            try:
                result[key] = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pass
    return result


def process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    return value * 1024


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def storage_record(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    spec = config[key]
    root = REPO / str(spec["path"])
    trace = root / "step_records.jsonl.gz"
    steps = int(spec["step_records"])
    trace_bytes = trace.stat().st_size
    return {
        "path": str(spec["path"]),
        "artifact_total_bytes": directory_size(root),
        "compressed_step_trace_bytes": trace_bytes,
        "step_records": steps,
        "compressed_bytes_per_step": trace_bytes / steps,
        "step_trace_sha256": sha256_file(trace),
    }


def economic_effects(statistics: Mapping[str, Any]) -> dict[str, Any]:
    aligned = statistics["method_summary"]["aligned_soft_rank"]
    random = statistics["method_summary"]["exact_random_soft_rank"]
    headline = float(
        statistics["contrasts"]["A_exact_random_minus_aligned"]["t_interval"]["mean"]
    )
    random_low = float(random["low_execution_half_spread"]["mean"])
    aligned_all = float(aligned["all_execution_half_spread"]["mean"])
    random_all = float(random["all_execution_half_spread"]["mean"])
    aligned_exposure = float(
        aligned["high_risk_announced_signal_activation_rate"]["mean"]
    )
    random_exposure = float(
        random["high_risk_announced_signal_activation_rate"]["mean"]
    )
    aligned_inventory = float(aligned["mean_inventory_std"]["mean"])
    random_inventory = float(random["mean_inventory_std"]["mean"])
    return {
        "comparison": "aligned_soft_rank versus exact_random_soft_rank",
        "low_risk_execution_improvement_ticks": headline,
        "low_risk_execution_relative_improvement": headline / random_low,
        "low_risk_customer_tick_units_saved_per_episode": float(
            random["low_customer_tick_units"]["mean"]
            - aligned["low_customer_tick_units"]["mean"]
        ),
        "high_risk_announced_exposure_relative_reduction": (
            random_exposure - aligned_exposure
        )
        / random_exposure,
        "aggregate_dealer_reward_difference": float(
            aligned["aggregate_dealer_reward"]["mean"]
            - random["aggregate_dealer_reward"]["mean"]
        ),
        "inventory_sd_relative_reduction": (random_inventory - aligned_inventory)
        / random_inventory,
        "all_state_execution_half_spread_aligned": aligned_all,
        "all_state_execution_half_spread_random": random_all,
        "all_state_execution_half_spread_aligned_minus_random": aligned_all
        - random_all,
        "all_state_direction": (
            "aligned_is_worse" if aligned_all > random_all else "aligned_is_better"
        ),
        "interpretation": "selective low-risk improvement, not all-state spread dominance",
    }


def write_proxy_records(
    output: Path,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for split, records in (
                ("train", train_records),
                ("validation", validation_records),
            ):
                for row in records:
                    compact = {
                        "split": split,
                        "seed": int(row["seed"]),
                        "episode": int(row["episode"]),
                        "step": int(row["step"]),
                        "simulator_low_risk_label": int(row["risk_state"] == "low"),
                        "public_volatility": float(row["features"][6]),
                        "absolute_lagged_return": float(row["features"][7]),
                    }
                    zipped.write((canonical_json(compact) + "\n").encode("utf-8"))


def proxy_validation(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    calibration_path = verify_source(config["sources"]["proxy_calibration"])
    calibration = load_json(calibration_path)
    train_path = REPO / calibration["train_records"]["path"]
    validation_path = REPO / calibration["validation_records"]["path"]
    if sha256_file(train_path) != calibration["train_records"]["sha256"]:
        raise RuntimeError("proxy train-record hash mismatch")
    if sha256_file(validation_path) != calibration["validation_records"]["sha256"]:
        raise RuntimeError("proxy validation-record hash mismatch")
    train_records = load_json(train_path)
    validation_records = load_json(validation_path)

    gate_path = verify_source(config["sources"]["gate_model"])
    full_model = ABIDESTwoStageGateModel.load(gate_path)
    full_indices = [FEATURE_NAMES.index(name) for name in full_model.feature_names]
    full_values = np.asarray(
        [
            [row["features"][index] for index in full_indices]
            for row in validation_records
        ],
        dtype=float,
    )
    full_logits = (
        (full_values - np.asarray(full_model.feature_mean))
        / np.asarray(full_model.feature_scale)
    ) @ np.asarray(full_model.coefficients) + full_model.intercept
    full_scores = expit(full_logits)
    labels = [int(row["risk_state"] == "low") for row in validation_records]
    full_auc = tie_aware_auc(labels, full_scores)
    if not np.isclose(
        full_auc, float(config["expected_proxy_auc"]), atol=1e-12, rtol=0.0
    ):
        raise RuntimeError(f"final proxy AUC mismatch: {full_auc}")

    single = {}
    for name in ("volatility_only", "return_only"):
        model_path = REPO / calibration["artifacts"][name]["path"]
        if sha256_file(model_path) != calibration["artifacts"][name]["sha256"]:
            raise RuntimeError(f"{name} proxy hash mismatch")
        model = ABIDESPublicRiskProxyModel.load(model_path)
        feature_index = FEATURE_NAMES.index(model.feature_name)
        values = np.asarray(
            [row["features"][feature_index] for row in validation_records],
            dtype=float,
        )
        logits = (
            model.coefficient * (values - model.feature_mean) / model.feature_scale
            + model.intercept
        )
        auc = tie_aware_auc(labels, expit(logits))
        runtime_scalar_auc = tie_aware_auc(
            labels,
            [model.probability(row["features"]) for row in validation_records],
        )
        expected = float(calibration["calibration"][name]["risk_auc"])
        if not np.isclose(auc, expected, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"{name} proxy AUC mismatch: {auc}")
        single[name] = {
            "risk_auc": auc,
            "runtime_scalar_auc": runtime_scalar_auc,
            "feature_name": model.feature_name,
            "activation_rate": calibration["calibration"][name]["validation_metrics"][
                "activation_rate"
            ],
            "source_path": calibration["artifacts"][name]["path"],
            "source_sha256": calibration["artifacts"][name]["sha256"],
        }

    record_export = output_dir / "proxy_records.jsonl.gz"
    write_proxy_records(record_export, train_records, validation_records)

    dose_path = verify_source(config["sources"]["dose_statistics"])
    dose = load_json(dose_path)
    spread_only = {}
    for key in ("dose_010", "dose_020", "dose_030"):
        group = dose["groups"][key]
        contrast = group["contrasts"]["spread_only_budget_matched"][
            "policy_seed_t_interval"
        ]
        budget = group["formal_realized_budget_diagnostics"][
            "spread_only_budget_matched"
        ]
        spread_only[key] = {
            "target_dose": group["dose_target"],
            "spread_only_minus_risk_aware": contrast,
            "realized_budget_within_passive_tolerances": budget[
                "formal_realized_budget_within_passive_tolerances"
            ],
            "routing_tv_ratio": budget["routing_tv_ratio_comparator_over_risk_aware"],
        }

    return {
        "training_label": calibration["training_label"],
        "offline_only_label": True,
        "runtime_feature_names": list(full_model.feature_names),
        "runtime_reads_latent_risk": False,
        "train_records": {
            "count": len(train_records),
            "source_sha256": calibration["train_records"]["sha256"],
        },
        "validation_records": {
            "count": len(validation_records),
            "source_sha256": calibration["validation_records"]["sha256"],
        },
        "split_is_disjoint": not bool(
            {(row["seed"], row["episode"], row["step"]) for row in train_records}
            & {(row["seed"], row["episode"], row["step"]) for row in validation_records}
        ),
        "final_two_feature": {
            "risk_auc": full_auc,
            "runtime_scalar_auc": tie_aware_auc(
                labels,
                [full_model.probability(row["features"]) for row in validation_records],
            ),
            "auc_definition": "sklearn-compatible batch expit ROC AUC",
            "probability_threshold": full_model.probability_threshold,
            "source_path": config["sources"]["gate_model"]["path"],
            "source_sha256": config["sources"]["gate_model"]["sha256"],
        },
        "single_feature": single,
        "spread_only_control": {
            "role": "outcome comparator, not a risk classifier",
            "dose_cells": spread_only,
            "source_path": config["sources"]["dose_statistics"]["path"],
            "source_sha256": config["sources"]["dose_statistics"]["sha256"],
        },
        "record_export": {
            "path": record_export.name,
            "sha256": sha256_file(record_export),
            "rows": len(train_records) + len(validation_records),
        },
    }


def git_metadata() -> dict[str, Any]:
    command = ["git", "-c", f"safe.directory={REPO}"]
    commit = subprocess.run(
        [*command, "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = os.environ.get("ACE_RL_PROFILE_WORKTREE_DIRTY")
    if dirty not in {"true", "false"}:
        raise RuntimeError("ACE_RL_PROFILE_WORKTREE_DIRTY must be explicitly supplied")
    return {
        "commit_before_profile": commit,
        "worktree_dirty_before_profile": dirty == "true",
        "status_source": "explicit host launch environment",
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_readme(
    output: Path,
    profile: Mapping[str, Any],
    effects: Mapping[str, Any],
    proxy: Mapping[str, Any],
) -> None:
    latency = profile["latency"]
    overhead = profile["end_to_end_overhead"]
    params = profile["model_parameters"]
    text = f"""# ABCE Frozen Mechanism Profile v1

This is a non-outcome, single-thread microbenchmark of the ABCE online control
path in the frozen formal Docker image. It is not full ABIDES step latency.

## Latency

| Module | Median us | p95 us | p99 us |
| --- | ---: | ---: | ---: |
"""
    labels = (
        ("M1 logistic core", "m1_logistic_core"),
        ("M1 state to proposal", "m1_state_to_proposal"),
        ("M2 pacing", "m2_threshold_pacing"),
        ("M3 soft-rank", "m3_soft_rank"),
        ("M3 hard-best", "m3_hard_best"),
        ("Contract end-to-end", "contract_on_end_to_end"),
        ("Announced no-routing end-to-end", "announced_no_routing_end_to_end"),
        ("No-contract allocation", "no_contract_allocation"),
    )
    for label, key in labels:
        row = latency[key]
        text += f"| {label} | {row['median_microseconds']:.4f} | {row['p95_microseconds']:.4f} | {row['p99_microseconds']:.4f} |\n"
    routing_overhead = profile["routing_incremental_overhead"]
    text += f"""

Contract-on minus announced no-routing median overhead is
`{routing_overhead['median_absolute_microseconds']:.4f}` microseconds
(`{routing_overhead['median_ratio']:.2f}x`). Contract-on minus the minimal
no-contract allocation branch is
`{overhead['median_absolute_microseconds']:.4f}` microseconds
(`{overhead['median_ratio']:.2f}x`). These values are specific to the recorded
emulated `linux/amd64` container and do not justify a production-efficiency
claim.

## Parameters And Memory

- base actor-critic: `{params['base_actor_critic_total']}` parameters;
- actor adapter: `{params['actor_adapter']}` parameters;
- response-trainable including critic: `{params['response_trainable_including_critic']}` parameters;
- profiler process peak RSS: `{profile['resource']['process_peak_rss_bytes']}` bytes.

## Existing Evidence Export

- low-risk improvement over exact random: `{effects['low_risk_execution_improvement_ticks']:.5f}` ticks
  (`{100 * effects['low_risk_execution_relative_improvement']:.3f}%`);
- low-risk customer cost saved: `{effects['low_risk_customer_tick_units_saved_per_episode']:.2f}` tick-units/episode;
- high-risk announced exposure reduction: `{100 * effects['high_risk_announced_exposure_relative_reduction']:.1f}%`;
- aggregate dealer reward difference: `{effects['aggregate_dealer_reward_difference']:.2f}`;
- inventory SD reduction: `{100 * effects['inventory_sd_relative_reduction']:.1f}%`;
- all-state spread difference (aligned minus random): `{effects['all_state_execution_half_spread_aligned_minus_random']:.5f}` ticks, so the result remains selective.

The frozen two-feature public-risk proxy has validation AUC
`{proxy['final_two_feature']['risk_auc']:.4f}` on a disjoint 1,000/1,000
passive split. Runtime uses only public volatility and absolute lagged return;
the simulator risk label is offline-only.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(
        item
        for item in output.iterdir()
        if item.is_file() and item.name != "SHA256SUMS"
    ):
        rows.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(config_path: Path) -> Path:
    config = load_json(config_path)
    if (
        config["formal_outcome_experiment"]
        or config["status"] != "frozen_non_outcome_profile"
    ):
        raise RuntimeError("profiling config is not frozen as non-outcome")
    for source in config["sources"].values():
        verify_source(source)
    for name, expected in config["thread_limits"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread limit {name} must equal {expected}")
    actual_image = os.environ.get("ACE_RL_PROFILE_IMAGE_ID")
    if actual_image != config["docker_image_id"]:
        raise RuntimeError(
            "Docker image ID was not supplied or does not match the frozen image"
        )

    output = REPO / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for old in output.iterdir():
        if old.is_file():
            old.unlink()

    model = ABIDESTwoStageGateModel.load(REPO / config["sources"]["gate_model"]["path"])
    state = fixed_state(config)
    lagged_features = abides_gate_features(state, config["fixed_lagged_actions"])
    proposal = model.announced_strength(state)
    if not proposal["proposed"] or float(proposal["strength"]) <= 0.0:
        raise RuntimeError(
            "frozen profiling input must exercise an active contract proposal"
        )
    half_spreads = np.asarray(config["fixed_half_spreads"], dtype=float)
    warmup = int(config["warmup_iterations"])
    iterations = int(config["measured_iterations"])
    batch = int(config["batch_size"])
    rate = float(config["pacing_target_rate"])
    kappa = float(config["routing_target_kappa"])
    fraction = float(config["routing_routable_fraction"])

    pacing = PacingCallable(
        int(config["episode_steps"]),
        rate,
        float(proposal["risk_probability"]),
        model.probability_threshold,
        float(proposal["severity"]),
    )
    contract = ContractCallable(
        model,
        state,
        half_spreads,
        int(config["episode_steps"]),
        rate,
        kappa,
        fraction,
    )
    announced_no_routing = AnnouncedNoRoutingCallable(
        model,
        state,
        int(config["n_agents"]),
        int(config["episode_steps"]),
        rate,
    )
    callables: dict[str, Callable[[], Any]] = {
        "m1_logistic_core": lambda: model.probability(lagged_features),
        "m1_state_to_proposal": lambda: model.announced_strength(state),
        "m2_threshold_pacing": pacing,
        "m3_soft_rank": lambda: soft_rank_routing_weights(
            half_spreads, float(proposal["strength"]), kappa, fraction
        ),
        "m3_hard_best": lambda: hard_best_weights(
            half_spreads, float(proposal["strength"]), fraction
        ),
        "contract_on_end_to_end": contract,
        "announced_no_routing_end_to_end": announced_no_routing,
        "no_contract_allocation": lambda: uniform_routing_weights(
            int(config["n_agents"])
        ),
    }
    latency = {
        name: benchmark(callable_, warmup, iterations, batch)
        for name, callable_ in callables.items()
    }
    contract_latency = latency["contract_on_end_to_end"]
    announced_no_routing_latency = latency["announced_no_routing_end_to_end"]
    baseline_latency = latency["no_contract_allocation"]

    statistics = load_json(verify_source(config["sources"]["primary_statistics"]))
    effects = economic_effects(statistics)
    proxy = proxy_validation(config, output)
    storage = {
        "primary": storage_record(config, "primary_artifact"),
        "exact_learner": storage_record(config, "exact_learner_artifact"),
    }
    storage["combined_artifact_total_bytes"] = (
        storage["primary"]["artifact_total_bytes"]
        + storage["exact_learner"]["artifact_total_bytes"]
    )

    cgroup = get_cgroup_memory()
    profile = {
        "protocol_version": config["protocol_version"],
        "scope": "pure online control-path microbenchmark; excludes ABIDES exchange step",
        "measurement": {
            "warmup_iterations_per_callable": warmup,
            "measured_iterations_per_callable": iterations,
            "batch_size": batch,
            "n_agents": int(config["n_agents"]),
            "single_cpu": True,
            "thread_limits": config["thread_limits"],
        },
        "latency": latency,
        "end_to_end_overhead": {
            "reference": "minimal uniform no-contract allocation construction",
            "median_absolute_microseconds": float(
                contract_latency["median_microseconds"]
                - baseline_latency["median_microseconds"]
            ),
            "p95_absolute_microseconds": float(
                contract_latency["p95_microseconds"]
                - baseline_latency["p95_microseconds"]
            ),
            "p99_absolute_microseconds": float(
                contract_latency["p99_microseconds"]
                - baseline_latency["p99_microseconds"]
            ),
            "median_ratio": float(
                contract_latency["median_microseconds"]
                / baseline_latency["median_microseconds"]
            ),
            "p95_ratio": float(
                contract_latency["p95_microseconds"]
                / baseline_latency["p95_microseconds"]
            ),
            "p99_ratio": float(
                contract_latency["p99_microseconds"]
                / baseline_latency["p99_microseconds"]
            ),
        },
        "routing_incremental_overhead": {
            "reference": "announced no-routing path with M1 and M2 retained",
            "median_absolute_microseconds": float(
                contract_latency["median_microseconds"]
                - announced_no_routing_latency["median_microseconds"]
            ),
            "p95_absolute_microseconds": float(
                contract_latency["p95_microseconds"]
                - announced_no_routing_latency["p95_microseconds"]
            ),
            "p99_absolute_microseconds": float(
                contract_latency["p99_microseconds"]
                - announced_no_routing_latency["p99_microseconds"]
            ),
            "median_ratio": float(
                contract_latency["median_microseconds"]
                / announced_no_routing_latency["median_microseconds"]
            ),
            "p95_ratio": float(
                contract_latency["p95_microseconds"]
                / announced_no_routing_latency["p95_microseconds"]
            ),
            "p99_ratio": float(
                contract_latency["p99_microseconds"]
                / announced_no_routing_latency["p99_microseconds"]
            ),
        },
        "model_parameters": parameter_counts(config["model_dimensions"]),
        "resource": {
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "cgroup_memory": cgroup,
        },
        "storage": storage,
        "runtime": {
            "docker_image": config["docker_image"],
            "docker_image_id": actual_image,
            "configured_platform": config["container_platform"],
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "host_architecture_emulation_expected": config["container_platform"]
            == "linux/amd64",
        },
    }
    manifest = {
        "protocol_version": config["protocol_version"],
        "config_path": str(config_path.relative_to(REPO)),
        "config_sha256": sha256_file(config_path),
        "profiler_source": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "sources": config["sources"],
        "git": git_metadata(),
        "outcome_data_generated": False,
        "training_performed": False,
    }
    write_json(output / "profile.json", profile)
    write_json(output / "economic_effects.json", effects)
    write_json(output / "proxy_validation.json", proxy)
    write_json(output / "manifest.json", manifest)
    write_readme(output, profile, effects, proxy)
    write_checksums(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("revision/configs/abides_abce_mechanism_profile_v1.json"),
    )
    args = parser.parse_args()
    output = run(
        (REPO / args.config).resolve() if not args.config.is_absolute() else args.config
    )
    print(output)


if __name__ == "__main__":
    main()
