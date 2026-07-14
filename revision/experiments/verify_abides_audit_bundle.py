#!/usr/bin/env python3
"""Independently verify the committed ABIDES domain audit bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.utils.abides_audit import (  # noqa: E402
    DEFAULT_CONTRASTS,
    aggregate_per_seed,
    build_statistics,
    coerce_per_path_row,
)


DEFAULT_BUNDLE = "revision/paper/artifacts/abides_audit_bundle_v1"
DEFAULT_CONFIG = "revision/configs/abides_domain_benchmark_v1.json"
PAPER_PER_PATH = "revision/paper/data/abides_domain_benchmark_v1_per_path.csv"
PAPER_PER_SEED = "revision/paper/data/abides_domain_benchmark_v1_per_seed.csv"
V5_METHODS = {
    "risk_aware",
    "matched_random",
    "spread_only_threshold",
    "always_on",
    "no_routing",
}
NEW_METHODS = {
    "spread_only_budget_matched",
    "brero_threshold_buybox",
    "brero_threshold_buybox_budget_matched",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_fingerprint(config: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in config.items()
        if key not in {"workers", "output_dir"}
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_checksums(bundle: Path) -> dict[str, str]:
    checksum_file = bundle / "SHA256SUMS"
    if not checksum_file.is_file():
        raise RuntimeError("audit bundle has no SHA256SUMS")
    checksums: dict[str, str] = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise RuntimeError(f"invalid SHA256SUMS line: {line!r}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in checksums:
            raise RuntimeError(f"unsafe or duplicate checksum path: {relative}")
        checksums[relative] = digest
    required = {
        "manifest.json",
        "checks.json",
        "per_path.csv",
        "per_seed.csv",
        "step_records.jsonl.gz",
        "README.md",
    }
    if set(checksums) != required:
        raise RuntimeError("SHA256SUMS does not cover the exact bundle payload")
    for relative, expected in checksums.items():
        path = bundle / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"checksum mismatch: {relative}")
    return checksums


def assert_nested_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise RuntimeError(f"mapping mismatch at {path}")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"list mismatch at {path}")
        for index, value in enumerate(expected):
            assert_nested_close(actual[index], value, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(f"numeric mismatch at {path}: {actual} != {expected}")
        return
    if actual != expected:
        raise RuntimeError(f"value mismatch at {path}: {actual!r} != {expected!r}")


def expected_model_digest(
    manifest: Mapping[str, Any], policy_seed: int, method: str
) -> str:
    source = manifest["source_model_digests"][str(policy_seed)]
    if method == "no_intervention":
        return str(source["base"])
    if method in V5_METHODS:
        return str(source["v5"][method])
    if method in NEW_METHODS:
        return str(source["new"][method])
    raise RuntimeError(f"unexpected method: {method}")


def verify_per_path(
    bundle: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    methods = list(config["methods"])
    policy_seeds = {int(seed) for seed in config["policy_seeds"]}
    evaluation_seeds = {int(seed) for seed in config["evaluation_seeds"]}
    rows = [coerce_per_path_row(row) for row in read_csv(bundle / "per_path.csv")]
    expected_count = len(policy_seeds) * len(evaluation_seeds) * len(methods)
    provenance_seeds = manifest["source_model_digests"]
    if set(provenance_seeds) != {str(seed) for seed in policy_seeds}:
        raise RuntimeError("model provenance does not cover the source policy seeds")
    for source in provenance_seeds.values():
        if set(source["v5"]) != V5_METHODS or set(source["new"]) != NEW_METHODS:
            raise RuntimeError("model provenance has an incomplete method set")
    keys = {
        (int(row["policy_seed"]), int(row["evaluation_seed"]), str(row["method"]))
        for row in rows
    }
    expected_keys = {
        (policy_seed, evaluation_seed, method)
        for policy_seed in policy_seeds
        for evaluation_seed in evaluation_seeds
        for method in methods
    }
    if len(rows) != expected_count or keys != expected_keys:
        raise RuntimeError("per-path identity grid is incomplete or duplicated")
    if {str(row["protocol_version"]) for row in rows} != {
        str(config["protocol_version"])
    }:
        raise RuntimeError("per-path protocol version mismatch")

    path_groups: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in rows:
        policy_seed = int(row["policy_seed"])
        evaluation_seed = int(row["evaluation_seed"])
        method = str(row["method"])
        path_groups[(policy_seed, evaluation_seed)].add(str(row["path_digest"]))
        if str(row["model_digest"]) != expected_model_digest(
            manifest, policy_seed, method
        ):
            raise RuntimeError("per-path model digest differs from provenance")
        if not math.isclose(float(row["mean_volume"]), 20.0, abs_tol=1e-12):
            raise RuntimeError("per-path volume is not 20")
    if any(len(values) != 1 for values in path_groups.values()):
        raise RuntimeError("evaluation exogenous path digest differs across methods")

    derived_per_seed = aggregate_per_seed(rows, methods, DEFAULT_CONTRASTS)
    committed_per_seed = read_csv(bundle / "per_seed.csv")
    if len(committed_per_seed) != len(derived_per_seed):
        raise RuntimeError("per-seed row count mismatch")
    normalized_committed = []
    for row in committed_per_seed:
        normalized_committed.append(
            {
                key: int(value) if key == "policy_seed" else float(value)
                for key, value in row.items()
            }
        )
    assert_nested_close(normalized_committed, derived_per_seed, "per_seed")
    return (
        rows,
        derived_per_seed,
        {
            "source_seed_grid_complete": True,
            "evaluation_paths_equal": True,
            "method_volume_is_20": True,
            "per_path_row_count_exact": True,
            "paper_per_seed_recomputed": True,
            "source_v5_digests_unchanged": True,
        },
    )


def verify_step_records(
    bundle: Path,
    config: Mapping[str, Any],
    per_path: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    methods = list(config["methods"])
    expected_steps = int(config["episode_steps"])
    expected_count = (
        len(config["policy_seeds"])
        * len(config["evaluation_seeds"])
        * len(methods)
        * expected_steps
    )
    per_path_digests = {
        (int(row["policy_seed"]), int(row["evaluation_seed"]), str(row["method"])): str(
            row["path_digest"]
        )
        for row in per_path
    }
    seen: set[tuple[int, int, str, int]] = set()
    common: dict[tuple[int, int, int], dict[str, set[str]]] = defaultdict(
        lambda: {"path": set(), "uniform": set(), "risk": set()}
    )
    episode_flow: dict[tuple[int, int, str], float] = defaultdict(float)
    count = 0
    with gzip.open(bundle / "step_records.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            policy_seed = int(row["policy_seed"])
            evaluation_seed = int(row["evaluation_seed"])
            method = str(row["method"])
            step = int(row["step"])
            identity = (policy_seed, evaluation_seed, method, step)
            if identity in seen:
                raise RuntimeError(f"duplicate step record: {identity}")
            seen.add(identity)
            if method not in methods or not 0 <= step < expected_steps:
                raise RuntimeError(f"invalid method or step: {identity}")
            path_digest = str(row["path_digest"])
            if path_digest != per_path_digests[identity[:3]]:
                raise RuntimeError(f"step/per-path digest mismatch: {identity}")
            weights = [float(value) for value in row["routing_weights"]]
            if len(weights) != int(config["market"]["n_agents"]):
                raise RuntimeError(f"routing-weight length mismatch: {identity}")
            if any(value < -1e-15 for value in weights) or not math.isclose(
                sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(f"invalid routing weights: {identity}")
            flow = float(row["realized_flow_quantity"])
            if not math.isclose(flow, 20.0, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"per-step demand is not conserved: {identity}")
            episode_flow[identity[:3]] += flow
            key = (policy_seed, evaluation_seed, step)
            common[key]["path"].add(path_digest)
            common[key]["uniform"].add(str(row["action_uniform_digest"]))
            common[key]["risk"].add(str(row["latent_risk_state"]))
            count += 1
    if count != expected_count or len(seen) != expected_count:
        raise RuntimeError("step-record identity grid is incomplete")
    if any(
        any(len(values) != 1 for values in fields.values())
        for fields in common.values()
    ):
        raise RuntimeError("paired path, action uniform, or latent risk differs")
    expected_episode_flow = 20.0 * expected_steps
    if any(
        not math.isclose(value, expected_episode_flow, abs_tol=1e-12)
        for value in episode_flow.values()
    ):
        raise RuntimeError("per-episode demand is not conserved")
    return {
        "evaluation_action_uniforms_equal": True,
        "evaluation_latent_paths_equal": True,
        "per_step_demand_conserved": True,
        "per_episode_demand_conserved": True,
        "routing_weights_valid": True,
        "step_record_count_exact": True,
    }


def verify_execution_and_budget(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, bool]:
    policy_seeds = [int(seed) for seed in config["policy_seeds"]]
    records = manifest["execution_records"]
    if set(records) != {str(seed) for seed in policy_seeds}:
        raise RuntimeError("execution records do not cover every policy seed")
    if any(
        int(record["completed_episodes"]) != int(config["response_episodes"])
        for record in records.values()
    ):
        raise RuntimeError("response episode budget is incomplete")
    if any(not record["training_paths_equal"] for record in records.values()):
        raise RuntimeError("training exogenous paths were not paired")
    if any(not record["training_action_uniforms_equal"] for record in records.values()):
        raise RuntimeError("training action uniforms were not paired")
    if any(not record["gate_off_exact"] for record in records.values()):
        raise RuntimeError("gate-off actions did not reproduce the base policy")
    if any(not record["convergence"]["stable"] for record in records.values()):
        raise RuntimeError("a response arm failed the frozen convergence rule")
    if any(not record["initial_adapter_digest"] for record in records.values()):
        raise RuntimeError("initial adapter digest is missing")

    tolerances = config["budget_tolerances"]
    for method in (
        "spread_only_budget_matched",
        "brero_threshold_buybox_budget_matched",
    ):
        parameters = config["mechanism_parameters"][method]
        status = str(parameters["budget_match_status"])
        active_difference = abs(
            float(parameters["calibration_active_routing_rate"])
            - float(config["calibration_reference_budget"]["active_routing_rate"])
        )
        tv_ratio = float(parameters["calibration_mean_tv_ratio"])
        within = active_difference <= float(
            tolerances["maximum_active_rate_difference"]
        ) and float(tolerances["minimum_mean_tv_ratio"]) <= tv_ratio <= float(
            tolerances["maximum_mean_tv_ratio"]
        )
        if status == "matched" and not within:
            raise RuntimeError(f"{method} is incorrectly labeled matched")
        if status == "unmatched" and within:
            raise RuntimeError(f"{method} is incorrectly labeled unmatched")
        if status not in {"matched", "unmatched"}:
            raise RuntimeError(f"invalid budget status for {method}: {status}")

    calibration = set(int(seed) for seed in config["calibration_environment_seeds"])
    evaluation = set(int(seed) for seed in config["evaluation_seeds"])
    if calibration & evaluation:
        raise RuntimeError("calibration and evaluation seed namespaces overlap")
    if manifest["pending_policy_seeds"] or manifest["failures"]:
        raise RuntimeError("formal run has pending or failed seeds")
    if not isinstance(manifest["retained_failure_events"], list):
        raise RuntimeError("historical failure-event ledger is missing")
    if not manifest["completed_resume_verified"]:
        raise RuntimeError("no-container completed resume was not verified")
    if manifest["implementation_status"] != "pass":
        raise RuntimeError("runner implementation status is not pass")
    return {
        "new_adapter_initialization_recorded": True,
        "training_paths_equal": True,
        "training_action_uniforms_equal": True,
        "calibration_evaluation_disjoint": True,
        "budget_status_valid": True,
        "gate_off_exact": True,
        "response_episodes_complete": True,
        "frozen_convergence_passed": True,
        "failures_and_pending_retained": True,
        "completed_resume_verified": True,
    }


def verify_provenance(
    bundle: Path,
    config_path: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checksums: Mapping[str, str],
    allow_unarchived: bool,
) -> dict[str, bool]:
    if sha256_file(config_path) != str(manifest["formal_config_sha256"]):
        raise RuntimeError("formal config hash mismatch")
    if protocol_fingerprint(config) != str(manifest["protocol_fingerprint"]):
        raise RuntimeError("protocol fingerprint mismatch")
    required_strings = (
        "protocol_freeze_commit",
        "execution_commit",
        "source_base_fingerprint",
        "source_v5_fingerprint",
        "abides_commit",
        "docker_image",
        "docker_image_id",
        "started_at",
        "finished_at",
        "command",
    )
    if any(not str(manifest.get(field, "")) for field in required_strings):
        raise RuntimeError("manifest provenance is incomplete")
    if str(manifest["abides_commit"]) != "f9cbe51342b7dedd9587e4e069040d68a5c6477f":
        raise RuntimeError("ABIDES provenance changed")
    archival_commit = manifest.get("result_archival_commit")
    if not allow_unarchived and not archival_commit:
        raise RuntimeError("result archival commit has not been recorded")
    if archival_commit is not None and len(str(archival_commit)) != 40:
        raise RuntimeError("invalid result archival commit")
    artifact_hashes = manifest["artifact_hashes"]
    for field in (
        "gate_artifact_sha256",
        "calibration_artifact_sha256",
        "docker_image_id",
        "source_base_fingerprint",
        "source_v5_fingerprint",
    ):
        if not artifact_hashes.get(field):
            raise RuntimeError(f"missing artifact provenance: {field}")
    gate_path = REPO / str(config["gate_artifact"])
    calibration_path = REPO / str(config["calibration_artifact"])
    if sha256_file(gate_path) != str(artifact_hashes["gate_artifact_sha256"]):
        raise RuntimeError("gate artifact hash mismatch")
    if sha256_file(calibration_path) != str(
        artifact_hashes["calibration_artifact_sha256"]
    ):
        raise RuntimeError("calibration artifact hash mismatch")
    for relative, digest in manifest["output_file_sha256"].items():
        if checksums.get(relative) != digest:
            raise RuntimeError(f"manifest output hash mismatch: {relative}")
    if (bundle / "per_path.csv").read_bytes() != (REPO / PAPER_PER_PATH).read_bytes():
        raise RuntimeError("paper per-path export is not audit-derived")
    if (bundle / "per_seed.csv").read_bytes() != (REPO / PAPER_PER_SEED).read_bytes():
        raise RuntimeError("paper per-seed export is not audit-derived")
    return {
        "provenance_complete": True,
        "bundle_hashes_valid": True,
        "paper_exports_audit_derived": True,
        "external_baseline_no_latent_risk_access": bool(
            json.loads((bundle / "checks.json").read_text(encoding="utf-8"))[
                "runner_checks"
            ]["external_baselines_no_latent_risk_access"]
        ),
    }


def verify_bundle(
    bundle: Path, config_path: Path, allow_unarchived: bool = False
) -> dict[str, Any]:
    checksums = parse_checksums(bundle)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    checks = json.loads((bundle / "checks.json").read_text(encoding="utf-8"))
    if bool(config.get("engineering_smoke")):
        raise RuntimeError("engineering smoke is not eligible for audit export")
    if manifest["protocol_version"] != config["protocol_version"]:
        raise RuntimeError("bundle/config protocol version mismatch")
    per_path, _per_seed, path_checks = verify_per_path(bundle, config, manifest)
    step_checks = verify_step_records(bundle, config, per_path)
    execution_checks = verify_execution_and_budget(config, manifest)
    provenance_checks = verify_provenance(
        bundle,
        config_path,
        config,
        manifest,
        checksums,
        allow_unarchived,
    )
    recomputed_statistics = build_statistics(
        per_path,
        list(config["methods"]),
        DEFAULT_CONTRASTS,
        ("spread_only_budget_matched", "matched_random"),
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]),
    )
    assert_nested_close(
        checks["statistics"], recomputed_statistics, "checks.statistics"
    )
    all_checks = {
        **path_checks,
        **step_checks,
        **execution_checks,
        **provenance_checks,
        "statistics_recomputed": True,
        "implementation_science_separated": (
            checks["implementation_status"] == "pass"
            and checks["scientific_result"] == manifest["scientific_result"]
        ),
    }
    if not all(all_checks.values()):
        failed = sorted(name for name, passed in all_checks.items() if not passed)
        raise RuntimeError(f"audit integrity checks failed: {failed}")
    directive_checks = {
        "01_source_policy_seeds_complete": path_checks["source_seed_grid_complete"],
        "02_common_base_and_initial_adapter_digests": execution_checks[
            "new_adapter_initialization_recorded"
        ],
        "03_source_v5_digests_unchanged": path_checks["source_v5_digests_unchanged"],
        "04_training_exogenous_paths_equal": execution_checks["training_paths_equal"],
        "05_evaluation_exogenous_paths_equal": path_checks["evaluation_paths_equal"],
        "06_action_uniforms_equal": (
            execution_checks["training_action_uniforms_equal"]
            and step_checks["evaluation_action_uniforms_equal"]
        ),
        "07_calibration_evaluation_seeds_disjoint": execution_checks[
            "calibration_evaluation_disjoint"
        ],
        "08_external_baselines_no_latent_risk_access": provenance_checks[
            "external_baseline_no_latent_risk_access"
        ],
        "09_demand_conserved_per_step_and_episode": (
            step_checks["per_step_demand_conserved"]
            and step_checks["per_episode_demand_conserved"]
        ),
        "10_routing_weights_valid": step_checks["routing_weights_valid"],
        "11_budget_match_status_valid": execution_checks["budget_status_valid"],
        "12_gate_off_exact_base_actions": execution_checks["gate_off_exact"],
        "13_response_episodes_complete": execution_checks["response_episodes_complete"],
        "14_frozen_convergence_rule_passed": execution_checks[
            "frozen_convergence_passed"
        ],
        "15_provenance_complete": provenance_checks["provenance_complete"],
        "16_completed_resume_verified": execution_checks["completed_resume_verified"],
        "17_failures_and_pending_retained": execution_checks[
            "failures_and_pending_retained"
        ],
        "18_method_volume_is_20": path_checks["method_volume_is_20"],
        "19_per_path_row_count_exact": path_checks["per_path_row_count_exact"],
        "20_paper_exports_audit_derived": provenance_checks[
            "paper_exports_audit_derived"
        ],
    }
    if len(directive_checks) != 20 or not all(directive_checks.values()):
        raise RuntimeError("one or more directive integrity checks failed")
    return {
        "status": "pass",
        "protocol_version": str(config["protocol_version"]),
        "protocol_fingerprint": str(manifest["protocol_fingerprint"]),
        "implementation_status": str(manifest["implementation_status"]),
        "scientific_result": str(manifest["scientific_result"]),
        "integrity_checks": all_checks,
        "directive_integrity_checks": directive_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--allow-unarchived", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_bundle(
        REPO / args.bundle,
        REPO / args.config,
        allow_unarchived=args.allow_unarchived,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
