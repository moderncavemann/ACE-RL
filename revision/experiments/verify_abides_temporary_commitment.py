#!/usr/bin/env python3
"""Independently verify the committed temporary-commitment audit bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from verify_abides_audit_bundle import (  # noqa: E402
    assert_nested_close,
    parse_checksums,
    protocol_fingerprint,
    sha256_file,
)
from src.utils.abides_audit import (  # noqa: E402
    aggregate_per_seed,
    build_statistics,
    coerce_per_path_row,
)


DEFAULT_BUNDLE = (
    "revision/paper/artifacts/abides_temporary_commitment_benchmark_v1"
)
DEFAULT_CONFIG = "revision/configs/abides_temporary_commitment_benchmark_v1.json"
PAPER_PER_PATH = (
    "revision/paper/data/abides_temporary_commitment_benchmark_v1_per_path.csv"
)
PAPER_PER_SEED = (
    "revision/paper/data/abides_temporary_commitment_benchmark_v1_per_seed.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_contract_path(
    rows: Sequence[Mapping[str, Any]], parameters: Mapping[str, Any]
) -> tuple[bool, int]:
    duration = int(parameters["duration"])
    expected_events = int(round(20 * float(parameters["protected_fraction"])))
    expected_remaining = 0
    takeups = 0
    valid = True
    for row in sorted(rows, key=lambda item: int(item["step"])):
        mechanism = row["mechanism_parameters"]
        takeup = bool(mechanism.get("takeup", False))
        remaining = int(mechanism.get("remaining_periods_including_current", 0))
        active = remaining > 0
        if takeup:
            valid &= expected_remaining == 0
            expected_remaining = duration
            takeups += 1
        if expected_remaining > 0:
            valid &= active and remaining == expected_remaining
            valid &= int(row["targeted_event_count"]) == expected_events
            valid &= math.isclose(
                float(row["protected_execution_quantity"]),
                float(expected_events),
                abs_tol=1e-12,
            )
            valid &= int(row["protected_routing_fallback_count"]) == 0
            tags = list(row["protected_matched_order_tags"])
            prices = [int(value) for value in row["protected_execution_prices"]]
            prefix = mechanism.get("protected_order_tag_prefix")
            reference = mechanism.get("frozen_reference_price")
            half_spread = mechanism.get("frozen_half_spread")
            valid &= prefix is not None and len(tags) == expected_events
            valid &= all(str(tag).startswith(str(prefix)) for tag in tags)
            valid &= (
                reference is not None
                and half_spread is not None
                and len(prices) == expected_events
                and all(
                    abs(price - int(reference)) == int(half_spread)
                    for price in prices
                )
            )
            expected_remaining -= 1
        else:
            valid &= not active
            valid &= int(row["targeted_event_count"]) == 0
            valid &= math.isclose(
                float(row["protected_execution_quantity"]), 0.0, abs_tol=1e-12
            )
    return bool(valid), takeups


def verify_steps(
    bundle: Path, config: Mapping[str, Any]
) -> dict[str, bool | int]:
    expected_methods = set(config["methods"])
    expected_keys = {
        (int(policy_seed), int(evaluation_seed), method, step)
        for policy_seed in config["policy_seeds"]
        for evaluation_seed in config["evaluation_seeds"]
        for method in expected_methods
        for step in range(int(config["episode_steps"]))
    }
    actual_keys = set()
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    common: dict[tuple[int, int, int], tuple[str, str]] = {}
    common_valid = True
    volume_valid = True
    routing_valid = True
    with gzip.open(bundle / "step_records.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                int(row["policy_seed"]),
                int(row["evaluation_seed"]),
                str(row["method"]),
                int(row["step"]),
            )
            actual_keys.add(key)
            grouped[key[:3]].append(row)
            pair_key = (key[0], key[1], key[3])
            pair_value = (str(row["path_digest"]), str(row["action_uniform_digest"]))
            if pair_key in common:
                common_valid &= common[pair_key] == pair_value
            else:
                common[pair_key] = pair_value
            volume_valid &= math.isclose(
                float(row["realized_flow_quantity"]), 20.0, abs_tol=1e-12
            )
            weights = [float(value) for value in row["routing_weights"]]
            routing_valid &= all(value >= 0.0 for value in weights)
            routing_valid &= math.isclose(sum(weights), 1.0, abs_tol=1e-12)
    contract_valid = True
    total_takeups = 0
    for key, rows in grouped.items():
        method = key[2]
        if method.startswith("banerjee_temporary_commitment"):
            valid, takeups = validate_contract_path(
                rows, config["mechanism_parameters"][method]
            )
            contract_valid &= valid
            total_takeups += takeups
    return {
        "step_key_grid_exact": actual_keys == expected_keys,
        "step_record_count": len(actual_keys),
        "common_paths_and_action_uniforms": common_valid,
        "volume_conserved": volume_valid,
        "routing_weights_valid": routing_valid,
        "temporary_commitment_integrity": contract_valid,
        "temporary_commitment_exercised": total_takeups > 0,
    }


def verify_archival_commit(bundle: Path, manifest: Mapping[str, Any]) -> bool:
    commit = manifest.get("result_archival_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        return False
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=REPO,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=REPO,
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError:
        return False
    relative = str(bundle.relative_to(REPO))
    return any(path.startswith(relative + "/") for path in paths)


def verify(bundle: Path, config_path: Path) -> dict[str, Any]:
    parse_checksums(bundle)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    checks = json.loads((bundle / "checks.json").read_text(encoding="utf-8"))
    methods = list(config["methods"])
    contrasts = list(config["contrast_methods"])
    per_path = [coerce_per_path_row(row) for row in read_csv(bundle / "per_path.csv")]
    per_seed = [
        {
            key: int(value) if key == "policy_seed" else float(value)
            for key, value in row.items()
        }
        for row in read_csv(bundle / "per_seed.csv")
    ]
    recomputed_seed = aggregate_per_seed(per_path, methods, contrasts)
    recomputed_statistics = build_statistics(
        per_path,
        methods,
        contrasts,
        (
            "banerjee_temporary_commitment",
            "banerjee_temporary_commitment_budget_matched",
        ),
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]),
    )
    expected_grid = {
        (int(policy_seed), int(evaluation_seed), method)
        for policy_seed in config["policy_seeds"]
        for evaluation_seed in config["evaluation_seeds"]
        for method in methods
    }
    actual_grid = {
        (int(row["policy_seed"]), int(row["evaluation_seed"]), str(row["method"]))
        for row in per_path
    }
    verification: dict[str, bool] = {
        "protocol_fingerprint_valid": manifest["protocol_fingerprint"]
        == protocol_fingerprint(config),
        "formal_config_hash_valid": manifest["formal_config_sha256"]
        == sha256_file(config_path),
        "calibration_hash_valid": manifest["calibration_artifact_sha256"]
        == sha256_file(REPO / str(config["calibration_artifact"])),
        "per_path_grid_exact": actual_grid == expected_grid,
        "paper_per_path_matches": (bundle / "per_path.csv").read_bytes()
        == (REPO / PAPER_PER_PATH).read_bytes(),
        "paper_per_seed_matches": (bundle / "per_seed.csv").read_bytes()
        == (REPO / PAPER_PER_SEED).read_bytes(),
        "completed_resume_verified": bool(manifest["completed_resume_verified"]),
        "implementation_failure_retained": (
            manifest["implementation_status"] == "fail"
            and manifest["scientific_result"] == "not_evaluated_integrity_fail"
            and checks["runner_checks"]["response_converged"] is False
        ),
        "result_archival_commit_valid": verify_archival_commit(bundle, manifest),
    }
    try:
        assert_nested_close(per_seed, recomputed_seed)
        verification["per_seed_recomputed"] = True
    except RuntimeError:
        verification["per_seed_recomputed"] = False
    try:
        assert_nested_close(checks["statistics"], recomputed_statistics)
        verification["statistics_recomputed"] = True
    except RuntimeError:
        verification["statistics_recomputed"] = False
    step_checks = verify_steps(bundle, config)
    verification.update(
        {
            key: bool(value)
            for key, value in step_checks.items()
            if key != "step_record_count"
        }
    )
    verification["step_record_count_exact"] = int(
        step_checks["step_record_count"]
    ) == int(checks["step_record_count"]["expected"])
    status = "pass" if all(verification.values()) else "fail"
    return {
        "status": status,
        "checks": verification,
        "per_path_rows": len(per_path),
        "step_record_rows": int(step_checks["step_record_count"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(REPO / args.bundle, REPO / args.config)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
