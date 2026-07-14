#!/usr/bin/env python3
"""Export the temporary-commitment formal run as a self-contained audit bundle."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from export_abides_audit_bundle import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    check_export as _unused_check_export,
    combine_step_records,
    git_output,
    logbook_events,
    read_csv,
    resume_verified,
    sha256_file,
    validate_archival_commit,
    write_csv,
)
from src.utils.abides_audit import (  # noqa: E402
    aggregate_per_seed,
    build_statistics,
    coerce_per_path_row,
)


DEFAULT_SUMMARY = (
    "revision/experiments/outputs/abides_temporary_commitment_benchmark_v1/"
    "summary.json"
)
DEFAULT_CONFIG = "revision/configs/abides_temporary_commitment_benchmark_v1.json"
DEFAULT_BUNDLE = (
    "revision/paper/artifacts/abides_temporary_commitment_benchmark_v1"
)
PAPER_PER_PATH = (
    "revision/paper/data/abides_temporary_commitment_benchmark_v1_per_path.csv"
)
PAPER_PER_SEED = (
    "revision/paper/data/abides_temporary_commitment_benchmark_v1_per_seed.csv"
)


def export_bundle(
    summary_path: Path,
    config_path: Path,
    bundle: Path,
    result_archival_commit: str | None,
) -> dict[str, Any]:
    output = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    run_state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    results = json.loads((output / "rows.json").read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != manifest["config"]:
        raise RuntimeError("temporary commitment config differs from execution manifest")
    if bool(config.get("engineering_smoke")):
        raise RuntimeError("engineering smoke cannot be exported")
    methods = list(config["methods"])
    contrasts = list(config["contrast_methods"])
    per_path = [coerce_per_path_row(row) for row in read_csv(output / "per_path.csv")]
    per_path.sort(
        key=lambda row: (
            int(row["policy_seed"]),
            int(row["evaluation_seed"]),
            methods.index(str(row["method"])),
        )
    )
    per_seed = aggregate_per_seed(per_path, methods, contrasts)
    statistics = build_statistics(
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
    expected_paths = (
        len(config["policy_seeds"])
        * len(config["evaluation_seeds"])
        * len(methods)
    )
    checks: dict[str, Any] = {
        "runner_checks": dict(summary["checks"]),
        "per_path_row_count": {
            "actual": len(per_path),
            "expected": expected_paths,
            "pass": len(per_path) == expected_paths,
        },
        "per_seed_row_count": {
            "actual": len(per_seed),
            "expected": len(config["policy_seeds"]),
            "pass": len(per_seed) == len(config["policy_seeds"]),
        },
        "budget_calibration": {
            method: dict(config["mechanism_parameters"][method])
            for method in config["new_adapter_methods"]
        },
        "statistics": statistics,
        "implementation_status": str(summary["implementation_status"]),
        "runner_scientific_result": str(summary["scientific_result"]),
        "scientific_result": "not_evaluated_integrity_fail",
    }
    bundle.mkdir(parents=True, exist_ok=True)
    write_csv(bundle / "per_path.csv", per_path)
    write_csv(bundle / "per_seed.csv", per_seed)
    step_count = combine_step_records(results, bundle / "step_records.jsonl.gz")
    checks["step_record_count"] = {
        "actual": step_count,
        "expected": expected_paths * int(config["episode_steps"]),
        "pass": step_count == expected_paths * int(config["episode_steps"]),
    }
    atomic_write_json(bundle / "checks.json", checks)
    atomic_write_text(
        bundle / "README.md",
        """# ABIDES Temporary Commitment Audit Bundle v1

This bundle contains the primitive per-path table, derived per-seed table,
paired inference, and compressed per-decision records for the frozen ABIDES
adaptation of Banerjee-style temporary buyer commitment.

The frozen implementation status is `fail` because the literature-form
adapter missed the response-convergence threshold in two policy seeds. All
efficacy statistics are retained for auditability but are not confirmatory.
The calibrated operating point is activation-unmatched and is never a
same-budget superiority test.
""",
    )
    source_files = (
        "per_path.csv",
        "per_seed.csv",
        "step_records.jsonl.gz",
        "checks.json",
        "README.md",
    )
    events = logbook_events(output / "logbook.jsonl")
    latest = events[-1] if events else None
    audit_manifest: dict[str, Any] = {
        "protocol_version": str(config["protocol_version"]),
        "protocol_fingerprint": str(manifest["protocol_fingerprint"]),
        "protocol_freeze_commit": git_output(
            "log", "-1", "--format=%H", "--", str(config_path.relative_to(REPO))
        ),
        "execution_commit": str(manifest["git_commit"]),
        "result_archival_commit": validate_archival_commit(
            result_archival_commit
        ),
        "formal_config_sha256": sha256_file(config_path),
        "calibration_artifact_sha256": str(
            config["calibration_artifact_sha256"]
        ),
        "source_base_fingerprint": str(config["source_base_fingerprint"]),
        "source_v5_fingerprint": str(config["source_v5_fingerprint"]),
        "artifact_hashes": dict(manifest["artifacts"]),
        "source_model_digests": {
            str(result["policy_seed"]): {
                "base": result["base_model_digest"],
                "v5": result["v5_model_digests"],
                "new": result["final_model_digests"],
            }
            for result in results
        },
        "execution_records": {
            str(result["policy_seed"]): {
                "completed_episodes": int(result["completed_episodes"]),
                "initial_adapter_digest": result["initial_adapter_digest"],
                "convergence": result["convergence"],
                "temporary_commitment_integrity": result["final_evaluation"][
                    "temporary_commitment_integrity"
                ],
                "temporary_commitment_takeups": result["final_evaluation"][
                    "temporary_commitment_takeups"
                ],
                "step_records_sha256": result["step_records_sha256"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
            for result in results
        },
        "abides_commit": "f9cbe51342b7dedd9587e4e069040d68a5c6477f",
        "docker_image": str(config["docker_image"]),
        "docker_image_id": str(manifest["artifacts"]["docker_image_id"]),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "policy_seeds": list(config["policy_seeds"]),
        "calibration_environment_seeds": list(
            config["calibration_environment_seeds"]
        ),
        "evaluation_seeds": list(config["evaluation_seeds"]),
        "started_at": manifest["started_at"],
        "finished_at": None if latest is None else latest.get("timestamp"),
        "completed_policy_seeds": list(run_state["completed_policy_seeds"]),
        "pending_policy_seeds": list(run_state["pending_policy_seeds"]),
        "failures": list(run_state["failures"]),
        "completed_resume_verified": resume_verified(output / "logbook.jsonl"),
        "implementation_status": str(summary["implementation_status"]),
        "scientific_result": "not_evaluated_integrity_fail",
        "git_dirty_status_at_export": git_output("status", "--porcelain"),
        "output_file_sha256": {
            name: sha256_file(bundle / name) for name in source_files
        },
    }
    atomic_write_json(bundle / "manifest.json", audit_manifest)
    checksum_files = (*source_files, "manifest.json")
    atomic_write_text(
        bundle / "SHA256SUMS",
        "".join(
            f"{sha256_file(bundle / name)}  {name}\n" for name in checksum_files
        ),
    )
    paper_path = REPO / PAPER_PER_PATH
    paper_seed = REPO / PAPER_PER_SEED
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / "per_path.csv", paper_path)
    shutil.copyfile(bundle / "per_seed.csv", paper_seed)
    return audit_manifest


def check_export(bundle: Path) -> None:
    for source, target in (
        (bundle / "per_path.csv", REPO / PAPER_PER_PATH),
        (bundle / "per_seed.csv", REPO / PAPER_PER_SEED),
    ):
        if not source.is_file() or not target.is_file():
            raise RuntimeError("temporary commitment paper export is missing")
        if source.read_bytes() != target.read_bytes():
            raise RuntimeError("temporary commitment paper export differs from bundle")
    print(canonical_json({"status": "export_check_pass", "bundle": str(bundle)}))


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--result-archival-commit")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = REPO / args.bundle
    if args.check:
        check_export(bundle)
        return
    manifest = export_bundle(
        REPO / args.summary,
        REPO / args.config,
        bundle,
        args.result_archival_commit,
    )
    print(
        canonical_json(
            {
                "status": "exported",
                "bundle": str(bundle.relative_to(REPO)),
                "protocol_fingerprint": manifest["protocol_fingerprint"],
            }
        )
    )


if __name__ == "__main__":
    main()
