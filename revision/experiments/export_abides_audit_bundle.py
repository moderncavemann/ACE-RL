#!/usr/bin/env python3
"""Export the formal ABIDES result into a self-contained audit bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.utils.abides_audit import (  # noqa: E402
    DEFAULT_CONTRASTS,
    aggregate_per_seed,
    build_statistics,
    coerce_per_path_row,
)


DEFAULT_DOMAIN_SUMMARY = (
    "revision/experiments/outputs/abides_domain_benchmark_v1/summary.json"
)
DEFAULT_BUNDLE = "revision/paper/artifacts/abides_audit_bundle_v1"
PAPER_PER_PATH = "revision/paper/data/abides_domain_benchmark_v1_per_path.csv"
PAPER_PER_SEED = "revision/paper/data/abides_domain_benchmark_v1_per_seed.csv"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot export an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def logbook_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def resume_verified(path: Path) -> bool:
    events = logbook_events(path)
    return any(
        event.get("event") == "completed_resume_verified"
        and int(event.get("containers_started", -1)) == 0
        and bool(event.get("aggregate_sha256"))
        for event in events
    )


def combine_step_records(results: Sequence[Mapping[str, Any]], output: Path) -> int:
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as target:
                for result in sorted(results, key=lambda row: int(row["policy_seed"])):
                    source = REPO / str(result["step_records_file"])
                    if sha256_file(source) != str(result["step_records_sha256"]):
                        raise RuntimeError(
                            f"step-record source hash mismatch: {source}"
                        )
                    with gzip.open(source, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            target.write(line)
                            count += 1
    temporary.replace(output)
    return count


def validate_archival_commit(commit: str | None) -> str | None:
    if commit is None:
        return None
    resolved = git_output("rev-parse", f"{commit}^{{commit}}")
    if resolved != commit:
        raise ValueError("result archival commit must be a full commit SHA")
    return resolved


def export_bundle(
    domain_summary: Path,
    bundle: Path,
    result_archival_commit: str | None = None,
) -> dict[str, Any]:
    output_dir = domain_summary.parent
    summary = json.loads(domain_summary.read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    run_state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    results = json.loads((output_dir / "rows.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    methods = list(config["methods"])
    if bool(config.get("engineering_smoke")):
        raise RuntimeError("engineering smoke cannot be exported as audit evidence")
    if summary.get("implementation_status") != "pass":
        raise RuntimeError("domain runner implementation status is not pass")
    if set(methods) != {
        "no_intervention",
        "risk_aware",
        "matched_random",
        "spread_only_budget_matched",
        "brero_threshold_buybox",
        "brero_threshold_buybox_budget_matched",
        "spread_only_threshold",
        "always_on",
        "no_routing",
    }:
        raise RuntimeError("formal method set changed")

    per_path = [
        coerce_per_path_row(row) for row in read_csv(output_dir / "per_path.csv")
    ]
    per_path.sort(
        key=lambda row: (
            int(row["policy_seed"]),
            int(row["evaluation_seed"]),
            methods.index(str(row["method"])),
        )
    )
    per_seed = aggregate_per_seed(per_path, methods, DEFAULT_CONTRASTS)
    statistics = build_statistics(
        per_path,
        methods,
        DEFAULT_CONTRASTS,
        ("spread_only_budget_matched", "matched_random"),
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]),
    )
    expected_paths = (
        len(config["policy_seeds"]) * len(config["evaluation_seeds"]) * len(methods)
    )
    budget = {
        method: dict(config["mechanism_parameters"][method])
        for method in (
            "spread_only_budget_matched",
            "brero_threshold_buybox_budget_matched",
        )
    }
    checks = {
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
        "budget_calibration": budget,
        "statistics": statistics,
        "implementation_status": summary["implementation_status"],
        "scientific_result": summary["scientific_result"],
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
    readme = """# ACE-RL ABIDES Audit Bundle v1

This bundle is the committed source for the domain-benchmark tables and
inference. `per_path.csv` is the primitive result table; `per_seed.csv` is
derived from it. `step_records.jsonl.gz` contains the paired per-decision
audit trail. Run `python revision/experiments/verify_abides_audit_bundle.py
--bundle revision/paper/artifacts/abides_audit_bundle_v1` to verify hashes,
recompute statistics, and rerun integrity checks.

The Brero threshold adaptation is calibration-unmatched and is therefore an
operating-point comparison, not a same-budget superiority test.
"""
    atomic_write_text(bundle / "README.md", readme)

    source_files = (
        "per_path.csv",
        "per_seed.csv",
        "step_records.jsonl.gz",
        "checks.json",
        "README.md",
    )
    freeze_commit = git_output(
        "log",
        "-1",
        "--format=%H",
        "--",
        "revision/configs/abides_domain_benchmark_v1.json",
    )
    events = logbook_events(output_dir / "logbook.jsonl")
    latest = events[-1] if events else None
    execution_records = {
        str(result["policy_seed"]): {
            "completed_episodes": int(result["completed_episodes"]),
            "initial_adapter_digest": str(result["initial_adapter_digest"]),
            "training_paths_equal": bool(result["training_paths_equal"]),
            "training_action_uniforms_equal": bool(
                result["training_action_uniforms_equal"]
            ),
            "gate_off_exact": bool(result["gate_off_exact"]),
            "convergence": dict(result["convergence"]),
            "step_records_sha256": str(result["step_records_sha256"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
        }
        for result in results
    }
    formal_config_path = REPO / "revision/configs/abides_domain_benchmark_v1.json"
    audit_manifest = {
        "protocol_version": str(config["protocol_version"]),
        "protocol_fingerprint": str(manifest["protocol_fingerprint"]),
        "protocol_freeze_commit": freeze_commit,
        "execution_commit": str(manifest["git_commit"]),
        "result_archival_commit": validate_archival_commit(result_archival_commit),
        "formal_config_sha256": sha256_file(formal_config_path),
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
        "execution_records": execution_records,
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
        "calibration_environment_seeds": list(config["calibration_environment_seeds"]),
        "evaluation_seeds": list(config["evaluation_seeds"]),
        "response_environment_seed_namespace": int(
            config["response_environment_seed_namespace"]
        ),
        "response_action_namespace": int(config["response_action_namespace"]),
        "evaluation_action_namespace": int(config["evaluation_action_namespace"]),
        "started_at": manifest["started_at"],
        "finished_at": None if latest is None else latest.get("timestamp"),
        "completed_policy_seeds": list(run_state["completed_policy_seeds"]),
        "pending_policy_seeds": list(run_state["pending_policy_seeds"]),
        "failures": list(run_state["failures"]),
        "retained_failure_events": [
            event for event in events if event.get("event") == "seed_failed"
        ],
        "command": (
            "python revision/experiments/run_abides_domain_benchmark.py "
            "--config revision/configs/abides_domain_benchmark_v1.json"
        ),
        "git_dirty_status_at_export": git_output("status", "--porcelain"),
        "completed_resume_verified": resume_verified(output_dir / "logbook.jsonl"),
        "implementation_status": str(summary["implementation_status"]),
        "scientific_result": str(summary["scientific_result"]),
        "output_file_sha256": {
            name: sha256_file(bundle / name) for name in source_files
        },
    }
    atomic_write_json(bundle / "manifest.json", audit_manifest)
    checksum_files = (*source_files, "manifest.json")
    checksum_text = "".join(
        f"{sha256_file(bundle / name)}  {name}\n" for name in checksum_files
    )
    atomic_write_text(bundle / "SHA256SUMS", checksum_text)

    paper_per_path = REPO / PAPER_PER_PATH
    paper_per_seed = REPO / PAPER_PER_SEED
    paper_per_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / "per_path.csv", paper_per_path)
    shutil.copyfile(bundle / "per_seed.csv", paper_per_seed)
    return audit_manifest


def check_export(bundle: Path) -> None:
    paper_per_path = REPO / PAPER_PER_PATH
    paper_per_seed = REPO / PAPER_PER_SEED
    for source, target in (
        (bundle / "per_path.csv", paper_per_path),
        (bundle / "per_seed.csv", paper_per_seed),
    ):
        if not source.exists() or not target.exists():
            raise RuntimeError(f"committed export is missing: {source} or {target}")
        if source.read_bytes() != target.read_bytes():
            raise RuntimeError(f"paper export differs from audit source: {target}")
    print(canonical_json({"status": "export_check_pass", "bundle": str(bundle)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-summary", default=DEFAULT_DOMAIN_SUMMARY)
    parser.add_argument("--output", default=DEFAULT_BUNDLE)
    parser.add_argument("--result-archival-commit")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = REPO / args.output
    if args.check:
        check_export(bundle)
        return
    manifest = export_bundle(
        REPO / args.domain_summary,
        bundle,
        result_archival_commit=args.result_archival_commit,
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
