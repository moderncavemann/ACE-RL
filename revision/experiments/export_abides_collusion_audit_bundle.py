#!/usr/bin/env python3
"""Export the frozen ABIDES deviation diagnostic as a deterministic audit bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from run_abides_mappo_full_main import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


DEFAULT_OUTPUT = "revision/experiments/outputs/abides_collusion_diagnostic_v1"
DEFAULT_BUNDLE = "revision/paper/artifacts/abides_collusion_diagnostic_v1"
PAPER_PER_EVENT = "revision/paper/data/abides_collusion_diagnostic_v1_per_event.csv"
PAPER_PER_SEED = "revision/paper/data/abides_collusion_diagnostic_v1_per_seed.csv"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalize_csv(source: Path, target: Path) -> None:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise RuntimeError(f"CSV has no header: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


def validate_archival_commit(commit: str | None) -> str | None:
    if commit is None:
        return None
    full = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=REPO, text=True
    ).strip()
    if len(full) != 40:
        raise RuntimeError("result archival commit did not resolve to a full SHA")
    return full


def load_seed_results(output_dir: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "seed_results").glob("seed_*.json"))
    ]
    if not rows:
        raise RuntimeError("diagnostic output has no seed results")
    return sorted(rows, key=lambda row: int(row["policy_seed"]))


def export_bundle(
    output_dir: Path, bundle: Path, result_archival_commit: str | None
) -> dict[str, Any]:
    output_manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    aggregate = json.loads(
        (output_dir / "aggregate.json").read_text(encoding="utf-8")
    )
    completion = json.loads(
        (output_dir / "completion.json").read_text(encoding="utf-8")
    )
    results = load_seed_results(output_dir)
    if aggregate["implementation_status"] != "pass":
        raise RuntimeError("cannot archive a failed diagnostic implementation")
    if not completion["completed_resume_verification"] or not completion[
        "aggregate_unchanged_on_completed_resume"
    ]:
        raise RuntimeError("completed-resume verification is missing")
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ("aggregate.json", "completion.json"):
        payload = json.loads((output_dir / name).read_text(encoding="utf-8"))
        atomic_write_json(bundle / name, payload)
    for name in ("per_event.csv", "per_seed.csv", "trajectory.csv"):
        normalize_csv(output_dir / name, bundle / name)
    readme = """# ABIDES Collusion Diagnostic v1 Audit Bundle

This bundle is the repo-visible, audit-derived export of the preregistered
forced-undercut event study. It contains 20 policy seeds, 400 paired events,
and step records for sham, forced-undercut, and upward-deviation arms from
`t0` through `t0+10`.

External interpretation is fail-closed. The result supports only
`high-spread coordination`; it does not establish punishment-supported
collusion. Run `python revision/experiments/verify_abides_collusion_diagnostic.py`
to recompute path integrity, policy-seed intervals, exact sign tests, and the
claim boundary.
"""
    atomic_write_text(bundle / "README.md", readme)
    source_files = (
        "aggregate.json",
        "completion.json",
        "per_event.csv",
        "per_seed.csv",
        "trajectory.csv",
        "README.md",
    )
    manifest = {
        "protocol_version": output_manifest["protocol_version"],
        "protocol_fingerprint": output_manifest["protocol_fingerprint"],
        "config": output_manifest["config"],
        "config_sha256": output_manifest["config_sha256"],
        "execution_commit": output_manifest["execution_commit"],
        "result_archival_commit": validate_archival_commit(result_archival_commit),
        "docker_image_id": output_manifest["docker_image_id"],
        "source_base_manifest_sha256": output_manifest[
            "source_base_manifest_sha256"
        ],
        "source_policy_artifacts": {
            str(result["policy_seed"]): {
                "source_checkpoint_sha256": result["source_checkpoint_sha256"],
                "model_digest": result["model_digest"],
            }
            for result in results
        },
        "policy_seeds": [int(result["policy_seed"]) for result in results],
        "evaluation_seeds": list(output_manifest["config"]["evaluation_seeds"]),
        "event_count": int(aggregate["eligible_event_count"]),
        "trajectory_row_count": int(aggregate["trajectory_row_count"]),
        "implementation_status": aggregate["implementation_status"],
        "scientific_result": aggregate["scientific_result"],
        "required_wording": aggregate["required_wording"],
        "completed_resume_verified": bool(
            completion["completed_resume_verification"]
            and completion["aggregate_unchanged_on_completed_resume"]
            and int(completion["containers_started"]) == 0
        ),
        "output_file_sha256": {
            name: sha256_file(bundle / name) for name in source_files
        },
        "command": (
            "python revision/experiments/run_abides_collusion_diagnostic.py "
            "--config revision/configs/abides_collusion_diagnostic_v1.json"
        ),
    }
    atomic_write_json(bundle / "manifest.json", manifest)
    checksum_files = (*source_files, "manifest.json")
    atomic_write_text(
        bundle / "SHA256SUMS",
        "".join(
            f"{sha256_file(bundle / name)}  {name}\n" for name in checksum_files
        ),
    )
    for source_name, paper_path in (
        ("per_event.csv", REPO / PAPER_PER_EVENT),
        ("per_seed.csv", REPO / PAPER_PER_SEED),
    ):
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / source_name, paper_path)
    return manifest


def check_export(bundle: Path) -> None:
    for source_name, paper_path in (
        ("per_event.csv", REPO / PAPER_PER_EVENT),
        ("per_seed.csv", REPO / PAPER_PER_SEED),
    ):
        if (bundle / source_name).read_bytes() != paper_path.read_bytes():
            raise RuntimeError(f"paper export differs from audit source: {paper_path}")
    checksums = (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in checksums:
        expected, name = line.split("  ", 1)
        if sha256_file(bundle / name) != expected:
            raise RuntimeError(f"audit checksum mismatch: {name}")
    print(canonical_json({"status": "export_check_pass", "bundle": str(bundle)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--result-archival-commit")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bundle = REPO / args.bundle
    if args.check:
        check_export(bundle)
        return
    manifest = export_bundle(
        REPO / args.output, bundle, args.result_archival_commit
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
