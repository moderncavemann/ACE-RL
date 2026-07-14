#!/usr/bin/env python3
"""Export standard-learner and ecology holdouts into one audit bundle."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.utils.abides_audit import (  # noqa: E402
    IDENTITY_FIELDS,
    METRIC_FIELDS,
    aggregate_per_seed,
    build_statistics,
    coerce_per_path_row,
)


PROTOCOL_VERSION = "abides_holdout_audit_bundle_v1"
BUNDLE = REPO / "revision/paper/artifacts/abides_holdout_audit_bundle_v1"
PAPER_PER_PATH = REPO / "revision/paper/data/abides_holdout_v1_per_path.csv"
PAPER_PER_SEED = REPO / "revision/paper/data/abides_holdout_v1_per_seed.csv"
OFFICIAL_CONFIG = REPO / "revision/configs/abides_standard_learner_holdout_v1.json"
OFFICIAL_OUTPUT = REPO / "revision/experiments/outputs/abides_standard_learner_holdout_v1"
ECOLOGY_AUDIT_OUTPUT = (
    REPO / "revision/experiments/outputs/abides_holdout_audit_evaluation_v3"
)
ECOLOGY_CONFIGS = {
    "ecology_e1": REPO / "revision/configs/abides_ecology_e1_response_v1.json",
    "ecology_e2": REPO / "revision/configs/abides_ecology_e2_response_v1.json",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_714
ABIDES_COMMIT = "f9cbe51342b7dedd9587e4e069040d68a5c6477f"
OFFICIAL_METHODS = (
    "no_intervention",
    "risk_aware",
    "matched_random",
    "brero_threshold_buybox_budget_matched",
)
ECOLOGY_METHODS = (
    "no_intervention",
    "risk_aware",
    "matched_random",
    "spread_only_threshold",
    "always_on",
    "no_routing",
)
GROUPS = (
    "official_full_policy_MAPPO",
    "official_IPPO",
    "ecology_e1",
    "ecology_e2",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    if not rows:
        raise ValueError("cannot export an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def logbook_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_event_timestamp(path: Path, event_name: str) -> str:
    matches = [
        str(event["timestamp"])
        for event in logbook_events(path)
        if event.get("event") == event_name and event.get("timestamp")
    ]
    if not matches:
        raise RuntimeError(f"logbook has no {event_name} event: {path}")
    return matches[-1]


def validate_archival_commit(commit: str | None) -> str | None:
    if commit is None:
        return None
    resolved = git_output("rev-parse", f"{commit}^{{commit}}")
    if resolved != commit:
        raise ValueError("result archival commit must be a full commit SHA")
    return resolved


def official_results() -> list[dict[str, Any]]:
    rows = json.loads((OFFICIAL_OUTPUT / "rows.json").read_text(encoding="utf-8"))
    return sorted(rows, key=lambda row: (str(row["learner"]), int(row["policy_seed"])))


def ecology_audit_results() -> list[dict[str, Any]]:
    rows = json.loads(
        (ECOLOGY_AUDIT_OUTPUT / "rows.json").read_text(encoding="utf-8")
    )
    return sorted(rows, key=lambda row: (str(row["cell"]), int(row["policy_seed"])))


def normalized_path_row(
    row: Mapping[str, Any], holdout: str, learner: str
) -> dict[str, Any]:
    normalized = coerce_per_path_row(row)
    return {"holdout": holdout, "learner": learner, **normalized}


def collect_per_path(
    official: Sequence[Mapping[str, Any]],
    ecology: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for result in official:
        learner = str(result["learner"])
        rows.extend(
            normalized_path_row(row, "standard_learner", learner)
            for row in result["final_evaluation"]["per_path"]
        )
    for result in ecology:
        rows.extend(
            normalized_path_row(
                row, str(result["cell"]), str(result["learner"])
            )
            for row in result["per_path"]
        )
    method_order = {method: index for index, method in enumerate(
        (*OFFICIAL_METHODS, *[m for m in ECOLOGY_METHODS if m not in OFFICIAL_METHODS])
    )}
    rows.sort(
        key=lambda row: (
            str(row["holdout"]),
            str(row["learner"]),
            int(row["policy_seed"]),
            int(row["evaluation_seed"]),
            method_order[str(row["method"])],
        )
    )
    return rows


def group_rows(per_path: Sequence[Mapping[str, Any]], group: str) -> list[dict[str, Any]]:
    if group.startswith("official_"):
        return [
            dict(row)
            for row in per_path
            if row["holdout"] == "standard_learner" and row["learner"] == group
        ]
    return [dict(row) for row in per_path if row["holdout"] == group]


def group_contract(group: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if group.startswith("official_"):
        return (
            OFFICIAL_METHODS,
            ("matched_random", "brero_threshold_buybox_budget_matched"),
            ("brero_threshold_buybox_budget_matched",),
        )
    return (
        ECOLOGY_METHODS,
        (
            "matched_random",
            "no_intervention",
            "spread_only_threshold",
            "always_on",
            "no_routing",
        ),
        (
            "no_intervention",
            "spread_only_threshold",
            "always_on",
            "no_routing",
        ),
    )


def derive_statistics(
    per_path: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    statistics = {}
    per_seed = []
    for group in GROUPS:
        rows = group_rows(per_path, group)
        methods, contrasts, secondary = group_contract(group)
        statistics[group] = build_statistics(
            rows,
            methods,
            contrasts,
            secondary,
            BOOTSTRAP_REPLICATES,
            BOOTSTRAP_SEED,
        )
        learner = (
            group
            if group.startswith("official_")
            else "compact_MAPPO_style_intervention_adapter"
        )
        holdout = "standard_learner" if group.startswith("official_") else group
        per_seed.extend(
            {"holdout": holdout, "learner": learner, **row}
            for row in aggregate_per_seed(rows, methods, contrasts)
        )
    per_seed.sort(
        key=lambda row: (
            str(row["holdout"]), str(row["learner"]), int(row["policy_seed"])
        )
    )
    return statistics, per_seed


def canonical_step_source(
    official: Sequence[Mapping[str, Any]], ecology: Sequence[Mapping[str, Any]]
) -> Iterable[dict[str, Any]]:
    for result in official:
        source = REPO / str(result["step_records_file"])
        if sha256_file(source) != str(result["step_records_sha256"]):
            raise RuntimeError(f"official step source hash mismatch: {source}")
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["holdout"] = "standard_learner"
                row["learner"] = str(result["learner"])
                yield row
    for result in ecology:
        source = REPO / str(result["step_records_file"])
        if sha256_file(source) != str(result["step_records_sha256"]):
            raise RuntimeError(f"ecology step source hash mismatch: {source}")
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def write_combined_steps(
    official: Sequence[Mapping[str, Any]], ecology: Sequence[Mapping[str, Any]], path: Path
) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    stream_hash = hashlib.sha256()
    count = 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as target:
                for row in canonical_step_source(official, ecology):
                    line = canonical_json(row) + "\n"
                    target.write(line)
                    stream_hash.update(line.encode("utf-8"))
                    count += 1
    temporary.replace(path)
    return count, stream_hash.hexdigest()


def claim_classification(statistics: Mapping[str, Any]) -> dict[str, str]:
    output = {}
    for group in GROUPS:
        interval = statistics[group]["contrasts"]["matched_random"]["t_interval"]
        if float(interval["ci_low"]) > 0.0:
            output[group] = "favorable_timing_interval"
        elif float(interval["mean"]) > 0.0:
            output[group] = "positive_direction_inconclusive_interval"
        else:
            output[group] = "nonpositive_timing_direction"
    return output


def source_provenance(
    official: Sequence[Mapping[str, Any]], ecology: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    official_manifest = json.loads(
        (OFFICIAL_OUTPUT / "formal/manifest.json").read_text(encoding="utf-8")
    )
    audit_manifest = json.loads(
        (ECOLOGY_AUDIT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
    )
    official_config = json.loads(OFFICIAL_CONFIG.read_text(encoding="utf-8"))
    ecology_configs = {
        cell: json.loads(path.read_text(encoding="utf-8"))
        for cell, path in ECOLOGY_CONFIGS.items()
    }
    ecology_base_configs = {
        cell: REPO
        / "revision/configs"
        / f"abides_{cell}_base_v1.json"
        for cell in ECOLOGY_CONFIGS
    }
    if {
        json.loads(path.read_text(encoding="utf-8"))["abides_source_commit"]
        for path in ecology_base_configs.values()
    } != {ABIDES_COMMIT}:
        raise RuntimeError("ecology ABIDES source commit differs")
    return {
        "official": {
            "config_sha256": sha256_file(OFFICIAL_CONFIG),
            "selection_sha256": sha256_file(
                REPO
                / "revision/configs/abides_standard_learner_holdout_v1_selection.json"
            ),
            "protocol_fingerprint": official_manifest["protocol_fingerprint"],
            "execution_commit": official_manifest["git_commit"],
            "summary_sha256": sha256_file(OFFICIAL_OUTPUT / "summary.json"),
            "result_sha256": {
                f"{row['learner']}:{row['policy_seed']}": sha256_file(
                    OFFICIAL_OUTPUT
                    / "formal"
                    / str(row["learner"])
                    / f"seed_{int(row['policy_seed'])}"
                    / "result.json"
                )
                for row in official
            },
            "official_repository": "https://github.com/marlbenchmark/on-policy",
            "official_commit": "de66d7a4b23fac2513f56f96f73b3f5cb96695ac",
            "official_license": "MIT",
            "abides_commit": ABIDES_COMMIT,
            "docker_image": official_config["docker_image"],
            "docker_image_id": official_manifest["artifacts"]["docker_image_id"],
            "started_at": official_manifest["started_at"],
            "finished_at": latest_event_timestamp(
                OFFICIAL_OUTPUT / "logbook.jsonl", "formal_finished"
            ),
        },
        "ecology_audit": {
            "evaluator_version": audit_manifest["evaluator_version"],
            "audit_fingerprint": audit_manifest["audit_fingerprint"],
            "execution_commit": audit_manifest["git_commit"],
            "summary_sha256": sha256_file(ECOLOGY_AUDIT_OUTPUT / "summary.json"),
            "result_sha256": {
                f"{row['cell']}:{row['policy_seed']}": sha256_file(
                    ECOLOGY_AUDIT_OUTPUT
                    / str(row["cell"])
                    / f"seed_{int(row['policy_seed'])}"
                    / "result.json"
                )
                for row in ecology
            },
            "response_configs": {
                cell: sha256_file(path) for cell, path in ECOLOGY_CONFIGS.items()
            },
            "source_descriptors": audit_manifest["sources"],
            "abides_commit": ABIDES_COMMIT,
            "base_config_sha256": {
                cell: sha256_file(path)
                for cell, path in ecology_base_configs.items()
            },
            "docker_images": {
                cell: ecology_configs[cell]["docker_image"]
                for cell in ECOLOGY_CONFIGS
            },
            "docker_image_ids": audit_manifest["docker_images"],
            "started_at": audit_manifest["started_at"],
            "finished_at": latest_event_timestamp(
                ECOLOGY_AUDIT_OUTPUT / "logbook.jsonl", "audit_evaluation_finished"
            ),
        },
    }


def model_digest_provenance(
    per_path: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in per_path:
        key = ":".join(
            (
                str(row["holdout"]),
                str(row["learner"]),
                str(row["policy_seed"]),
                str(row["method"]),
            )
        )
        digest_value = str(row["model_digest"])
        if key in output and output[key] != digest_value:
            raise RuntimeError(f"one policy/method has multiple model digests: {key}")
        output[key] = digest_value
    return output


def export_bundle(result_archival_commit: str | None = None) -> dict[str, Any]:
    official = official_results()
    ecology = ecology_audit_results()
    official_summary = json.loads(
        (OFFICIAL_OUTPUT / "summary.json").read_text(encoding="utf-8")
    )
    ecology_audit_summary = json.loads(
        (ECOLOGY_AUDIT_OUTPUT / "summary.json").read_text(encoding="utf-8")
    )
    ecology_summaries = {
        cell: json.loads(
            (REPO / json.loads(path.read_text(encoding="utf-8"))["output_dir"] / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for cell, path in ECOLOGY_CONFIGS.items()
    }
    per_path = collect_per_path(official, ecology)
    statistics, per_seed = derive_statistics(per_path)
    classification = claim_classification(statistics)

    BUNDLE.mkdir(parents=True, exist_ok=True)
    per_path_fields = ("holdout", "learner", *IDENTITY_FIELDS, *METRIC_FIELDS)
    write_csv(BUNDLE / "per_path.csv", per_path, per_path_fields)
    seed_fields = ["holdout", "learner", "policy_seed"] + sorted(
        {key for row in per_seed for key in row} - {"holdout", "learner", "policy_seed"}
    )
    write_csv(BUNDLE / "per_seed.csv", per_seed, seed_fields)
    atomic_write_json(BUNDLE / "statistics.json", statistics)
    step_count, step_stream_sha = write_combined_steps(
        official, ecology, BUNDLE / "step_records.jsonl.gz"
    )

    official_resume = any(
        event.get("event") == "completed_resume_verified"
        and int(event.get("containers_started", -1)) == 0
        and int(event.get("result_count", -1)) == 20
        for event in logbook_events(OFFICIAL_OUTPUT / "logbook.jsonl")
    )
    ecology_resume = {
        cell: any(
            event.get("event") == "run_resumed"
            and not event.get("pending_policy_seeds")
            for event in logbook_events(
                REPO
                / json.loads(path.read_text(encoding="utf-8"))["output_dir"]
                / "logbook.jsonl"
            )
        )
        for cell, path in ECOLOGY_CONFIGS.items()
    }
    checks = {
        "official_runner_checks": dict(official_summary["checks"]),
        "official_implementation_status": official_summary["implementation_status"],
        "ecology_runner_checks": {
            cell: dict(summary["checks"])
            for cell, summary in ecology_summaries.items()
        },
        "ecology_status": {
            cell: summary["status"] for cell, summary in ecology_summaries.items()
        },
        "ecology_audit_status": ecology_audit_summary["status"],
        "ecology_audit_all_checks_pass": ecology_audit_summary["all_checks_pass"],
        "completed_resume_verified": {
            "official": official_resume,
            **ecology_resume,
            "ecology_audit": bool(ecology_audit_summary["completed_resume"]),
        },
        "per_path_count": {"actual": len(per_path), "expected": 2_000},
        "per_seed_count": {"actual": len(per_seed), "expected": 40},
        "step_record_count": {"actual": step_count, "expected": 200_000},
        "step_stream_sha256": step_stream_sha,
        "claim_classification": classification,
        "all_integrity_checks_pass": (
            official_summary["implementation_status"] == "pass"
            and all(official_summary["checks"].values())
            and all(summary["status"] == "pass" for summary in ecology_summaries.values())
            and all(
                all(summary["checks"].values()) for summary in ecology_summaries.values()
            )
            and ecology_audit_summary["status"] == "pass"
            and bool(ecology_audit_summary["all_checks_pass"])
            and official_resume
            and all(ecology_resume.values())
            and bool(ecology_audit_summary["completed_resume"])
            and len(per_path) == 2_000
            and len(per_seed) == 40
            and step_count == 200_000
        ),
    }
    atomic_write_json(BUNDLE / "checks.json", checks)
    readme = """# ACE-RL ABIDES Holdout Audit Bundle v1

This bundle contains the standard-learner and two pre-frozen ecology holdouts.
`per_path.csv` is the primitive table. `per_seed.csv` and `statistics.json`
are derived only from that table. `step_records.jsonl.gz` contains 200,000
decision records with holdout and learner identifiers.

The official learner source is `marlbenchmark/on-policy` at pinned commit
`de66d7a4b23fac2513f56f96f73b3f5cb96695ac`. The ecology policies use the
repository's compact MAPPO-style intervention-adapter and are not renamed as
standard MAPPO. Compact ecology records expose realized flow shares rather
than latent routing probabilities; the step records label that provenance.

The cross-group audit uses an equal-weight mean over the ten evaluation paths
inside each policy seed. The compact ecology runner's preregistered source
summary pools low/high-risk steps instead. The verifier reconstructs that
source estimand as a separate integrity check; both definitions preserve the
reported direction, and they are not silently treated as identical numbers.

Inference uses policy-seed paired t intervals, exact sign tests, and a
10,000-replicate two-way paired bootstrap with RNG seed 20260714. Secondary
contrasts use Holm adjustment. Groups are never pooled to hide a reversal.
Run `python revision/experiments/verify_abides_holdout_audit_bundle.py` to
verify source correspondence, hashes, grids, statistics, and paper exports.
"""
    atomic_write_text(BUNDLE / "README.md", readme)

    payload_files = (
        "per_path.csv",
        "per_seed.csv",
        "statistics.json",
        "step_records.jsonl.gz",
        "checks.json",
        "README.md",
    )
    provenance = source_provenance(official, ecology)
    source_fingerprint = hashlib.sha256(
        canonical_json(provenance).encode("utf-8")
    ).hexdigest()
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": source_fingerprint,
        "protocol_freeze_commits": {
            "official_selection": git_output(
                "log", "-1", "--format=%H", "--", str(OFFICIAL_CONFIG.relative_to(REPO))
            ),
            "ecology": git_output(
                "log",
                "-1",
                "--format=%H",
                "--",
                "revision/configs/abides_ecology_e1_response_v1.json",
                "revision/configs/abides_ecology_e2_response_v1.json",
            ),
        },
        "source_provenance": provenance,
        "source_model_digests": model_digest_provenance(per_path),
        "result_archival_commit": validate_archival_commit(result_archival_commit),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "groups": list(GROUPS),
        "policy_seeds": {
            "standard_learner": list(range(550000, 550010)),
            "ecology_e1": list(range(560000, 560010)),
            "ecology_e2": list(range(570000, 570010)),
        },
        "evaluation_seeds": {
            "standard_learner": list(range(552000, 552010)),
            "ecology_e1": list(range(562000, 562010)),
            "ecology_e2": list(range(572000, 572010)),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python revision/experiments/export_abides_holdout_audit_bundle.py",
        "source_completion": {
            "official": {
                "result_count": len(official),
                "pending": [],
                "failures": json.loads(
                    (OFFICIAL_OUTPUT / "failures.json").read_text(encoding="utf-8")
                )["failures"],
                "completed_resume": official_resume,
            },
            "ecology_audit": {
                "result_count": len(ecology),
                "pending": [],
                "failures": ecology_audit_summary["failures"],
                "completed_resume": ecology_audit_summary["completed_resume"],
            },
        },
        "git_dirty_status_at_export": git_output("status", "--porcelain"),
        "claim_classification": classification,
        "output_file_sha256": {
            name: sha256_file(BUNDLE / name) for name in payload_files
        },
    }
    atomic_write_json(BUNDLE / "manifest.json", manifest)
    checksum_files = (*payload_files, "manifest.json")
    atomic_write_text(
        BUNDLE / "SHA256SUMS",
        "".join(
            f"{sha256_file(BUNDLE / name)}  {name}\n" for name in checksum_files
        ),
    )
    PAPER_PER_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BUNDLE / "per_path.csv", PAPER_PER_PATH)
    shutil.copyfile(BUNDLE / "per_seed.csv", PAPER_PER_SEED)
    return manifest


def parse_checksums(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", 1)
        output[name] = value
    return output


def check_export() -> None:
    for name, expected in parse_checksums(BUNDLE / "SHA256SUMS").items():
        if sha256_file(BUNDLE / name) != expected:
            raise RuntimeError(f"bundle hash differs: {name}")
    for source, paper in (
        (BUNDLE / "per_path.csv", PAPER_PER_PATH),
        (BUNDLE / "per_seed.csv", PAPER_PER_SEED),
    ):
        if source.read_bytes() != paper.read_bytes():
            raise RuntimeError(f"paper export differs: {paper}")
    print(canonical_json({"status": "export_check_pass", "bundle": str(BUNDLE)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-archival-commit")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        check_export()
        return
    manifest = export_bundle(args.result_archival_commit)
    print(
        canonical_json(
            {
                "status": "exported",
                "bundle": str(BUNDLE.relative_to(REPO)),
                "protocol_fingerprint": manifest["protocol_fingerprint"],
            }
        )
    )


if __name__ == "__main__":
    main()
