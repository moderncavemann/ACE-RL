#!/usr/bin/env python3
"""Verify hashes, data counts, scope, and double-blind metadata."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifact-manifest.json"
ALLOWED_GIT_IDENTITY = "Anonymous Authors\x00anonymous@invalid"
ALLOWED_EXTERNAL_COMMITS = {
    "f9cbe51342b7dedd9587e4e069040d68a5c6477f",  # ABIDES
    "de66d7a4b23fac2513f56f96f73b3f5cb96695ac",  # official MAPPO/IPPO
}
TEXT_NAMES = {"Dockerfile", ".gitignore"}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
COMMIT_FIELD = re.compile(
    r'"[^"\n]*commit[^"\n]*"\s*:\s*"([0-9a-f]{7,40})"', re.I
)
FORBIDDEN_PATTERNS = {
    "absolute_user_path": re.compile(r"/(?:Users|home)/[^/\s]+", re.I),
    "legacy_archive": re.compile(r"archive/v1_rejected_submission|REVISION_TRIAGE", re.I),
    "submission_history": re.compile(r"post-rejection|rejected submission", re.I),
    "source_remote_metadata": re.compile(r"remote_at_formal_run|origin_main", re.I),
}
IGNORED_RUNTIME_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_text(path: Path) -> bool:
    return path.name in TEXT_NAMES or path.suffix.lower() in TEXT_SUFFIXES


def is_ignored_runtime_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return bool(IGNORED_RUNTIME_PARTS.intersection(relative.parts)) or path.suffix == ".pyc"


def row_count(path: Path) -> int | None:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.reader(handle)) - 1
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    return None


def walk_json(value: Any, path: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if "commit" in str(key).lower() and isinstance(child, str):
                if re.fullmatch(r"[0-9a-f]{7,40}", child, re.I):
                    if child.lower() not in ALLOWED_EXTERNAL_COMMITS:
                        violations.append(child_path)
            violations.extend(walk_json(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(walk_json(child, f"{path}[{index}]"))
    return violations


def verify_git_identity() -> list[str]:
    if not (ROOT / ".git").exists():
        return []
    completed = subprocess.run(
        ["git", "log", "--format=%an%x00%ae"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    identities = {line for line in completed.stdout.splitlines() if line}
    return sorted(identity for identity in identities if identity != ALLOWED_GIT_IDENTITY)


def scan_compressed_records(path: Path, relative: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    categories: set[str] = set()
    internal_commits: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            for category, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(line):
                    categories.add(category)
            if any(email.lower() != "anonymous@invalid" for email in EMAIL.findall(line)):
                categories.add("email_address")
            for match in COMMIT_FIELD.finditer(line):
                commit = match.group(1).lower()
                if commit not in ALLOWED_EXTERNAL_COMMITS:
                    internal_commits.add(commit)
    errors.extend({"check": category, "file": relative} for category in sorted(categories))
    errors.extend(
        {"check": "internal_commit_reference", "file": relative, "value": commit}
        for commit in sorted(internal_commits)
    )
    return errors


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    declared = dict(manifest["files"])
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and path.name != MANIFEST.name
        and not is_ignored_runtime_file(path)
    }
    errors: list[dict[str, Any]] = []
    if actual != set(declared):
        errors.append(
            {
                "check": "file_allowlist",
                "missing": sorted(set(declared) - actual),
                "unexpected": sorted(actual - set(declared)),
            }
        )

    for relative, expected in declared.items():
        path = ROOT / relative
        if not path.exists():
            continue
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            errors.append({"check": "file_hash", "file": relative})

    for relative, expected in manifest["row_counts"].items():
        observed = row_count(ROOT / relative)
        if observed != expected:
            errors.append(
                {
                    "check": "row_count",
                    "file": relative,
                    "expected": expected,
                    "observed": observed,
                }
            )

    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.relative_to(ROOT).parts
            or is_ignored_runtime_file(path)
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.name.endswith(".jsonl.gz"):
            errors.extend(scan_compressed_records(path, relative))
            continue
        if not is_text(path):
            continue
        text = path.read_text(encoding="utf-8")
        if relative != "scripts/verify_release.py":
            for category, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(text):
                    errors.append({"check": category, "file": relative})
        for email in EMAIL.findall(text):
            if email.lower() != "anonymous@invalid":
                errors.append({"check": "email_address", "file": relative})
        if path.suffix == ".json":
            try:
                commit_paths = walk_json(json.loads(text))
            except json.JSONDecodeError:
                commit_paths = []
            for commit_path in commit_paths:
                errors.append(
                    {
                        "check": "internal_commit_reference",
                        "file": relative,
                        "json_path": commit_path,
                    }
                )

    for identity in verify_git_identity():
        errors.append({"check": "git_author_identity", "identity": identity})

    result = {
        "status": "pass" if not errors else "fail",
        "file_count": len(declared),
        "dataset_count": len(manifest["row_counts"]),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
