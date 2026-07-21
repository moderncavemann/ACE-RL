#!/usr/bin/env python3
"""Build a fresh-history, double-blind release from an explicit allowlist."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "release" / "anonymous"
DEFAULT_OUTPUT = ROOT.parent / "ACE-RL-anonymous"
MARKER = ".anonymous-release-root"

SOURCE_FILES = {
    # Current ABIDES implementation.
    "revision/src/__init__.py",
    "revision/src/agents/__init__.py",
    "revision/src/agents/abides_mappo.py",
    "revision/src/agents/mappo.py",
    "revision/src/agents/official_on_policy_mappo.py",
    "revision/src/envs/__init__.py",
    "revision/src/envs/abides_docker_env.py",
    "revision/src/interventions/__init__.py",
    "revision/src/interventions/abides_gate.py",
    "revision/src/interventions/abides_abce.py",
    "revision/src/interventions/abides_literature_baselines.py",
    "revision/src/utils/__init__.py",
    "revision/src/utils/abides_audit.py",
    "revision/src/utils/abides_abce_audit.py",
    # Dockerized ABIDES matching-engine integration used by the final runs.
    "revision/infra/abides/Dockerfile",
    "revision/infra/abides/Dockerfile.abce",
    "revision/infra/abides/Dockerfile.domain-benchmark",
    "revision/infra/abides/README.md",
    "revision/infra/abides/ace_abides/__init__.py",
    "revision/infra/abides/ace_abides/adapter.py",
    "revision/infra/abides/ace_abides/economic_env.py",
    "revision/infra/abides/ace_abides/routing.py",
    "revision/infra/abides/requirements-gym.lock",
    "revision/infra/abides/run_smoke.sh",
    "revision/infra/abides/smoke_domain_benchmark.py",
    "revision/infra/abides/smoke_gym.py",
    "revision/infra/abides/smoke_multiagent.py",
    "revision/infra/abides/smoke_rmsc.py",
    "revision/infra/abides/smoke_routing.py",
    "revision/infra/abides/smoke_temporary_commitment.py",
    "revision/infra/abides/validate_economics.py",
    "revision/infra/abides/validate_oracle_timing.py",
    "revision/infra/abides/worker.py",
    # Frozen current configurations. Development v1-v4 cells are excluded. The
    # three current smoke configs are retained only as regression-test fixtures.
    "revision/configs/abides_adaptive_gate_shift_diagnostic_v1.json",
    "revision/configs/abides_abce_primary_followup_smoke_v1.json",
    "revision/configs/abides_abce_primary_followup_v1.json",
    "revision/configs/abides_collusion_diagnostic_v1.json",
    "revision/configs/abides_domain_baseline_calibration_v1.json",
    "revision/configs/abides_domain_baseline_calibration_v1_artifact.json",
    "revision/configs/abides_domain_benchmark_v1.json",
    "revision/configs/abides_domain_benchmark_smoke_v1.json",
    "revision/configs/abides_dose_routing_sensitivity_v1.json",
    "revision/configs/abides_dose_routing_sensitivity_v1_calibration.json",
    "revision/configs/abides_ecology_e1_base_v1.json",
    "revision/configs/abides_ecology_e1_response_v1.json",
    "revision/configs/abides_ecology_e2_base_v1.json",
    "revision/configs/abides_ecology_e2_response_v1.json",
    "revision/configs/abides_ecology_holdout_v1.json",
    "revision/configs/abides_mappo_adaptive_v5_full_main_primary_base_v1.json",
    "revision/configs/abides_mappo_adaptive_v5_full_main_primary_v1.json",
    "revision/configs/abides_mappo_adaptive_v5_supplement_v1.json",
    "revision/configs/abides_mappo_adaptive_v5_supplement_smoke_v1.json",
    "revision/configs/abides_public_risk_proxies_v1/calibration_summary.json",
    "revision/configs/abides_public_risk_proxies_v1/return_only.json",
    "revision/configs/abides_public_risk_proxies_v1/volatility_only.json",
    "revision/configs/abides_standard_learner_holdout_v1.json",
    "revision/configs/abides_standard_learner_exact_risk_only_holdout_v1.json",
    "revision/configs/abides_standard_learner_holdout_v1_selection.json",
    "revision/configs/abides_temporary_commitment_benchmark_v1.json",
    "revision/configs/abides_temporary_commitment_benchmark_smoke_v1.json",
    "revision/configs/abides_temporary_commitment_calibration_v1.json",
    "revision/configs/abides_temporary_commitment_calibration_v1_artifact.json",
    "revision/configs/abides_two_stage_gate_v1.json",
    # Current runners plus helper modules used by the v5 and holdout pipelines.
    "revision/experiments/calibrate_abides_domain_baselines.py",
    "revision/experiments/calibrate_abides_dose_routing_sensitivity.py",
    "revision/experiments/calibrate_abides_temporary_commitment.py",
    "revision/experiments/diagnose_abides_adaptive_gate_shift.py",
    "revision/experiments/export_abides_audit_bundle.py",
    "revision/experiments/export_abides_abce_exact_risk_only_holdout_v1.py",
    "revision/experiments/export_abides_abce_primary_followup.py",
    "revision/experiments/export_abides_collusion_audit_bundle.py",
    "revision/experiments/export_abides_dose_routing_audit_bundle.py",
    "revision/experiments/export_abides_holdout_audit_bundle.py",
    "revision/experiments/export_abides_mappo_adaptive_v5_supplement.py",
    "revision/experiments/export_abides_temporary_commitment_audit_bundle.py",
    "revision/experiments/fit_abides_public_risk_proxies.py",
    "revision/experiments/run_abides_collusion_diagnostic.py",
    "revision/experiments/run_abides_abce_pipeline.py",
    "revision/experiments/run_abides_abce_primary_followup.py",
    "revision/experiments/run_abides_domain_benchmark.py",
    "revision/experiments/run_abides_dose_routing_sensitivity.py",
    "revision/experiments/run_abides_holdout_audit_evaluation.py",
    "revision/experiments/run_abides_learned_gate_screen.py",
    "revision/experiments/run_abides_mappo_adaptive_pilot.py",
    "revision/experiments/run_abides_mappo_adaptive_v3_full_main.py",
    "revision/experiments/run_abides_mappo_adaptive_v3_learned.py",
    "revision/experiments/run_abides_mappo_adaptive_v3_response.py",
    "revision/experiments/run_abides_mappo_adaptive_v5_supplement.py",
    "revision/experiments/run_abides_mappo_full_main.py",
    "revision/experiments/run_abides_standard_learner_holdout.py",
    "revision/experiments/setup_official_on_policy.py",
    "revision/experiments/smoke_abides_docker_env.py",
    "revision/experiments/smoke_abides_mappo.py",
    "revision/experiments/train_abides_mappo_base_v3.py",
    "revision/experiments/train_abides_mappo_passive.py",
    "revision/experiments/validate_abides_announced_contract.py",
    "revision/experiments/verify_abides_audit_bundle.py",
    "revision/experiments/verify_abides_abce_docker_parity.py",
    "revision/experiments/verify_abides_abce_exact_risk_only_holdout_v1.py",
    "revision/experiments/verify_abides_abce_primary_followup_v1.py",
    "revision/experiments/verify_abides_collusion_diagnostic.py",
    "revision/experiments/verify_abides_dose_routing_sensitivity.py",
    "revision/experiments/verify_abides_holdout_audit_bundle.py",
    "revision/experiments/verify_abides_temporary_commitment.py",
    # Tests for the current ABIDES implementation and final experiment families.
    "revision/tests/test_abides_adaptive_v5_supplement.py",
    "revision/tests/test_abides_abce_exact_risk_only_holdout_bundle.py",
    "revision/tests/test_abides_abce_protocol.py",
    "revision/tests/test_abides_announced_contract.py",
    "revision/tests/test_abides_audit_bundle.py",
    "revision/tests/test_abides_collusion_diagnostic.py",
    "revision/tests/test_abides_docker_env_cleanup.py",
    "revision/tests/test_abides_domain_benchmark.py",
    "revision/tests/test_abides_dose_calibration.py",
    "revision/tests/test_abides_dose_sensitivity.py",
    "revision/tests/test_abides_ecology_holdout.py",
    "revision/tests/test_abides_holdout_audit.py",
    "revision/tests/test_abides_literature_baselines.py",
    "revision/tests/test_abides_standard_learner_holdout.py",
    "revision/tests/test_mappo_intervention_adapter.py",
    "revision/tests/test_official_on_policy_mappo.py",
    # Current external-baseline correspondence note. Keep its original path
    # because the public regression test treats that path as part of the API.
    "revision/notes/literature_baseline_mapping_v1.md",
    # Current paper-facing LaTeX figures and tables.
    "revision/paper/latex/icaif26_assets/README.md",
    "revision/paper/latex/icaif26_assets/figure_abce_trace.tex",
    "revision/paper/latex/icaif26_assets/figure_framework.tex",
    "revision/paper/latex/icaif26_assets/figure_paired_seeds.tex",
    "revision/paper/latex/icaif26_assets/figure_sensitivity.tex",
    "revision/paper/latex/icaif26_assets/preamble.tex",
    "revision/paper/latex/icaif26_assets/preview.tex",
    "revision/paper/latex/icaif26_assets/table_behavioral_boundary.tex",
    "revision/paper/latex/icaif26_assets/table_main_results.tex",
    "revision/paper/latex/icaif26_assets/table_transfer_results.tex",
}

EXPERIMENT_DOCS = {
    "revision/notes/abides_abce_execution_amendment_v1.md",
    "revision/notes/abides_abce_formal_experiment_protocol_v1.md",
    "revision/notes/abides_mappo_adaptive_v5_full_main_protocol.md",
    "revision/notes/abides_mappo_adaptive_v5_supplement_protocol.md",
    "revision/notes/abides_standard_learner_holdout_v1_protocol.md",
    "revision/notes/abides_ecology_holdout_v1_protocol.md",
    "revision/notes/abides_dose_routing_sensitivity_v1_protocol.md",
    "revision/notes/abides_collusion_diagnostic_v1_protocol.md",
    "revision/notes/abides_domain_benchmark_v1_protocol.md",
    "revision/notes/abides_temporary_commitment_benchmark_v1_protocol.md",
}

DOC_MAP: dict[str, str] = {}

TEST_FILES = tuple(sorted(path for path in SOURCE_FILES if "/tests/" in path))

ARTIFACT_DIRS = {
    "revision/paper/artifacts/abides_abce_exact_risk_only_learner_holdout_v1",
    "revision/paper/artifacts/abides_abce_primary_followup_v1",
    "revision/paper/artifacts/abides_audit_bundle_v1",
    "revision/paper/artifacts/abides_collusion_diagnostic_v1",
    "revision/paper/artifacts/abides_dose_routing_sensitivity_v1",
    "revision/paper/artifacts/abides_holdout_audit_bundle_v1",
    "revision/paper/artifacts/abides_temporary_commitment_benchmark_v1",
}

# Original manifests contain source-repository commit IDs. Primitive data and
# derived statistics are retained; anonymous release hashes replace old sums.
ARTIFACT_EXCLUDED_NAMES = {
    "README.md",
    "SHA256SUMS",
    "manifest.json",
    "completion.json",
}

TEXT_NAMES = {"Dockerfile", ".gitattributes", ".gitignore", "SHA256SUMS"}
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
HEX_TOKEN = re.compile(r"(?<![0-9a-f])([0-9a-f]{7,40})(?![0-9a-f])", re.I)
EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
IGNORED_GENERATED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}


def git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def tracked_files() -> set[str]:
    return set(git("ls-files").splitlines())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_text(path: Path) -> bool:
    return path.name in TEXT_NAMES or path.suffix.lower() in TEXT_SUFFIXES


def is_generated_cache(path: Path) -> bool:
    return bool(IGNORED_GENERATED_PARTS.intersection(path.parts)) or path.suffix == ".pyc"


def local_identity_tokens() -> list[str]:
    values = {
        str(ROOT),
        str(Path.home()),
        Path.home().name,
        git("config", "--get", "user.name", check=False),
        git("config", "--get", "user.email", check=False),
        git("remote", "get-url", "origin", check=False),
    }
    remote = git("remote", "get-url", "origin", check=False)
    match = re.search(r"github\.com[/:]([^/]+)/", remote)
    if match:
        values.add(match.group(1))
    return sorted((value for value in values if len(value.strip()) >= 4), key=len, reverse=True)


def local_commit_tokens() -> set[str]:
    tokens: set[str] = set()
    for commit in git("rev-list", "--all").splitlines():
        lowered = commit.lower()
        tokens.add(lowered)
        for length in range(7, 13):
            tokens.add(lowered[:length])
    return tokens


def sanitize_text(
    text: str, identity_tokens: Iterable[str], commit_tokens: set[str]
) -> tuple[str, int]:
    replacements = 0
    for token in identity_tokens:
        pattern = re.compile(re.escape(token), re.I)
        text, count = pattern.subn("ANONYMIZED_IDENTITY", text)
        replacements += count

    def replace_commit(match: re.Match[str]) -> str:
        nonlocal replacements
        token = match.group(1)
        if token.lower() in commit_tokens:
            replacements += 1
            return "ANONYMIZED_INTERNAL_COMMIT"
        return token

    text = HEX_TOKEN.sub(replace_commit, text)
    return text, replacements


def sanitize_compressed_jsonl(
    path: Path, identity_tokens: Iterable[str], commit_tokens: set[str]
) -> int:
    """Redact identities and internal commits inside deterministic gzip streams."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        if not any(
            sanitize_text(line, identity_tokens, commit_tokens)[1] for line in source
        ):
            return 0

    temporary = path.with_name(f".{path.name}.tmp")
    replacements = 0
    with gzip.open(path, "rt", encoding="utf-8") as source:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as target:
                    for line in source:
                        sanitized, count = sanitize_text(
                            line, identity_tokens, commit_tokens
                        )
                        target.write(sanitized)
                        replacements += count
    temporary.replace(path)
    return replacements


def prepare_output(output: Path, force: bool) -> None:
    if output.exists():
        if not force:
            raise SystemExit(f"output exists; rerun with --force: {output}")
        if not (output / MARKER).exists():
            raise SystemExit(f"refusing to delete unmarked directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / MARKER).write_text("double-blind release root\n", encoding="utf-8")


def selected_artifact_files(tracked: set[str]) -> set[str]:
    selected: set[str] = set()
    for path in tracked:
        candidate = Path(path)
        if any(path.startswith(f"{root}/") for root in ARTIFACT_DIRS):
            if candidate.name not in ARTIFACT_EXCLUDED_NAMES:
                selected.add(path)
        elif path.startswith("revision/paper/data/"):
            if candidate.name != "abides_mappo_adaptive_v5_supplement_v1_reproducibility.json":
                selected.add(path)
        elif path.startswith("revision/paper/figures/"):
            selected.add(path)
    return selected


def source_to_destination(path: str) -> str:
    if path in DOC_MAP:
        return DOC_MAP[path]
    return path


def copy_release_files(output: Path) -> tuple[int, int]:
    tracked = tracked_files()
    selected = set(SOURCE_FILES) | set(EXPERIMENT_DOCS) | set(DOC_MAP)
    selected |= selected_artifact_files(tracked)
    missing = sorted(selected - tracked)
    if missing:
        raise SystemExit(f"allowlisted files are not tracked: {missing}")

    copied = 0
    for relative in sorted(selected):
        destination = output / source_to_destination(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        copied += 1

    # Frozen configs and tests address protocols by their source-tree paths.
    # Keep those machine-verifiable paths and add reviewer-facing copies.
    for relative in sorted(EXPERIMENT_DOCS):
        destination = output / "docs" / "protocols" / Path(relative).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        copied += 1

    for source in sorted(TEMPLATE_ROOT.rglob("*")):
        if source.is_dir() or is_generated_cache(source.relative_to(TEMPLATE_ROOT)):
            continue
        destination = output / source.relative_to(TEMPLATE_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

    identity_tokens = local_identity_tokens()
    commit_tokens = local_commit_tokens()
    replacements = 0
    for path in sorted(output.rglob("*")):
        if not path.is_file() or not is_text(path):
            continue
        text = path.read_text(encoding="utf-8")
        sanitized, count = sanitize_text(text, identity_tokens, commit_tokens)
        if count:
            path.write_text(sanitized, encoding="utf-8")
            replacements += count
    for path in sorted(output.rglob("*.jsonl.gz")):
        replacements += sanitize_compressed_jsonl(
            path, identity_tokens, commit_tokens
        )
    return copied, replacements


def rewrite_sanitized_hash_references(output: Path) -> None:
    """Keep frozen config-to-artifact checks valid after identity redaction."""
    references = (
        (
            "revision/configs/abides_domain_benchmark_v1.json",
            "calibration_artifact_sha256",
            "revision/configs/abides_domain_baseline_calibration_v1_artifact.json",
        ),
        (
            "revision/configs/abides_dose_routing_sensitivity_v1.json",
            "calibration_artifact_sha256",
            "revision/configs/abides_dose_routing_sensitivity_v1_calibration.json",
        ),
        (
            "revision/configs/abides_temporary_commitment_benchmark_v1.json",
            "calibration_artifact_sha256",
            "revision/configs/abides_temporary_commitment_calibration_v1_artifact.json",
        ),
    )
    for config_name, field, artifact_name in references:
        config_path = output / config_name
        artifact_path = output / artifact_name
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config[field] = sha256(artifact_path)
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def row_count(path: Path) -> int | None:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.reader(handle)) - 1
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    return None


def write_release_metadata(output: Path, copied: int, replacements: int) -> None:
    report = {
        "status": "pass",
        "scope": "current_final_abides_experiment_chain_only",
        "fresh_history_required": True,
        "copied_file_count_before_generated_metadata": copied,
        "internal_identity_or_commit_replacements": replacements,
        "source_history_included": False,
        "legacy_archive_included": False,
        "internal_commit_provenance": "withheld_for_double_blind_review",
    }
    (output / "ANONYMITY_SCAN.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    files: dict[str, dict[str, int | str]] = {}
    row_counts: dict[str, int] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact-manifest.json":
            continue
        relative = path.relative_to(output).as_posix()
        files[relative] = {"sha256": sha256(path), "size": path.stat().st_size}
        count = row_count(path)
        if count is not None:
            row_counts[relative] = count
    manifest = {
        "schema_version": 1,
        "release_scope": "anonymous_current_abides_artifact",
        "files": files,
        "row_counts": row_counts,
    }
    (output / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def scan_output(output: Path) -> list[dict[str, str]]:
    identity_tokens = local_identity_tokens()
    commit_tokens = local_commit_tokens()
    identity_pattern = (
        re.compile("|".join(re.escape(token) for token in identity_tokens), re.I)
        if identity_tokens
        else None
    )
    violations: list[dict[str, str]] = []
    static_patterns = {
        "absolute_user_path": re.compile(r"/(?:Users|home)/[^/\s]+", re.I),
        "legacy_archive": re.compile(r"archive/v1_rejected_submission|REVISION_TRIAGE", re.I),
        "submission_history": re.compile(r"post-rejection|rejected submission", re.I),
        "source_remote_metadata": re.compile(r"remote_at_formal_run|origin_main", re.I),
    }

    def scan_chunks(relative: str, chunks: Iterable[str]) -> set[str]:
        categories: set[str] = set()
        for text in chunks:
            if relative != "scripts/verify_release.py":
                for category, pattern in static_patterns.items():
                    if pattern.search(text):
                        categories.add(category)
            if identity_pattern is not None and identity_pattern.search(text):
                categories.add("source_identity")
            if any(email.lower() != "anonymous@invalid" for email in EMAIL.findall(text)):
                categories.add("email_address")
            if "commit" in text.lower() and any(
                match.group(1).lower() in commit_tokens for match in HEX_TOKEN.finditer(text)
            ):
                categories.add("internal_commit_reference")
        return categories

    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if is_text(path):
            categories = scan_chunks(
                relative, (path.read_text(encoding="utf-8"),)
            )
        elif path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                categories = scan_chunks(relative, handle)
        else:
            continue
        violations.extend(
            {"file": relative, "category": category}
            for category in sorted(categories)
        )
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    prepare_output(output, args.force)
    copied, replacements = copy_release_files(output)
    rewrite_sanitized_hash_references(output)
    write_release_metadata(output, copied, replacements)
    violations = scan_output(output)
    if violations:
        print(json.dumps({"status": "fail", "violations": violations}, indent=2))
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "copied_files": copied,
                "sanitized_references": replacements,
                "tests": list(TEST_FILES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
