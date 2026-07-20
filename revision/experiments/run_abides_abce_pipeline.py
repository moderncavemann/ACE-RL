#!/usr/bin/env python3
"""Resume-safe orchestration for the frozen ABCE formal experiment chain."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_abce_primary_followup import (  # noqa: E402
    METHODS,
    formal_worktree_violations,
    result_path as primary_result_path,
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
from run_abides_standard_learner_holdout import (  # noqa: E402
    formal_result_paths,
    validate_config as validate_official_config,
)
from verify_abides_abce_primary_followup_v1 import (  # noqa: E402
    verify_bundle,
    verify_strict_raw,
)
from verify_abides_abce_docker_parity import verify_parity  # noqa: E402


PIPELINE_VERSION = "abides_abce_formal_pipeline_v1"
STAGE_ORDER = (
    "primary_abce",
    "official_risk_only_holdout",
    "export_audit_bundle",
    "verify_strict_raw",
    "verify_bundle_only",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_clean_worktree(allowed_untracked_root: str | None) -> None:
    if formal_worktree_violations(allowed_untracked_root):
        raise RuntimeError("ABCE formal pipeline requires a clean committed worktree")


def ensure_tracked(path: Path) -> None:
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(REPO))],
        cwd=REPO,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def validate_smoke(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    output = REPO / str(config["output_dir"])
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("two-seed ABCE engineering smoke has not completed")
    summary = load_json(summary_path)
    if summary.get("implementation_status") != "pass" or not all(
        bool(value) for value in summary.get("checks", {}).values()
    ):
        raise RuntimeError("two-seed ABCE engineering smoke failed integrity")
    converged = int(summary.get("converged_policy_seeds", 0))
    if converged < int(config["minimum_converged_policy_seeds"]):
        raise RuntimeError("two-seed ABCE engineering smoke failed convergence")
    if set(summary.get("classifications", {}).values()) != {"smoke_only"}:
        raise RuntimeError("engineering smoke was incorrectly interpreted as efficacy")
    results = [
        primary_result_path(output, int(seed)) for seed in config["policy_seeds"]
    ]
    if not all(path.is_file() for path in results):
        raise RuntimeError("two-seed ABCE smoke result grid is incomplete")
    return {
        "summary_sha256": sha256_file(summary_path),
        "policy_seed_count": len(results),
        "converged_policy_seed_count": converged,
    }


def validate_primary(config: Mapping[str, Any]) -> dict[str, Any]:
    output = REPO / str(config["output_dir"])
    summary_path = output / "summary.json"
    summary = load_json(summary_path)
    if summary.get("implementation_status") != "pass" or not all(
        bool(value) for value in summary.get("checks", {}).values()
    ):
        raise RuntimeError("formal ABCE primary integrity gates did not pass")
    paths = [
        primary_result_path(output, int(seed)) for seed in config["policy_seeds"]
    ]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("formal ABCE primary policy-seed grid is incomplete")
    for path in paths:
        row = load_json(path)
        if int(row["completed_episodes"]) != int(config["response_episodes"]):
            raise RuntimeError(f"formal ABCE budget mismatch: {path}")
        if len(row["final_evaluation"]["per_path"]) != len(
            config["evaluation_seeds"]
        ) * len(METHODS):
            raise RuntimeError(f"formal ABCE evaluation grid mismatch: {path}")
    return {
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "policy_seed_count": len(paths),
        "implementation_status": "pass",
    }


def validate_official(config: Mapping[str, Any]) -> dict[str, Any]:
    output = REPO / str(config["output_dir"])
    summary_path = output / "summary.json"
    summary = load_json(summary_path)
    if bool(config.get("include_lagged_spread_severity", True)):
        raise RuntimeError("official holdout is not the frozen risk-only signal")
    if summary.get("implementation_status") != "pass" or not all(
        bool(value) for value in summary.get("checks", {}).values()
    ):
        raise RuntimeError("official MAPPO/IPPO integrity gates did not pass")
    paths = formal_result_paths(config)
    if not all(path.is_file() for path in paths):
        raise RuntimeError("official MAPPO/IPPO result grid is incomplete")
    for path in paths:
        row = load_json(path)
        if int(row["base_completed_episodes"]) != int(config["base_episodes"]):
            raise RuntimeError(f"official base budget mismatch: {path}")
        if int(row["response_completed_episodes"]) != int(
            config["response_episodes"]
        ):
            raise RuntimeError(f"official response budget mismatch: {path}")
    return {
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(output / "formal" / "manifest.json"),
        "task_count": len(paths),
        "implementation_status": "pass",
    }


def artifact_digest(artifact: Path) -> str:
    return sha256_file(artifact / "SHA256SUMS")


def validate_export(artifact: Path) -> dict[str, Any]:
    result = verify_bundle(artifact)
    return {**result, "sha256sums_sha256": artifact_digest(artifact)}


def validate_strict(artifact: Path) -> dict[str, Any]:
    result = verify_strict_raw(artifact)
    return {**result, "sha256sums_sha256": artifact_digest(artifact)}


def pipeline_commands(
    primary_config: Path, official_config: Path, artifact: Path
) -> dict[str, list[str]]:
    python = sys.executable
    return {
        "primary_abce": [
            python,
            "revision/experiments/run_abides_abce_primary_followup.py",
            "--config",
            str(primary_config.relative_to(REPO)),
        ],
        "official_risk_only_holdout": [
            python,
            "revision/experiments/run_abides_standard_learner_holdout.py",
            "--config",
            str(official_config.relative_to(REPO)),
            "--stage",
            "formal",
        ],
        "export_audit_bundle": [
            python,
            "revision/experiments/export_abides_abce_primary_followup.py",
            "--config",
            str(primary_config.relative_to(REPO)),
            "--artifact",
            str(artifact.relative_to(REPO)),
        ],
        "verify_strict_raw": [
            python,
            "revision/experiments/verify_abides_abce_primary_followup_v1.py",
            "--artifact",
            str(artifact.relative_to(REPO)),
            "--strict-raw",
        ],
        "verify_bundle_only": [
            python,
            "revision/experiments/verify_abides_abce_primary_followup_v1.py",
            "--artifact",
            str(artifact.relative_to(REPO)),
            "--bundle-only",
        ],
    }


def render_logbook(state: Mapping[str, Any]) -> str:
    lines = [
        "# ABCE Formal Pipeline Logbook",
        "",
        f"Pipeline: `{PIPELINE_VERSION}`",
        f"Commit: `{state['git_commit']}`",
        "",
        "| Stage | Status | Started | Finished |",
        "| --- | --- | --- | --- |",
    ]
    stage_state = state.get("stages", {})
    for stage in STAGE_ORDER:
        row = stage_state.get(stage, {})
        lines.append(
            f"| {stage} | {row.get('status', 'pending')} | "
            f"{row.get('started_at', '')} | {row.get('finished_at', '')} |"
        )
    return "\n".join(lines) + "\n"


def persist_state(output: Path, state: Mapping[str, Any]) -> None:
    atomic_write_json(output / "pipeline_state.json", state)
    atomic_write_text(output / "LOGBOOK.md", render_logbook(state))


def run_command_stage(
    stage: str,
    command: Sequence[str],
    validator: Callable[[], Mapping[str, Any]],
    output: Path,
    state: dict[str, Any],
) -> None:
    existing = state["stages"].get(stage, {})
    if existing.get("status") == "complete":
        receipt = dict(validator())
        state["stages"][stage]["receipt"] = receipt
        state["stages"][stage]["reverified_at"] = utc_now()
        persist_state(output, state)
        append_logbook(
            output / "logbook.jsonl",
            {"event": "stage_reverified", "stage": stage, "receipt": receipt},
        )
        return
    log_path = output / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state["status"] = "running"
    state["current_stage"] = stage
    state["stages"][stage] = {
        "status": "running",
        "started_at": utc_now(),
        "command": list(command),
        "log": str(log_path.relative_to(REPO)),
    }
    persist_state(output, state)
    append_logbook(
        output / "logbook.jsonl",
        {"event": "stage_started", "stage": stage, "command": list(command)},
    )
    started = time.perf_counter()
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] START {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                list(command),
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            state["stages"][stage]["pid"] = process.pid
            persist_state(output, state)

            def terminate_child(signum: int, _frame: object) -> None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                raise SystemExit(128 + signum)

            previous_term = signal.signal(signal.SIGTERM, terminate_child)
            previous_int = signal.signal(signal.SIGINT, terminate_child)
            try:
                return_code = process.wait()
            finally:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
            log.write(f"[{utc_now()}] EXIT {return_code}\n")
        if return_code != 0:
            raise RuntimeError(f"pipeline stage {stage} exited {return_code}")
        receipt = dict(validator())
        state["stages"][stage].update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "receipt": receipt,
            }
        )
        state["current_stage"] = None
        persist_state(output, state)
        atomic_write_json(output / "receipts" / f"{stage}.json", receipt)
        append_logbook(
            output / "logbook.jsonl",
            {"event": "stage_complete", "stage": stage, "receipt": receipt},
        )
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, SystemExit) else "failed"
        state["stages"][stage].update(
            {
                "status": state["status"],
                "finished_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        persist_state(output, state)
        append_logbook(
            output / "logbook.jsonl",
            {
                "event": "stage_failed",
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def initialize_state(
    output: Path, fingerprint: str, commands: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    state_path = output / "pipeline_state.json"
    commit = current_git_commit()
    if state_path.exists():
        state = load_json(state_path)
        if (
            state.get("pipeline_fingerprint") != fingerprint
            or state.get("git_commit") != commit
        ):
            raise RuntimeError("ABCE pipeline resume fingerprint changed")
        return state
    return {
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_fingerprint": fingerprint,
        "git_commit": commit,
        "status": "initialized",
        "current_stage": None,
        "created_at": utc_now(),
        "commands": {key: list(value) for key, value in commands.items()},
        "stages": {},
    }


def preflight(
    smoke_config_path: Path,
    primary_config_path: Path,
    official_config_path: Path,
    artifact: Path,
) -> dict[str, Any]:
    primary = load_json(primary_config_path)
    official = load_json(official_config_path)
    ensure_clean_worktree(str(primary.get("managed_artifact_output", "")) or None)
    if artifact != (REPO / str(primary["managed_artifact_output"])).resolve():
        raise RuntimeError("pipeline artifact path differs from the frozen managed output")
    for path in (
        primary_config_path,
        official_config_path,
        REPO / "revision/experiments/run_abides_abce_pipeline.py",
    ):
        ensure_tracked(path)
    smoke = validate_smoke(smoke_config_path)
    if not bool(primary.get("formal")) or primary.get("freeze_status") != "frozen_pre_outcome":
        raise RuntimeError("ABCE primary config is not frozen for formal execution")
    if official.get("freeze_status") != "frozen_pre_outcome" or not bool(
        official.get("formal_execution_allowed")
    ):
        raise RuntimeError("official learner holdout is not frozen for formal execution")
    image_id = resolve_docker_image_id(str(primary["docker_image"]))
    if image_id != str(primary["docker_image_id"]) or image_id != str(
        official["docker_image_id"]
    ):
        raise RuntimeError("ABCE formal configs do not pin the installed Docker image")
    official_artifacts = validate_official_config(official, "formal")
    docker_parity = verify_parity(primary)
    primary_preflight = subprocess.run(
        [
            sys.executable,
            "revision/experiments/run_abides_abce_primary_followup.py",
            "--config",
            str(primary_config_path.relative_to(REPO)),
            "--preflight-only",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "smoke": smoke,
        "docker_image_id": image_id,
        "official_artifacts": official_artifacts,
        "docker_parity": docker_parity,
        "primary_preflight": json.loads(primary_preflight.stdout),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-config",
        default="revision/configs/abides_abce_primary_followup_smoke_v1.json",
    )
    parser.add_argument(
        "--primary-config",
        default="revision/configs/abides_abce_primary_followup_v1.json",
    )
    parser.add_argument(
        "--official-config",
        default="revision/configs/abides_standard_learner_exact_risk_only_holdout_v1.json",
    )
    parser.add_argument(
        "--artifact",
        default="revision/paper/artifacts/abides_abce_primary_followup_v1",
    )
    parser.add_argument(
        "--output",
        default="revision/experiments/outputs/abides_abce_formal_pipeline_v1",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke_config_path = (REPO / args.smoke_config).resolve()
    primary_config_path = (REPO / args.primary_config).resolve()
    official_config_path = (REPO / args.official_config).resolve()
    artifact = (REPO / args.artifact).resolve()
    output = (REPO / args.output).resolve()
    if args.status:
        state = load_json(output / "pipeline_state.json")
        pid = state.get("stages", {}).get(state.get("current_stage"), {}).get("pid")
        state["current_pid_alive"] = bool(pid) and _pid_alive(int(pid))
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    output.mkdir(parents=True, exist_ok=True)
    preflight_receipt = preflight(
        smoke_config_path, primary_config_path, official_config_path, artifact
    )
    if args.preflight_only:
        print(json.dumps({"status": "pass", **preflight_receipt}, indent=2, sort_keys=True))
        return
    primary = load_json(primary_config_path)
    official = load_json(official_config_path)
    commands = pipeline_commands(primary_config_path, official_config_path, artifact)
    fingerprint = protocol_fingerprint(
        {
            "pipeline_version": PIPELINE_VERSION,
            "git_commit": current_git_commit(),
            "smoke_config_sha256": sha256_file(smoke_config_path),
            "primary_config_sha256": sha256_file(primary_config_path),
            "official_config_sha256": sha256_file(official_config_path),
            "commands": commands,
        }
    )
    state = initialize_state(output, fingerprint, commands)
    state["preflight"] = preflight_receipt
    persist_state(output, state)
    validators: dict[str, Callable[[], Mapping[str, Any]]] = {
        "primary_abce": lambda: validate_primary(primary),
        "official_risk_only_holdout": lambda: validate_official(official),
        "export_audit_bundle": lambda: validate_export(artifact),
        "verify_strict_raw": lambda: validate_strict(artifact),
        "verify_bundle_only": lambda: validate_export(artifact),
    }
    with RunLock(output / ".pipeline.lock", fingerprint, args.force_unlock):
        for stage in STAGE_ORDER:
            run_command_stage(
                stage, commands[stage], validators[stage], output, state
            )
        state["status"] = "complete"
        state["current_stage"] = None
        state["completed_at"] = utc_now()
        persist_state(output, state)
        completion = {
            "status": "complete",
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_fingerprint": fingerprint,
            "git_commit": state["git_commit"],
            "completed_at": state["completed_at"],
            "artifact_sha256sums_sha256": artifact_digest(artifact),
            "stage_receipts": {
                stage: state["stages"][stage]["receipt"] for stage in STAGE_ORDER
            },
        }
        atomic_write_json(output / "completion.json", completion)
        append_logbook(
            output / "logbook.jsonl",
            {"event": "pipeline_complete", **completion},
        )
    print(json.dumps(completion, indent=2, sort_keys=True))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
