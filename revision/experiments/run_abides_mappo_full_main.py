#!/usr/bin/env python3
"""Frozen 20-seed ABIDES/MAPPO full-main runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_learned_gate_screen import (  # noqa: E402
    circular_within_risk,
    exact_dose,
    load_trainer,
    paired_summary,
    shuffled_within_risk,
    summarize,
)
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.envs.abides_docker_env import ABIDESDockerDealerEnv  # noqa: E402
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESGateActivationController,
    ABIDESLearnedGateModel,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_fingerprint(config: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in config.items()
        if key not in {"workers", "output_dir"}
    }
    payload = json.dumps(
        stable, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_logbook(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": utc_now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
    ).strip()


def validate_artifacts(config: dict[str, Any]) -> dict[str, str]:
    checks = {}
    for path_key, hash_key in (
        ("policy_checkpoint", "policy_checkpoint_sha256"),
        ("gate_artifact", "gate_artifact_sha256"),
    ):
        path = Path(config[path_key])
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen artifact: {path}")
        actual = sha256_file(path)
        expected = str(config[hash_key])
        if actual != expected:
            raise RuntimeError(
                f"artifact hash mismatch for {path}: expected {expected}, got {actual}"
            )
        checks[path_key] = actual
    subprocess.run(
        ["docker", "inspect", "--type", "image", str(config["docker_image"])],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    checks["docker_image"] = str(config["docker_image"])
    return checks


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock:
    def __init__(self, path: Path, fingerprint: str, force_unlock: bool = False):
        self.path = path
        self.fingerprint = fingerprint
        self.force_unlock = force_unlock
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
            same_host = existing.get("host") == socket.gethostname()
            active = same_host and pid_is_alive(int(existing.get("pid", -1)))
            if active and not self.force_unlock:
                raise RuntimeError(
                    f"full-main run is already active under pid {existing.get('pid')}"
                )
            if not same_host and not self.force_unlock:
                raise RuntimeError(
                    "run lock belongs to another host; use --force-unlock only after "
                    "confirming that run is no longer active"
                )
            self.path.unlink()
        descriptor = os.open(
            self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started_at": utc_now(),
                    "protocol_fingerprint": self.fingerprint,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink()


def checkpoint_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "checkpoints" / f"seed_{seed}.json"


def load_checkpoints(
    output_dir: Path,
    seeds: Sequence[int],
    fingerprint: str,
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        path = checkpoint_path(output_dir, seed)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"checkpoint fingerprint mismatch: {path}")
        row = payload.get("row")
        if not isinstance(row, dict) or int(row.get("seed", -1)) != seed:
            raise RuntimeError(f"invalid seed checkpoint: {path}")
        rows.append(row)
    return rows


def write_seed_checkpoint(
    output_dir: Path,
    row: dict[str, Any],
    fingerprint: str,
) -> None:
    atomic_write_json(
        checkpoint_path(output_dir, int(row["seed"])),
        {
            "protocol_fingerprint": fingerprint,
            "completed_at": utc_now(),
            "row": row,
        },
    )


def render_markdown_logbook(
    manifest: dict[str, Any],
    state: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> str:
    by_seed = {int(row["seed"]): row for row in rows}
    lines = [
        "# ABIDES/MAPPO Full-Main Logbook",
        "",
        f"- Status: `{state['status']}`",
        f"- Protocol: `{manifest['protocol_version']}`",
        f"- Fingerprint: `{manifest['protocol_fingerprint']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Started: `{manifest['started_at']}`",
        f"- Updated: `{state['updated_at']}`",
        f"- Completed: `{len(state['completed_seeds'])}/{len(manifest['seeds'])}` seeds",
        "",
        "| Seed | Status | Runtime (s) |",
        "| ---: | --- | ---: |",
    ]
    for seed in manifest["seeds"]:
        row = by_seed.get(int(seed))
        if row is None:
            lines.append(f"| {seed} | pending |  |")
        else:
            lines.append(
                f"| {seed} | complete | {float(row.get('elapsed_seconds', 0.0)):.2f} |"
            )
    lines.extend(
        [
            "",
            "Machine-readable events: `logbook.jsonl`  ",
            "Resume state: `run_state.json`  ",
            "Per-seed checkpoints: `checkpoints/seed_<seed>.json`",
            "",
        ]
    )
    return "\n".join(lines)


def persist_run_state(
    output_dir: Path,
    manifest: dict[str, Any],
    status: str,
    rows: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    completed = sorted(int(row["seed"]) for row in rows)
    expected = [int(seed) for seed in manifest["seeds"]]
    state = {
        "status": status,
        "updated_at": utc_now(),
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "completed_seeds": completed,
        "pending_seeds": [seed for seed in expected if seed not in completed],
        "failures": list(failures),
    }
    atomic_write_json(output_dir / "run_state.json", state)
    atomic_write_text(
        output_dir / "LOGBOOK.md",
        render_markdown_logbook(manifest, state, rows),
    )
    return state


def has_modal_undercutter(actions: Sequence[int]) -> bool:
    half_spreads = np.asarray(actions, dtype=float) + 1.0
    values, counts = np.unique(half_spreads, return_counts=True)
    modal = float(np.max(values[counts == np.max(counts)]))
    return bool(np.any(half_spreads <= modal - 1.0))


def has_excess_spread(actions: Sequence[int]) -> bool:
    half_spreads = np.asarray(actions, dtype=float) + 1.0
    return bool(np.mean(half_spreads) > 4.0)


def run_seed(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    episode_steps = int(config["episode_steps"])
    trainer = load_trainer(Path(config["policy_checkpoint"]), "cpu")
    model = ABIDESLearnedGateModel.load(config["gate_artifact"])
    action_namespace = int(config["action_namespace"])
    max_rate = float(config["max_activation_rate"])

    with ABIDESDockerDealerEnv(
        {"episode_steps": episode_steps}
    ) as env:
        learned_controller = ABIDESGateActivationController(
            episode_steps, max_activation_rate=max_rate
        )

        def learned_decision(state, actions, step):
            return learned_controller.decide(
                bool(model.decide(state, actions)["active"])
            )

        learned_rollout = rollout_episode(
            env,
            trainer,
            seed=seed,
            episode_steps=episode_steps,
            train=False,
            gate_decision=learned_decision,
            action_namespace=action_namespace,
        )
        learned_schedule = [
            bool(row["gate_active"]) for row in learned_rollout["records"]
        ]
        risk_path = [
            str(row["info"]["last_step_risk"])
            for row in learned_rollout["records"]
        ]
        matched_schedule = shuffled_within_risk(
            learned_schedule, risk_path, seed ^ 0xA117
        )
        circular_schedule = circular_within_risk(
            learned_schedule, risk_path, seed ^ 0xC117
        )
        if not exact_dose(learned_schedule, matched_schedule, risk_path):
            raise RuntimeError("matched-random dose mismatch")
        if not exact_dose(learned_schedule, circular_schedule, risk_path):
            raise RuntimeError("circular-shuffled dose mismatch")

        def scheduled(schedule, routing_enabled=True):
            return rollout_episode(
                env,
                trainer,
                seed=seed,
                episode_steps=episode_steps,
                train=False,
                gate_schedule=schedule,
                routing_enabled=routing_enabled,
                action_namespace=action_namespace,
            )

        no_intervention = scheduled([False] * episode_steps)
        matched_random = scheduled(matched_schedule)
        circular_shuffled = scheduled(circular_schedule)
        no_routing = scheduled(learned_schedule, routing_enabled=False)
        always_on = scheduled([True] * episode_steps)

        threshold_controller = ABIDESGateActivationController(
            episode_steps, max_activation_rate=max_rate
        )

        def threshold_decision(state, actions, step):
            proposal = has_excess_spread(actions) and has_modal_undercutter(actions)
            return threshold_controller.decide(proposal)

        spread_only_threshold = rollout_episode(
            env,
            trainer,
            seed=seed,
            episode_steps=episode_steps,
            train=False,
            gate_decision=threshold_decision,
            action_namespace=action_namespace,
        )

        oracle_controller = ABIDESGateActivationController(
            episode_steps, max_activation_rate=max_rate
        )

        def oracle_decision(state, actions, step):
            proposal = (
                state["info"].get("diagnostic_current_risk") == "low"
                and has_excess_spread(actions)
                and has_modal_undercutter(actions)
            )
            return oracle_controller.decide(proposal)

        oracle_diagnostic = rollout_episode(
            env,
            trainer,
            seed=seed,
            episode_steps=episode_steps,
            train=False,
            gate_decision=oracle_decision,
            action_namespace=action_namespace,
        )

    arm_rollouts = {
        "no_intervention": no_intervention,
        "learned_gate": learned_rollout,
        "matched_random": matched_random,
        "circular_shuffled": circular_shuffled,
        "spread_only_threshold": spread_only_threshold,
        "always_on": always_on,
        "no_routing": no_routing,
        "oracle_diagnostic": oracle_diagnostic,
    }
    return {
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "methods": {
            method: summarize(rollout) for method, rollout in arm_rollouts.items()
        },
        "learned_raw_proposal_rate": learned_controller.proposal_rate,
        "learned_schedule": learned_schedule,
        "risk_path": risk_path,
        "matched_schedule": matched_schedule,
        "circular_schedule": circular_schedule,
    }


def aggregate(
    rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    required_seed_count: int,
) -> dict[str, Any]:
    metrics = (
        "low_execution_half_spread",
        "mean_reward",
        "mean_volume",
        "activation_rate",
        "high_risk_activation_rate",
        "targeted_events",
    )
    method_summary = {}
    for method in methods:
        method_summary[method] = {
            metric: float(mean(row["methods"][method][metric] for row in rows))
            for metric in metrics
        }

    learned = [
        row["methods"]["learned_gate"]["low_execution_half_spread"]
        for row in rows
    ]
    contrasts = {}
    for comparator in (
        "matched_random",
        "circular_shuffled",
        "no_intervention",
        "spread_only_threshold",
        "always_on",
        "oracle_diagnostic",
    ):
        delta = [
            row["methods"][comparator]["low_execution_half_spread"]
            - learned_value
            for row, learned_value in zip(rows, learned)
        ]
        contrasts[f"{comparator}_minus_learned_gate"] = paired_summary(delta)
    integrity = {
        "required_seed_count": len(rows) == required_seed_count,
        "activation_in_range": all(
            0.05 <= row["methods"]["learned_gate"]["activation_rate"] <= 0.35
            for row in rows
        ),
        "volume_conserved": all(
            abs(row["methods"][method]["mean_volume"] - 20.0) < 1e-9
            for row in rows
            for method in methods
        ),
        "no_routing_equals_no_intervention": all(
            abs(
                row["methods"]["no_routing"]["low_execution_half_spread"]
                - row["methods"]["no_intervention"]["low_execution_half_spread"]
            )
            < 1e-9
            and abs(
                row["methods"]["no_routing"]["mean_reward"]
                - row["methods"]["no_intervention"]["mean_reward"]
            )
            < 1e-9
            for row in rows
        ),
    }
    return {
        "method_summary": method_summary,
        "paired_contrasts": contrasts,
        "integrity": integrity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="revision/configs/abides_mappo_full_main_v1.json"
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def configured_run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.seeds:
        config["seeds"] = [int(value) for value in args.seeds.split(",")]
    if args.workers is not None:
        config["workers"] = args.workers
    if args.episode_steps is not None:
        config["episode_steps"] = args.episode_steps
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    seeds = [int(value) for value in config["seeds"]]
    if args.allow_partial:
        config["required_seed_count"] = len(seeds)
    return config


def build_manifest(
    config: dict[str, Any],
    fingerprint: str,
    artifact_checks: dict[str, str],
) -> dict[str, Any]:
    return {
        "protocol_version": config["protocol_version"],
        "protocol_fingerprint": fingerprint,
        "started_at": utc_now(),
        "git_commit": current_git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "command": sys.argv,
        "seeds": [int(seed) for seed in config["seeds"]],
        "episode_steps": int(config["episode_steps"]),
        "methods": list(config["methods"]),
        "artifact_checks": artifact_checks,
        "config": config,
    }


def run_pending_seeds(
    config: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    fingerprint: str,
    rows: list[dict[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed = {int(row["seed"]) for row in rows}
    pending = [int(seed) for seed in config["seeds"] if int(seed) not in completed]
    failures: list[dict[str, Any]] = []
    logbook_path = output_dir / "logbook.jsonl"
    if not pending:
        return rows, failures

    if workers == 1:
        for seed in pending:
            append_logbook(logbook_path, {"event": "seed_started", "seed": seed})
            try:
                row = run_seed(seed, config)
            except BaseException as exc:
                failure = {
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                append_logbook(logbook_path, {"event": "seed_failed", **failure})
                persist_run_state(output_dir, manifest, "failed", rows, failures)
                break
            write_seed_checkpoint(output_dir, row, fingerprint)
            rows.append(row)
            rows.sort(key=lambda item: item["seed"])
            append_logbook(
                logbook_path,
                {
                    "event": "seed_completed",
                    "seed": seed,
                    "elapsed_seconds": row["elapsed_seconds"],
                },
            )
            persist_run_state(output_dir, manifest, "running", rows, failures)
            print(json.dumps({"seed": seed, "status": "complete"}), flush=True)
        return rows, failures

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for seed in pending:
            append_logbook(logbook_path, {"event": "seed_started", "seed": seed})
            futures[executor.submit(run_seed, seed, config)] = seed
        for future in as_completed(futures):
            seed = futures[future]
            try:
                row = future.result()
            except BaseException as exc:
                failure = {
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                append_logbook(logbook_path, {"event": "seed_failed", **failure})
            else:
                write_seed_checkpoint(output_dir, row, fingerprint)
                rows.append(row)
                rows.sort(key=lambda item: item["seed"])
                append_logbook(
                    logbook_path,
                    {
                        "event": "seed_completed",
                        "seed": seed,
                        "elapsed_seconds": row["elapsed_seconds"],
                    },
                )
                print(json.dumps({"seed": seed, "status": "complete"}), flush=True)
            persist_run_state(
                output_dir,
                manifest,
                "failed" if failures else "running",
                rows,
                failures,
            )
    return rows, failures


def execute(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = protocol_fingerprint(config)
    artifact_checks = validate_artifacts(config)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT.parent, text=True
    ).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "refusing to launch a formal run from a dirty worktree; commit or use "
            "--allow-dirty only for a smoke test"
        )

    lock_path = output_dir / ".run.lock"
    with RunLock(lock_path, fingerprint, force_unlock=args.force_unlock):
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("protocol_fingerprint") != fingerprint:
                raise RuntimeError(
                    "existing run manifest belongs to a different protocol"
                )
            resume = True
        else:
            manifest = build_manifest(config, fingerprint, artifact_checks)
            atomic_write_json(manifest_path, manifest)
            resume = False

        rows = load_checkpoints(output_dir, config["seeds"], fingerprint)
        workers = min(
            int(config["workers"]),
            max(1, len(config["seeds"]) - len(rows)),
        )
        append_logbook(
            output_dir / "logbook.jsonl",
            {
                "event": "run_resumed" if resume else "run_started",
                "protocol_fingerprint": fingerprint,
                "completed_seeds": sorted(int(row["seed"]) for row in rows),
                "workers": workers,
            },
        )
        persist_run_state(output_dir, manifest, "running", rows)

        rows, failures = run_pending_seeds(
            config,
            manifest,
            output_dir,
            fingerprint,
            rows,
            workers,
        )
        if failures:
            persist_run_state(output_dir, manifest, "failed", rows, failures)
            raise RuntimeError(
                f"{len(failures)} seed(s) failed; rerun the same command to resume"
            )

        rows.sort(key=lambda row: row["seed"])
        aggregated = aggregate(
            rows,
            config["methods"],
            int(config["required_seed_count"]),
        )
        summary = {
            "status": "complete"
            if all(aggregated["integrity"].values())
            else "integrity_fail",
            "protocol_version": config["protocol_version"],
            "protocol_fingerprint": fingerprint,
            "backend": "abides",
            "seeds": [int(seed) for seed in config["seeds"]],
            "workers": workers,
            "episode_steps": config["episode_steps"],
            "seed_runtime_seconds": {
                "mean": float(mean(row["elapsed_seconds"] for row in rows)),
                "min": float(min(row["elapsed_seconds"] for row in rows)),
                "max": float(max(row["elapsed_seconds"] for row in rows)),
            },
            **aggregated,
        }
        atomic_write_json(output_dir / "rows.json", rows)
        atomic_write_json(output_dir / "summary.json", summary)
        persist_run_state(output_dir, manifest, summary["status"], rows)
        append_logbook(
            output_dir / "logbook.jsonl",
            {"event": "run_finished", "status": summary["status"]},
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        if summary["status"] != "complete":
            raise SystemExit(2)
        return summary


def main() -> None:
    args = parse_args()
    config = configured_run(args)
    try:
        execute(config, args)
    except KeyboardInterrupt:
        print(
            "Run interrupted. Completed seed checkpoints are preserved; rerun the "
            "same command to resume.",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
