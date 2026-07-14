#!/usr/bin/env python3
"""Re-evaluate frozen ecology holdouts at audit-grade path and step granularity."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_domain_benchmark import audit_step_record, path_metrics  # noqa: E402
from run_abides_learned_gate_screen import summarize  # noqa: E402
from run_abides_mappo_adaptive_pilot import (  # noqa: E402
    model_digest,
    trainer_from_payload,
)
from run_abides_mappo_adaptive_v3_full_main import (  # noqa: E402
    EVALUATED_METHODS,
    rollout_methods,
)
from run_abides_mappo_adaptive_v3_response import market_config  # noqa: E402
from run_abides_mappo_full_main import (  # noqa: E402
    RunLock,
    append_logbook,
    atomic_write_json,
    current_git_commit,
    resolve_docker_image_id,
    sha256_file,
    utc_now,
)
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.interventions.abides_gate import ABIDESTwoStageGateModel  # noqa: E402


EVALUATOR_VERSION = "abides_holdout_audit_evaluation_v2"
OUTPUT_DIR = REPO / "revision/experiments/outputs/abides_holdout_audit_evaluation_v2"
CELL_CONFIGS = {
    "ecology_e1": REPO / "revision/configs/abides_ecology_e1_response_v1.json",
    "ecology_e2": REPO / "revision/configs/abides_ecology_e2_response_v1.json",
}
LEARNER_LABEL = "compact_MAPPO_style_intervention_adapter"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def source_descriptor(cell: str, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = REPO / str(config["output_dir"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    result_paths = [
        source / "response_results" / f"seed_{int(seed)}.json"
        for seed in config["policy_seeds"]
    ]
    if not all(path.exists() for path in result_paths):
        raise RuntimeError(f"{cell} source response results are incomplete")
    return {
        "cell": cell,
        "config": str(config_path.relative_to(REPO)),
        "config_sha256": sha256_file(config_path),
        "source_output": str(source.relative_to(REPO)),
        "source_protocol_fingerprint": str(manifest["protocol_fingerprint"]),
        "source_execution_commit": str(manifest["git_commit"]),
        "source_summary_sha256": sha256_file(source / "summary.json"),
        "source_result_sha256": {
            path.name: sha256_file(path) for path in result_paths
        },
        "policy_seeds": [int(seed) for seed in config["policy_seeds"]],
        "evaluation_seeds": [int(seed) for seed in config["evaluation_seeds"]],
    }


def evaluation_fingerprint(descriptors: Sequence[Mapping[str, Any]]) -> str:
    return digest(
        {
            "evaluator_version": EVALUATOR_VERSION,
            "sources": list(descriptors),
        }
    )


def task_paths(cell: str, policy_seed: int) -> tuple[Path, Path]:
    root = OUTPUT_DIR / cell / f"seed_{int(policy_seed)}"
    return root / "result.json", root / "step_records.jsonl.gz"


def write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical_json(row) + "\n")
    temporary.replace(path)


def normalize_compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map compact-runner flow instrumentation into the common audit schema."""
    normalized = dict(record)
    info = dict(record["info"])
    flow_shares = [float(value) for value in info.get("flow_shares", [])]
    if not flow_shares:
        raise RuntimeError("compact audit record has no realized flow shares")
    uniform = 1.0 / len(flow_shares)
    info["routing_weights"] = flow_shares
    info["routing_total_variation"] = 0.5 * sum(
        abs(value - uniform) for value in flow_shares
    )
    info["routing_mechanism_parameters"] = {
        "routing_effective_alpha": float(info.get("routing_effective_alpha", 0.0)),
        "routing_severity": float(info.get("routing_severity", 0.0)),
        "audit_weights_source": "realized_flow_shares",
    }
    normalized["info"] = info
    return normalized


def assert_nested_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise RuntimeError(f"{path} keys differ")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
            raise RuntimeError(f"{path} differs: {actual} != {expected}")
        return
    if actual != expected:
        raise RuntimeError(f"{path} differs: {actual!r} != {expected!r}")


def load_frozen_trainers(
    config: Mapping[str, Any], policy_seed: int, source_fingerprint: str
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    source = REPO / str(config["output_dir"])
    checkpoint = source / "response_seeds" / f"seed_{policy_seed}" / "training_state.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol_fingerprint") != source_fingerprint:
        raise RuntimeError(f"source checkpoint fingerprint mismatch: {checkpoint}")
    if int(payload.get("completed_episodes", -1)) != int(config["response_episodes"]):
        raise RuntimeError(f"source checkpoint is incomplete: {checkpoint}")
    trainers = {
        method: trainer_from_payload(values, str(config["device"]))
        for method, values in payload["trainers"].items()
    }
    source_result = json.loads(
        (source / "response_results" / f"seed_{policy_seed}.json").read_text(
            encoding="utf-8"
        )
    )
    for method, trainer in trainers.items():
        expected = str(source_result["final_model_digests"][method])
        if model_digest(trainer) != expected:
            raise RuntimeError(f"{method} model digest differs for seed {policy_seed}")
    return trainers, source_result


def evaluate_task(
    cell: str,
    config_path_raw: str,
    policy_seed: int,
    fingerprint: str,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    result_path, step_path = task_paths(cell, policy_seed)
    if result_path.exists() and step_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("audit_fingerprint") != fingerprint:
            raise RuntimeError(f"audit result fingerprint mismatch: {result_path}")
        if cached.get("step_records_sha256") != sha256_file(step_path):
            raise RuntimeError(f"audit step hash mismatch: {step_path}")
        return cached

    config_path = Path(config_path_raw)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = REPO / str(config["output_dir"])
    source_manifest = json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    source_fingerprint = str(source_manifest["protocol_fingerprint"])
    trainers, source_result = load_frozen_trainers(
        config, policy_seed, source_fingerprint
    )
    gate_model = ABIDESTwoStageGateModel.load(REPO / str(config["gate_artifact"]))
    scoped = dict(config)
    scoped["active_policy_seed"] = int(policy_seed)
    per_path: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    all_records = {method: [] for method in EVALUATED_METHODS}
    path_checks = []
    dose_checks = []
    no_routing_checks = []
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(market_config(config), docker) as env:
        for evaluation_seed in config["evaluation_seeds"]:
            rollouts, paths_equal, dose_exact, no_routing_exact = rollout_methods(
                env,
                scoped,
                int(evaluation_seed),
                trainers,
                gate_model,
                train=False,
                include_base=True,
            )
            path_checks.append(paths_equal)
            dose_checks.append(dose_exact)
            no_routing_checks.append(no_routing_exact)
            for method, rollout in rollouts.items():
                records = list(rollout["records"])
                all_records[method].extend(records)
                audit_records = [normalize_compact_record(record) for record in records]
                model_hash = (
                    str(source_result["base_model_digest"])
                    if method == "no_intervention"
                    else str(source_result["final_model_digests"][method])
                )
                row = path_metrics(
                    str(config["protocol_version"]),
                    int(policy_seed),
                    int(evaluation_seed),
                    method,
                    model_hash,
                    audit_records,
                )
                row["holdout"] = cell
                row["learner"] = LEARNER_LABEL
                per_path.append(row)
                for record in audit_records:
                    step = audit_step_record(
                        int(policy_seed), int(evaluation_seed), method, record
                    )
                    step["holdout"] = cell
                    step["learner"] = LEARNER_LABEL
                    step["flow_shares"] = list(record["info"]["flow_shares"])
                    step["routing_weights_source"] = "realized_flow_shares"
                    step_rows.append(step)

    recomputed = {
        method: summarize({"records": records})
        for method, records in all_records.items()
    }
    assert_nested_close(
        recomputed, source_result["final_evaluation"]["methods"], "final_evaluation"
    )
    checks = {
        "source_aggregate_reproduced": True,
        "evaluation_paths_equal": bool(all(path_checks)),
        "evaluation_dose_exact": bool(all(dose_checks)),
        "evaluation_no_routing_dose_exact": bool(all(no_routing_checks)),
        "per_path_count": len(per_path)
        == len(EVALUATED_METHODS) * len(config["evaluation_seeds"]),
        "step_record_count": len(step_rows)
        == len(EVALUATED_METHODS)
        * len(config["evaluation_seeds"])
        * int(config["episode_steps"]),
        "volume_conserved": all(
            math.isclose(float(row["mean_volume"]), 20.0, abs_tol=1e-12)
            for row in per_path
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{cell} seed {policy_seed} audit checks failed: {checks}")
    write_jsonl_gz(step_path, step_rows)
    result = {
        "audit_fingerprint": fingerprint,
        "cell": cell,
        "learner": LEARNER_LABEL,
        "policy_seed": int(policy_seed),
        "source_protocol_fingerprint": source_fingerprint,
        "source_result_sha256": sha256_file(
            source / "response_results" / f"seed_{policy_seed}.json"
        ),
        "checks": checks,
        "per_path": per_path,
        "step_record_count": len(step_rows),
        "step_records_file": str(step_path.relative_to(REPO)),
        "step_records_sha256": sha256_file(step_path),
    }
    atomic_write_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError("holdout audit evaluation requires a clean worktree")
    descriptors = [
        source_descriptor(cell, path) for cell, path in CELL_CONFIGS.items()
    ]
    fingerprint = evaluation_fingerprint(descriptors)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RunLock(OUTPUT_DIR / ".run.lock", fingerprint, args.force_unlock):
        manifest_path = OUTPUT_DIR / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("audit_fingerprint") != fingerprint:
                raise RuntimeError("holdout audit manifest fingerprint mismatch")
        else:
            manifest = {
                "evaluator_version": EVALUATOR_VERSION,
                "audit_fingerprint": fingerprint,
                "git_commit": current_git_commit(),
                "started_at": utc_now(),
                "sources": descriptors,
                "docker_images": {
                    cell: resolve_docker_image_id(
                        str(json.loads(path.read_text(encoding="utf-8"))["docker_image"])
                    )
                    for cell, path in CELL_CONFIGS.items()
                },
            }
            atomic_write_json(manifest_path, manifest)
        tasks = [
            (cell, str(path), int(seed))
            for cell, path in CELL_CONFIGS.items()
            for seed in json.loads(path.read_text(encoding="utf-8"))["policy_seeds"]
        ]
        complete_before = sum(
            task_paths(cell, seed)[0].exists() and task_paths(cell, seed)[1].exists()
            for cell, _, seed in tasks
        )
        append_logbook(
            OUTPUT_DIR / "logbook.jsonl",
            {
                "event": "audit_evaluation_started",
                "completed_before": int(complete_before),
                "pending_before": len(tasks) - int(complete_before),
            },
        )
        results = []
        failures = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            futures = {
                pool.submit(evaluate_task, cell, path, seed, fingerprint): (cell, seed)
                for cell, path, seed in tasks
            }
            for future in as_completed(futures):
                cell, seed = futures[future]
                try:
                    results.append(future.result())
                except BaseException as exc:
                    failures.append(
                        {
                            "cell": cell,
                            "policy_seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
        results.sort(key=lambda row: (str(row["cell"]), int(row["policy_seed"])))
        summary = {
            "status": "pass" if not failures else "fail",
            "audit_fingerprint": fingerprint,
            "task_count": len(results),
            "failure_count": len(failures),
            "per_path_count": sum(len(row["per_path"]) for row in results),
            "step_record_count": sum(int(row["step_record_count"]) for row in results),
            "all_checks_pass": bool(results)
            and all(all(row["checks"].values()) for row in results),
            "completed_resume": complete_before == len(tasks),
            "failures": failures,
        }
        atomic_write_json(OUTPUT_DIR / "rows.json", results)
        atomic_write_json(OUTPUT_DIR / "summary.json", summary)
        append_logbook(
            OUTPUT_DIR / "logbook.jsonl",
            {
                "event": "audit_evaluation_finished",
                "status": summary["status"],
                "completed_resume": summary["completed_resume"],
            },
        )
        print(canonical_json(summary))
        if failures:
            raise RuntimeError(f"holdout audit evaluation failures: {len(failures)}")


if __name__ == "__main__":
    main()
