#!/usr/bin/env python3
"""Collect passive ABIDES records and freeze routing-budget baseline choices."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_mappo_adaptive_pilot import (  # noqa: E402
    model_digest,
    trainer_from_payload,
)
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESPacedSignalController,
    ABIDESTwoStageGateModel,
    abides_lagged_actions,
)
from src.interventions.abides_literature_baselines import (  # noqa: E402
    calibrate_domain_baselines,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical_json(row) + "\n")
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def docker_image_id(image: str) -> str:
    return subprocess.check_output(
        ["docker", "inspect", image, "--format", "{{.Id}}"], text=True
    ).strip()


def market_config(config: Mapping[str, Any]) -> dict[str, Any]:
    v5 = json.loads(
        (
            ROOT / "configs" / "abides_mappo_adaptive_v5_full_main_primary_v1.json"
        ).read_text(encoding="utf-8")
    )
    market = dict(v5["market"])
    market.update(
        {
            "episode_steps": int(config["episode_steps"]),
            "routing_allocation_mode": str(config["routing_allocation_mode"]),
            "routing_target_kappa": float(config["routing_target_kappa"]),
            "routing_routable_fraction": float(config["routing_routable_fraction"]),
        }
    )
    return market


def load_base(config: Mapping[str, Any], policy_seed: int):
    checkpoint = (
        REPO
        / str(config["source_base_output_dir"])
        / "base_seeds"
        / f"seed_{policy_seed}"
        / "training_state.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = str(config["source_base_fingerprint"])
    if payload.get("protocol_fingerprint") != expected:
        raise RuntimeError(f"source base fingerprint mismatch: {checkpoint}")
    trainer = trainer_from_payload(payload["trainer"], "cpu")
    return trainer, model_digest(trainer)


def collect_policy_records(
    config: Mapping[str, Any], policy_seed: int
) -> dict[str, Any]:
    trainer, digest = load_base(config, policy_seed)
    gate_path = REPO / str(config["gate_artifact"])
    gate = ABIDESTwoStageGateModel.load(gate_path)
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    records: list[dict[str, Any]] = []
    with ABIDESDockerDealerEnv(market_config(config), docker) as env:
        for evaluation_seed in config["calibration_environment_seeds"]:
            evaluation_seed = int(evaluation_seed)
            rollout = rollout_episode(
                env,
                trainer,
                seed=evaluation_seed,
                episode_steps=int(config["episode_steps"]),
                train=False,
                announced_signal_schedule=[0.0] * int(config["episode_steps"]),
                reward_scale=100.0,
                action_namespace=int(config["evaluation_action_namespace"]),
            )
            controller = ABIDESPacedSignalController(
                int(config["episode_steps"]),
                target_rate=float(config["reference_controller_target_rate"]),
            )
            episode_id = f"{policy_seed}:{evaluation_seed}"
            for row in rollout["records"]:
                state = {
                    "observations": row["pre_observations"],
                    "info": row["pre_info"],
                }
                proposal = gate.announced_strength(state, include_spread_severity=False)
                risk_strength = controller.decide(float(proposal["strength"]))
                lagged_actions = abides_lagged_actions(state)
                lagged_spreads = lagged_actions.astype(float) + 1.0
                lagged_severity = float(
                    np.clip((float(np.mean(lagged_spreads)) - 4.0) / 2.0, 0.0, 1.0)
                )
                records.append(
                    {
                        "episode_id": episode_id,
                        "policy_seed": int(policy_seed),
                        "evaluation_seed": evaluation_seed,
                        "step": int(row["step"]),
                        "half_spreads": [int(action) + 1 for action in row["actions"]],
                        "risk_signal_strength": float(risk_strength),
                        "lagged_contract_severity": lagged_severity,
                        "latent_risk_state": str(
                            row["pre_info"].get("diagnostic_current_risk")
                        ),
                        "path_digest": str(row["info"]["path_digest"]),
                        "model_digest": digest,
                    }
                )
    return {"policy_seed": int(policy_seed), "model_digest": digest, "records": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_domain_baseline_calibration_v1.json",
    )
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = REPO / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = REPO / str(config["output_dir"])
    records_path = output_dir / "passive_records.jsonl.gz"
    summary_path = output_dir / "summary.json"
    artifact_path = REPO / str(config["output_artifact"])
    fingerprint = sha256_bytes(canonical_json(config).encode("utf-8"))

    if summary_path.exists() and artifact_path.exists() and records_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("completed calibration fingerprint mismatch")
        if summary.get("records_sha256") != sha256_file(records_path):
            raise RuntimeError("completed calibration record hash mismatch")
        print(canonical_json({"status": "already_complete", **summary}))
        return

    workers = int(args.workers or min(8, len(config["source_policy_seeds"])))
    collected: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(collect_policy_records, config, int(seed)): int(seed)
            for seed in config["source_policy_seeds"]
        }
        for future in as_completed(futures):
            seed = futures[future]
            collected[seed] = future.result()
            print(f"calibration policy {seed} complete", flush=True)

    rows = [
        row
        for seed in sorted(collected)
        for row in sorted(
            collected[seed]["records"],
            key=lambda value: (value["evaluation_seed"], value["step"]),
        )
    ]
    expected = (
        len(config["source_policy_seeds"])
        * len(config["calibration_environment_seeds"])
        * int(config["episode_steps"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"passive record count {len(rows)} != {expected}")
    write_jsonl_gz(records_path, rows)
    calibration = calibrate_domain_baselines(rows, config)
    artifact = {
        "protocol_version": str(config["protocol_version"]),
        "protocol_fingerprint": fingerprint,
        "config_sha256": sha256_file(config_path),
        "source_base_fingerprint": str(config["source_base_fingerprint"]),
        "source_v5_fingerprint": str(config["source_v5_fingerprint"]),
        "gate_artifact_sha256": str(config["gate_artifact_sha256"]),
        "source_model_digests": {
            str(seed): collected[seed]["model_digest"] for seed in sorted(collected)
        },
        "calibration_environment_seeds": list(config["calibration_environment_seeds"]),
        "forbidden_formal_evaluation_seeds": list(
            config["forbidden_formal_evaluation_seeds"]
        ),
        "records_sha256": sha256_file(records_path),
        "execution_commit": git_commit(),
        "docker_image": str(config["docker_image"]),
        "docker_image_id": docker_image_id(str(config["docker_image"])),
        "selection": calibration,
    }
    atomic_write_json(artifact_path, artifact)
    summary = {
        "status": "complete",
        "protocol_fingerprint": fingerprint,
        "record_count": len(rows),
        "records_sha256": sha256_file(records_path),
        "artifact": str(config["output_artifact"]),
        "artifact_sha256": sha256_file(artifact_path),
        "source_policy_count": len(collected),
    }
    atomic_write_json(summary_path, summary)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
