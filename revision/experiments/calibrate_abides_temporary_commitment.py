#!/usr/bin/env python3
"""Calibrate temporary commitment routing budget on passive quote records."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.interventions.abides_literature_baselines import (  # noqa: E402
    calibrate_temporary_commitment,
)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="revision/configs/abides_temporary_commitment_calibration_v1.json",
    )
    args = parser.parse_args()
    config_path = REPO / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records_path = REPO / config["passive_records"]
    if sha256_file(records_path) != config["passive_records_sha256"]:
        raise RuntimeError("passive quote records changed")
    source_artifact = REPO / config["source_calibration_artifact"]
    if sha256_file(source_artifact) != config["source_calibration_artifact_sha256"]:
        raise RuntimeError("source calibration artifact changed")
    records = []
    with gzip.open(records_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    selection = calibrate_temporary_commitment(records, config)
    artifact = {
        "protocol_version": config["protocol_version"],
        "calibration_status": "pass",
        "config_sha256": sha256_file(config_path),
        "passive_records_sha256": sha256_file(records_path),
        "selection": selection,
    }
    output = REPO / config["output_artifact"]
    atomic_json(output, artifact)
    output_dir = REPO / config["output_dir"]
    atomic_json(output_dir / "summary.json", artifact)
    print(
        json.dumps(
            {
                "status": "pass",
                "artifact": str(output.relative_to(REPO)),
                "artifact_sha256": sha256_file(output),
                "budget_match_status": selection["budget_matched"]["selected"][
                    "budget_match_status"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
