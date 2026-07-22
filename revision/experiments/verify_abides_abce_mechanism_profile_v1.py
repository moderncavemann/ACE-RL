#!/usr/bin/env python3
"""Independent bundle verifier for the frozen ABCE mechanism profile."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

REVISION = Path(__file__).resolve().parents[1]
REPO = REVISION.parent
sys.path.insert(0, str(REVISION))
sys.path.insert(0, str(REVISION / "experiments"))

from profile_abides_abce_mechanism_v1 import (  # noqa: E402
    economic_effects,
    parameter_counts,
    sha256_file,
    tie_aware_auc,
)
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESPublicRiskProxyModel,
    ABIDESTwoStageGateModel,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checksums(artifact: Path) -> bool:
    rows = (artifact / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if not rows:
        return False
    for row in rows:
        expected, name = row.split("  ", 1)
        if sha256_file(artifact / name) != expected:
            return False
    return True


def read_proxy_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=0.0))


def verify(config_path: Path, artifact: Path) -> dict[str, bool]:
    config = load_json(config_path)
    profile = load_json(artifact / "profile.json")
    proxy = load_json(artifact / "proxy_validation.json")
    effects = load_json(artifact / "economic_effects.json")
    manifest = load_json(artifact / "manifest.json")

    source_hashes = all(
        sha256_file(REPO / source["path"]) == source["sha256"]
        for source in config["sources"].values()
    )
    records = read_proxy_records(artifact / proxy["record_export"]["path"])
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    labels = np.asarray(
        [row["simulator_low_risk_label"] for row in validation], dtype=int
    )

    gate = ABIDESTwoStageGateModel.load(REPO / config["sources"]["gate_model"]["path"])
    values = np.asarray(
        [
            [row["public_volatility"], row["absolute_lagged_return"]]
            for row in validation
        ],
        dtype=float,
    )
    full_probabilities = expit(
        ((values - np.asarray(gate.feature_mean)) / np.asarray(gate.feature_scale))
        @ np.asarray(gate.coefficients)
        + gate.intercept
    )
    full_auc = tie_aware_auc(labels, full_probabilities)

    single_auc_checks = []
    for name, column in (("volatility_only", 0), ("return_only", 1)):
        source = proxy["single_feature"][name]
        model = ABIDESPublicRiskProxyModel.load(REPO / source["source_path"])
        probabilities = expit(
            model.coefficient
            * (values[:, column] - model.feature_mean)
            / model.feature_scale
            + model.intercept
        )
        single_auc_checks.append(
            close(tie_aware_auc(labels, probabilities), source["risk_auc"])
        )

    statistics = load_json(REPO / config["sources"]["primary_statistics"]["path"])
    recomputed_effects = economic_effects(statistics)
    effects_match = all(
        (
            recomputed_effects[key] == value
            if not isinstance(value, float)
            else close(recomputed_effects[key], value)
        )
        for key, value in effects.items()
    )

    latency_ordered = all(
        row["median_microseconds"] > 0.0
        and row["p95_microseconds"] >= row["median_microseconds"]
        and row["p99_microseconds"] >= row["p95_microseconds"]
        and row["sample_batches"]
        == config["measured_iterations"] // config["batch_size"]
        for row in profile["latency"].values()
    )
    checks = {
        "checksums_match": verify_checksums(artifact),
        "config_hash_matches_manifest": sha256_file(config_path)
        == manifest["config_sha256"],
        "source_hashes_match": source_hashes,
        "non_outcome_scope": manifest["outcome_data_generated"] is False
        and manifest["training_performed"] is False,
        "fixed_runtime_contract": profile["measurement"]["n_agents"] == 10
        and profile["measurement"]["single_cpu"] is True
        and profile["runtime"]["docker_image_id"] == config["docker_image_id"],
        "latency_quantiles_valid": latency_ordered,
        "parameter_counts_match": profile["model_parameters"]
        == parameter_counts(config["model_dimensions"]),
        "proxy_rows_complete": len(train) == 1000 and len(validation) == 1000,
        "proxy_split_disjoint": proxy["split_is_disjoint"] is True,
        "full_proxy_auc_recomputes": close(
            full_auc, proxy["final_two_feature"]["risk_auc"]
        ),
        "single_proxy_aucs_recompute": all(single_auc_checks),
        "runtime_excludes_latent_risk": proxy["runtime_reads_latent_risk"] is False
        and proxy["runtime_feature_names"]
        == ["public_volatility", "absolute_lagged_return"],
        "economic_effects_recompute": effects_match,
        "selective_boundary_retained": effects["all_state_direction"]
        == "aligned_is_worse",
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "revision/configs/abides_abce_mechanism_profile_v1.json",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=REPO / "revision/paper/artifacts/abides_abce_mechanism_profile_v1",
    )
    args = parser.parse_args()
    checks = verify(args.config.resolve(), args.artifact.resolve())
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
