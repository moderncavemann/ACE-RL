#!/usr/bin/env python3
"""Export the completed exact risk-only MAPPO/IPPO holdout as an audit bundle."""

from __future__ import annotations

import argparse
import copy
import gzip
import io
import json
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_abce_primary_followup import canonical_json, write_csv  # noqa: E402
from run_abides_mappo_full_main import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from run_abides_standard_learner_holdout import formal_result_paths  # noqa: E402
from src.utils.abides_audit import (  # noqa: E402
    METRIC_FIELDS,
    aggregate_per_seed,
    exact_sign_test,
    method_means,
    t_interval,
    two_way_paired_bootstrap,
)


ARTIFACT_VERSION = "abides_abce_exact_risk_only_learner_holdout_v1"
DEFAULT_CONFIG = ROOT / "configs" / "abides_standard_learner_exact_risk_only_holdout_v1.json"
DEFAULT_ARTIFACT = ROOT / "paper" / "artifacts" / ARTIFACT_VERSION
EXPECTED_LEARNERS = ("official_full_policy_MAPPO", "official_IPPO")
EXPECTED_METHODS = (
    "no_intervention",
    "risk_aware",
    "matched_random",
    "brero_threshold_buybox_adaptation",
)
PRIMARY_COMPARATOR = "matched_random"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for path in formal_result_paths(config):
        if not path.is_file():
            raise FileNotFoundError(f"missing exact learner result: {path}")
        row = load_json(path)
        row["_source_file"] = str(path.relative_to(REPO))
        rows.append(row)
    rows.sort(key=lambda row: (str(row["learner"]), int(row["policy_seed"])))
    return rows


def validate_source(
    config: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, bool]:
    expected = {
        (str(learner), int(seed))
        for learner in config["learners"]
        for seed in config["policy_seeds"]
    }
    observed = {(str(row["learner"]), int(row["policy_seed"])) for row in results}
    runtime = REPO / str(config["output_dir"])
    summary = load_json(runtime / "summary.json")
    checks = {
        "source_task_grid_exact": observed == expected and len(results) == len(expected),
        "source_summary_passed": summary.get("implementation_status") == "pass"
        and all(bool(value) for value in summary.get("checks", {}).values()),
        "risk_only_signal_exact": config.get("include_lagged_spread_severity") is False,
        "source_learners_exact": tuple(config["learners"]) == EXPECTED_LEARNERS,
        "source_methods_exact": tuple(config["methods"]) == EXPECTED_METHODS,
        "base_budget_complete": all(
            int(row["base_completed_episodes"]) == int(config["base_episodes"])
            for row in results
        ),
        "response_budget_complete": all(
            int(row["response_completed_episodes"]) == int(config["response_episodes"])
            for row in results
        ),
        "official_commit_exact": all(
            str(row["official_commit"]) == str(config["official_commit"])
            for row in results
        ),
        "protocol_fingerprint_exact": len(
            {str(row["protocol_fingerprint"]) for row in results}
        )
        == 1,
        "training_paths_equal": all(bool(row["training_paths_equal"]) for row in results),
        "training_action_uniforms_equal": all(
            bool(row["training_action_uniforms_equal"]) for row in results
        ),
        "training_random_dose_exact": all(
            bool(row["training_random_dose_exact"]) for row in results
        ),
        "evaluation_paths_equal": all(
            bool(row["final_evaluation"]["paths_equal"]) for row in results
        ),
        "evaluation_action_uniforms_equal": all(
            bool(row["final_evaluation"]["action_uniforms_equal"])
            for row in results
        ),
        "evaluation_random_dose_exact": all(
            bool(row["final_evaluation"]["random_dose_exact"]) for row in results
        ),
        "per_path_row_count_exact": all(
            len(row["final_evaluation"]["per_path"])
            == len(config["response_evaluation_seeds"]) * len(config["methods"])
            for row in results
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"exact learner source checks failed: {failed}")
    return checks


def collect_per_path(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        learner = str(result["learner"])
        for source in result["final_evaluation"]["per_path"]:
            row = dict(source)
            row["learner"] = learner
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["learner"]),
            int(row["policy_seed"]),
            int(row["evaluation_seed"]),
            str(row["method"]),
        ),
    )


def build_per_seed(
    per_path: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for learner in config["learners"]:
        rows = [row for row in per_path if row["learner"] == learner]
        output.extend(
            {"learner": learner, **row}
            for row in aggregate_per_seed(
                rows, config["methods"], contrasts=(PRIMARY_COMPARATOR,)
            )
        )
    return output


def classify(interval: Mapping[str, Any]) -> str:
    if float(interval["ci_low"]) > 0.0:
        return "favorable_timing_interval"
    if float(interval["mean"]) > 0.0:
        return "positive_direction_inconclusive_interval"
    return "nonpositive_timing_direction"


def build_statistics(
    per_path: Sequence[Mapping[str, Any]],
    per_seed: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "primary_contrast": "matched_random - risk_aware low-risk execution half-spread",
        "inference_unit": "policy_seed",
        "learners": {},
        "claim_boundary": (
            "MAPPO is supported when its policy-seed interval excludes zero; IPPO "
            "is retained as directional and inconclusive. The Brero-style ABIDES "
            "adaptation is unmatched and descriptive only."
        ),
    }
    for learner_index, learner in enumerate(config["learners"]):
        learner_paths = [row for row in per_path if row["learner"] == learner]
        learner_seeds = [row for row in per_seed if row["learner"] == learner]
        effects = [
            float(row[f"{PRIMARY_COMPARATOR}__minus_risk_aware_spread"])
            for row in learner_seeds
        ]
        interval = t_interval(effects)
        output["learners"][learner] = {
            "method_summary": method_means(learner_paths, config["methods"]),
            "matched_random_minus_risk_aware": {
                "t_interval": interval,
                "exact_sign_test": exact_sign_test(effects),
                "two_way_bootstrap": two_way_paired_bootstrap(
                    learner_paths,
                    PRIMARY_COMPARATOR,
                    int(config["bootstrap_replicates"]),
                    int(config["bootstrap_seed"]) + learner_index,
                ),
            },
            "classification": classify(interval),
            "external_baseline_status": "unmatched_descriptive_only",
        }
    return output


def concatenate_steps(
    results: Sequence[Mapping[str, Any]], destination: Path
) -> tuple[int, dict[str, bool]]:
    temporary = destination.with_name(f".{destination.name}.tmp")
    count = 0
    checks = {
        "source_step_hashes_valid": True,
        "step_rows_finite": True,
        "step_volume_conserved": True,
        "step_routing_weights_valid": True,
    }
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as target:
                for result in results:
                    source = REPO / str(result["step_records_file"])
                    checks["source_step_hashes_valid"] &= (
                        source.is_file()
                        and sha256_file(source) == str(result["step_records_sha256"])
                    )
                    with gzip.open(source, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            row = json.loads(line)
                            values = row["routing_weights"]
                            checks["step_volume_conserved"] &= abs(
                                float(row["realized_flow_quantity"]) - 20.0
                            ) <= 1e-12
                            checks["step_routing_weights_valid"] &= (
                                all(float(value) >= 0.0 for value in values)
                                and abs(sum(float(value) for value in values) - 1.0)
                                <= 1e-12
                            )
                            checks["step_rows_finite"] &= all(
                                value is None
                                or not isinstance(value, float)
                                or (value == value and abs(value) != float("inf"))
                                for value in row.values()
                            )
                            target.write(canonical_json(row) + "\n")
                            count += 1
    temporary.replace(destination)
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"exact learner step export checks failed: {failed}")
    return count, checks


def build_model_digests(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for result in results:
        learner = str(result["learner"])
        seed = str(int(result["policy_seed"]))
        output.setdefault(learner, {})[seed] = {
            "base_model_digest": result["base_model_digest"],
            "initial_response_model_digest": result["initial_response_model_digest"],
            "final_model_digests": result["final_model_digests"],
            "source_result_sha256": sha256_file(REPO / str(result["_source_file"])),
            "source_step_records_sha256": result["step_records_sha256"],
        }
    return output


def build_path_digests(per_path: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    grouped: dict[tuple[str, int, int], set[str]] = {}
    for row in per_path:
        key = (str(row["learner"]), int(row["policy_seed"]), int(row["evaluation_seed"]))
        grouped.setdefault(key, set()).add(str(row["path_digest"]))
    for key, values in grouped.items():
        if len(values) != 1:
            raise RuntimeError(f"paired path digest mismatch: {key}")
        output["/".join(map(str, key))] = next(iter(values))
    return output


def write_sha256sums(artifact: Path) -> None:
    files = sorted(
        path for path in artifact.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(artifact)}" for path in files]
    atomic_write_text(artifact / "SHA256SUMS", "\n".join(lines) + "\n")


def export(config_path: Path, artifact: Path) -> dict[str, Any]:
    config = load_json(config_path)
    results = load_results(config)
    source_checks = validate_source(config, results)
    per_path = collect_per_path(results)
    expected_paths = (
        len(config["learners"])
        * len(config["policy_seeds"])
        * len(config["response_evaluation_seeds"])
        * len(config["methods"])
    )
    if len(per_path) != expected_paths:
        raise RuntimeError("exact learner per-path grid is incomplete")
    per_seed = build_per_seed(per_path, config)
    statistics = build_statistics(per_path, per_seed, config)

    if artifact.exists():
        shutil.rmtree(artifact)
    artifact.mkdir(parents=True)
    shutil.copyfile(REPO / str(config["protocol_file"]), artifact / "PROTOCOL.md")
    shutil.copyfile(
        REPO / str(config["execution_amendment_file"]),
        artifact / "EXECUTION_AMENDMENT.md",
    )
    protocol = copy.deepcopy(config)
    protocol["official_source_root"] = (
        "~/.cache/ace-rl/on-policy/" + str(config["official_commit"])
    )
    atomic_write_json(artifact / "protocol.json", protocol)
    write_csv(artifact / "per_path.csv", per_path)
    write_csv(artifact / "per_seed.csv", per_seed)
    atomic_write_json(artifact / "statistics.json", statistics)
    atomic_write_json(artifact / "model_digests.json", build_model_digests(results))
    atomic_write_json(artifact / "path_digests.json", build_path_digests(per_path))
    trace_count, trace_checks = concatenate_steps(
        results, artifact / "step_records.jsonl.gz"
    )
    expected_traces = expected_paths * int(config["episode_steps"])
    checks = {
        **source_checks,
        **trace_checks,
        "artifact_per_path_count_exact": len(per_path) == expected_paths,
        "artifact_per_seed_count_exact": len(per_seed)
        == len(config["learners"]) * len(config["policy_seeds"]),
        "artifact_trace_count_exact": trace_count == expected_traces,
        "all_methods_volume_conserved": all(
            abs(float(row["mean_volume"]) - 20.0) <= 1e-12 for row in per_path
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("exact learner artifact integrity failed")
    atomic_write_json(artifact / "checks.json", checks)
    runtime = REPO / str(config["output_dir"])
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "protocol_version": config["protocol_version"],
        "protocol_fingerprint": results[0]["protocol_fingerprint"],
        "official_commit": config["official_commit"],
        "source_git_commit": load_json(runtime / "formal" / "manifest.json")["git_commit"],
        "source_manifest_sha256": sha256_file(runtime / "formal" / "manifest.json"),
        "source_summary_sha256": sha256_file(runtime / "summary.json"),
        "source_result_sha256": {
            f"{row['learner']}/{int(row['policy_seed'])}": sha256_file(
                REPO / str(row["_source_file"])
            )
            for row in results
        },
        "learner_count": len(config["learners"]),
        "policy_seed_count": len(config["policy_seeds"]),
        "per_path_rows": len(per_path),
        "per_seed_rows": len(per_seed),
        "trace_rows": trace_count,
        "primary_inference_unit": "policy_seed",
        "external_baseline_status": "unmatched_descriptive_only",
    }
    atomic_write_json(artifact / "manifest.json", manifest)
    atomic_write_text(
        artifact / "README.md",
        "# Exact Risk-only Official Learner Holdout v1\n\n"
        "This self-contained audit bundle seals the completed exact primary "
        "risk-only holdout under official full-policy MAPPO and IPPO. It contains "
        "800 paired path rows, 20 learner-policy rows, 80,000 primitive step "
        "records, model/path digests, deterministic statistics, and integrity "
        "checks. MAPPO is supported; IPPO is directional but inconclusive. The "
        "Brero-style ABIDES adaptation is unmatched and descriptive only.\n",
    )
    write_sha256sums(artifact)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export(args.config.resolve(), args.artifact.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
