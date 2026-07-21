#!/usr/bin/env python3
"""Independently verify the exact risk-only MAPPO/IPPO audit bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, t, ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_mappo_full_main import sha256_file  # noqa: E402
from run_abides_standard_learner_holdout import formal_result_paths  # noqa: E402


ARTIFACT_VERSION = "abides_abce_exact_risk_only_learner_holdout_v1"
DEFAULT_ARTIFACT = ROOT / "paper" / "artifacts" / ARTIFACT_VERSION
DEFAULT_CONFIG = ROOT / "configs" / "abides_standard_learner_exact_risk_only_holdout_v1.json"
EXPECTED_LEARNERS = ("official_full_policy_MAPPO", "official_IPPO")
EXPECTED_METHODS = (
    "no_intervention",
    "risk_aware",
    "matched_random",
    "brero_threshold_buybox_adaptation",
)
PRIMARY_COMPARATOR = "matched_random"
METRIC_FIELDS = (
    "low_execution_half_spread",
    "high_execution_half_spread",
    "all_state_execution_half_spread",
    "mean_reward",
    "dealer_reward_units_per_episode",
    "low_customer_cost_tick_units_per_episode",
    "all_customer_cost_tick_units_per_episode",
    "activation_rate",
    "high_risk_activation_rate",
    "mean_signal_strength",
    "mean_routing_total_variation",
    "mean_flow_hhi",
    "mean_inventory_std",
    "mean_volume",
    "low_risk_steps",
    "high_risk_steps",
)
PATH_STRING_FIELDS = {
    "protocol_version",
    "learner",
    "method",
    "model_digest",
    "path_digest",
}
PATH_INTEGER_FIELDS = {"policy_seed", "evaluation_seed"}
TRACE_FIELDS = {
    "action_uniform_digest",
    "actions",
    "eligible_dealers",
    "evaluation_seed",
    "execution_half_spread",
    "inventory_summary",
    "latent_risk_state",
    "learner",
    "mechanism_parameters",
    "mechanism_signal",
    "method",
    "path_digest",
    "policy_seed",
    "pre_state_digest",
    "protected_execution_prices",
    "protected_execution_quantity",
    "protected_matched_order_tags",
    "protected_routing_fallback_count",
    "realized_flow_quantity",
    "rewards",
    "routing_threshold",
    "routing_weights",
    "step",
    "targeted_event_count",
}


def read_csv(path: Path, per_seed: bool = False) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in source.items():
                if key == "learner" or (not per_seed and key in PATH_STRING_FIELDS):
                    row[key] = value
                elif key == "policy_seed" or (not per_seed and key in PATH_INTEGER_FIELDS):
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def assert_nested_close(
    observed: Any, expected: Any, label: str, rel_tol: float = 1e-10
) -> None:
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        if set(observed) != set(expected):
            raise RuntimeError(f"{label} keys differ")
        for key in sorted(observed):
            assert_nested_close(observed[key], expected[key], f"{label}.{key}", rel_tol)
        return
    if isinstance(observed, Sequence) and not isinstance(observed, (str, bytes)):
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
            raise RuntimeError(f"{label} types differ")
        if len(observed) != len(expected):
            raise RuntimeError(f"{label} lengths differ")
        for index, (left, right) in enumerate(zip(observed, expected)):
            assert_nested_close(left, right, f"{label}[{index}]", rel_tol)
        return
    if (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        if not math.isclose(
            float(observed), float(expected), rel_tol=rel_tol, abs_tol=1e-12
        ):
            raise RuntimeError(f"{label} values differ: {observed} != {expected}")
        return
    if observed != expected:
        raise RuntimeError(f"{label} values differ")


def verify_sums(artifact: Path) -> None:
    sums = artifact / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError("exact learner bundle has no SHA256SUMS")
    listed = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = artifact / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"exact learner bundle hash mismatch: {relative}")
        listed.add(relative)
    actual = {
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if listed != actual:
        raise RuntimeError("exact learner SHA256SUMS file set is incomplete")


def path_key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["learner"]),
        int(row["policy_seed"]),
        int(row["evaluation_seed"]),
        str(row["method"]),
    )


def aggregate_per_seed_independent(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(str(row["learner"]), int(row["policy_seed"]))][
            str(row["method"])
        ].append(row)
    output = []
    for (learner, policy_seed), method_rows in sorted(grouped.items()):
        if set(method_rows) != set(methods):
            raise RuntimeError(f"incomplete method set for {learner}/{policy_seed}")
        evaluation_sets = {
            method: {int(row["evaluation_seed"]) for row in values}
            for method, values in method_rows.items()
        }
        if len({tuple(sorted(values)) for values in evaluation_sets.values()}) != 1:
            raise RuntimeError(f"unpaired paths for {learner}/{policy_seed}")
        result: dict[str, Any] = {"learner": learner, "policy_seed": policy_seed}
        for method in methods:
            for metric in METRIC_FIELDS:
                result[f"{method}__{metric}"] = float(
                    mean(float(row[metric]) for row in method_rows[method])
                )
        result[f"{PRIMARY_COMPARATOR}__minus_risk_aware_spread"] = (
            result[f"{PRIMARY_COMPARATOR}__low_execution_half_spread"]
            - result["risk_aware__low_execution_half_spread"]
        )
        output.append(result)
    return output


def t_interval_independent(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    average = float(np.mean(data))
    sample_sd = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    if len(data) == 1 or sample_sd <= 1e-15:
        standard_error = 0.0
        ci_low = ci_high = average
        p_value = 0.0 if len(data) > 1 and abs(average) > 1e-12 else 1.0
    else:
        standard_error = sample_sd / math.sqrt(len(data))
        critical = float(t.ppf(0.975, len(data) - 1))
        ci_low = average - critical * standard_error
        ci_high = average + critical * standard_error
        p_value = float(ttest_1samp(data, 0.0).pvalue)
    return {
        "mean": average,
        "se": standard_error,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value_two_sided": p_value,
        "positive": int(np.sum(data > 1e-12)),
        "ties": int(np.sum(np.abs(data) <= 1e-12)),
        "negative": int(np.sum(data < -1e-12)),
        "n": len(data),
    }


def exact_sign_test_independent(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    positive = int(np.sum(data > 1e-12))
    negative = int(np.sum(data < -1e-12))
    ties = int(len(data) - positive - negative)
    non_ties = positive + negative
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "non_ties": non_ties,
        "p_value_two_sided": (
            float(binomtest(positive, non_ties, 0.5).pvalue) if non_ties else 1.0
        ),
    }


def two_way_bootstrap_independent(
    rows: Sequence[Mapping[str, Any]], replicates: int, rng_seed: int
) -> dict[str, float | int]:
    lookup = {path_key(row): row for row in rows}
    policy_seeds = np.asarray(sorted({int(row["policy_seed"]) for row in rows}))
    differences: dict[int, np.ndarray] = {}
    learner = str(rows[0]["learner"])
    for policy_seed in policy_seeds:
        evaluations = sorted(
            {
                int(row["evaluation_seed"])
                for row in rows
                if int(row["policy_seed"]) == int(policy_seed)
            }
        )
        differences[int(policy_seed)] = np.asarray(
            [
                float(
                    lookup[(learner, int(policy_seed), evaluation, PRIMARY_COMPARATOR)][
                        "low_execution_half_spread"
                    ]
                )
                - float(
                    lookup[(learner, int(policy_seed), evaluation, "risk_aware")][
                        "low_execution_half_spread"
                    ]
                )
                for evaluation in evaluations
            ]
        )
    rng = np.random.default_rng(rng_seed)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(policy_seeds, size=len(policy_seeds), replace=True)
        estimates[replicate] = float(
            np.mean(
                [
                    np.mean(
                        rng.choice(
                            differences[int(seed)],
                            size=len(differences[int(seed)]),
                            replace=True,
                        )
                    )
                    for seed in sampled_seeds
                ]
            )
        )
    return {
        "replicates": replicates,
        "rng_seed": rng_seed,
        "mean": float(np.mean(estimates)),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
    }


def classify(interval: Mapping[str, Any]) -> str:
    if float(interval["ci_low"]) > 0.0:
        return "favorable_timing_interval"
    if float(interval["mean"]) > 0.0:
        return "positive_direction_inconclusive_interval"
    return "nonpositive_timing_direction"


def build_statistics_independent(
    rows: Sequence[Mapping[str, Any]],
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
        learner_paths = [row for row in rows if row["learner"] == learner]
        learner_seeds = [row for row in per_seed if row["learner"] == learner]
        effects = [
            float(row[f"{PRIMARY_COMPARATOR}__minus_risk_aware_spread"])
            for row in learner_seeds
        ]
        interval = t_interval_independent(effects)
        summaries = {}
        for method in config["methods"]:
            method_rows = [row for row in learner_paths if row["method"] == method]
            summaries[method] = {
                metric: float(mean(float(row[metric]) for row in method_rows))
                for metric in METRIC_FIELDS
            }
        output["learners"][learner] = {
            "method_summary": summaries,
            "matched_random_minus_risk_aware": {
                "t_interval": interval,
                "exact_sign_test": exact_sign_test_independent(effects),
                "two_way_bootstrap": two_way_bootstrap_independent(
                    learner_paths,
                    int(config["bootstrap_replicates"]),
                    int(config["bootstrap_seed"]) + learner_index,
                ),
            },
            "classification": classify(interval),
            "external_baseline_status": "unmatched_descriptive_only",
        }
    return output


def finite_nested(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_nested(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(finite_nested(child) for child in value)
    return True


def verify_trace(
    artifact: Path, config: Mapping[str, Any]
) -> tuple[int, dict[str, bool]]:
    seen = set()
    trajectory_counts: Counter[tuple[str, int, int, str]] = Counter()
    action_digests: dict[tuple[str, int, int, int], set[str]] = defaultdict(set)
    path_digests: dict[tuple[int, int], set[str]] = defaultdict(set)
    signals: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    checks = {
        "trace_schema_complete": True,
        "trace_rows_finite": True,
        "trace_volume_conserved": True,
        "trace_routing_weights_valid": True,
        "trace_no_intervention_exact": True,
    }
    count = 0
    with gzip.open(artifact / "step_records.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            checks["trace_schema_complete"] &= set(row) == TRACE_FIELDS
            checks["trace_rows_finite"] &= finite_nested(row)
            learner = str(row["learner"])
            policy_seed = int(row["policy_seed"])
            evaluation_seed = int(row["evaluation_seed"])
            method = str(row["method"])
            step = int(row["step"])
            key = (learner, policy_seed, evaluation_seed, method, step)
            if key in seen:
                raise RuntimeError(f"duplicate exact learner trace row: {key}")
            seen.add(key)
            trajectory = key[:4]
            trajectory_counts[trajectory] += 1
            action_digests[(learner, policy_seed, evaluation_seed, step)].add(
                str(row["action_uniform_digest"])
            )
            path_digests[(policy_seed, evaluation_seed)].add(str(row["path_digest"]))
            signals[trajectory].append(float(row["mechanism_signal"]))
            weights = [float(value) for value in row["routing_weights"]]
            checks["trace_volume_conserved"] &= math.isclose(
                float(row["realized_flow_quantity"]), 20.0, abs_tol=1e-12
            )
            checks["trace_routing_weights_valid"] &= (
                len(weights) == int(config["market"]["n_agents"])
                and all(value >= 0.0 for value in weights)
                and math.isclose(sum(weights), 1.0, abs_tol=1e-12)
            )
            if method == "no_intervention":
                checks["trace_no_intervention_exact"] &= math.isclose(
                    float(row["mechanism_signal"]), 0.0, abs_tol=1e-12
                ) and all(
                    math.isclose(value, 1.0 / len(weights), abs_tol=1e-12)
                    for value in weights
                )
            count += 1
    checks["trace_trajectory_horizon_exact"] = all(
        value == int(config["episode_steps"]) for value in trajectory_counts.values()
    )
    checks["trace_trajectory_grid_exact"] = len(trajectory_counts) == (
        len(config["learners"])
        * len(config["policy_seeds"])
        * len(config["response_evaluation_seeds"])
        * len(config["methods"])
    )
    checks["trace_common_action_uniforms"] = all(
        len(values) == 1 for values in action_digests.values()
    )
    checks["trace_common_paths"] = all(
        len(values) == 1 for values in path_digests.values()
    )
    checks["trace_exact_random_signal_multiset"] = all(
        sorted(signals[(learner, seed, path, "risk_aware")])
        == sorted(signals[(learner, seed, path, "matched_random")])
        for learner in config["learners"]
        for seed in config["policy_seeds"]
        for path in config["response_evaluation_seeds"]
    )
    return count, checks


def verify_model_digests(artifact: Path, config: Mapping[str, Any]) -> None:
    payload = json.loads((artifact / "model_digests.json").read_text(encoding="utf-8"))
    if set(payload) != set(config["learners"]):
        raise RuntimeError("exact learner model digest learner grid differs")
    for learner in config["learners"]:
        if set(map(int, payload[learner])) != set(map(int, config["policy_seeds"])):
            raise RuntimeError(f"model digest seed grid differs for {learner}")
        for row in payload[learner].values():
            values = [
                row["base_model_digest"],
                row["initial_response_model_digest"],
                row["source_result_sha256"],
                row["source_step_records_sha256"],
                *row["final_model_digests"].values(),
            ]
            if not all(
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
                for value in values
            ):
                raise RuntimeError("invalid exact learner model/result digest")


def verify_bundle(artifact: Path) -> dict[str, Any]:
    verify_sums(artifact)
    config = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    if config.get("protocol_version") != "abides_standard_learner_exact_risk_only_holdout_v1":
        raise RuntimeError("exact learner protocol version changed")
    if config.get("include_lagged_spread_severity") is not False:
        raise RuntimeError("exact learner bundle is not risk-only")
    if tuple(config["learners"]) != EXPECTED_LEARNERS:
        raise RuntimeError("exact learner grid changed")
    if tuple(config["methods"]) != EXPECTED_METHODS:
        raise RuntimeError("exact learner method grid changed")

    per_path = sorted(read_csv(artifact / "per_path.csv"), key=path_key)
    expected_keys = {
        (learner, int(seed), int(path), method)
        for learner in config["learners"]
        for seed in config["policy_seeds"]
        for path in config["response_evaluation_seeds"]
        for method in config["methods"]
    }
    keys = [path_key(row) for row in per_path]
    if len(keys) != 800 or len(set(keys)) != 800 or set(keys) != expected_keys:
        raise RuntimeError("exact learner per-path grid is incomplete or duplicated")
    if any(not math.isclose(float(row["mean_volume"]), 20.0, abs_tol=1e-12) for row in per_path):
        raise RuntimeError("exact learner path volume differs from 20")
    paired_paths: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in per_path:
        paired_paths[(int(row["policy_seed"]), int(row["evaluation_seed"]))].add(
            str(row["path_digest"])
        )
    if any(len(values) != 1 for values in paired_paths.values()):
        raise RuntimeError("exact learner paired path digests differ")

    expected_seed = aggregate_per_seed_independent(per_path, config["methods"])
    stored_seed = sorted(
        read_csv(artifact / "per_seed.csv", per_seed=True),
        key=lambda row: (str(row["learner"]), int(row["policy_seed"])),
    )
    assert_nested_close(stored_seed, expected_seed, "per_seed")
    expected_statistics = build_statistics_independent(
        per_path, expected_seed, config
    )
    stored_statistics = json.loads(
        (artifact / "statistics.json").read_text(encoding="utf-8")
    )
    assert_nested_close(stored_statistics, expected_statistics, "statistics")

    trace_count, trace_checks = verify_trace(artifact, config)
    expected_trace_count = 800 * int(config["episode_steps"])
    if trace_count != expected_trace_count:
        raise RuntimeError("exact learner trace row count differs")
    stored_checks = json.loads((artifact / "checks.json").read_text(encoding="utf-8"))
    if not all(bool(value) for value in stored_checks.values()):
        raise RuntimeError("exact learner stored checks contain a failure")
    for key, value in trace_checks.items():
        if not value:
            raise RuntimeError(f"exact learner trace check failed: {key}")
    verify_model_digests(artifact, config)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_version") != ARTIFACT_VERSION
        or int(manifest.get("per_path_rows", -1)) != 800
        or int(manifest.get("per_seed_rows", -1)) != 20
        or int(manifest.get("trace_rows", -1)) != expected_trace_count
    ):
        raise RuntimeError("exact learner manifest counts differ")
    for required in ("README.md", "PROTOCOL.md", "EXECUTION_AMENDMENT.md", "path_digests.json"):
        if not (artifact / required).is_file():
            raise FileNotFoundError(f"exact learner bundle missing {required}")
    return {
        "status": "pass",
        "mode": "bundle_only",
        "per_path_rows": len(per_path),
        "per_seed_rows": len(stored_seed),
        "trace_rows": trace_count,
        "classifications": {
            learner: stored_statistics["learners"][learner]["classification"]
            for learner in config["learners"]
        },
        "checks": trace_checks,
    }


def verify_strict_raw(artifact: Path, config_path: Path) -> dict[str, Any]:
    result = verify_bundle(artifact)
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle_config = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    assert_nested_close(bundle_config, source_config, "protocol")
    source_results = []
    for path in formal_result_paths(source_config):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_source_file"] = str(path.relative_to(REPO))
        source_results.append(row)
    source_results.sort(key=lambda row: (str(row["learner"]), int(row["policy_seed"])))
    source_paths = sorted(
        [
            {**path, "learner": str(row["learner"])}
            for row in source_results
            for path in row["final_evaluation"]["per_path"]
        ],
        key=path_key,
    )
    committed_paths = sorted(read_csv(artifact / "per_path.csv"), key=path_key)
    assert_nested_close(committed_paths, source_paths, "source_per_path")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    runtime = REPO / str(source_config["output_dir"])
    if manifest["source_summary_sha256"] != sha256_file(runtime / "summary.json"):
        raise RuntimeError("exact learner source summary hash differs")
    if manifest["source_manifest_sha256"] != sha256_file(runtime / "formal" / "manifest.json"):
        raise RuntimeError("exact learner source manifest hash differs")
    for row in source_results:
        key = f"{row['learner']}/{int(row['policy_seed'])}"
        if manifest["source_result_sha256"][key] != sha256_file(
            REPO / str(row["_source_file"])
        ):
            raise RuntimeError(f"exact learner source result hash differs: {key}")
        source_steps = REPO / str(row["step_records_file"])
        if sha256_file(source_steps) != str(row["step_records_sha256"]):
            raise RuntimeError(f"exact learner source step hash differs: {key}")
    result["mode"] = "strict_raw"
    result["checks"] = {
        **result["checks"],
        "source_protocol_exact": True,
        "source_per_path_exact": True,
        "source_result_hashes_valid": True,
        "source_step_hashes_valid": True,
        "source_manifest_hash_valid": True,
        "source_summary_hash_valid": True,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict-raw", action="store_true")
    mode.add_argument("--bundle-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    result = (
        verify_strict_raw(artifact, args.config.resolve())
        if args.strict_raw
        else verify_bundle(artifact)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
