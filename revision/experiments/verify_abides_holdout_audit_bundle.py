#!/usr/bin/env python3
"""Independently verify the combined ABIDES learner/ecology holdout bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, t, ttest_1samp


REPO = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPO / "revision/paper/artifacts/abides_holdout_audit_bundle_v1"
OFFICIAL_CONFIG = REPO / "revision/configs/abides_standard_learner_holdout_v1.json"
OFFICIAL_OUTPUT = REPO / "revision/experiments/outputs/abides_standard_learner_holdout_v1"
ECOLOGY_AUDIT_OUTPUT = (
    REPO / "revision/experiments/outputs/abides_holdout_audit_evaluation_v3"
)
ECOLOGY_CONFIGS = {
    "ecology_e1": REPO / "revision/configs/abides_ecology_e1_response_v1.json",
    "ecology_e2": REPO / "revision/configs/abides_ecology_e2_response_v1.json",
}
PAPER_PER_PATH = REPO / "revision/paper/data/abides_holdout_v1_per_path.csv"
PAPER_PER_SEED = REPO / "revision/paper/data/abides_holdout_v1_per_seed.csv"
PROTOCOL_VERSION = "abides_holdout_audit_bundle_v1"
OFFICIAL_COMMIT = "de66d7a4b23fac2513f56f96f73b3f5cb96695ac"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_714
OFFICIAL_METHODS = (
    "no_intervention",
    "risk_aware",
    "matched_random",
    "brero_threshold_buybox_budget_matched",
)
ECOLOGY_METHODS = (
    "no_intervention",
    "risk_aware",
    "matched_random",
    "spread_only_threshold",
    "always_on",
    "no_routing",
)
GROUPS = (
    "official_full_policy_MAPPO",
    "official_IPPO",
    "ecology_e1",
    "ecology_e2",
)
IDENTITY_FIELDS = (
    "protocol_version",
    "policy_seed",
    "evaluation_seed",
    "method",
    "model_digest",
    "path_digest",
)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def git_blob_sha256(commit: str, relative: str) -> str:
    process = subprocess.Popen(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    _stdout, stderr = process.communicate()
    if process.returncode:
        raise RuntimeError(
            f"archival commit lacks {relative}: {stderr.decode('utf-8', 'replace')}"
        )
    return digest.hexdigest()


def parse_checksums(bundle: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise RuntimeError(f"invalid SHA256SUMS line: {line!r}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in checksums:
            raise RuntimeError(f"unsafe or duplicate checksum path: {relative}")
        checksums[relative] = digest
    required = {
        "README.md",
        "checks.json",
        "manifest.json",
        "per_path.csv",
        "per_seed.csv",
        "statistics.json",
        "step_records.jsonl.gz",
    }
    if set(checksums) != required:
        raise RuntimeError("SHA256SUMS does not cover the exact bundle payload")
    for relative, expected in checksums.items():
        path = bundle / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"bundle checksum mismatch: {relative}")
    return checksums


def assert_nested_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise RuntimeError(f"mapping mismatch at {path}")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"list mismatch at {path}")
        for index, value in enumerate(expected):
            assert_nested_close(actual[index], value, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(f"numeric mismatch at {path}: {actual} != {expected}")
        return
    if actual != expected:
        raise RuntimeError(f"value mismatch at {path}: {actual!r} != {expected!r}")


def normalize_path_row(
    row: Mapping[str, Any], holdout: str, learner: str
) -> dict[str, Any]:
    normalized = {
        "holdout": str(holdout),
        "learner": str(learner),
        **{field: row[field] for field in IDENTITY_FIELDS},
    }
    normalized["policy_seed"] = int(normalized["policy_seed"])
    normalized["evaluation_seed"] = int(normalized["evaluation_seed"])
    for field in METRIC_FIELDS:
        normalized[field] = float(row[field])
    return normalized


def path_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row["holdout"]),
        str(row["learner"]),
        int(row["policy_seed"]),
        int(row["evaluation_seed"]),
        str(row["method"]),
    )


def load_source_results() -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    official_config = read_json(OFFICIAL_CONFIG)
    ecology_configs = {cell: read_json(path) for cell, path in ECOLOGY_CONFIGS.items()}
    official = []
    for learner in official_config["learners"]:
        for policy_seed in official_config["policy_seeds"]:
            path = (
                OFFICIAL_OUTPUT
                / "formal"
                / str(learner)
                / f"seed_{int(policy_seed)}"
                / "result.json"
            )
            result = read_json(path)
            if (
                result["learner"] != learner
                or int(result["policy_seed"]) != int(policy_seed)
                or result["official_commit"] != OFFICIAL_COMMIT
            ):
                raise RuntimeError(f"official source identity mismatch: {path}")
            result["_source_file"] = str(path.relative_to(REPO))
            official.append(result)
    ecology = []
    for cell, config in ecology_configs.items():
        for policy_seed in config["policy_seeds"]:
            path = (
                ECOLOGY_AUDIT_OUTPUT
                / cell
                / f"seed_{int(policy_seed)}"
                / "result.json"
            )
            result = read_json(path)
            if result["cell"] != cell or int(result["policy_seed"]) != int(
                policy_seed
            ):
                raise RuntimeError(f"ecology source identity mismatch: {path}")
            response_path = (
                REPO
                / str(config["output_dir"])
                / "response_results"
                / f"seed_{int(policy_seed)}.json"
            )
            if sha256_file(response_path) != str(result["source_result_sha256"]):
                raise RuntimeError(
                    f"ecology audit/source response hash mismatch: {response_path}"
                )
            result["_source_file"] = str(path.relative_to(REPO))
            result["_response_result_file"] = str(response_path.relative_to(REPO))
            ecology.append(result)
    official.sort(key=lambda row: (str(row["learner"]), int(row["policy_seed"])))
    ecology.sort(key=lambda row: (str(row["cell"]), int(row["policy_seed"])))
    return official_config, ecology_configs, official, ecology


def source_path_rows(
    official: Sequence[Mapping[str, Any]], ecology: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for result in official:
        learner = str(result["learner"])
        rows.extend(
            normalize_path_row(row, "standard_learner", learner)
            for row in result["final_evaluation"]["per_path"]
        )
    for result in ecology:
        rows.extend(
            normalize_path_row(row, str(result["cell"]), str(result["learner"]))
            for row in result["per_path"]
        )
    return sorted(rows, key=path_key)


def expected_path_grid(
    official_config: Mapping[str, Any], ecology_configs: Mapping[str, Mapping[str, Any]]
) -> set[tuple[str, str, int, int, str]]:
    expected = {
        (
            "standard_learner",
            str(learner),
            int(policy_seed),
            int(evaluation_seed),
            str(method),
        )
        for learner in official_config["learners"]
        for policy_seed in official_config["policy_seeds"]
        for evaluation_seed in official_config["response_evaluation_seeds"]
        for method in OFFICIAL_METHODS
    }
    for cell, config in ecology_configs.items():
        expected.update(
            (
                cell,
                "compact_MAPPO_style_intervention_adapter",
                int(policy_seed),
                int(evaluation_seed),
                str(method),
            )
            for policy_seed in config["policy_seeds"]
            for evaluation_seed in config["evaluation_seeds"]
            for method in ECOLOGY_METHODS
        )
    return expected


def verify_per_path(
    bundle: Path,
    official_config: Mapping[str, Any],
    ecology_configs: Mapping[str, Mapping[str, Any]],
    official: Sequence[Mapping[str, Any]],
    ecology: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    committed = sorted(
        (
            normalize_path_row(row, str(row["holdout"]), str(row["learner"]))
            for row in read_csv(bundle / "per_path.csv")
        ),
        key=path_key,
    )
    expected_rows = source_path_rows(official, ecology)
    expected_keys = expected_path_grid(official_config, ecology_configs)
    keys = [path_key(row) for row in committed]
    if len(committed) != 2_000 or len(set(keys)) != 2_000 or set(keys) != expected_keys:
        raise RuntimeError("combined per-path identity grid is incomplete or duplicated")
    assert_nested_close(committed, expected_rows, "per_path_source_correspondence")

    protocols = {
        "standard_learner": str(official_config["protocol_version"]),
        **{
            cell: str(config["protocol_version"])
            for cell, config in ecology_configs.items()
        },
    }
    paired_paths: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for row in committed:
        if row["protocol_version"] != protocols[str(row["holdout"])]:
            raise RuntimeError("per-path protocol version mismatch")
        if not math.isclose(float(row["mean_volume"]), 20.0, abs_tol=1e-12):
            raise RuntimeError("per-path volume differs from frozen demand")
        paired_paths[path_key(row)[:4]].add(str(row["path_digest"]))
    if any(len(digests) != 1 for digests in paired_paths.values()):
        raise RuntimeError("exogenous path digest differs across a paired method grid")
    return committed, {
        "per_path_grid_exact": True,
        "per_path_matches_source_results": True,
        "paired_path_digests_equal": True,
        "per_path_volume_conserved": True,
    }


def group_spec(group: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if group.startswith("official_"):
        return (
            OFFICIAL_METHODS,
            ("matched_random", "brero_threshold_buybox_budget_matched"),
            ("brero_threshold_buybox_budget_matched",),
        )
    return (
        ECOLOGY_METHODS,
        (
            "matched_random",
            "no_intervention",
            "spread_only_threshold",
            "always_on",
            "no_routing",
        ),
        ("no_intervention", "spread_only_threshold", "always_on", "no_routing"),
    )


def rows_for_group(rows: Sequence[Mapping[str, Any]], group: str) -> list[dict[str, Any]]:
    if group.startswith("official_"):
        return [
            dict(row)
            for row in rows
            if row["holdout"] == "standard_learner" and row["learner"] == group
        ]
    return [dict(row) for row in rows if row["holdout"] == group]


def aggregate_per_seed_independent(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    contrasts: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row["policy_seed"])][str(row["method"])].append(row)
    output = []
    for policy_seed in sorted(grouped):
        method_rows = grouped[policy_seed]
        if set(method_rows) != set(methods):
            raise RuntimeError(f"incomplete method set for policy seed {policy_seed}")
        evaluation_sets = {
            method: {int(row["evaluation_seed"]) for row in values}
            for method, values in method_rows.items()
        }
        if len({tuple(sorted(values)) for values in evaluation_sets.values()}) != 1:
            raise RuntimeError(f"unpaired evaluation paths for policy seed {policy_seed}")
        result: dict[str, Any] = {"policy_seed": policy_seed}
        for method in methods:
            for metric in METRIC_FIELDS:
                result[f"{method}__{metric}"] = float(
                    mean(float(row[metric]) for row in method_rows[method])
                )
        for comparator in contrasts:
            result[f"{comparator}__minus_risk_aware_spread"] = (
                result[f"{comparator}__low_execution_half_spread"]
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
        p_value = 0.0 if abs(average) > 1e-12 and len(data) > 1 else 1.0
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
    rows: Sequence[Mapping[str, Any]], comparator: str
) -> dict[str, float | int]:
    lookup = {
        (int(row["policy_seed"]), int(row["evaluation_seed"]), str(row["method"])): row
        for row in rows
    }
    policy_seeds = np.asarray(sorted({int(row["policy_seed"]) for row in rows}))
    differences = {}
    for policy_seed in policy_seeds:
        evaluation_seeds = sorted(
            {
                int(row["evaluation_seed"])
                for row in rows
                if int(row["policy_seed"]) == int(policy_seed)
            }
        )
        differences[int(policy_seed)] = np.asarray(
            [
                float(lookup[(int(policy_seed), seed, comparator)]["low_execution_half_spread"])
                - float(lookup[(int(policy_seed), seed, "risk_aware")]["low_execution_half_spread"])
                for seed in evaluation_seeds
            ]
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = rng.choice(policy_seeds, size=len(policy_seeds), replace=True)
        estimates[replicate] = float(
            np.mean(
                [
                    np.mean(
                        rng.choice(
                            differences[int(policy_seed)],
                            size=len(differences[int(policy_seed)]),
                            replace=True,
                        )
                    )
                    for policy_seed in sampled_seeds
                ]
            )
        )
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": BOOTSTRAP_SEED,
        "mean": float(np.mean(estimates)),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
    }


def holm_adjust_independent(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((float(value), str(key)) for key, value in p_values.items())
    running = 0.0
    adjusted = {}
    for rank, (p_value, key) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * p_value)
        adjusted[key] = min(1.0, running)
    return adjusted


def build_statistics_independent(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    contrasts: Sequence[str],
    secondary: Sequence[str],
) -> dict[str, Any]:
    per_seed = aggregate_per_seed_independent(rows, methods, contrasts)
    contrast_results = {}
    secondary_p = {}
    for comparator in contrasts:
        values = [
            float(row[f"{comparator}__minus_risk_aware_spread"])
            for row in per_seed
        ]
        interval = t_interval_independent(values)
        contrast_results[comparator] = {
            "t_interval": interval,
            "exact_sign_test": exact_sign_test_independent(values),
            "two_way_bootstrap": two_way_bootstrap_independent(rows, comparator),
        }
        if comparator in secondary:
            secondary_p[comparator] = float(interval["p_value_two_sided"])
    for comparator, adjusted in holm_adjust_independent(secondary_p).items():
        contrast_results[comparator]["holm_adjusted_p_value"] = adjusted
    method_means = {
        method: {
            metric: float(
                mean(float(row[metric]) for row in rows if row["method"] == method)
            )
            for metric in METRIC_FIELDS
        }
        for method in methods
    }
    return {
        "method_means": method_means,
        "contrasts": contrast_results,
        "secondary_holm_family": list(secondary),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def classify_statistics(statistics: Mapping[str, Any]) -> dict[str, str]:
    classification = {}
    for group in GROUPS:
        interval = statistics[group]["contrasts"]["matched_random"]["t_interval"]
        if float(interval["ci_low"]) > 0.0:
            classification[group] = "favorable_timing_interval"
        elif float(interval["mean"]) > 0.0:
            classification[group] = "positive_direction_inconclusive_interval"
        else:
            classification[group] = "nonpositive_timing_direction"
    return classification


def verify_statistics(
    bundle: Path, per_path: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, bool]]:
    expected_statistics = {}
    expected_per_seed = []
    for group in GROUPS:
        methods, contrasts, secondary = group_spec(group)
        rows = rows_for_group(per_path, group)
        expected_statistics[group] = build_statistics_independent(
            rows, methods, contrasts, secondary
        )
        holdout = "standard_learner" if group.startswith("official_") else group
        learner = (
            group
            if group.startswith("official_")
            else "compact_MAPPO_style_intervention_adapter"
        )
        expected_per_seed.extend(
            {"holdout": holdout, "learner": learner, **row}
            for row in aggregate_per_seed_independent(rows, methods, contrasts)
        )
    expected_per_seed.sort(
        key=lambda row: (
            str(row["holdout"]), str(row["learner"]), int(row["policy_seed"])
        )
    )
    committed_seed = []
    for row in read_csv(bundle / "per_seed.csv"):
        committed_seed.append(
            {
                key: (
                    value
                    if key in {"holdout", "learner"}
                    else int(value)
                    if key == "policy_seed"
                    else float(value)
                )
                for key, value in row.items()
                if value != ""
            }
        )
    if len(committed_seed) != 40:
        raise RuntimeError("combined per-seed table does not contain 40 rows")
    assert_nested_close(committed_seed, expected_per_seed, "per_seed")
    committed_statistics = read_json(bundle / "statistics.json")
    assert_nested_close(committed_statistics, expected_statistics, "statistics")
    classification = classify_statistics(expected_statistics)
    return expected_statistics, {
        "per_seed_recomputed_from_per_path": True,
        "t_intervals_recomputed": True,
        "exact_sign_tests_recomputed": True,
        "two_way_bootstrap_recomputed": True,
        "secondary_holm_recomputed": True,
        "claim_classification_recomputed": bool(classification),
    }


def source_step_rows(
    official: Sequence[Mapping[str, Any]], ecology: Sequence[Mapping[str, Any]]
) -> Iterable[dict[str, Any]]:
    for result in official:
        path = REPO / str(result["step_records_file"])
        if sha256_file(path) != str(result["step_records_sha256"]):
            raise RuntimeError(f"official source step hash mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["holdout"] = "standard_learner"
                row["learner"] = str(result["learner"])
                yield row
    for result in ecology:
        path = REPO / str(result["step_records_file"])
        if sha256_file(path) != str(result["step_records_sha256"]):
            raise RuntimeError(f"ecology source step hash mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def verify_step_records(
    bundle: Path,
    per_path: Sequence[Mapping[str, Any]],
    official: Sequence[Mapping[str, Any]],
    ecology: Sequence[Mapping[str, Any]],
    checks: Mapping[str, Any],
) -> dict[str, bool]:
    expected_source = iter(source_step_rows(official, ecology))
    path_digests = {path_key(row): str(row["path_digest"]) for row in per_path}
    seen: set[tuple[str, str, int, int, str, int]] = set()
    paired: dict[tuple[str, str, int, int, int], dict[str, set[str]]] = defaultdict(
        lambda: {"path": set(), "uniform": set(), "risk": set()}
    )
    episode_flow: dict[tuple[str, str, int, int, str], float] = defaultdict(float)
    episode_active: dict[tuple[str, str, int, int, str], int] = defaultdict(int)
    episode_high_active: dict[tuple[str, str, int, int, str], int] = defaultdict(int)
    episode_high_steps: dict[tuple[str, str, int, int, str], int] = defaultdict(int)
    targeted_events: dict[tuple[str, int, str], int] = defaultdict(int)
    stream_hash = hashlib.sha256()
    count = 0
    with gzip.open(bundle / "step_records.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            actual = json.loads(line)
            try:
                expected = next(expected_source)
            except StopIteration as error:
                raise RuntimeError("bundle has more step rows than source files") from error
            if canonical_json(actual) != canonical_json(expected):
                raise RuntimeError(f"step source mismatch at combined row {count}")
            canonical_line = canonical_json(actual) + "\n"
            stream_hash.update(canonical_line.encode("utf-8"))
            identity = (
                str(actual["holdout"]),
                str(actual["learner"]),
                int(actual["policy_seed"]),
                int(actual["evaluation_seed"]),
                str(actual["method"]),
                int(actual["step"]),
            )
            if identity in seen or not 0 <= identity[-1] < 100:
                raise RuntimeError(f"duplicate or invalid step identity: {identity}")
            seen.add(identity)
            if path_digests[identity[:5]] != str(actual["path_digest"]):
                raise RuntimeError(f"step/per-path digest mismatch: {identity}")
            weights = [float(value) for value in actual["routing_weights"]]
            if len(weights) != 10 or any(value < -1e-15 for value in weights):
                raise RuntimeError(f"invalid routing vector: {identity}")
            if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"routing vector does not sum to one: {identity}")
            flow = float(actual["realized_flow_quantity"])
            if not math.isclose(flow, 20.0, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"per-step flow is not conserved: {identity}")
            episode_flow[identity[:5]] += flow
            paired_key = (*identity[:4], identity[-1])
            paired[paired_key]["path"].add(str(actual["path_digest"]))
            paired[paired_key]["uniform"].add(str(actual["action_uniform_digest"]))
            paired[paired_key]["risk"].add(str(actual["latent_risk_state"]))
            if identity[0].startswith("ecology_"):
                if "gate_active" not in actual:
                    raise RuntimeError("compact ecology activation provenance is missing")
                active = bool(actual["gate_active"])
                episode_active[identity[:5]] += int(active)
                if str(actual["latent_risk_state"]) == "high":
                    episode_high_steps[identity[:5]] += 1
                    episode_high_active[identity[:5]] += int(active)
                targeted_events[(identity[0], identity[2], identity[4])] += int(
                    actual["targeted_event_count"]
                )
                if actual.get("routing_weights_source") != "realized_flow_shares":
                    raise RuntimeError("compact ecology routing provenance is missing")
                if [float(value) for value in actual["flow_shares"]] != weights:
                    raise RuntimeError("compact ecology flow shares differ from audit weights")
                if (
                    actual.get("mechanism_parameters", {}).get("audit_weights_source")
                    != "realized_flow_shares"
                ):
                    raise RuntimeError("compact ecology mechanism provenance is missing")
            count += 1
    try:
        next(expected_source)
    except StopIteration:
        pass
    else:
        raise RuntimeError("source files have more step rows than the bundle")
    if count != 200_000 or len(seen) != 200_000:
        raise RuntimeError("combined step identity grid is incomplete")
    if any(
        any(len(values) != 1 for values in fields.values())
        for fields in paired.values()
    ):
        raise RuntimeError("paired exogenous path/action/risk streams differ")
    if any(
        not math.isclose(value, 2_000.0, rel_tol=0.0, abs_tol=1e-12)
        for value in episode_flow.values()
    ):
        raise RuntimeError("episode flow is not conserved")
    if stream_hash.hexdigest() != str(checks["step_stream_sha256"]):
        raise RuntimeError("combined canonical step stream hash differs")

    ecology_path_rows = {
        path_key(row): row
        for row in per_path
        if str(row["holdout"]).startswith("ecology_")
    }
    for identity, row in ecology_path_rows.items():
        if not math.isclose(
            float(row["activation_rate"]),
            episode_active[identity] / 100.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"ecology activation rate does not replay: {identity}")
        high_steps = episode_high_steps[identity]
        expected_high = (
            episode_high_active[identity] / high_steps if high_steps else 0.0
        )
        if not math.isclose(
            float(row["high_risk_activation_rate"]),
            expected_high,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"ecology high-risk activation does not replay: {identity}"
            )

    for result in ecology:
        source = read_json(REPO / str(result["_response_result_file"]))
        source_methods = source["final_evaluation"]["methods"]
        result_rows = list(result["per_path"])
        for method in ECOLOGY_METHODS:
            rows = [row for row in result_rows if row["method"] == method]
            if len(rows) != 10:
                raise RuntimeError("ecology source aggregate has incomplete paths")
            low_steps = sum(float(row["low_risk_steps"]) for row in rows)
            high_steps = sum(float(row["high_risk_steps"]) for row in rows)
            replayed = {
                "low_execution_half_spread": sum(
                    float(row["low_execution_half_spread"])
                    * float(row["low_risk_steps"])
                    for row in rows
                )
                / low_steps,
                "mean_reward": float(mean(float(row["mean_reward"]) for row in rows)),
                "mean_volume": float(mean(float(row["mean_volume"]) for row in rows)),
                "activation_rate": float(
                    mean(float(row["activation_rate"]) for row in rows)
                ),
                "high_risk_activation_rate": (
                    sum(
                        float(row["high_risk_activation_rate"])
                        * float(row["high_risk_steps"])
                        for row in rows
                    )
                    / high_steps
                    if high_steps
                    else 0.0
                ),
                "targeted_events": targeted_events[
                    (str(result["cell"]), int(result["policy_seed"]), method)
                ],
            }
            assert_nested_close(
                replayed,
                source_methods[method],
                f"ecology_source_aggregate.{result['cell']}.{result['policy_seed']}.{method}",
            )
            expected_model = (
                source["base_model_digest"]
                if method == "no_intervention"
                else source["final_model_digests"][method]
            )
            if {str(row["model_digest"]) for row in rows} != {str(expected_model)}:
                raise RuntimeError("ecology source model digest differs")
    return {
        "step_source_correspondence_exact": True,
        "step_identity_grid_exact": True,
        "routing_vectors_valid": True,
        "flow_conserved_per_step_and_episode": True,
        "paired_exogenous_streams_equal": True,
        "compact_flow_share_provenance_explicit": True,
        "compact_activation_recomputed_from_steps": True,
        "ecology_source_aggregates_recomputed": True,
        "canonical_step_stream_hash_recomputed": True,
    }


def logbook_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expected_provenance(
    official: Sequence[Mapping[str, Any]], ecology: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    official_manifest = read_json(OFFICIAL_OUTPUT / "formal/manifest.json")
    audit_manifest = read_json(ECOLOGY_AUDIT_OUTPUT / "manifest.json")
    return {
        "official": {
            "config_sha256": sha256_file(OFFICIAL_CONFIG),
            "selection_sha256": sha256_file(
                REPO / "revision/configs/abides_standard_learner_holdout_v1_selection.json"
            ),
            "protocol_fingerprint": official_manifest["protocol_fingerprint"],
            "execution_commit": official_manifest["git_commit"],
            "summary_sha256": sha256_file(OFFICIAL_OUTPUT / "summary.json"),
            "result_sha256": {
                f"{row['learner']}:{row['policy_seed']}": sha256_file(
                    REPO / str(row["_source_file"])
                )
                for row in official
            },
            "official_repository": "https://github.com/marlbenchmark/on-policy",
            "official_commit": OFFICIAL_COMMIT,
            "official_license": "MIT",
        },
        "ecology_audit": {
            "evaluator_version": audit_manifest["evaluator_version"],
            "audit_fingerprint": audit_manifest["audit_fingerprint"],
            "execution_commit": audit_manifest["git_commit"],
            "summary_sha256": sha256_file(ECOLOGY_AUDIT_OUTPUT / "summary.json"),
            "result_sha256": {
                f"{row['cell']}:{row['policy_seed']}": sha256_file(
                    REPO / str(row["_source_file"])
                )
                for row in ecology
            },
            "response_configs": {
                cell: sha256_file(path) for cell, path in ECOLOGY_CONFIGS.items()
            },
        },
    }


def verify_provenance(
    bundle: Path,
    manifest: Mapping[str, Any],
    checksums: Mapping[str, str],
    official_config: Mapping[str, Any],
    official: Sequence[Mapping[str, Any]],
    ecology: Sequence[Mapping[str, Any]],
    allow_unarchived: bool,
) -> dict[str, bool]:
    expected = expected_provenance(official, ecology)
    assert_nested_close(manifest["source_provenance"], expected, "source_provenance")
    expected_fingerprint = hashlib.sha256(
        canonical_json(expected).encode("utf-8")
    ).hexdigest()
    if manifest["protocol_fingerprint"] != expected_fingerprint:
        raise RuntimeError("holdout bundle protocol fingerprint differs")
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("holdout bundle protocol version differs")
    if expected["ecology_audit"]["evaluator_version"] != (
        "abides_holdout_audit_evaluation_v3"
    ):
        raise RuntimeError("ecology audit evaluator is not the corrected v3 schema")
    for relative, digest in manifest["output_file_sha256"].items():
        if checksums.get(relative) != digest:
            raise RuntimeError(f"manifest/checksum disagreement: {relative}")
    if (bundle / "per_path.csv").read_bytes() != PAPER_PER_PATH.read_bytes():
        raise RuntimeError("paper per-path CSV is not bundle-derived")
    if (bundle / "per_seed.csv").read_bytes() != PAPER_PER_SEED.read_bytes():
        raise RuntimeError("paper per-seed CSV is not bundle-derived")

    official_source = Path(str(official_config["official_source_root"])).expanduser()
    if not official_source.is_dir():
        raise RuntimeError("pinned official learner source has not been installed")
    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=official_source, text=True
    ).strip()
    if source_head != OFFICIAL_COMMIT:
        raise RuntimeError("installed official learner source commit differs")

    freeze_commits = {
        "official_selection": git_output(
            "log", "-1", "--format=%H", "--", str(OFFICIAL_CONFIG.relative_to(REPO))
        ),
        "ecology": git_output(
            "log",
            "-1",
            "--format=%H",
            "--",
            "revision/configs/abides_ecology_e1_response_v1.json",
            "revision/configs/abides_ecology_e2_response_v1.json",
        ),
    }
    if dict(manifest["protocol_freeze_commits"]) != freeze_commits:
        raise RuntimeError("protocol freeze commit provenance differs")

    archival_commit = manifest.get("result_archival_commit")
    if not archival_commit and not allow_unarchived:
        raise RuntimeError("result archival commit is not sealed")
    if archival_commit:
        resolved = git_output("rev-parse", f"{archival_commit}^{{commit}}")
        if resolved != archival_commit or len(str(archival_commit)) != 40:
            raise RuntimeError("result archival commit is invalid")
        for relative in (
            "README.md",
            "checks.json",
            "per_path.csv",
            "per_seed.csv",
            "statistics.json",
            "step_records.jsonl.gz",
        ):
            repository_path = str((bundle / relative).relative_to(REPO))
            if git_blob_sha256(str(archival_commit), repository_path) != sha256_file(
                bundle / relative
            ):
                raise RuntimeError(f"archived payload differs: {relative}")
        if str(manifest.get("git_dirty_status_at_export", "")):
            raise RuntimeError("sealed export was not produced from a clean worktree")
    return {
        "source_provenance_recomputed": True,
        "official_source_commit_exact": True,
        "protocol_freeze_commits_verified": True,
        "bundle_hashes_valid": True,
        "paper_exports_bundle_derived": True,
        "result_archival_commit_verified": bool(archival_commit) or allow_unarchived,
    }


def verify_runner_state(checks: Mapping[str, Any]) -> dict[str, bool]:
    official_summary = read_json(OFFICIAL_OUTPUT / "summary.json")
    audit_summary = read_json(ECOLOGY_AUDIT_OUTPUT / "summary.json")
    ecology_summaries = {
        cell: read_json(REPO / config["output_dir"] / "summary.json")
        for cell, config in (
            (cell, read_json(path)) for cell, path in ECOLOGY_CONFIGS.items()
        )
    }
    official_resume = any(
        event.get("event") == "completed_resume_verified"
        and int(event.get("containers_started", -1)) == 0
        and int(event.get("result_count", -1)) == 20
        for event in logbook_events(OFFICIAL_OUTPUT / "logbook.jsonl")
    )
    ecology_resume = {
        cell: any(
            event.get("event") == "run_resumed"
            and not event.get("pending_policy_seeds")
            for event in logbook_events(REPO / config["output_dir"] / "logbook.jsonl")
        )
        for cell, config in (
            (cell, read_json(path)) for cell, path in ECOLOGY_CONFIGS.items()
        )
    }
    if official_summary["implementation_status"] != "pass" or not all(
        official_summary["checks"].values()
    ):
        raise RuntimeError("official learner runner integrity did not pass")
    if any(
        summary["status"] != "pass" or not all(summary["checks"].values())
        for summary in ecology_summaries.values()
    ):
        raise RuntimeError("an ecology runner integrity check did not pass")
    if (
        audit_summary["status"] != "pass"
        or not audit_summary["all_checks_pass"]
        or not audit_summary["completed_resume"]
    ):
        raise RuntimeError("ecology audit replay integrity did not pass")
    if not official_resume or not all(ecology_resume.values()):
        raise RuntimeError("a completed no-container resume is missing")
    if dict(checks["official_runner_checks"]) != official_summary["checks"]:
        raise RuntimeError("bundle official runner checks differ from source")
    if dict(checks["ecology_runner_checks"]) != {
        cell: summary["checks"] for cell, summary in ecology_summaries.items()
    }:
        raise RuntimeError("bundle ecology runner checks differ from source")
    return {
        "official_runner_checks_pass": True,
        "ecology_runner_checks_pass": True,
        "ecology_audit_replay_pass": True,
        "completed_resumes_verified": True,
    }


def verify_bundle(bundle: Path, allow_unarchived: bool = False) -> dict[str, Any]:
    checksums = parse_checksums(bundle)
    manifest = read_json(bundle / "manifest.json")
    checks = read_json(bundle / "checks.json")
    official_config, ecology_configs, official, ecology = load_source_results()
    per_path, path_checks = verify_per_path(
        bundle, official_config, ecology_configs, official, ecology
    )
    statistics, statistic_checks = verify_statistics(bundle, per_path)
    classification = classify_statistics(statistics)
    if classification != manifest["claim_classification"]:
        raise RuntimeError("manifest claim classification differs from recomputation")
    if classification != checks["claim_classification"]:
        raise RuntimeError("checks claim classification differs from recomputation")
    step_checks = verify_step_records(
        bundle, per_path, official, ecology, checks
    )
    provenance_checks = verify_provenance(
        bundle,
        manifest,
        checksums,
        official_config,
        official,
        ecology,
        allow_unarchived,
    )
    runner_checks = verify_runner_state(checks)
    declared_counts = {
        "per_path_count": {"actual": 2_000, "expected": 2_000},
        "per_seed_count": {"actual": 40, "expected": 40},
        "step_record_count": {"actual": 200_000, "expected": 200_000},
    }
    for name, expected in declared_counts.items():
        if checks[name] != expected:
            raise RuntimeError(f"declared count differs: {name}")
    all_checks = {
        **path_checks,
        **statistic_checks,
        **step_checks,
        **provenance_checks,
        **runner_checks,
        "claim_groups_not_pooled": set(classification) == set(GROUPS),
        "declared_counts_exact": True,
        "bundle_integrity_status_true": bool(checks["all_integrity_checks_pass"]),
    }
    if not all(all_checks.values()):
        failed = sorted(name for name, passed in all_checks.items() if not passed)
        raise RuntimeError(f"holdout audit checks failed: {failed}")
    return {
        "audit_verification_status": "pass",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "result_archival_commit": manifest.get("result_archival_commit"),
        "claim_classification": classification,
        "counts": {"per_path": 2_000, "per_seed": 40, "per_step": 200_000},
        "checks": all_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--allow-unarchived", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = args.bundle if args.bundle.is_absolute() else REPO / args.bundle
    result = verify_bundle(bundle, allow_unarchived=args.allow_unarchived)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
