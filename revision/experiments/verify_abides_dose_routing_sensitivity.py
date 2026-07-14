#!/usr/bin/env python3
"""Independently verify the committed ABIDES dose/routing audit bundle."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, t, ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BUNDLE = ROOT / "paper" / "artifacts" / "abides_dose_routing_sensitivity_v1"
CONFIG = ROOT / "configs" / "abides_dose_routing_sensitivity_v1.json"
PAPER_PER_PATH = ROOT / "paper" / "data" / "abides_dose_routing_sensitivity_v1_per_path.csv"
PAPER_PER_SEED = ROOT / "paper" / "data" / "abides_dose_routing_sensitivity_v1_per_seed.csv"
PAPER_DOSE_FIGURE = ROOT / "paper" / "figures" / "abides_dose_curve.png"
PAPER_ROUTING_FIGURE = ROOT / "paper" / "figures" / "abides_routing_sensitivity.png"


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def protocol_fingerprint(config: Mapping[str, Any]) -> str:
    stable = {
        key: value for key, value in config.items() if key not in {"workers", "output_dir"}
    }
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def dose_group_id(value: float) -> str:
    return f"dose_{int(round(float(value) * 100)):03d}"


def routing_group_id(kappa: float, fraction: float) -> str:
    return f"routing_k{float(kappa):g}_f{int(round(float(fraction) * 100)):03d}"


def specifications(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for target_value in config["dose_targets"]:
        target = float(target_value)
        specs.append(
            {
                "group_id": dose_group_id(target),
                "study": "dose",
                "dose_target": target,
                "routing_target_kappa": float(config["routing_target_kappa"]),
                "routing_routable_fraction": float(
                    config["routing_routable_fraction"]
                ),
                "methods": list(config["dose_methods"]),
            }
        )
    for setting in config["routing_settings"]:
        specs.append(
            {
                "group_id": routing_group_id(
                    float(setting["routing_target_kappa"]),
                    float(setting["routing_routable_fraction"]),
                ),
                "study": "routing",
                "dose_target": float(config["routing_activation_target"]),
                "routing_target_kappa": float(setting["routing_target_kappa"]),
                "routing_routable_fraction": float(
                    setting["routing_routable_fraction"]
                ),
                "methods": list(config["routing_methods"]),
            }
        )
    return specs


def interval(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("interval input is invalid")
    average = float(np.mean(data))
    if len(data) == 1:
        se = 0.0
        low = high = average
        p_value = 1.0
    else:
        sd = float(np.std(data, ddof=1))
        if sd <= 1e-15:
            se = 0.0
            low = high = average
            p_value = 0.0 if abs(average) > 1e-12 else 1.0
        else:
            se = sd / math.sqrt(len(data))
            critical = float(t.ppf(0.975, len(data) - 1))
            low, high = average - critical * se, average + critical * se
            p_value = float(ttest_1samp(data, 0.0).pvalue)
    return {
        "mean": average,
        "se": se,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value_two_sided": p_value,
        "positive": int(np.sum(data > 1e-12)),
        "ties": int(np.sum(np.abs(data) <= 1e-12)),
        "negative": int(np.sum(data < -1e-12)),
        "n": len(data),
    }


def sign_test(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    positive = int(np.sum(data > 1e-12))
    negative = int(np.sum(data < -1e-12))
    ties = len(data) - positive - negative
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


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((float(value), str(key)) for key, value in p_values.items())
    adjusted = {}
    running = 0.0
    for rank, (p_value, key) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * p_value)
        adjusted[key] = min(1.0, running)
    return adjusted


def seed_metric_values(
    rows: Sequence[Mapping[str, str]], group_id: str, method: str, metric: str
) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["group_id"] == group_id and row["method"] == method:
            grouped[int(row["policy_seed"])].append(float(row[metric]))
    return [float(mean(grouped[seed])) for seed in sorted(grouped)]


def paired_bootstrap(
    rows: Sequence[Mapping[str, str]], comparator: str, replicates: int, seed: int
) -> dict[str, float | int]:
    lookup = {
        (int(row["policy_seed"]), int(row["evaluation_seed"]), row["method"]): row
        for row in rows
    }
    policy_seeds = np.asarray(
        sorted({int(row["policy_seed"]) for row in rows}), dtype=int
    )
    differences: dict[int, np.ndarray] = {}
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
                float(lookup[(int(policy_seed), evaluation, comparator)]["low_execution_half_spread"])
                - float(lookup[(int(policy_seed), evaluation, "risk_aware")]["low_execution_half_spread"])
                for evaluation in evaluations
            ],
            dtype=float,
        )
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        sampled_seeds = rng.choice(policy_seeds, size=len(policy_seeds), replace=True)
        estimates[index] = float(
            mean(
                float(
                    np.mean(
                        rng.choice(
                            differences[int(policy_seed)],
                            size=len(differences[int(policy_seed)]),
                            replace=True,
                        )
                    )
                )
                for policy_seed in sampled_seeds
            )
        )
    return {
        "replicates": int(replicates),
        "rng_seed": int(seed),
        "mean": float(np.mean(estimates)),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
    }


def recompute_statistics(
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    output = {}
    raw_p_values = {}
    metrics = (
        "low_execution_half_spread",
        "all_state_execution_half_spread",
        "dealer_reward_units_per_episode",
        "activation_rate",
        "high_risk_activation_rate",
        "mean_routing_total_variation",
        "mean_volume",
    )
    for spec in specs:
        group_id = str(spec["group_id"])
        methods = ["no_intervention", *spec["methods"]]
        method_stats = {
            method: {
                metric: interval(seed_metric_values(rows, group_id, method, metric))
                for metric in metrics
            }
            for method in methods
        }
        no_reward = seed_metric_values(
            rows, group_id, "no_intervention", "dealer_reward_units_per_episode"
        )
        for method in methods:
            method_reward = seed_metric_values(
                rows, group_id, method, "dealer_reward_units_per_episode"
            )
            method_stats[method]["dealer_reward_loss_vs_no_intervention"] = interval(
                [base - value for base, value in zip(no_reward, method_reward)]
            )
        contrasts = {}
        risk_spread = seed_metric_values(
            rows, group_id, "risk_aware", "low_execution_half_spread"
        )
        group_rows = [row for row in rows if row["group_id"] == group_id]
        for comparator in spec["methods"]:
            if comparator == "risk_aware":
                continue
            comparator_spread = seed_metric_values(
                rows, group_id, comparator, "low_execution_half_spread"
            )
            differences = [left - right for left, right in zip(comparator_spread, risk_spread)]
            result = {
                "policy_seed_t_interval": interval(differences),
                "policy_seed_exact_sign_test": sign_test(differences),
                "two_way_paired_bootstrap": paired_bootstrap(
                    group_rows,
                    comparator,
                    int(config["bootstrap_replicates"]),
                    int(config["bootstrap_seed"]),
                ),
            }
            contrasts[comparator] = result
            raw_p_values[f"{group_id}:{comparator}"] = float(
                result["policy_seed_t_interval"]["p_value_two_sided"]
            )
        risk_activation = float(method_stats["risk_aware"]["activation_rate"]["mean"])
        risk_tv = float(
            method_stats["risk_aware"]["mean_routing_total_variation"]["mean"]
        )
        budget = {}
        for comparator in spec["methods"]:
            if comparator == "risk_aware":
                continue
            comparator_activation = float(method_stats[comparator]["activation_rate"]["mean"])
            comparator_tv = float(
                method_stats[comparator]["mean_routing_total_variation"]["mean"]
            )
            ratio = comparator_tv / risk_tv if risk_tv > 1e-15 else math.nan
            maximum_difference = float(
                config["calibration"]["maximum_pair_activation_difference"]
            )
            minimum_ratio = float(config["calibration"]["minimum_tv_ratio"])
            maximum_ratio = float(config["calibration"]["maximum_tv_ratio"])
            budget[comparator] = {
                "activation_absolute_difference": abs(
                    comparator_activation - risk_activation
                ),
                "routing_tv_ratio_comparator_over_risk_aware": ratio,
                "passive_calibration_tolerances": {
                    "activation_difference_maximum": maximum_difference,
                    "routing_tv_ratio_minimum": minimum_ratio,
                    "routing_tv_ratio_maximum": maximum_ratio,
                },
                "formal_realized_budget_within_passive_tolerances": bool(
                    abs(comparator_activation - risk_activation) <= maximum_difference
                    and math.isfinite(ratio)
                    and minimum_ratio <= ratio <= maximum_ratio
                ),
            }
        output[group_id] = {
            "study": spec["study"],
            "dose_target": float(spec["dose_target"]),
            "routing_target_kappa": float(spec["routing_target_kappa"]),
            "routing_routable_fraction": float(spec["routing_routable_fraction"]),
            "method_seed_statistics": method_stats,
            "contrasts": contrasts,
            "formal_realized_budget_diagnostics": budget,
        }
    adjusted = holm_adjust(raw_p_values)
    for contrast_id, p_value in adjusted.items():
        group_id, comparator = contrast_id.split(":", 1)
        output[group_id]["contrasts"][comparator][
            "holm_adjusted_p_value_global_family"
        ] = p_value
    statistics = {
        "groups": output,
        "holm_family": sorted(raw_p_values),
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "bootstrap_seed": int(config["bootstrap_seed"]),
    }
    statistics["routing_robustness"] = classify_routing(statistics)
    return statistics


def classify_routing(statistics: Mapping[str, Any]) -> dict[str, Any]:
    intervals = {
        group_id: values["contrasts"]["matched_random"]["policy_seed_t_interval"]
        for group_id, values in statistics["groups"].items()
        if values["study"] == "routing"
    }
    favorable = sum(float(values["mean"]) > 0.0 for values in intervals.values())
    adverse = sorted(
        group_id
        for group_id, values in intervals.items()
        if float(values["ci_high"]) < 0.0
    )
    center_supported = float(intervals["routing_k4_f050"]["ci_low"]) > 0.0
    if favorable >= 3 and not adverse:
        label = "broadly_robust"
    elif center_supported and not adverse:
        label = "locally_robust"
    else:
        label = "parameter_specific"
    return {
        "classification": label,
        "favorable_mean_settings": favorable,
        "total_settings": len(intervals),
        "significantly_adverse_settings": adverse,
        "center_supported": center_supported,
        "nonsignificance_is_not_equivalence": True,
    }


def values_close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            values_close(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            values_close(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-12)
    return actual == expected


def verify_per_seed(
    rows: Sequence[Mapping[str, str]], reported: Sequence[Mapping[str, str]]
) -> bool:
    metrics = (
        "low_execution_half_spread",
        "all_state_execution_half_spread",
        "mean_reward",
        "dealer_reward_units_per_episode",
        "activation_rate",
        "high_risk_activation_rate",
        "mean_signal_strength",
        "mean_routing_total_variation",
        "mean_volume",
    )
    grouped: dict[tuple[str, int, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["group_id"], int(row["policy_seed"]), row["method"])].append(row)
    lookup = {
        (row["group_id"], int(row["policy_seed"]), row["method"]): row
        for row in reported
    }
    if set(lookup) != set(grouped):
        return False
    for key, path_rows in grouped.items():
        output = lookup[key]
        if int(output["evaluation_path_count"]) != len(path_rows):
            return False
        for metric in metrics:
            value = float(mean(float(row[metric]) for row in path_rows))
            if not math.isclose(float(output[metric]), value, abs_tol=1e-12, rel_tol=1e-12):
                return False
    return True


def verify_step_records(
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    per_path: Sequence[Mapping[str, str]],
) -> dict[str, bool | int]:
    required = {
        "group_id",
        "study",
        "policy_seed",
        "evaluation_seed",
        "method",
        "step",
        "pre_state_digest",
        "actions",
        "mechanism_signal",
        "mechanism_parameters",
        "routing_threshold",
        "eligible_dealers",
        "routing_weights",
        "latent_risk_state",
        "execution_half_spread",
        "realized_flow_quantity",
        "rewards",
        "inventory_summary",
        "action_uniform_digest",
        "path_digest",
    }
    expected_methods = {
        spec["group_id"]: {"no_intervention", *spec["methods"]} for spec in specs
    }
    unique = set()
    path_counts: dict[tuple[str, int, int, str], int] = defaultdict(int)
    paired: dict[tuple[str, int, int, int], dict[str, set[str]]] = defaultdict(
        lambda: {"methods": set(), "paths": set(), "uniforms": set()}
    )
    valid_schema = True
    valid_values = True
    count = 0
    with gzip.open(BUNDLE / "step_records.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            valid_schema &= required <= set(row)
            key = (
                str(row["group_id"]),
                int(row["policy_seed"]),
                int(row["evaluation_seed"]),
                str(row["method"]),
                int(row["step"]),
            )
            unique.add(key)
            path_counts[key[:4]] += 1
            pair = paired[(key[0], key[1], key[2], key[4])]
            pair["methods"].add(key[3])
            pair["paths"].add(str(row["path_digest"]))
            pair["uniforms"].add(str(row["action_uniform_digest"]))
            weights = [float(value) for value in row["routing_weights"]]
            values = (
                float(row["mechanism_signal"]),
                float(row["execution_half_spread"]),
                float(row["realized_flow_quantity"]),
                *weights,
                *(float(value) for value in row["rewards"]),
            )
            valid_values &= all(math.isfinite(value) for value in values)
            valid_values &= len(weights) == int(config["market"]["n_agents"])
            valid_values &= all(value >= 0.0 for value in weights)
            valid_values &= abs(sum(weights) - 1.0) <= 1e-12
            valid_values &= abs(float(row["realized_flow_quantity"]) - 20.0) <= 1e-12
            valid_values &= len(row["actions"]) == int(config["market"]["n_agents"])
            valid_values &= len(row["rewards"]) == int(config["market"]["n_agents"])
            valid_values &= 0 <= int(row["step"]) < int(config["episode_steps"])
    expected_count = len(per_path) * int(config["episode_steps"])
    path_grid = {
        (row["group_id"], int(row["policy_seed"]), int(row["evaluation_seed"]), row["method"])
        for row in per_path
    }
    paths_complete = set(path_counts) == path_grid and all(
        value == int(config["episode_steps"]) for value in path_counts.values()
    )
    pairs_valid = all(
        values["methods"] == expected_methods[group_id]
        and len(values["paths"]) == 1
        and len(values["uniforms"]) == 1
        for (group_id, _, _, _), values in paired.items()
    )
    return {
        "step_record_count": count,
        "step_record_count_exact": count == expected_count,
        "step_keys_unique": len(unique) == count,
        "step_schema_complete": bool(valid_schema),
        "step_values_valid": bool(valid_values),
        "step_path_grid_complete": paths_complete,
        "step_common_paths_and_uniforms": pairs_valid,
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    specs = specifications(config)
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    aggregate = json.loads((BUNDLE / "aggregate.json").read_text(encoding="utf-8"))
    completion = json.loads((BUNDLE / "completion.json").read_text(encoding="utf-8"))
    reported_statistics = json.loads(
        (BUNDLE / "statistics.json").read_text(encoding="utf-8")
    )
    per_path = read_csv(BUNDLE / "per_path.csv")
    per_seed = read_csv(BUNDLE / "per_seed.csv")
    checks: dict[str, bool] = {}

    checksum_valid = True
    for line in (BUNDLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        checksum_valid &= sha256_file(BUNDLE / name) == expected
    checks["bundle_hashes_valid"] = checksum_valid
    checks["paper_exports_audit_derived"] = (
        (BUNDLE / "per_path.csv").read_bytes() == PAPER_PER_PATH.read_bytes()
        and (BUNDLE / "per_seed.csv").read_bytes() == PAPER_PER_SEED.read_bytes()
        and (BUNDLE / "abides_dose_curve.png").read_bytes()
        == PAPER_DOSE_FIGURE.read_bytes()
        and (BUNDLE / "abides_routing_sensitivity.png").read_bytes()
        == PAPER_ROUTING_FIGURE.read_bytes()
    )
    fingerprint = protocol_fingerprint(config)
    checks["protocol_fingerprint_valid"] = (
        manifest["protocol_fingerprint"]
        == aggregate["protocol_fingerprint"]
        == completion["protocol_fingerprint"]
        == fingerprint
    )
    checks["config_hash_valid"] = manifest["config_sha256"] == sha256_file(CONFIG)
    checks["calibration_hash_valid"] = manifest["calibration_artifact_sha256"] == sha256_file(
        REPO / config["calibration"]["artifact"]
    )
    checks["source_v5_provenance_valid"] = (
        manifest["source_v5_fingerprint"] == config["source_base_fingerprint"]
        and manifest["source_v5_manifest_sha256"]
        == config["source_base_manifest_sha256"]
        and set(manifest["source_policy_artifacts"])
        == {str(seed) for seed in config["source_policy_seeds"]}
    )
    checks["task_provenance_complete"] = (
        len(manifest["task_artifacts"]) == len(specs) * len(config["source_policy_seeds"])
        and all(
            int(values["completed_episodes"]) == int(config["response_episodes"])
            and len(values["step_records_sha256"]) == 64
            and all(len(value) == 64 for value in values["final_model_digests"].values())
            for values in manifest["task_artifacts"].values()
        )
    )
    archival_commit = manifest.get("result_archival_commit")
    checks["result_archival_commit_valid"] = archival_commit is not None and subprocess.run(
        ["git", "cat-file", "-e", f"{archival_commit}^{{commit}}"],
        cwd=REPO,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    checks["completed_resume_verified"] = bool(
        manifest["completed_resume_verified"]
        and completion["completed_resume_verification"]
        and completion["aggregate_unchanged_on_completed_resume"]
        and int(completion["containers_started"]) == 0
    )

    expected_path_count = 0
    expected_grid = set()
    for spec in specs:
        methods = {"no_intervention", *spec["methods"]}
        expected_path_count += (
            len(config["source_policy_seeds"])
            * len(config["evaluation_seeds"])
            * len(methods)
        )
        expected_grid.update(
            (spec["group_id"], int(policy_seed), int(evaluation_seed), method)
            for policy_seed in config["source_policy_seeds"]
            for evaluation_seed in config["evaluation_seeds"]
            for method in methods
        )
    actual_grid = {
        (row["group_id"], int(row["policy_seed"]), int(row["evaluation_seed"]), row["method"])
        for row in per_path
    }
    checks["per_path_grid_complete"] = (
        len(per_path) == expected_path_count and actual_grid == expected_grid
    )
    checks["per_seed_recomputed"] = verify_per_seed(per_path, per_seed)
    recomputed_statistics = recompute_statistics(config, specs, per_path)
    checks["statistics_recomputed"] = values_close(
        reported_statistics, recomputed_statistics
    )
    checks["routing_classification_recomputed"] = (
        reported_statistics["routing_robustness"]
        == aggregate["routing_robustness"]
        == recomputed_statistics["routing_robustness"]
    )
    checks["runner_integrity_checks_pass"] = (
        aggregate["implementation_status"] == "pass"
        and all(bool(value) for value in aggregate["checks"].values())
    )
    step_checks = verify_step_records(config, specs, per_path)
    checks.update(
        {key: bool(value) for key, value in step_checks.items() if key != "step_record_count"}
    )
    checks["manifest_step_count_valid"] = (
        int(manifest["step_record_count"]) == int(step_checks["step_record_count"])
    )
    checks["seed_namespaces_disjoint"] = not (
        set(config["source_policy_seeds"])
        & set(config["passive_calibration_seeds"])
        | set(config["source_policy_seeds"])
        & set(config["evaluation_seeds"])
        | set(config["passive_calibration_seeds"])
        & set(config["evaluation_seeds"])
    )
    checks["secondary_claim_boundary_retained"] = (
        "secondary sensitivity" in aggregate["claim_boundary"]
        and "nonsignificance is not equivalence" in aggregate["claim_boundary"]
    )
    status = "pass" if all(checks.values()) else "fail"
    print(
        json.dumps(
            {
                "status": status,
                "checks": checks,
                "routing_robustness": recomputed_statistics["routing_robustness"],
                "per_path_rows": len(per_path),
                "step_record_rows": int(step_checks["step_record_count"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
