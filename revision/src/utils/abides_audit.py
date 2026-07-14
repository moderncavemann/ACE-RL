"""Pure statistics and table derivation for the ABIDES audit bundle."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, t, ttest_1samp


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
DEFAULT_CONTRASTS = (
    "brero_threshold_buybox_budget_matched",
    "spread_only_budget_matched",
    "matched_random",
)


def coerce_per_path_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: row[field] for field in IDENTITY_FIELDS}
    result["policy_seed"] = int(result["policy_seed"])
    result["evaluation_seed"] = int(result["evaluation_seed"])
    for field in METRIC_FIELDS:
        result[field] = float(row[field])
    return result


def validate_unique_per_path(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [
        (int(row["policy_seed"]), int(row["evaluation_seed"]), str(row["method"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("per-path table contains duplicate seed/path/method rows")


def aggregate_per_seed(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    contrasts: Sequence[str] = DEFAULT_CONTRASTS,
) -> list[dict[str, Any]]:
    normalized = [coerce_per_path_row(row) for row in rows]
    validate_unique_per_path(normalized)
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in normalized:
        grouped[row["policy_seed"]][row["method"]].append(row)
    output = []
    for policy_seed in sorted(grouped):
        seed_methods = grouped[policy_seed]
        if set(seed_methods) != set(methods):
            raise ValueError(f"policy seed {policy_seed} has an incomplete method set")
        evaluation_sets = {
            method: {int(row["evaluation_seed"]) for row in method_rows}
            for method, method_rows in seed_methods.items()
        }
        if len({tuple(sorted(values)) for values in evaluation_sets.values()}) != 1:
            raise ValueError(
                f"policy seed {policy_seed} has unmatched evaluation paths"
            )
        result: dict[str, Any] = {"policy_seed": policy_seed}
        for method in methods:
            method_rows = seed_methods[method]
            for metric in METRIC_FIELDS:
                result[f"{method}__{metric}"] = float(
                    mean(float(row[metric]) for row in method_rows)
                )
        for comparator in contrasts:
            result[f"{comparator}__minus_risk_aware_spread"] = (
                result[f"{comparator}__low_execution_half_spread"]
                - result["risk_aware__low_execution_half_spread"]
            )
        output.append(result)
    return output


def t_interval(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("paired values must be a nonempty finite vector")
    average = float(np.mean(data))
    if len(data) == 1:
        standard_error = 0.0
        ci_low = ci_high = average
        p_value = 1.0
    else:
        standard_error = float(np.std(data, ddof=1) / math.sqrt(len(data)))
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


def exact_sign_test(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    positive = int(np.sum(data > 1e-12))
    negative = int(np.sum(data < -1e-12))
    ties = int(len(data) - positive - negative)
    non_ties = positive + negative
    p_value = (
        float(binomtest(positive, non_ties, 0.5, alternative="two-sided").pvalue)
        if non_ties
        else 1.0
    )
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "non_ties": non_ties,
        "p_value_two_sided": p_value,
    }


def paired_path_differences(
    rows: Sequence[Mapping[str, Any]], comparator: str
) -> dict[int, np.ndarray]:
    normalized = [coerce_per_path_row(row) for row in rows]
    lookup = {
        (row["policy_seed"], row["evaluation_seed"], row["method"]): row
        for row in normalized
    }
    policy_seeds = sorted({row["policy_seed"] for row in normalized})
    result = {}
    for policy_seed in policy_seeds:
        evaluation_seeds = sorted(
            {
                row["evaluation_seed"]
                for row in normalized
                if row["policy_seed"] == policy_seed
            }
        )
        values = []
        for evaluation_seed in evaluation_seeds:
            comparator_row = lookup.get((policy_seed, evaluation_seed, comparator))
            risk_row = lookup.get((policy_seed, evaluation_seed, "risk_aware"))
            if comparator_row is None or risk_row is None:
                raise ValueError("paired comparator/risk-aware path row is missing")
            values.append(
                float(comparator_row["low_execution_half_spread"])
                - float(risk_row["low_execution_half_spread"])
            )
        result[policy_seed] = np.asarray(values, dtype=float)
    return result


def two_way_paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    comparator: str,
    replicates: int,
    rng_seed: int,
) -> dict[str, float | int]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    differences = paired_path_differences(rows, comparator)
    policy_seeds = np.asarray(sorted(differences), dtype=int)
    if len(policy_seeds) == 0:
        raise ValueError("bootstrap has no policy seeds")
    rng = np.random.default_rng(int(rng_seed))
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_policy_seeds = rng.choice(
            policy_seeds, size=len(policy_seeds), replace=True
        )
        seed_means = []
        for policy_seed in sampled_policy_seeds:
            path_values = differences[int(policy_seed)]
            sampled_paths = rng.choice(path_values, size=len(path_values), replace=True)
            seed_means.append(float(np.mean(sampled_paths)))
        estimates[replicate] = float(np.mean(seed_means))
    return {
        "replicates": int(replicates),
        "rng_seed": int(rng_seed),
        "mean": float(np.mean(estimates)),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((float(value), str(key)) for key, value in p_values.items())
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (p_value, key) in enumerate(ordered):
        running = max(running, (count - rank) * p_value)
        adjusted[key] = min(1.0, running)
    return adjusted


def method_means(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> dict[str, dict[str, float]]:
    normalized = [coerce_per_path_row(row) for row in rows]
    output = {}
    for method in methods:
        method_rows = [row for row in normalized if row["method"] == method]
        if not method_rows:
            raise ValueError(f"method {method} has no per-path rows")
        output[method] = {
            metric: float(mean(float(row[metric]) for row in method_rows))
            for metric in METRIC_FIELDS
        }
    return output


def build_statistics(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    contrasts: Sequence[str],
    secondary_contrasts: Sequence[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seed_rows = aggregate_per_seed(rows, methods, contrasts)
    contrast_results = {}
    raw_secondary_p = {}
    for comparator in contrasts:
        values = [
            float(row[f"{comparator}__minus_risk_aware_spread"]) for row in seed_rows
        ]
        interval = t_interval(values)
        result = {
            "t_interval": interval,
            "exact_sign_test": exact_sign_test(values),
            "two_way_bootstrap": two_way_paired_bootstrap(
                rows,
                comparator,
                bootstrap_replicates,
                bootstrap_seed,
            ),
        }
        contrast_results[comparator] = result
        if comparator in secondary_contrasts:
            raw_secondary_p[comparator] = float(interval["p_value_two_sided"])
    adjusted = holm_adjust(raw_secondary_p)
    for comparator, adjusted_p in adjusted.items():
        contrast_results[comparator]["holm_adjusted_p_value"] = adjusted_p
    return {
        "method_means": method_means(rows, methods),
        "contrasts": contrast_results,
        "secondary_holm_family": list(secondary_contrasts),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
    }
