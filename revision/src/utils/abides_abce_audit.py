"""Trace-first metrics and paired inference for the ABCE protocol."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils.abides_audit import exact_sign_test, holm_adjust, t_interval


def _weighted_execution(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    quantity = 0.0
    tick_units = 0.0
    for row in rows:
        executions = row["info"]["execution_records"]
        step_quantity = sum(float(item["quantity"]) for item in executions)
        step_tick_units = sum(
            float(item["quantity"])
            * abs(float(item["price"]) - float(item["reference_mid"]))
            for item in executions
        )
        if abs(step_quantity - float(row["info"]["realized_flow_quantity"])) > 1e-12:
            raise ValueError("ABCE execution quantity does not match the step aggregate")
        if abs(
            step_tick_units
            - float(row["info"]["customer_execution_cost_tick_units"])
        ) > 1e-12:
            raise ValueError("ABCE primitive customer cost does not match the aggregate")
        quantity += step_quantity
        tick_units += step_tick_units
    return (tick_units / quantity if quantity else 0.0), tick_units


def _run_lengths(active: Sequence[bool]) -> list[int]:
    result: list[int] = []
    current = 0
    for value in active:
        if value:
            current += 1
        elif current:
            result.append(current)
            current = 0
    if current:
        result.append(current)
    return result


def _rank_correlation(weights: Sequence[float], quotes: Sequence[float]) -> float:
    left = np.asarray(weights, dtype=float)
    right = np.asarray(quotes, dtype=float)
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return 0.0
    # Better allocation means larger weight for a smaller quote, hence negate quote.
    return float(np.corrcoef(left, -right)[0, 1])


def path_metrics(
    protocol_version: str,
    policy_seed: int,
    path_seed: int,
    method: str,
    model_digest: str,
    base_model_digest: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("ABCE path has no step records")
    low = [row for row in records if row["info"].get("last_step_risk") == "low"]
    high = [row for row in records if row["info"].get("last_step_risk") == "high"]
    if not low:
        raise ValueError("ABCE final path has no low-risk observations")
    low_spread, low_cost = _weighted_execution(low)
    high_spread, high_cost = _weighted_execution(high)
    all_spread, all_cost = _weighted_execution(records)
    signal_active = [float(row["platform_applied_signal"]) > 0.0 for row in records]
    route_active = [
        float(row["info"]["routing_total_variation"]) > 1e-12 for row in records
    ]
    high_signal = [
        float(row["platform_applied_signal"]) > 0.0 for row in high
    ]
    high_route = [
        float(row["info"]["routing_total_variation"]) > 1e-12 for row in high
    ]
    dealer_rewards = np.sum(
        np.asarray([row["rewards_unscaled"] for row in records], dtype=float), axis=0
    )
    inventories = np.asarray(
        [row["info"]["inventory_by_dealer"] for row in records], dtype=float
    )
    n_agents = inventories.shape[1]
    uniform_hhi = 1.0 / n_agents
    normalized_hhi = [
        (float(row["info"]["realized_flow_hhi"]) - uniform_hhi)
        / (1.0 - uniform_hhi)
        for row in records
    ]
    correlations = [
        _rank_correlation(
            row["info"]["routing_weights"],
            np.asarray(row["actions"], dtype=float) + 1.0,
        )
        for row in records
        if float(row["info"]["routing_total_variation"]) > 1e-12
    ]
    weighted_routed_quotes = []
    for row in records:
        weights = np.asarray(row["info"]["routing_weights"], dtype=float)
        quotes = np.asarray(row["actions"], dtype=float) + 1.0
        weighted_routed_quotes.append(float(np.dot(weights, quotes)))
    path_digests = {str(row["info"]["path_digest"]) for row in records}
    multiset_hashes = {
        str(row.get("signal_audit", {}).get("signal_multiset_hash", ""))
        for row in records
    }
    return {
        "protocol_version": protocol_version,
        "policy_seed": int(policy_seed),
        "path_seed": int(path_seed),
        "method": method,
        "low_execution_half_spread": low_spread,
        "high_execution_half_spread": high_spread,
        "all_execution_half_spread": all_spread,
        "low_customer_tick_units": low_cost,
        "high_customer_tick_units": high_cost,
        "all_customer_tick_units": all_cost,
        "announced_signal_activation_rate": float(mean(signal_active)),
        "realized_routing_activation_rate": float(mean(route_active)),
        "high_risk_announced_signal_activation_rate": (
            float(mean(high_signal)) if high_signal else 0.0
        ),
        "high_risk_realized_routing_activation_rate": (
            float(mean(high_route)) if high_route else 0.0
        ),
        "mean_routing_total_variation": float(
            mean(float(row["info"]["routing_total_variation"]) for row in records)
        ),
        "high_risk_mean_routing_total_variation": (
            float(
                mean(float(row["info"]["routing_total_variation"]) for row in high)
            )
            if high
            else 0.0
        ),
        "targeted_customer_quantity": float(
            sum(float(row["info"]["realized_targeted_quantity"]) for row in records)
        ),
        "mean_fill_ratio": float(mean(float(row["info"]["fill_ratio"]) for row in records)),
        "mean_fallback_rate": float(
            mean(float(row["info"]["fallback_rate"]) for row in records)
        ),
        "volume_conserved": bool(
            all(bool(row["info"]["planned_to_realized_volume_exact"]) for row in records)
        ),
        "aggregate_dealer_reward": float(np.sum(dealer_rewards)),
        "mean_per_dealer_reward": float(np.mean(dealer_rewards)),
        "dealer_reward_dispersion": float(np.std(dealer_rewards)),
        "mean_inventory_std": float(mean(np.std(inventories, axis=1))),
        "maximum_absolute_inventory": float(np.max(np.abs(inventories))),
        "mean_normalized_flow_hhi": float(mean(normalized_hhi)),
        "mean_maximum_dealer_share": float(
            mean(float(row["info"]["realized_max_dealer_share"]) for row in records)
        ),
        "mean_quote_dispersion": float(
            mean(float(row["info"]["quote_dispersion"]) for row in records)
        ),
        "best_quote_tie_rate": float(
            mean(int(row["info"]["best_quote_tie_count"]) > 1 for row in records)
        ),
        "mean_weight_quote_rank_correlation": (
            float(mean(correlations)) if correlations else 0.0
        ),
        "mean_weighted_routed_quote": float(mean(weighted_routed_quotes)),
        "signal_run_lengths": _run_lengths(signal_active),
        "low_risk_steps": len(low),
        "high_risk_steps": len(high),
        "model_digest": model_digest,
        "base_model_digest": base_model_digest,
        "path_digest": next(iter(path_digests)) if len(path_digests) == 1 else "",
        "signal_multiset_hash": (
            next(iter(multiset_hashes)) if len(multiset_hashes) == 1 else ""
        ),
        "finite": bool(
            all(
                math.isfinite(float(value))
                for row in records
                for value in row["rewards_unscaled"]
            )
        ),
    }


def audit_step_record(
    protocol_version: str,
    protocol_fingerprint: str,
    source_commit: str,
    docker_image_id: str,
    policy_seed: int,
    path_seed: int,
    method: str,
    model_digest: str,
    base_model_digest: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    info = row["info"]
    signal = dict(row.get("signal_audit", {}))
    weights = np.asarray(info["routing_weights"], dtype=float)
    pre_weights = np.asarray(
        info["pre_permutation_routing_weights"], dtype=float
    )
    applied = float(row["platform_applied_signal"])
    observed = float(row["dealer_observed_signal"])
    return {
        "provenance": {
            "protocol_version": protocol_version,
            "protocol_fingerprint": protocol_fingerprint,
            "source_commit": source_commit,
            "docker_image_id": docker_image_id,
            "method": method,
            "policy_seed": int(policy_seed),
            "path_seed": int(path_seed),
            "step": int(row["step"]),
            "model_digest": model_digest,
            "base_model_digest": base_model_digest,
            "path_digest": str(info["path_digest"]),
            "action_uniform_id": str(row["action_uniform_digest"]),
            "route_uniform_id": str(info["route_uniform_digest"]),
            "target_uniform_id": str(info["target_uniform_digest"]),
        },
        "estimator": {
            "lagged_public_features": signal.get("public_features", []),
            "normalization_artifact": signal.get("normalization_artifact"),
            "proxy_score": signal.get("risk_probability"),
            "threshold": signal.get("probability_threshold"),
            "latent_risk": {
                "value": str(info.get("last_step_risk")),
                "usage": "audit_only",
            },
        },
        "scheduler": {
            "raw_proposal": signal.get("raw_proposal"),
            "token_balance_before": signal.get("token_balance_before"),
            "token_balance_after": signal.get("token_balance_after"),
            "platform_applied_signal": applied,
            "dealer_observed_signal": observed,
            "schedule_source": signal.get("schedule_source"),
            "scheduled_value_source_step": signal.get("scheduled_value_source_step"),
            "signal_permutation_id": signal.get("signal_permutation_id"),
            "circular_shift_id": signal.get("circular_shift_id"),
            "signal_multiset_hash": signal.get("signal_multiset_hash"),
            "announcement_timestamp_ns": int(info["decision_timestamp_ns"]),
            "announcement_phase": "before_quote",
        },
        "dealer": {
            "observation_hash": str(row["actor_observation_digest"]),
            "critic_observation_hash": str(row["critic_observation_digest"]),
            "action_uniform_id": str(row["action_uniform_digest"]),
            "actions": list(row["actions"]),
            "half_spreads": [int(value) + 1 for value in row["actions"]],
            "base_policy_digest": base_model_digest,
            "response_policy_digest": model_digest,
        },
        "allocation": {
            "mechanism": str(info["routing_mechanism"]),
            "base_weights": list(info["base_routing_weights"]),
            "quote_ranks": list(info["quote_ranks"]),
            "target_scores": list(info["target_scores"]),
            "pre_permutation_weights": pre_weights.tolist(),
            "final_weights": weights.tolist(),
            "identity_permutation": list(info["identity_permutation"]),
            "routing_total_variation": float(info["routing_total_variation"]),
            "selected_targets": list(info["eligible_agent_indices"]),
            "targeted_event_count": int(info["realized_targeted_event_count"]),
            "targeted_quantity": float(info["realized_targeted_quantity"]),
            "route_uniform_id": str(info["route_uniform_digest"]),
        },
        "outcomes": {
            "executions": list(info["execution_records"]),
            "customer_execution_cost_tick_units": float(
                info["customer_execution_cost_tick_units"]
            ),
            "reward_components": {
                "spread_revenue": list(info["spread_revenue_by_dealer"]),
                "adverse_selection": list(info["adverse_selection_by_dealer"]),
                "inventory_penalty": list(info["inventory_penalty_by_dealer"]),
            },
            "total_rewards": list(row["rewards_unscaled"]),
            "inventories": list(info["inventory_by_dealer"]),
            "flow_shares": list(info["flow_shares"]),
            "flow_hhi": float(info["realized_flow_hhi"]),
            "maximum_share": float(info["realized_max_dealer_share"]),
            "next_public_state": list(row["post_observations"]),
            "next_public_state_hash": str(row["post_state_digest"]),
        },
        "integrity": {
            "no_lookahead": signal.get("computed_before_action") is True,
            "weights_sum_to_one": bool(abs(float(np.sum(weights)) - 1.0) <= 1e-12),
            "fixed_volume": bool(info["planned_to_realized_volume_exact"]),
            "identity_weight_multiset_exact": bool(
                np.allclose(
                    np.sort(weights),
                    np.sort(pre_weights),
                    atol=1e-12,
                    rtol=0.0,
                )
            ),
            "missing": False,
            "finite": bool(
                np.all(np.isfinite(weights))
                and all(math.isfinite(float(value)) for value in row["rewards_unscaled"])
            ),
        },
    }


def aggregate_per_seed(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row["policy_seed"])][str(row["method"])].append(row)
    result = []
    for seed in sorted(grouped):
        if set(grouped[seed]) != set(methods):
            raise ValueError(f"policy seed {seed} has an incomplete ABCE method grid")
        path_sets = {
            method: {int(row["path_seed"]) for row in method_rows}
            for method, method_rows in grouped[seed].items()
        }
        if len({tuple(sorted(values)) for values in path_sets.values()}) != 1:
            raise ValueError(f"policy seed {seed} has unmatched ABCE paths")
        output: dict[str, Any] = {"policy_seed": seed}
        for method in methods:
            method_rows = grouped[seed][method]
            for metric in (
                "low_execution_half_spread",
                "high_execution_half_spread",
                "all_execution_half_spread",
                "low_customer_tick_units",
                "high_customer_tick_units",
                "all_customer_tick_units",
                "announced_signal_activation_rate",
                "realized_routing_activation_rate",
                "high_risk_announced_signal_activation_rate",
                "high_risk_realized_routing_activation_rate",
                "mean_routing_total_variation",
                "high_risk_mean_routing_total_variation",
                "targeted_customer_quantity",
                "mean_fill_ratio",
                "mean_fallback_rate",
                "aggregate_dealer_reward",
                "mean_per_dealer_reward",
                "dealer_reward_dispersion",
                "mean_inventory_std",
                "maximum_absolute_inventory",
                "mean_normalized_flow_hhi",
                "mean_maximum_dealer_share",
                "mean_quote_dispersion",
                "best_quote_tie_rate",
                "mean_weight_quote_rank_correlation",
                "mean_weighted_routed_quote",
            ):
                output[f"{method}__{metric}"] = float(
                    mean(float(row[metric]) for row in method_rows)
                )
        result.append(output)
    return result


def _two_way_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    replicates: int,
    rng_seed: int,
) -> dict[str, Any]:
    lookup = {
        (int(row["policy_seed"]), int(row["path_seed"]), str(row["method"])): row
        for row in rows
    }
    seeds = sorted({int(row["policy_seed"]) for row in rows})
    paths = {
        seed: sorted(
            {int(row["path_seed"]) for row in rows if int(row["policy_seed"]) == seed}
        )
        for seed in seeds
    }
    differences = {
        seed: np.asarray(
            [
                float(lookup[(seed, path, left)]["low_execution_half_spread"])
                - float(lookup[(seed, path, right)]["low_execution_half_spread"])
                for path in paths[seed]
            ],
            dtype=float,
        )
        for seed in seeds
    }
    rng = np.random.default_rng(int(rng_seed))
    estimates = np.empty(int(replicates), dtype=float)
    seed_array = np.asarray(seeds, dtype=int)
    for index in range(int(replicates)):
        sampled_seeds = rng.choice(seed_array, size=len(seed_array), replace=True)
        estimates[index] = mean(
            float(
                np.mean(
                    rng.choice(
                        differences[int(seed)],
                        size=len(differences[int(seed)]),
                        replace=True,
                    )
                )
            )
            for seed in sampled_seeds
        )
    return {
        "replicates": int(replicates),
        "rng_seed": int(rng_seed),
        "mean": float(np.mean(estimates)),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
    }


def build_statistics(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    contrasts: Mapping[str, tuple[str, str]],
    holm_family: Sequence[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    per_seed = aggregate_per_seed(rows, methods)
    metric_names = sorted(
        {
            key.split("__", 1)[1]
            for row in per_seed
            for key in row
            if "__" in key
        }
    )
    method_summary = {}
    for method in methods:
        method_summary[method] = {}
        for metric in metric_names:
            values = np.asarray(
                [float(row[f"{method}__{metric}"]) for row in per_seed],
                dtype=float,
            )
            method_summary[method][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_policy_seeds": int(len(values)),
            }
    results = {}
    raw_p_values = {}
    for index, (name, (left, right)) in enumerate(contrasts.items()):
        values = [
            float(row[f"{left}__low_execution_half_spread"])
            - float(row[f"{right}__low_execution_half_spread"])
            for row in per_seed
        ]
        interval = t_interval(values)
        results[name] = {
            "definition": f"{left} - {right}",
            "t_interval": interval,
            "exact_sign_test": exact_sign_test(values),
            "two_way_bootstrap": _two_way_bootstrap(
                rows,
                left,
                right,
                bootstrap_replicates,
                int(bootstrap_seed) + index,
            ),
        }
        if name in holm_family:
            raw_p_values[name] = float(interval["p_value_two_sided"])
    for name, adjusted in holm_adjust(raw_p_values).items():
        results[name]["holm_adjusted_p_value"] = adjusted
    return {
        "method_summary": method_summary,
        "contrasts": results,
        "holm_family": list(holm_family),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
    }
