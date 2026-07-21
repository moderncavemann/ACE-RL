#!/usr/bin/env python3
"""Verify the ABCE primary follow-up from raw outputs or bundle alone."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_abce_primary_followup import (  # noqa: E402
    CONTRASTS,
    HOLM_FAMILY,
    METHODS,
    canonical_json,
    result_path,
)
from run_abides_mappo_full_main import sha256_file  # noqa: E402
from src.utils.abides_abce_audit import (  # noqa: E402
    aggregate_per_seed,
    build_statistics,
)


STRING_FIELDS = {
    "protocol_version",
    "method",
    "model_digest",
    "base_model_digest",
    "path_digest",
    "signal_multiset_hash",
}
INTEGER_FIELDS = {"policy_seed", "path_seed", "low_risk_steps", "high_risk_steps"}
BOOLEAN_FIELDS = {"volume_conserved", "finite"}
JSON_FIELDS = {"signal_run_lengths"}

REQUIRED_TRACE_FIELDS = {
    "provenance": {
        "protocol_version",
        "protocol_fingerprint",
        "source_commit",
        "docker_image_id",
        "method",
        "policy_seed",
        "path_seed",
        "step",
        "model_digest",
        "base_model_digest",
        "path_digest",
        "action_uniform_id",
        "route_uniform_id",
        "target_uniform_id",
    },
    "estimator": {
        "lagged_public_features",
        "normalization_artifact",
        "proxy_score",
        "threshold",
        "latent_risk",
    },
    "scheduler": {
        "raw_proposal",
        "token_balance_before",
        "token_balance_after",
        "platform_applied_signal",
        "dealer_observed_signal",
        "schedule_source",
        "scheduled_value_source_step",
        "signal_permutation_id",
        "circular_shift_id",
        "signal_multiset_hash",
        "announcement_timestamp_ns",
        "announcement_phase",
    },
    "dealer": {
        "observation_hash",
        "critic_observation_hash",
        "action_uniform_id",
        "actions",
        "half_spreads",
        "base_policy_digest",
        "response_policy_digest",
    },
    "allocation": {
        "mechanism",
        "base_weights",
        "quote_ranks",
        "target_scores",
        "pre_permutation_weights",
        "final_weights",
        "identity_permutation",
        "routing_total_variation",
        "selected_targets",
        "targeted_event_count",
        "targeted_quantity",
        "route_uniform_id",
    },
    "outcomes": {
        "executions",
        "customer_execution_cost_tick_units",
        "reward_components",
        "total_rewards",
        "inventories",
        "flow_shares",
        "flow_hhi",
        "maximum_share",
        "next_public_state",
        "next_public_state_hash",
    },
    "integrity": {
        "no_lookahead",
        "weights_sum_to_one",
        "fixed_volume",
        "identity_weight_multiset_exact",
        "missing",
        "finite",
    },
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in STRING_FIELDS:
                    row[key] = value
                elif key in INTEGER_FIELDS:
                    row[key] = int(value)
                elif key in BOOLEAN_FIELDS:
                    row[key] = value == "True"
                elif key in JSON_FIELDS:
                    row[key] = json.loads(value)
                elif key == "policy_seed":
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def verify_sums(artifact: Path) -> None:
    sums = artifact / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError("ABCE bundle has no SHA256SUMS")
    listed = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = artifact / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"ABCE bundle hash mismatch: {relative}")
        listed.add(relative)
    actual = {
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if listed != actual:
        raise RuntimeError("ABCE SHA256SUMS file set is incomplete")


def compare_statistics(left: Any, right: Any, path: str = "statistics") -> None:
    """Compare derived statistics without requiring platform-identical float text."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise RuntimeError(f"ABCE statistics keys differ at {path}")
        for key in sorted(left):
            compare_statistics(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            raise RuntimeError(f"ABCE statistics types differ at {path}")
        if len(left) != len(right):
            raise RuntimeError(f"ABCE statistics lengths differ at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_statistics(left_item, right_item, f"{path}[{index}]")
        return
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if not math.isclose(
            float(left), float(right), rel_tol=1e-10, abs_tol=1e-12
        ):
            raise RuntimeError(f"ABCE statistics differ at {path}")
        return
    if left != right:
        raise RuntimeError(f"ABCE statistics differ at {path}")


def verify_trace(
    artifact: Path, config: Mapping[str, Any]
) -> tuple[int, dict[str, bool]]:
    count = 0
    checks = {
        "trace_no_lookahead": True,
        "trace_weights_sum_to_one": True,
        "trace_fixed_volume": True,
        "trace_identity_weight_multiset_exact": True,
        "trace_finite": True,
        "trace_schema_complete": True,
        "trace_customer_cost_recomputed": True,
    }
    seen = set()
    action_uniforms: dict[tuple[int, int, int], set[str]] = {}
    route_uniforms: dict[tuple[int, int, int], set[str]] = {}
    target_uniforms: dict[tuple[int, int, int], set[str]] = {}
    path_digests: dict[tuple[int, int], set[str]] = {}
    multiset_hashes: dict[tuple[int, int], set[str]] = {}
    n_agents = int(config["market"]["n_agents"])
    planned_quantity = float(
        int(config["market"]["flow_events_per_step"])
        * float(config["market"]["flow_quantity"])
    )
    with gzip.open(artifact / "step_records.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            checks["trace_schema_complete"] &= set(row) == set(
                REQUIRED_TRACE_FIELDS
            ) and all(
                fields <= set(row[section])
                for section, fields in REQUIRED_TRACE_FIELDS.items()
            )
            provenance = row["provenance"]
            key = (
                int(provenance["policy_seed"]),
                int(provenance["path_seed"]),
                str(provenance["method"]),
                int(provenance["step"]),
            )
            if key in seen:
                raise RuntimeError(f"duplicate ABCE trace row: {key}")
            seen.add(key)
            integrity = row["integrity"]
            checks["trace_no_lookahead"] &= bool(integrity["no_lookahead"])
            checks["trace_weights_sum_to_one"] &= bool(
                integrity["weights_sum_to_one"]
            )
            checks["trace_fixed_volume"] &= bool(integrity["fixed_volume"])
            checks["trace_identity_weight_multiset_exact"] &= bool(
                integrity["identity_weight_multiset_exact"]
            )
            checks["trace_finite"] &= bool(integrity["finite"])
            primitive_customer_cost = sum(
                float(execution["quantity"])
                * abs(
                    float(execution["price"])
                    - float(execution["reference_mid"])
                )
                for execution in row["outcomes"]["executions"]
            )
            checks["trace_customer_cost_recomputed"] &= abs(
                primitive_customer_cost
                - float(row["outcomes"]["customer_execution_cost_tick_units"])
            ) <= 1e-12 and all(
                abs(
                    float(execution["execution_half_spread"])
                    - abs(
                        float(execution["price"])
                        - float(execution["reference_mid"])
                    )
                )
                <= 1e-12
                for execution in row["outcomes"]["executions"]
            )
            checks["trace_schema_complete"] &= (
                provenance["protocol_version"] == config["protocol_version"]
                and provenance["method"] in METHODS
                and len(str(provenance["model_digest"])) == 64
                and len(str(provenance["base_model_digest"])) == 64
                and len(str(provenance["path_digest"])) == 64
                and row["estimator"]["latent_risk"]["usage"] == "audit_only"
                and row["scheduler"]["announcement_phase"] == "before_quote"
                and row["dealer"]["action_uniform_id"]
                == provenance["action_uniform_id"]
                and row["allocation"]["route_uniform_id"]
                == provenance["route_uniform_id"]
                and len(row["dealer"]["actions"]) == n_agents
                and len(row["allocation"]["final_weights"]) == n_agents
                and len(row["outcomes"]["total_rewards"]) == n_agents
                and len(row["outcomes"]["inventories"]) == n_agents
                and len(row["outcomes"]["flow_shares"]) == n_agents
                and row["integrity"]["missing"] is False
                and abs(
                    sum(
                        float(execution["quantity"])
                        for execution in row["outcomes"]["executions"]
                    )
                    - planned_quantity
                )
                <= 1e-12
                and all(
                    {
                        "eligible_dealer_indices",
                        "matched_dealer_index",
                        "price",
                        "reference_mid",
                        "execution_half_spread",
                        "quantity",
                        "route_kind",
                        "side",
                        "used_fallback",
                    }
                    <= set(execution)
                    for execution in row["outcomes"]["executions"]
                )
            )
            common_key = (key[0], key[1], key[3])
            action_uniforms.setdefault(common_key, set()).add(
                str(provenance["action_uniform_id"])
            )
            route_uniforms.setdefault(common_key, set()).add(
                str(provenance["route_uniform_id"])
            )
            target_uniforms.setdefault(common_key, set()).add(
                str(provenance["target_uniform_id"])
            )
            path_digests.setdefault((key[0], key[1]), set()).add(
                str(provenance["path_digest"])
            )
            multiset_hashes.setdefault((key[0], key[1]), set()).add(
                str(row["scheduler"]["signal_multiset_hash"])
            )
            count += 1
    expected = (
        len(config["policy_seeds"])
        * len(config["evaluation_seeds"])
        * len(METHODS)
        * int(config["episode_steps"])
    )
    expected_keys = {
        (int(policy_seed), int(path_seed), method, step)
        for policy_seed in config["policy_seeds"]
        for path_seed in config["evaluation_seeds"]
        for method in METHODS
        for step in range(int(config["episode_steps"]))
    }
    checks["trace_row_count"] = count == expected
    checks["trace_exact_grid"] = seen == expected_keys
    checks["trace_common_action_uniforms"] = all(
        len(values) == 1 for values in action_uniforms.values()
    )
    checks["trace_common_route_uniforms"] = all(
        len(values) == 1 for values in route_uniforms.values()
    )
    checks["trace_common_target_uniforms"] = all(
        len(values) == 1 for values in target_uniforms.values()
    )
    checks["trace_common_paths"] = all(
        len(values) == 1 for values in path_digests.values()
    )
    checks["trace_exact_signal_multisets"] = all(
        len(values) == 1 for values in multiset_hashes.values()
    )
    return count, checks


def verify_bundle(artifact: Path) -> dict[str, Any]:
    verify_sums(artifact)
    config = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    if tuple(config["methods"]) != METHODS:
        raise RuntimeError("ABCE bundle method grid changed")
    per_path = read_csv(artifact / "per_path.csv")
    expected_paths = (
        len(config["policy_seeds"]) * len(config["evaluation_seeds"]) * len(METHODS)
    )
    if len(per_path) != expected_paths:
        raise RuntimeError("ABCE per-path grid is incomplete")
    recomputed_seed = aggregate_per_seed(per_path, METHODS)
    stored_seed = read_csv(artifact / "per_seed.csv")
    if canonical_json(recomputed_seed) != canonical_json(stored_seed):
        raise RuntimeError("ABCE per-seed rows do not recompute")
    recomputed_stats = build_statistics(
        per_path,
        METHODS,
        CONTRASTS,
        HOLM_FAMILY,
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]),
    )
    stored_stats = json.loads(
        (artifact / "statistics.json").read_text(encoding="utf-8")
    )
    compare_statistics(recomputed_stats, stored_stats)
    trace_count, trace_checks = verify_trace(artifact, config)
    stored_checks = json.loads((artifact / "checks.json").read_text(encoding="utf-8"))
    for key, value in trace_checks.items():
        if not value or stored_checks.get(key) is not True:
            raise RuntimeError(f"ABCE trace integrity failed: {key}")
    for required in (
        "manifest.json",
        "PROTOCOL.md",
        "EXECUTION_AMENDMENT.md",
        "model_digests.json",
        "efficiency.json",
        "path_digests.json",
        "figures/representative_trace.pdf",
        "figures/representative_trace.png",
        "tables/abce_contrasts.tex",
        "tables/abce_method_summary.tex",
    ):
        if not (artifact / required).is_file():
            raise FileNotFoundError(f"ABCE bundle is missing {required}")
    efficiency = json.loads(
        (artifact / "efficiency.json").read_text(encoding="utf-8")
    )
    elapsed = {
        int(seed): float(value)
        for seed, value in efficiency["policy_seed_elapsed_seconds"].items()
    }
    if set(elapsed) != {int(seed) for seed in config["policy_seeds"]} or not all(
        math.isfinite(value) and value >= 0.0 for value in elapsed.values()
    ):
        raise RuntimeError("ABCE efficiency policy-seed grid is invalid")
    values = list(elapsed.values())
    expected_sd = stdev(values) if len(values) > 1 else 0.0
    if not (
        math.isclose(
            float(efficiency["mean_policy_seed_elapsed_seconds"]),
            mean(values),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(efficiency["sample_sd_policy_seed_elapsed_seconds"]),
            expected_sd,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(efficiency["total_policy_seed_compute_seconds"]),
            sum(values),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("ABCE efficiency aggregates do not recompute")
    return {
        "status": "pass",
        "mode": "bundle_only",
        "per_path_rows": len(per_path),
        "trace_rows": trace_count,
        "checks": trace_checks,
    }


def verify_strict_raw(artifact: Path) -> dict[str, Any]:
    result = verify_bundle(artifact)
    config = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    runtime = REPO / str(config["output_dir"])
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(runtime / "manifest.json") != str(
        manifest["runtime_manifest_sha256"]
    ):
        raise RuntimeError("ABCE runtime manifest differs from exported bundle")
    if sha256_file(runtime / "summary.json") != str(
        manifest["runtime_summary_sha256"]
    ):
        raise RuntimeError("ABCE runtime summary differs from exported bundle")
    efficiency = json.loads(
        (artifact / "efficiency.json").read_text(encoding="utf-8")
    )
    for seed in config["policy_seeds"]:
        expected = manifest["runtime_result_sha256"][str(seed)]
        raw_path = result_path(runtime, int(seed))
        if sha256_file(raw_path) != expected:
            raise RuntimeError(f"ABCE raw result changed for policy seed {seed}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if not math.isclose(
            float(raw["elapsed_seconds"]),
            float(efficiency["policy_seed_elapsed_seconds"][str(seed)]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"ABCE efficiency differs from raw seed {seed}")
    result["mode"] = "strict_raw"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        default="revision/paper/artifacts/abides_abce_primary_followup_v1",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--strict-raw", action="store_true")
    mode.add_argument("--bundle-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = (REPO / args.artifact).resolve()
    result = verify_strict_raw(artifact) if args.strict_raw else verify_bundle(artifact)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
