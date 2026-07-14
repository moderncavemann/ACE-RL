#!/usr/bin/env python3
"""Independently verify the committed ABIDES deviation diagnostic bundle."""

from __future__ import annotations

import csv
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
BUNDLE = ROOT / "paper" / "artifacts" / "abides_collusion_diagnostic_v1"
CONFIG = ROOT / "configs" / "abides_collusion_diagnostic_v1.json"
PAPER_PER_EVENT = ROOT / "paper" / "data" / "abides_collusion_diagnostic_v1_per_event.csv"
PAPER_PER_SEED = ROOT / "paper" / "data" / "abides_collusion_diagnostic_v1_per_seed.csv"
METRICS = (
    "rival_spread_compression_after_undercut",
    "deviator_continuation_flow_loss_after_undercut",
    "deviator_continuation_reward_loss_after_undercut",
    "high_spread_recovery_rate_after_undercut",
    "flow_loss_asymmetry",
    "reward_loss_asymmetry",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def as_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid serialized boolean: {value}")
    return value == "True"


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
        "ci_low": low,
        "ci_high": high,
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


def feasible(row: Mapping[str, str], metric: str) -> bool:
    if metric in {"flow_loss_asymmetry", "reward_loss_asymmetry"}:
        return as_bool(row["forced_undercut_feasible"]) and as_bool(
            row["upward_deviation_feasible"]
        )
    return as_bool(row["forced_undercut_feasible"])


def assert_close(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-12):
            raise RuntimeError(f"{name} mismatch: {actual} != {expected}")
    elif actual != expected:
        raise RuntimeError(f"{name} mismatch: {actual} != {expected}")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    aggregate = json.loads((BUNDLE / "aggregate.json").read_text(encoding="utf-8"))
    completion = json.loads((BUNDLE / "completion.json").read_text(encoding="utf-8"))
    events = read_csv(BUNDLE / "per_event.csv")
    seeds = read_csv(BUNDLE / "per_seed.csv")
    trajectories = read_csv(BUNDLE / "trajectory.csv")
    checks: dict[str, bool] = {}

    checksum_valid = True
    for line in (BUNDLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        checksum_valid &= sha256_file(BUNDLE / name) == expected
    checks["bundle_hashes_valid"] = checksum_valid
    checks["paper_exports_audit_derived"] = (
        (BUNDLE / "per_event.csv").read_bytes() == PAPER_PER_EVENT.read_bytes()
        and (BUNDLE / "per_seed.csv").read_bytes() == PAPER_PER_SEED.read_bytes()
    )
    expected_fingerprint = protocol_fingerprint(config)
    checks["protocol_fingerprint_valid"] = (
        manifest["protocol_fingerprint"]
        == aggregate["protocol_fingerprint"]
        == completion["protocol_fingerprint"]
        == expected_fingerprint
    )
    checks["config_hash_valid"] = manifest["config_sha256"] == sha256_file(CONFIG)
    checks["source_provenance_complete"] = (
        set(manifest["source_policy_artifacts"])
        == {str(seed) for seed in config["source_policy_seeds"]}
        and all(
            len(values["source_checkpoint_sha256"]) == 64
            and len(values["model_digest"]) == 64
            for values in manifest["source_policy_artifacts"].values()
        )
    )
    archival_commit = manifest.get("result_archival_commit")
    checks["result_archival_commit_valid"] = archival_commit is not None
    if archival_commit is not None:
        checks["result_archival_commit_valid"] = subprocess.run(
            ["git", "cat-file", "-e", f"{archival_commit}^{{commit}}"],
            cwd=REPO,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    checks["completed_resume_verified"] = bool(
        completion["completed_resume_verification"]
        and completion["aggregate_unchanged_on_completed_resume"]
        and int(completion["containers_started"]) == 0
        and manifest["completed_resume_verified"]
    )

    policy_grid = {int(seed) for seed in config["source_policy_seeds"]}
    evaluation_grid = {int(seed) for seed in config["evaluation_seeds"]}
    event_keys = {(int(row["policy_seed"]), int(row["evaluation_seed"])) for row in events}
    checks["event_grid_complete"] = event_keys == {
        (policy_seed, evaluation_seed)
        for policy_seed in policy_grid
        for evaluation_seed in evaluation_grid
    }
    checks["all_events_found"] = all(as_bool(row["event_found"]) for row in events)
    checks["runner_pair_integrity_retained"] = all(
        as_bool(row["paired_path_exact"]) and as_bool(row["arm_integrity_pass"])
        for row in events
    )

    event_lookup = {
        (int(row["policy_seed"]), int(row["evaluation_seed"])): row for row in events
    }
    trajectory_groups: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in trajectories:
        trajectory_groups[
            (int(row["policy_seed"]), int(row["evaluation_seed"]), row["arm"])
        ].append(row)
    horizon = int(config["event_horizon"])
    checks["trajectory_grid_complete"] = len(trajectory_groups) == len(events) * 3
    checks["trajectory_horizon_exact"] = all(
        sorted(int(row["relative_step"]) for row in rows)
        == list(range(horizon + 1))
        for rows in trajectory_groups.values()
    )
    step_integrity = True
    arm_pairing = True
    for event_key, event in event_lookup.items():
        by_arm = {
            arm: sorted(
                trajectory_groups[(*event_key, arm)],
                key=lambda row: int(row["relative_step"]),
            )
            for arm in ("sham", "forced_undercut", "upward_deviation")
        }
        dealer = int(event["dealer_index"])
        for relative_step in range(horizon + 1):
            rows = [by_arm[arm][relative_step] for arm in by_arm]
            arm_pairing &= len({row["path_digest"] for row in rows}) == 1
            arm_pairing &= len({row["action_uniform_digest"] for row in rows}) == 1
            arm_pairing &= len({row["risk_state"] for row in rows}) == 1
            if relative_step == 0:
                arm_pairing &= len({row["pre_state_digest"] for row in rows}) == 1
                originals = [json.loads(row["original_actions"]) for row in rows]
                arm_pairing &= originals[0] == originals[1] == originals[2]
                sham_actual = json.loads(rows[0]["actual_actions"])
                forced_actual = json.loads(rows[1]["actual_actions"])
                upward_actual = json.loads(rows[2]["actual_actions"])
                arm_pairing &= sham_actual == originals[0]
                expected_forced = originals[0].copy()
                expected_forced[dealer] = max(expected_forced[dealer] - 1, 0)
                expected_upward = originals[0].copy()
                expected_upward[dealer] = min(expected_upward[dealer] + 1, 6)
                arm_pairing &= forced_actual == expected_forced
                arm_pairing &= upward_actual == expected_upward
            for row in rows:
                step_integrity &= not as_bool(row["gate_active"])
                step_integrity &= abs(float(row["routing_total_variation"])) <= 1e-12
                step_integrity &= abs(float(row["realized_flow_quantity"]) - 20.0) <= 1e-12
                step_integrity &= all(
                    math.isfinite(float(row[name]))
                    for name in (
                        "rivals_mean_half_spread",
                        "deviator_half_spread",
                        "joint_mean_half_spread",
                        "deviator_flow_share",
                        "deviator_reward",
                        "aggregate_dealer_reward",
                        "customer_execution_half_spread",
                    )
                )
    checks["arm_paths_and_uniforms_paired"] = arm_pairing
    checks["step_routing_gate_volume_valid"] = step_integrity

    recomputed_seed_rows: dict[int, dict[str, float]] = {}
    for policy_seed in sorted(policy_grid):
        policy_events = [row for row in events if int(row["policy_seed"]) == policy_seed]
        recomputed_seed_rows[policy_seed] = {}
        for metric in METRICS:
            values = [
                float(row[metric]) for row in policy_events if feasible(row, metric)
            ]
            if not values:
                raise RuntimeError(f"seed {policy_seed} has no values for {metric}")
            recomputed_seed_rows[policy_seed][metric] = float(mean(values))
    seed_csv = {int(row["policy_seed"]): row for row in seeds}
    seed_match = set(seed_csv) == policy_grid
    for policy_seed, metrics in recomputed_seed_rows.items():
        for metric, value in metrics.items():
            seed_match &= math.isclose(
                float(seed_csv[policy_seed][metric]), value, abs_tol=1e-12, rel_tol=1e-12
            )
    checks["per_seed_recomputed"] = seed_match

    statistics_match = True
    derived_statistics = {}
    for metric in METRICS:
        values = [recomputed_seed_rows[seed][metric] for seed in sorted(policy_grid)]
        derived_interval = interval(values)
        derived_sign = sign_test(values)
        derived_statistics[metric] = derived_interval
        reported = aggregate["statistics"][metric]
        for name, value in derived_interval.items():
            try:
                assert_close(reported["policy_seed_t_interval"][name], value, f"{metric}.{name}")
            except RuntimeError:
                statistics_match = False
        for name, value in derived_sign.items():
            try:
                assert_close(reported["policy_seed_exact_sign_test"][name], value, f"{metric}.sign.{name}")
            except RuntimeError:
                statistics_match = False
    checks["statistics_recomputed"] = statistics_match

    recovery_bound = float(
        config["estimand_rules"]["minimum_recovery_rate_ci_lower_bound"]
    )
    requirements = {
        "rival_spread_compression_after_undercut": derived_statistics[
            "rival_spread_compression_after_undercut"
        ]["ci_low"]
        > 0.0,
        "deviator_continuation_loss": (
            derived_statistics[
                "deviator_continuation_flow_loss_after_undercut"
            ]["ci_low"]
            > 0.0
            and derived_statistics[
                "deviator_continuation_reward_loss_after_undercut"
            ]["ci_low"]
            > 0.0
        ),
        "high_spread_state_recovery": derived_statistics[
            "high_spread_recovery_rate_after_undercut"
        ]["ci_low"]
        > recovery_bound,
        "upward_deviation_asymmetry": (
            derived_statistics["flow_loss_asymmetry"]["ci_low"] > 0.0
            and derived_statistics["reward_loss_asymmetry"]["ci_low"] > 0.0
        ),
        "policy_seed_ci_support": True,
    }
    derived_science = (
        "punishment_supported_collusion_pattern_supported"
        if all(requirements.values())
        else "high_spread_coordination_only"
    )
    checks["claim_boundary_recomputed"] = (
        requirements == aggregate["strong_collusion_requirements"]
        and derived_science == aggregate["scientific_result"]
        and aggregate["required_wording"]
        == (
            "punishment-supported collusion"
            if all(requirements.values())
            else config["fallback_wording"]
        )
    )
    checks["bundle_counts_exact"] = (
        len(events) == int(aggregate["eligible_event_count"]) == 400
        and len(trajectories) == int(aggregate["trajectory_row_count"]) == 13200
        and len(seeds) == 20
    )
    status = "pass" if all(checks.values()) else "fail"
    result = {
        "verification_status": status,
        "protocol_fingerprint": expected_fingerprint,
        "scientific_result": derived_science,
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
