from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REVISION_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REVISION_ROOT / "experiments"
if str(REVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(REVISION_ROOT))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from src.utils.abides_audit import (  # noqa: E402
    aggregate_per_seed,
    exact_sign_test,
    holm_adjust,
    two_way_paired_bootstrap,
)
from export_abides_audit_bundle import resume_verified  # noqa: E402
from verify_abides_audit_bundle import parse_checksums  # noqa: E402


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_audit_schemas_cover_required_identity_and_metrics() -> None:
    config = load_config("abides_domain_benchmark_v1.json")
    assert set(config["per_path_schema"]) >= {
        "policy_seed",
        "evaluation_seed",
        "method",
        "low_execution_half_spread",
        "mean_reward",
        "mean_routing_total_variation",
        "mean_volume",
        "model_digest",
        "path_digest",
    }
    assert set(config["step_record_schema"]) >= {
        "pre_state_digest",
        "actions",
        "mechanism_signal",
        "eligible_dealers",
        "routing_weights",
        "realized_flow_quantity",
        "latent_risk_state",
        "action_uniform_digest",
    }


def test_all_new_seed_namespaces_are_disjoint() -> None:
    domain = load_config("abides_domain_benchmark_v1.json")
    diagnostic = load_config("abides_collusion_diagnostic_v1.json")
    sensitivity = load_config("abides_dose_routing_sensitivity_v1.json")
    groups = [
        set(domain["calibration_environment_seeds"]),
        set(domain["evaluation_seeds"]),
        set(diagnostic["evaluation_seeds"]),
        set(sensitivity["passive_calibration_seeds"]),
        set(sensitivity["evaluation_seeds"]),
    ]
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            assert left.isdisjoint(right)


def test_standard_learner_and_ecology_are_fresh_holdouts() -> None:
    learner = load_config("abides_standard_learner_holdout_v1.json")
    ecology = load_config("abides_ecology_holdout_v1.json")
    assert learner["official_commit"] == ("de66d7a4b23fac2513f56f96f73b3f5cb96695ac")
    assert learner["policy_seeds"] == list(range(550000, 550010))
    assert ecology["cells"]["E1_fast_transition"]["policy_seeds"] == list(
        range(560000, 560010)
    )
    assert ecology["cells"]["E2_weaker_informed_flow"]["policy_seeds"] == list(
        range(570000, 570010)
    )


def synthetic_per_path_rows() -> list[dict]:
    rows = []
    methods = ("risk_aware", "matched_random")
    for policy_seed in (1, 2):
        for evaluation_seed in (10, 11):
            for method in methods:
                base = 2.0 + 0.1 * policy_seed + 0.01 * evaluation_seed
                spread = base if method == "risk_aware" else base + 0.25
                row = {
                    "protocol_version": "synthetic",
                    "policy_seed": policy_seed,
                    "evaluation_seed": evaluation_seed,
                    "method": method,
                    "model_digest": f"model-{policy_seed}-{method}",
                    "path_digest": f"path-{policy_seed}-{evaluation_seed}",
                    "low_execution_half_spread": spread,
                }
                for metric in (
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
                ):
                    row[metric] = 20.0 if metric == "mean_volume" else 1.0
                rows.append(row)
    return rows


def test_per_seed_is_derived_from_paired_paths() -> None:
    rows = synthetic_per_path_rows()
    result = aggregate_per_seed(
        rows,
        ("risk_aware", "matched_random"),
        ("matched_random",),
    )
    assert [row["policy_seed"] for row in result] == [1, 2]
    assert all(
        row["matched_random__minus_risk_aware_spread"] == pytest.approx(0.25)
        for row in result
    )


def test_exact_sign_and_holm_are_reproducible() -> None:
    sign = exact_sign_test([1.0, 2.0, 3.0, 4.0, -1.0, 0.0])
    assert sign == {
        "positive": 4,
        "negative": 1,
        "ties": 1,
        "non_ties": 5,
        "p_value_two_sided": pytest.approx(0.375),
    }
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == {
        "a": pytest.approx(0.03),
        "c": pytest.approx(0.06),
        "b": pytest.approx(0.06),
    }


def test_two_way_bootstrap_is_deterministic() -> None:
    rows = synthetic_per_path_rows()
    first = two_way_paired_bootstrap(rows, "matched_random", 500, 20260714)
    second = two_way_paired_bootstrap(rows, "matched_random", 500, 20260714)
    assert first == second
    assert first["ci_low"] == pytest.approx(0.25)
    assert first["ci_high"] == pytest.approx(0.25)


def test_checksum_verifier_rejects_tampering(tmp_path: Path) -> None:
    payloads = {
        "manifest.json": b"{}\n",
        "checks.json": b"{}\n",
        "per_path.csv": b"a\n1\n",
        "per_seed.csv": b"a\n1\n",
        "step_records.jsonl.gz": b"not-a-real-gzip",
        "README.md": b"readme\n",
    }
    import hashlib

    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
    (tmp_path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
            for name, payload in payloads.items()
        ),
        encoding="utf-8",
    )
    parse_checksums(tmp_path)
    (tmp_path / "per_path.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        parse_checksums(tmp_path)


def test_completed_resume_requires_aggregate_hash_and_zero_containers(
    tmp_path: Path,
) -> None:
    logbook = tmp_path / "logbook.jsonl"
    logbook.write_text(
        json.dumps(
            {
                "event": "run_resumed",
                "pending_policy_seeds": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert not resume_verified(logbook)
    with logbook.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "completed_resume_verified",
                    "aggregate_sha256": "a" * 64,
                    "containers_started": 0,
                }
            )
            + "\n"
        )
    assert resume_verified(logbook)
