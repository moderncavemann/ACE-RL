from __future__ import annotations

import json
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]


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
    assert learner["official_commit"] == (
        "de66d7a4b23fac2513f56f96f73b3f5cb96695ac"
    )
    assert learner["policy_seeds"] == list(range(550000, 550010))
    assert ecology["cells"]["E1_fast_transition"]["policy_seeds"] == list(
        range(560000, 560010)
    )
    assert ecology["cells"]["E2_weaker_informed_flow"]["policy_seeds"] == list(
        range(570000, 570010)
    )
