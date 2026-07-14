from __future__ import annotations

import json
import sys
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from experiments.run_abides_domain_benchmark import (  # noqa: E402
    NEW_METHODS,
    action_uniforms_equal,
    configured_contrasts,
    configured_methods,
    configured_new_methods,
    load_mechanism_parameters,
    spread_budget_decision,
    temporary_commitment_integrity,
)


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_formal_domain_cell_is_frozen_from_passive_calibration() -> None:
    config = load_config("abides_domain_benchmark_v1.json")
    assert config["freeze_status"] == "frozen_after_passive_calibration"
    assert config["formal_execution_allowed"] is True
    assert config["calibration_artifact"].endswith("_artifact.json")
    assert len(config["calibration_artifact_sha256"]) == 64
    assert (
        config["mechanism_parameters"]["spread_only_budget_matched"][
            "budget_match_status"
        ]
        == "matched"
    )
    assert (
        config["mechanism_parameters"]["brero_threshold_buybox_budget_matched"][
            "budget_match_status"
        ]
        == "unmatched"
    )
    assert config["external_same_budget_superiority_claim_allowed"] is False


def test_domain_benchmark_uses_frozen_sources_and_fresh_paths() -> None:
    config = load_config("abides_domain_benchmark_v1.json")
    calibration = set(config["calibration_environment_seeds"])
    evaluation = set(config["evaluation_seeds"])
    assert config["policy_seeds"] == list(range(530000, 530020))
    assert evaluation == set(range(534000, 534010))
    assert calibration == set(range(535000, 535010))
    assert calibration.isdisjoint(evaluation)
    assert config["source_base_fingerprint"].startswith("8a409007")
    assert config["source_v5_fingerprint"].startswith("8636827c")


def test_domain_method_set_and_inference_are_preregistered() -> None:
    config = load_config("abides_domain_benchmark_v1.json")
    assert set(config["methods"]) >= {
        "no_intervention",
        "risk_aware",
        "matched_random",
        "spread_only_budget_matched",
        "brero_threshold_buybox",
        "brero_threshold_buybox_budget_matched",
        "spread_only_threshold",
        "always_on",
        "no_routing",
    }
    assert config["primary_contrast"] == (
        "brero_threshold_buybox_budget_matched_minus_risk_aware"
    )
    assert config["required_policy_seed_count"] == 20
    assert config["minimum_positive_policy_seeds"] == 14
    assert config["bootstrap_replicates"] == 10000
    assert config["bootstrap_seed"] == 20260714


def test_smoke_is_separate_and_cannot_be_paper_evidence() -> None:
    smoke = load_config("abides_domain_benchmark_smoke_v1.json")
    formal = load_config("abides_domain_benchmark_v1.json")
    assert smoke["engineering_smoke"] is True
    assert smoke["output_dir"] != formal["output_dir"]
    assert set(smoke["evaluation_seeds"]).isdisjoint(formal["evaluation_seeds"])
    assert smoke["response_episodes"] < formal["response_episodes"]


def test_smoke_parameters_cover_each_new_adapter() -> None:
    smoke = load_config("abides_domain_benchmark_smoke_v1.json")
    assert tuple(smoke["new_adapter_methods"]) == NEW_METHODS
    assert set(load_mechanism_parameters(smoke)) == set(NEW_METHODS)
    assert smoke["engineering_smoke"] is True


def test_action_uniform_comparator_requires_exact_step_digests() -> None:
    matching = {
        "left": {
            "records": [{"action_uniform_digest": "a"}, {"action_uniform_digest": "b"}]
        },
        "right": {
            "records": [{"action_uniform_digest": "a"}, {"action_uniform_digest": "b"}]
        },
    }
    assert action_uniforms_equal(matching)
    matching["right"]["records"][1]["action_uniform_digest"] = "changed"
    assert not action_uniforms_equal(matching)


def test_spread_only_budget_rule_cannot_read_latent_risk() -> None:
    observations = [[0.0] * 6 for _ in range(10)]
    for observation in observations:
        observation[1] = 4.0 / 6.0
    parameters = {
        "activation_target": 0.2,
        "severity_threshold": 0.0,
        "signal_scale": 0.75,
    }
    low = spread_budget_decision(parameters, 100)
    high = spread_budget_decision(parameters, 100)
    assert low({"observations": observations, "latent_risk_state": "low"}, 0) == high(
        {"observations": observations, "latent_risk_state": "high"}, 0
    )


def test_temporary_commitment_formal_cell_is_frozen_without_same_budget_claim() -> None:
    config = load_config("abides_temporary_commitment_benchmark_v1.json")
    assert config["freeze_status"] == "frozen_after_passive_calibration"
    assert config["formal_execution_allowed"] is True
    assert configured_new_methods(config) == (
        "banerjee_temporary_commitment",
        "banerjee_temporary_commitment_budget_matched",
    )
    assert set(configured_new_methods(config)).issubset(configured_methods(config))
    assert configured_contrasts(config) == (
        "matched_random",
        "banerjee_temporary_commitment",
        "banerjee_temporary_commitment_budget_matched",
    )
    assert config["primary_external_comparator"] == (
        "banerjee_temporary_commitment_budget_matched"
    )
    selected = config["mechanism_parameters"][
        "banerjee_temporary_commitment_budget_matched"
    ]
    assert selected["budget_match_status"] == "unmatched"
    assert selected["reference_window"] == 10
    assert selected["qualification_steps"] == 3
    assert selected["protected_fraction"] == 0.25
    assert selected["duration"] == 1


def test_temporary_commitment_smoke_is_separate_from_formal_paths() -> None:
    smoke = load_config("abides_temporary_commitment_benchmark_smoke_v1.json")
    formal = load_config("abides_temporary_commitment_benchmark_v1.json")
    assert smoke["engineering_smoke"] is True
    assert smoke["output_dir"] != formal["output_dir"]
    assert set(smoke["evaluation_seeds"]).isdisjoint(formal["evaluation_seeds"])
    assert smoke["response_episodes"] < formal["response_episodes"]
    assert set(load_mechanism_parameters(smoke)) == set(
        configured_new_methods(smoke)
    )


def test_temporary_commitment_integrity_requires_exact_protected_execution() -> None:
    parameters = {
        "reference_window": 10,
        "qualification_steps": 3,
        "protected_fraction": 0.5,
        "duration": 2,
    }

    def record(takeup: bool, remaining: int, active: bool) -> dict:
        return {
            "info": {
                "active_routing": active,
                "routing_mechanism_parameters": {
                    "takeup": takeup,
                    "remaining_periods_including_current": remaining,
                    "protected_order_tag_prefix": "ace_protected_c10",
                    "frozen_reference_price": 100_000,
                    "frozen_half_spread": 2,
                },
                "targeted_event_count": 10 if active else 0,
                "protected_execution_quantity": 10.0 if active else 0.0,
                "protected_execution_prices": (
                    [99_998, 100_002] * 5 if active else []
                ),
                "protected_matched_order_tags": (
                    [f"ace_protected_c10_{index}" for index in range(10)]
                    if active
                    else []
                ),
                "protected_routing_fallback_count": 0,
            }
        }

    records = [
        record(False, 0, False),
        record(True, 2, True),
        record(False, 1, True),
        record(False, 0, False),
    ]
    valid, takeups = temporary_commitment_integrity(records, parameters)
    assert valid
    assert takeups == 1

    records[2]["info"]["protected_execution_prices"][0] = 100_000
    valid, takeups = temporary_commitment_integrity(records, parameters)
    assert not valid
    assert takeups == 1
