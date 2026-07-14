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
    load_mechanism_parameters,
)


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_formal_domain_cell_is_blocked_until_calibration_freeze() -> None:
    config = load_config("abides_domain_benchmark_v1.json")
    assert config["freeze_status"] == "calibration_pending"
    assert config["formal_execution_allowed"] is False
    assert config["calibration_artifact"] is None
    assert config["calibration_artifact_sha256"] is None


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
