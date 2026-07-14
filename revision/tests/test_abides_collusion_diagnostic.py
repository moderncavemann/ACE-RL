from __future__ import annotations

import json
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_deviation_event_study_is_frozen_and_paired() -> None:
    config = load_config("abides_collusion_diagnostic_v1.json")
    assert config["event_horizon"] == 10
    assert config["maximum_events_per_episode"] == 1
    assert config["event_risk_state"] == "low"
    assert config["minimum_mean_half_spread"] == 6.0
    assert config["arms"] == ["sham", "forced_undercut", "upward_deviation"]
    assert config["event_selection"] == "first_eligible_state"


def test_collusion_paths_are_disjoint_from_domain_paths() -> None:
    diagnostic = load_config("abides_collusion_diagnostic_v1.json")
    domain = load_config("abides_domain_benchmark_v1.json")
    diagnostic_paths = set(diagnostic["evaluation_seeds"])
    assert diagnostic_paths.isdisjoint(domain["evaluation_seeds"])
    assert diagnostic_paths.isdisjoint(domain["calibration_environment_seeds"])
    assert len(diagnostic["strong_collusion_requirements"]) == 5
    assert diagnostic["fallback_wording"] == "high-spread coordination"


def test_dose_and_routing_grids_are_fixed() -> None:
    config = load_config("abides_dose_routing_sensitivity_v1.json")
    assert config["dose_targets"] == [0.1, 0.2, 0.3]
    settings = {
        (cell["routing_target_kappa"], cell["routing_routable_fraction"])
        for cell in config["routing_settings"]
    }
    assert settings == {(2.0, 0.5), (4.0, 0.5), (8.0, 0.5), (4.0, 0.25), (4.0, 0.75)}
    assert len(settings) == 5
