from __future__ import annotations

import json
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_brero_candidate_family_is_frozen_and_complete() -> None:
    config = load_config("abides_domain_baseline_calibration_v1.json")
    brero = config["brero_threshold_buybox"]
    assert brero["order_statistic_k_candidates"] == list(range(1, 11))
    assert brero["margin_tick_candidates"] == [0, 1]
    assert brero["literature_form"] == {
        "order_statistic_k": 1,
        "margin_ticks": 0,
    }
    assert brero["empty_rule"] == "include_all_narrowest_ties"


def test_calibration_forbids_efficacy_fields() -> None:
    config = load_config("abides_domain_baseline_calibration_v1.json")
    forbidden = set(config["forbidden_calibration_fields"])
    assert {
        "low_execution_half_spread",
        "mean_reward",
        "dealer_reward_units_per_episode",
        "customer_cost",
    }.issubset(forbidden)
    assert config["selection_rule"].startswith("minimum_normalized_budget_distance")


def test_literature_mapping_precedes_implementation() -> None:
    mapping = REVISION_ROOT / "notes" / "literature_baseline_mapping_v1.md"
    text = mapping.read_text(encoding="utf-8")
    assert "arXiv:2202.07106v2" in text
    assert "arXiv:2303.02576v3" in text
    assert "ABIDES adaptation" in text
    assert "not an exact reproduction" in text
